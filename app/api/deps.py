"""Dependency wiring: one service singleton shared by every request.

The FastAPI dependencies defined here are thin accessors over :class:`AppServices`,
the container built once by the lifespan in ``app/main.py``.  Keeping construction
out of the route functions has two benefits:

* the services layer stays importable and runnable without FastAPI (a plain
  ``build_services()`` call gives a CLI or script the exact same objects the API
  uses), and
* tests can inject fakes by replacing ``app.state.services`` before issuing a
  request, without patching import bindings inside the route module.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, Request

from app.config import (
    AUTHENTIC_CLASSES,
    AUTHENTIC_IMAGE_DIRNAME,
    AUTHENTIC_MASK_DIRNAME,
    AppSettings,
    load_settings,
)
from app.schemas import (
    ClassStatistics,
    EvaluationDatasetInfo,
    HealthResponse,
    ModelInfo,
    utc_iso,
)
from app.services import (
    AttributionService,
    BriefingService,
    DetectionService,
    DriftService,
    PipelineService,
    SpillSegmentationModel,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AppServices",
    "build_services",
    "evaluation_dataset_info",
    "health_report",
    "get_services",
    "get_settings",
    "get_model",
    "get_detection_service",
    "get_drift_service",
    "get_attribution_service",
    "get_pipeline_service",
]


@dataclass
class AppServices:
    """Singleton services for one application process."""

    settings: AppSettings
    model: SpillSegmentationModel
    detection: DetectionService
    drift: DriftService
    attribution: AttributionService
    briefing: BriefingService
    pipeline: PipelineService
    started_at_perf: float = field(default_factory=time.perf_counter)
    started_at_utc: str = field(default_factory=utc_iso)

    @property
    def uptime_seconds(self) -> float:
        """Wall-clock seconds since the services were built."""
        return round(time.perf_counter() - self.started_at_perf, 3)

    def close(self) -> None:
        """Release the loaded checkpoint (called on application shutdown)."""
        self.model.close()


def build_services(
    settings: Optional[AppSettings] = None,
    *,
    load_model: bool = True,
) -> AppServices:
    """Resolve settings and instantiate every service exactly once.

    Args:
        settings: Pre-resolved settings; loaded from ``config.yaml`` when omitted.
        load_model: Warm the segmentation checkpoint at startup.  The load is
            deliberately non-fatal: a failure is recorded on the model and
            reported by :func:`health_report` instead of preventing startup, so
            the drift/attribution endpoints and the docs stay usable without a
            checkpoint.

    Returns:
        The fully wired :class:`AppServices` container.
    """
    resolved = settings if settings is not None else load_settings()
    created = resolved.ensure_directories()
    logger.info("Writable artifact directories ready: %s", ", ".join(created))

    model = SpillSegmentationModel(resolved)
    if load_model:
        try:
            info = model.load()
        except Exception as exc:  # noqa: BLE001 - reported through /health
            logger.error("Segmentation checkpoint could not be loaded: %s", exc)
        else:
            logger.info(
                "Segmentation model ready on %s (%.0f ms, %s parameters)",
                info.device,
                info.load_duration_ms,
                f"{info.parameters:,}" if info.parameters else "unknown",
            )

    detection = DetectionService(resolved, model)
    drift = DriftService(resolved)
    attribution = AttributionService(resolved)
    briefing = BriefingService(resolved)
    logger.info(
        "Services ready (drift engine chosen per request: auto/lagrangian/opendrift; "
        "default AIS source: %s; briefing enabled=%s)",
        resolved.ais_csv or "none",
        resolved.briefing.enabled,
    )

    return AppServices(
        settings=resolved,
        model=model,
        detection=detection,
        drift=drift,
        attribution=attribution,
        briefing=briefing,
        pipeline=PipelineService(resolved, detection, drift, attribution, briefing),
    )


# --------------------------------------------------------------------- health
def evaluation_dataset_info(settings: AppSettings) -> EvaluationDatasetInfo:
    """Describe the unseen ``authentic_testing`` corpus used by the test suite."""
    root = settings.authentic_test_root
    if not root or not root.is_dir():
        return EvaluationDatasetInfo(available=False)

    statistics: List[ClassStatistics] = []
    scenes = 0
    for class_name in AUTHENTIC_CLASSES:
        image_dir = root / AUTHENTIC_IMAGE_DIRNAME / class_name
        if not image_dir.is_dir():
            image_dir = root / class_name / AUTHENTIC_IMAGE_DIRNAME
        if not image_dir.is_dir():
            matches = sorted(
                glob(str(root / class_name / "*" / AUTHENTIC_IMAGE_DIRNAME))
            )
            if not matches:
                matches = sorted(glob(str(root / "*" / AUTHENTIC_IMAGE_DIRNAME)))
            if not matches:
                continue
            image_dir = Path(matches[0])
        mask_dir = image_dir.parent / AUTHENTIC_MASK_DIRNAME
        if not mask_dir.is_dir():
            mask_dir = root / AUTHENTIC_MASK_DIRNAME / class_name
        if not mask_dir.is_dir():
            mask_dir = root / class_name / AUTHENTIC_MASK_DIRNAME
        images = [
            scene
            for pattern in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
            for scene in image_dir.glob(pattern)
        ]
        masks = (
            [
                scene
                for pattern in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
                for scene in mask_dir.glob(pattern)
            ]
            if mask_dir.is_dir()
            else []
        )
        scenes += len(images)
        statistics.append(
            ClassStatistics(
                class_name=class_name,
                scene_count=len(images),
                mask_count=len(masks),
                images_dir=str(image_dir),
                masks_dir=str(mask_dir) if mask_dir.is_dir() else "",
            )
        )
    return EvaluationDatasetInfo(
        available=True,
        root=str(root),
        class_statistics=statistics,
        total_scenes=scenes,
    )


def _metocean_summary(settings: AppSettings) -> Dict[str, Any]:
    """Report the metocean sources drift requests will draw on.

    Mirrors :meth:`app.services.drift_service.DriftService._resolve_field`:
    request samples win, then the configured manifest, then these constants - so
    an operator reading ``/health`` can tell whether a run used real ocean data
    or the analytical fallback field.
    """
    metocean = settings.metocean
    manifest = metocean.environment_json
    summary: Dict[str, Any] = {
        "environment_manifest": str(manifest) if manifest else None,
        "environment_manifest_available": bool(manifest and manifest.is_file()),
        "environment_manifest_samples": 0,
        "fallback_region": metocean.fallback_region,
        "fallback_current_mps": [
            metocean.fallback_current_u_mps,
            metocean.fallback_current_v_mps,
        ],
        "fallback_wind_mps": [metocean.fallback_wind_u_mps, metocean.fallback_wind_v_mps],
        "wind_drift_factor": metocean.wind_drift_factor,
        "horizontal_diffusivity_m2_s": metocean.horizontal_diffusivity,
        "hindcast_hours": metocean.fallback_hindcast_hours,
        "forecast_hours": metocean.fallback_forecast_hours,
        "weathering_model": metocean.weathering_model,
        "opendrift_installed": importlib.util.find_spec("opendrift") is not None,
        "adios_installed": importlib.util.find_spec("simple_adios") is not None,
    }
    if manifest and manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            samples = (
                payload.get("samples")
                or payload.get("time_series")
                or payload.get("metocean")
            )
            if isinstance(samples, (list, tuple)):
                summary["environment_manifest_samples"] = len(samples)
            elif isinstance(samples, dict):
                summary["environment_manifest_samples"] = len(samples)
            elif "u_current_mps" in payload:
                summary["environment_manifest_samples"] = 1
            try:
                if payload.get("wind_drift_factor") is not None:
                    summary["environment_manifest_wind_drift_factor"] = float(
                        payload["wind_drift_factor"]
                    )
            except (TypeError, ValueError):
                pass  # /health reports, it never rejects a run over a bad number
    if str(metocean.weathering_model).lower().startswith("adios") and not summary[
        "adios_installed"
    ]:
        summary["weathering_model_effective"] = "none"
    return summary


def health_report(services: AppServices) -> HealthResponse:
    """Build the readiness report from live service state (never raises)."""
    settings = services.settings
    model_info: ModelInfo = services.model.info()
    evaluation = evaluation_dataset_info(settings)

    metocean = _metocean_summary(settings)

    checks = {
        "checkpoint_present": model_info.ready,
        "output_dir_writable": _is_writable(settings.output_dir),
        "artifact_dir_writable": _is_writable(settings.artifact_dir),
        "ais_source_available": bool(settings.ais_csv and settings.ais_csv.is_file()),
        "evaluation_corpus_available": evaluation.available,
        "opendrift_available": metocean["opendrift_installed"],
        "metocean_manifest_available": metocean["environment_manifest_available"],
    }
    warnings: List[str] = []
    if model_info.error:
        warnings.append(f"Segmentation model unavailable: {model_info.error}")
    elif not model_info.ready:
        warnings.append(
            "Segmentation checkpoint is not loaded yet; it loads on the first "
            "detection request."
        )
    if not checks["ais_source_available"]:
        warnings.append(
            "No default AIS source configured; send ais_csv_path or ais_records "
            "with each attribution request."
        )
    if not evaluation.available:
        warnings.append("authentic_testing evaluation corpus not found.")
    if metocean["environment_manifest"] and not metocean["environment_manifest_available"]:
        warnings.append(
            f"Configured metocean manifest {metocean['environment_manifest']} is missing; "
            "drift runs fall back to the config constants."
        )
    elif not metocean["environment_manifest"]:
        warnings.append(
            "No metocean manifest is configured (metocean.environment_json); drift runs "
            "use the analytical config field unless samples accompany each request."
        )
    elif metocean["environment_manifest_samples"] == 1:
        warnings.append(
            "The metocean manifest holds a single uniform vector: the field is "
            "time-invariant, so hindcasts cannot resolve tidal or seasonal veering. "
            "Supply a 'samples' time series for operational accuracy."
        )
    if not metocean["opendrift_installed"]:
        warnings.append("OpenDrift is not installed; only the built-in engine can run.")
    if metocean.get("weathering_model_effective") == "none":
        warnings.append(
            "The ADIOS library is missing: OpenDrift runs with weathering disabled ('none')."
        )

    return HealthResponse(
        status="ok" if model_info.ready else "degraded",
        version=settings.server.version,
        uptime_seconds=services.uptime_seconds,
        model=model_info,
        paths=settings.describe(),
        evaluation_dataset=evaluation,
        checks=checks,
        metocean=metocean,
        warnings=warnings,
    )


def _is_writable(directory: Path) -> bool:
    """Best-effort writability probe used by the health checks."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    probe = directory / ".write-probe"
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        return True
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


# -------------------------------------------------------------- dependencies
def get_services(request: Request) -> AppServices:
    """Return the application-wide service container."""
    services = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - only if the lifespan never ran
        services = build_services()
        request.app.state.services = services
    return services


def get_settings(services: AppServices = Depends(get_services)) -> AppSettings:
    """Resolved ``config.yaml`` settings."""
    return services.settings


def get_model(services: AppServices = Depends(get_services)) -> SpillSegmentationModel:
    """Loaded (or lazily loadable) segmentation model."""
    return services.model


def get_detection_service(
    services: AppServices = Depends(get_services),
) -> DetectionService:
    """SAR slick-detection service."""
    return services.detection


def get_drift_service(services: AppServices = Depends(get_services)) -> DriftService:
    """Bidirectional (hindcast + forecast) drift service."""
    return services.drift


def get_attribution_service(
    services: AppServices = Depends(get_services),
) -> AttributionService:
    """AIS vessel-attribution service."""
    return services.attribution


def get_briefing_service(
    services: AppServices = Depends(get_services),
) -> BriefingService:
    """Optional Gemini briefing service (never blocks the pipeline)."""
    return services.briefing


def get_pipeline_service(
    services: AppServices = Depends(get_services),
) -> PipelineService:
    """End-to-end orchestration service."""
    return services.pipeline

