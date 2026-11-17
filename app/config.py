"""Typed application settings resolved from ``config.yaml`` + environment.

Design notes
------------
* ``config.yaml`` remains the single operational source of truth.  Every
  numeric default used by the API also exists in the YAML file; the Python
  defaults below are only safety nets so the service still boots when a key
  is absent (mirroring ``src/run_pipeline.py`` semantics).
* Environment variables (prefixed ``OILSPILL_``) override individual paths so
  the same image can be deployed against a different model/data volume
  without editing the YAML file.
* ``config_path`` is honoured through ``PIPELINE_CONFIG`` (the variable the
  legacy ``backend/main.py`` already used).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from app import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Default configuration file name (resolved against the project root).
DEFAULT_CONFIG_FILENAME = "config.yaml"

#: Recognised directory names for the unseen-data evaluation corpus.  The
#: shipped corpus is ``data/authentic testing/02_Test_images_and_ground_truth``
#: but deployments may flatten it, so the resolver probes several spellings.
AUTHENTIC_TEST_CANDIDATES: Sequence[str] = (
    "data/authentic testing",
    "data/authentic_testing",
    "data/authentic-testing",
    "authentic testing",
    "authentic_testing",
    "authentic-testing",
)

#: Class sub-directory names used by the evaluation corpus.
AUTHENTIC_CLASSES: Sequence[str] = ("Oil", "No oil", "Lookalike")

#: Sub-directory names holding the scenes and their ground-truth masks.
AUTHENTIC_IMAGE_DIRNAME = "Images"
AUTHENTIC_MASK_DIRNAME = "Mask"


def _env_str(name: str) -> Optional[str]:
    """Return a non-empty environment variable value or ``None``."""
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, falling back to ``default``."""
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer for %s=%r", name, raw)
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable, falling back to ``default``."""
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float for %s=%r", name, raw)
        return default


def _upload_limit_bytes(api_cfg: Dict[str, Any]) -> int:
    """Resolve the upload cap from env, YAML, or the 200 MiB safety default.

    Env var ``OILSPILL_MAX_UPLOAD_BYTES`` wins (absolute bytes); otherwise
    ``config.yaml -> api.max_upload_mb`` is honoured.  A non-positive result
    would disable every upload endpoint, so it is clamped back to the default.
    """
    env_raw = _env_str("OILSPILL_MAX_UPLOAD_BYTES")
    if env_raw is not None:
        try:
            limit = int(env_raw)
        except ValueError:
            logger.warning("Ignoring invalid integer for OILSPILL_MAX_UPLOAD_BYTES=%r", env_raw)
            limit = 0
        if limit > 0:
            return limit
    raw = api_cfg.get("max_upload_mb", 200)
    try:
        limit = int(float(raw) * 1024 * 1024)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid api.max_upload_mb=%r; using 200 MiB", raw)
        return ServerSettings.max_upload_bytes
    if limit <= 0:
        logger.warning("Ignoring non-positive api.max_upload_mb=%r; using 200 MiB", raw)
        return ServerSettings.max_upload_bytes
    return limit


def _as_dict(section: Any) -> Dict[str, Any]:
    """Coerce a YAML section to a dict (``None``/scalars become ``{}``)."""
    return dict(section) if isinstance(section, dict) else {}


def _resolve_path(value: Any, base: Path, default: Optional[Path] = None) -> Path:
    """Resolve a possibly-relative configured path against ``base``."""
    if value in (None, ""):
        if default is None:
            raise ValueError("No path value and no default supplied")
        return Path(default)
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate)



# --------------------------------------------------------------------------- #
#  Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSettings:
    """Architecture and class-label contract of the trained checkpoint."""

    encoder: str = "resnet34"
    architecture: str = "deeplabv3plus"
    in_channels: int = 3
    classes: int = 5
    class_names: List[str] = field(
        default_factory=lambda: ["sea_surface", "oil_spill", "look_alike", "ship", "land"]
    )
    background_class_index: int = 0
    oil_class_index: int = 1
    lookalike_class_index: int = 2
    ship_class_index: int = 3
    land_class_index: int = 4
    input_size: List[int] = field(default_factory=lambda: [512, 512])
    confidence_threshold: float = 0.50

    @property
    def oil_class_name(self) -> str:
        """Name of the target class as declared by the checkpoint."""
        if 0 <= self.oil_class_index < len(self.class_names):
            return self.class_names[self.oil_class_index]
        return "oil_spill"


@dataclass(frozen=True)
class AgeSettings:
    """Heuristic spill-age inversion (Richardson turbulent spreading law)."""

    #: Fallback turbulent diffusivity (m^2/s) when no metocean field is given.
    diffusivity_m2_s: float = 10.0
    min_age_hours: float = 1.0
    max_age_hours: float = 240.0
    #: Confidence floor/ceiling of the returned estimate.
    min_confidence: float = 0.10
    max_confidence: float = 0.90
    enabled: bool = True


@dataclass(frozen=True)
class InferenceSettings:
    """Sliding-window inference profile."""

    tile_size: int = 256
    stride: int = 128
    #: Tiles bundled into a single forward pass (batched CPU inference is
    #: ~1.5x faster per tile than one-at-a-time on the reference hardware).
    tile_batch_size: int = 4
    #: Integer downscaling factor applied before inference (1 = native).
    downscale: int = 1
    min_spill_area_pixels: int = 50
    crs: str = "EPSG:4326"
    #: Normalisation mode for SAR input: "per_tile" (training-matched) or
    #: "scene" (whole-scene min-max).  Mirrors the config.yaml key
    #: ``inference.normalisation``; ``OILSPILL_NORMALISATION`` env var wins.
    normalisation: str = "per_tile"
    #: Morphological speckle suppression applied to the oil mask.
    morphological_cleanup: bool = True
    #: Elliptical speckle-suppression kernel edge length (odd int >= 1).
    #: Tunable via ``config.yaml -> inference.morphology_kernel_size`` or
    #: ``OILSPILL_MORPH_KERNEL``; mirrors ``src/detect_and_vectorize`` (3x3).
    morphology_kernel_size: int = 3
    #: Contour simplification epsilon as a fraction of the contour perimeter.
    #: Tunable via ``config.yaml -> inference.contour_epsilon_fraction`` or
    #: ``OILSPILL_CONTOUR_EPSILON``; mirrors ``src/detect_and_vectorize`` (0.005).
    contour_epsilon_fraction: float = 0.005
    save_artifacts: bool = True
    #: Longest edge of the base64 PNG preview returned with detections.
    preview_max_edge: int = 512
    #: Highest confidence threshold accepted from a request.
    confidence_bound: float = 1.0
    #: 5-class post-processing thresholds (config.yaml -> inference).
    min_slick_area_km2: float = 0.05
    ship_buffer_km: float = 2.0

    def effective_stride(self) -> int:
        """Stride actually usable for the configured tile size."""
        return max(1, min(self.stride, self.tile_size))

    def validate(self) -> None:
        """Raise :class:`ValueError` for a self-inconsistent profile."""
        if self.tile_size < 8:
            raise ValueError(f"tile_size must be >= 8, got {self.tile_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.tile_batch_size < 1:
            raise ValueError(f"tile_batch_size must be >= 1, got {self.tile_batch_size}")
        if self.downscale < 1:
            raise ValueError(f"downscale must be >= 1, got {self.downscale}")
        if self.min_spill_area_pixels < 0:
            raise ValueError("min_spill_area_pixels must be >= 0")
        if self.morphology_kernel_size < 1 or self.morphology_kernel_size % 2 == 0:
            raise ValueError(
                f"morphology_kernel_size must be an odd int >= 1, "
                f"got {self.morphology_kernel_size}"
            )
        if not 0.0 <= self.contour_epsilon_fraction <= 0.1:
            raise ValueError(
                "contour_epsilon_fraction must be within [0, 0.1], "
                f"got {self.contour_epsilon_fraction}"
            )
        if self.normalisation not in ("per_tile", "scene"):
            raise ValueError(
                f"normalisation must be 'per_tile' or 'scene', got {self.normalisation}"
            )



@dataclass(frozen=True)
class DetectionSettings:
    """Detection-stage defaults and radiometric validation thresholds."""

    #: Ground sample distance assumed for non-georeferenced scenes
    #: (Sentinel-1 IW GRD ~ 10 m).  Used for pixel -> metric conversion.
    pixel_spacing_m: float = 10.0
    #: Reject scenes smaller than this many pixels per side.
    min_scene_dim: int = 32
    #: Reject constant/blank rasters whose dynamic range is below this.
    min_dynamic_range: float = 1e-6
    #: When True, an optical-looking RGB scene is rejected with HTTP 400;
    #: when False it only raises a warning in the response envelope.
    strict_radiometric_validation: bool = False
    age: AgeSettings = field(default_factory=AgeSettings)


@dataclass(frozen=True)
class LookalikeFilterSettings:
    """Post-processing lookalike filtering configuration.

    Geometric, textural, and contextual heuristics to distinguish true petroleum
    discharges from common false-positive lookalikes (biogenic films, wind slicks,
    atmospheric artifacts, land boundary distortions).
    """

    # Geometry-based heuristics
    min_confirmed_area_px: float = 200.0
    max_confirmed_elongation: float = 5.0
    min_confirmed_solidity: float = 0.5

    # Boundary gradient heuristics
    min_edge_gradient_magnitude: float = 10.0
    edge_sample_count: int = 32

    # Spatial coherence & morphology
    max_confirmed_bbox_fill_fraction: float = 0.8
    max_confirmed_component_count: int = 2

    # Confidence scoring
    min_confident_scene_pixels: int = 2_000_000
    max_confirmed_oil_coverage_fraction: float = 0.05

    # Metadata context
    detect_lookalike_dataset_paths: bool = True

    # Overall sensitivity
    heuristic_threshold_count: int = 2
    use_lookalike_status: bool = True
    enabled: bool = True


@dataclass(frozen=True)
class MetoceanSettings:
    """Physics parameters and the config-driven fallback drift field."""

    wind_drift_factor: float = 0.032
    horizontal_diffusivity: float = 10.0
    weathering_model: str = "adios"
    confidence_level: float = 0.95
    oil_type: str = "GENERIC MEDIUM CRUDE"
    #: Optional per-incident metocean manifest (JSON).  When present and readable
    #: it replaces the constants below for built-in-engine runs, and may carry a
    #: whole time series instead of one uniform vector.
    environment_json: Optional[Path] = None
    #: Anchor region used when a scene carries no georeferencing.
    fallback_region: str = "Persian Gulf TSS"
    fallback_hindcast_hours: float = 4.0
    fallback_forecast_hours: float = 24.0
    fallback_current_u_mps: float = 0.22
    fallback_current_v_mps: float = 0.14
    fallback_wind_u_mps: float = 5.20
    fallback_wind_v_mps: float = 3.10
    particles: int = 2000
    time_step_minutes: int = 15
    random_seed: int = 42
    max_steps: int = 20_000
    max_endpoint_samples: int = 200
    #: Seed-region dilation (degrees) used when the source degenerates to a point.
    point_seed_buffer_deg: float = 0.01
    #: Live metocean fetch controls (Copernicus currents + ERA5 winds).
    fetch_timeout_sec: float = 25.0
    #: Server-side NetCDF forcing paths used by OpenDrift when live fetch is off.
    current_nc: Optional[Path] = None
    wind_nc: Optional[Path] = None

    def fallback_sample(self) -> Dict[str, float]:
        """Uniform fallback metocean sample (u/v current and wind, m/s)."""
        return {
            "u_current_mps": self.fallback_current_u_mps,
            "v_current_mps": self.fallback_current_v_mps,
            "u_wind_mps": self.fallback_wind_u_mps,
            "v_wind_mps": self.fallback_wind_v_mps,
        }


@dataclass(frozen=True)
class BriefingSettings:
    """Optional LLM synthesis layer (Gemini 2.5 Flash) for dossier narration."""

    enabled: bool = True
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    timeout_sec: float = 30.0
    #: Environment variable carrying the Gemini API key.
    api_key_env: str = "GEMINI_API_KEY"


@dataclass(frozen=True)
class AisSettings:
    """AIS ingestion / pruning profile."""

    time_window_hours: float = 6.0
    buffer_distance_m: float = 5000.0
    chunksize: int = 250_000
    metric_crs: str = "EPSG:3857"
    geographic_crs: str = "EPSG:4326"
    #: Minimum uncertainty ellipse accepted by the pruner (meters).
    min_ellipse_semi_major_m: float = 500.0
    min_ellipse_semi_minor_m: float = 250.0
    #: Default AIS CSV path used when a request doesn't specify one and the
    #: scene is geographic.  Resolved against the project root.
    default_csv: Optional[str] = None
    #: Hard-block: never fabricate AIS vessel tracks when real data has no
    #: spatio-temporal overlap with the drift envelope.  When true, FastAPI
    #: startup raises RuntimeError to fail-closed.
    allow_synthetic_ais: bool = False


@dataclass(frozen=True)
class ServerSettings:
    """HTTP surface configuration."""

    title: str = "Marine Oil Spill Detection & Vessel Attribution API"
    version: str = "1.0.0"
    description: str = (
        "FastAPI orchestration of the trained 3-class SAR U-Net, Lagrangian "
        "spill-drift modelling and multi-factor AIS vessel attribution."
    )
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    #: Maximum accepted upload size in bytes (guards against unbounded
    #: in-memory reads).  Mirrors ``OILSPILL_MAX_UPLOAD_BYTES`` and
    #: ``config.yaml -> api.max_upload_mb``.
    max_upload_bytes: int = 200 * 1024 * 1024

    @property
    def base_url(self) -> str:
        """URL used when composing artifact links."""
        host = self.host if self.host not in ("0.0.0.0", "::") else "localhost"
        return f"http://{host}:{self.port}"



# --------------------------------------------------------------------------- #
#  Aggregate settings object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppSettings:
    """Fully resolved, immutable application settings."""

    project_root: Path
    config_path: Path
    config: Dict[str, Any]
    model_checkpoint: Path
    output_dir: Path
    ais_csv: Optional[Path]
    authentic_test_root: Optional[Path]
    spill_geojson: Path
    culprit_geojson: Path
    device: str
    model: ModelSettings
    inference: InferenceSettings
    detection: DetectionSettings
    lookalike_filter: LookalikeFilterSettings
    metocean: MetoceanSettings
    ais: AisSettings
    server: ServerSettings
    briefing: BriefingSettings = field(default_factory=BriefingSettings)
    attribution: Dict[str, Any] = field(default_factory=dict)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def artifact_dir(self) -> Path:
        """Directory holding API-generated artifacts (mask PNG, GeoJSON)."""
        return self.output_dir / "api"

    def ensure_directories(self) -> List[str]:
        """Create the writable directories the API needs; return their paths."""
        created: List[str] = []
        for directory in (self.output_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory))
        return created

    def describe(self) -> Dict[str, Any]:
        """Compact, JSON-safe summary used by ``GET /health``."""
        return {
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "model_checkpoint": str(self.model_checkpoint),
            "model_checkpoint_exists": self.model_checkpoint.is_file(),
            "output_dir": str(self.output_dir),
            "authentic_test_root": (
                str(self.authentic_test_root) if self.authentic_test_root else None
            ),
            "authentic_test_exists": bool(
                self.authentic_test_root and self.authentic_test_root.is_dir()
            ),
            "ais_csv": str(self.ais_csv) if self.ais_csv else None,
            "device": self.device,
            "classes": list(self.model.class_names),
            "regions": [region.get("name") for region in self.regions],
        }


def resolve_authentic_test_root(
    project_root: Path, configured: Optional[Any] = None
) -> Optional[Path]:
    """Locate the unseen evaluation corpus (``authentic_testing``).

    Resolution order:

    1. an explicit ``configured`` path (config or env) if it exists,
    2. the well-known candidate spellings under the project root,
    3. any ``*authentic*`` directory under ``<root>/data`` that contains the
       ``Images``/``Mask`` (or nested ``*/Images``) class layout.

    Returns:
        The directory that directly contains the class folders, or ``None``
        when nothing usable is present (the API then reports the evaluation
        dataset as unavailable instead of failing at startup).
    """
    if configured not in (None, ""):
        resolved = _find_corpus_root(_resolve_path(configured, project_root))
        if resolved is not None:
            return resolved

    for relative in AUTHENTIC_TEST_CANDIDATES:
        resolved = _find_corpus_root(project_root / relative)
        if resolved is not None:
            return resolved

    data_dir = project_root / "data"
    if data_dir.is_dir():
        for child in sorted(data_dir.iterdir()):
            if child.is_dir() and "authentic" in child.name.casefold():
                resolved = _find_corpus_root(child)
                if resolved is not None:
                    return resolved
    return None


def _find_corpus_root(candidate: Path) -> Optional[Path]:
    """Return the directory holding ``Images``/``Mask`` under ``candidate``.

    Handles both the flat layout (``<root>/Images`` + ``<root>/Mask``) and the
    nested layout shipped with the project
    (``<root>/02_Test_images_and_ground_truth/Images``).
    """
    if not candidate.is_dir():
        return None
    if (candidate / AUTHENTIC_IMAGE_DIRNAME).is_dir() and (
        candidate / AUTHENTIC_MASK_DIRNAME
    ).is_dir():
        return candidate
    for child in sorted(candidate.iterdir()):
        if not child.is_dir():
            continue
        if (child / AUTHENTIC_IMAGE_DIRNAME).is_dir() and (
            child / AUTHENTIC_MASK_DIRNAME
        ).is_dir():
            return child
    return None


# --------------------------------------------------------------------------- #
#  Loader
# --------------------------------------------------------------------------- #
def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file; absence/malformation degrades to ``{}``.

    Mirrors the tolerant behaviour of ``src/run_pipeline.py``: configuration is
    advisory and must never prevent the service from starting.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        logger.warning("No configuration file at %s; using built-in defaults.", path)
        return {}
    except Exception as exc:  # pragma: no cover - malformed YAML
        logger.error("Could not parse config %s (%s); using built-in defaults.", path, exc)
        return {}


def _optional_path(value: Any, base: Path) -> Optional[Path]:
    """Resolve an optional path; ``None`` when unset."""
    if value in (None, ""):
        return None
    return _resolve_path(value, base)


def load_settings(
    config_path: Optional[str | Path] = None,
    project_root: Optional[str | Path] = None,
) -> AppSettings:
    """Build :class:`AppSettings` from ``config.yaml`` + env overrides.

    Args:
        config_path: Explicit config path; defaults to ``PIPELINE_CONFIG`` and
            then to ``<project_root>/config.yaml``.
        project_root: Service root; defaults to the parent of the ``app``
            package (``SIH26143``).

    Returns:
        An immutable :class:`AppSettings` instance.

    Raises:
        ValueError: If the configured inference profile is invalid.
    """
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT

    resolved_config_path = Path(
        _env_str("PIPELINE_CONFIG") or config_path or (root / DEFAULT_CONFIG_FILENAME)
    )
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path

    config = load_yaml_config(resolved_config_path)
    paths_cfg = _as_dict(config.get("paths"))
    model_cfg = _as_dict(config.get("model"))
    inference_cfg = _as_dict(config.get("inference"))
    detection_cfg = _as_dict(config.get("detection"))
    age_cfg = _as_dict(detection_cfg.get("age"))
    metocean_cfg = _as_dict(config.get("metocean"))
    ais_cfg = _as_dict(config.get("ais"))
    api_cfg = _as_dict(config.get("api"))
    backend_cfg = _as_dict(config.get("backend"))
    attribution_cfg = _as_dict(config.get("attribution"))
    briefing_cfg = _as_dict(config.get("briefing"))

    class_names = list(
        model_cfg.get("class_names") or ["background", "oil_spill", "lookalike"]
    )

    model = ModelSettings(
        architecture=str(model_cfg.get("architecture", "deeplabv3plus")),
        encoder=str(model_cfg.get("encoder", "resnet34")),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", len(class_names))),
        class_names=class_names,
        background_class_index=int(model_cfg.get("background_class_index", 0)),
        oil_class_index=int(model_cfg.get("oil_class_index", 1)),
        lookalike_class_index=int(model_cfg.get("lookalike_class_index", 2)),
        ship_class_index=int(model_cfg.get("ship_class_index", 3)),
        land_class_index=int(model_cfg.get("land_class_index", 4)),
        input_size=[int(v) for v in (model_cfg.get("input_size") or [512, 512])],
        confidence_threshold=float(model_cfg.get("confidence_threshold", 0.5)),
    )

    inference = InferenceSettings(
        tile_size=int(inference_cfg.get("tile_size", model.input_size[0])),
        stride=int(inference_cfg.get("stride", model.input_size[0] // 2)),
        tile_batch_size=_env_int(
            "OILSPILL_TILE_BATCH", int(inference_cfg.get("tile_batch_size", 4))
        ),
        downscale=int(inference_cfg.get("downscale", 1)),
        min_spill_area_pixels=int(inference_cfg.get("min_spill_area_pixels", 50)),
        crs=str(inference_cfg.get("crs", "EPSG:4326")),
        normalisation=str(
            _env_str("OILSPILL_NORMALISATION")
            or inference_cfg.get("normalisation", "per_tile")
        ),
        morphological_cleanup=bool(inference_cfg.get("morphological_cleanup", True)),
        morphology_kernel_size=_env_int(
            "OILSPILL_MORPH_KERNEL",
            int(inference_cfg.get("morphology_kernel_size", 3)),
        ),
        contour_epsilon_fraction=_env_float(
            "OILSPILL_CONTOUR_EPSILON",
            float(inference_cfg.get("contour_epsilon_fraction", 0.005)),
        ),
        save_artifacts=bool(inference_cfg.get("save_artifacts", True)),
        preview_max_edge=int(inference_cfg.get("preview_max_edge", 512)),
        min_slick_area_km2=float(inference_cfg.get("min_slick_area_km2", 0.05)),
        ship_buffer_km=float(inference_cfg.get("ship_buffer_km", 2.0)),
    )
    inference.validate()


    detection = DetectionSettings(
        pixel_spacing_m=float(
            detection_cfg.get("pixel_spacing_m", inference_cfg.get("pixel_spacing_m", 10.0))
        ),
        min_scene_dim=int(detection_cfg.get("min_scene_dim", 32)),
        min_dynamic_range=float(detection_cfg.get("min_dynamic_range", 1e-6)),
        strict_radiometric_validation=bool(
            detection_cfg.get("strict_radiometric_validation", False)
        ),
        age=AgeSettings(
            diffusivity_m2_s=float(
                age_cfg.get(
                    "diffusivity_m2_s", metocean_cfg.get("horizontal_diffusivity", 10.0)
                )
            ),
            min_age_hours=float(age_cfg.get("min_age_hours", 1.0)),
            max_age_hours=float(age_cfg.get("max_age_hours", 240.0)),
            min_confidence=float(age_cfg.get("min_confidence", 0.10)),
            max_confidence=float(age_cfg.get("max_confidence", 0.90)),
            enabled=bool(age_cfg.get("enabled", True)),
        ),
    )

    lookalike_filter_cfg = _as_dict(config.get("lookalike_filter"))
    lookalike_filter = LookalikeFilterSettings(
        min_confirmed_area_px=float(
            lookalike_filter_cfg.get("min_confirmed_area_px", 200.0)
        ),
        max_confirmed_elongation=float(
            lookalike_filter_cfg.get("max_confirmed_elongation", 5.0)
        ),
        min_confirmed_solidity=float(
            lookalike_filter_cfg.get("min_confirmed_solidity", 0.5)
        ),
        min_edge_gradient_magnitude=float(
            lookalike_filter_cfg.get("min_edge_gradient_magnitude", 10.0)
        ),
        edge_sample_count=int(lookalike_filter_cfg.get("edge_sample_count", 32)),
        max_confirmed_bbox_fill_fraction=float(
            lookalike_filter_cfg.get("max_confirmed_bbox_fill_fraction", 0.8)
        ),
        max_confirmed_component_count=int(
            lookalike_filter_cfg.get("max_confirmed_component_count", 2)
        ),
        min_confident_scene_pixels=int(
            lookalike_filter_cfg.get("min_confident_scene_pixels", 2_000_000)
        ),
        max_confirmed_oil_coverage_fraction=float(
            lookalike_filter_cfg.get("max_confirmed_oil_coverage_fraction", 0.05)
        ),
        detect_lookalike_dataset_paths=bool(
            lookalike_filter_cfg.get("detect_lookalike_dataset_paths", True)
        ),
        heuristic_threshold_count=int(
            lookalike_filter_cfg.get("heuristic_threshold_count", 2)
        ),
        use_lookalike_status=bool(
            lookalike_filter_cfg.get("use_lookalike_status", True)
        ),
        enabled=bool(lookalike_filter_cfg.get("enabled", True)),
    )

    metocean = MetoceanSettings(
        wind_drift_factor=float(metocean_cfg.get("wind_drift_factor", 0.032)),
        horizontal_diffusivity=float(metocean_cfg.get("horizontal_diffusivity", 10.0)),
        weathering_model=str(metocean_cfg.get("weathering_model", "adios")),
        confidence_level=float(metocean_cfg.get("confidence_level", 0.95)),
        oil_type=str(metocean_cfg.get("oil_type", "GENERIC MEDIUM CRUDE")),
        fallback_region=str(
            metocean_cfg.get("fallback_region")
            or _as_dict(config.get("metadata")).get("fallback_region", "Persian Gulf TSS")
        ),
        fallback_hindcast_hours=float(metocean_cfg.get("fallback_hindcast_hours", 4.0)),
        fallback_forecast_hours=float(metocean_cfg.get("fallback_forecast_hours", 24.0)),
        fallback_current_u_mps=float(metocean_cfg.get("fallback_current_u_mps", 0.22)),
        fallback_current_v_mps=float(metocean_cfg.get("fallback_current_v_mps", 0.14)),
        fallback_wind_u_mps=float(metocean_cfg.get("fallback_wind_u_mps", 5.20)),
        fallback_wind_v_mps=float(metocean_cfg.get("fallback_wind_v_mps", 3.10)),
        particles=int(metocean_cfg.get("particles", 2000)),
        time_step_minutes=int(metocean_cfg.get("time_step_minutes", 15)),
        random_seed=int(metocean_cfg.get("random_seed", 42)),
        max_steps=_env_int(
            "OILSPILL_MAX_STEPS", int(metocean_cfg.get("max_steps", 20_000))
        ),
        max_endpoint_samples=_env_int(
            "OILSPILL_MAX_ENDPOINTS",
            int(metocean_cfg.get("max_endpoint_samples", 200)),
        ),
        point_seed_buffer_deg=float(
            # ``release_seed_buffer_deg`` is the pre-rename key; keep honouring it
            # so existing deployments keep their value after the rename.
            metocean_cfg.get(
                "point_seed_buffer_deg", metocean_cfg.get("release_seed_buffer_deg", 0.01)
            )
        ),
        fetch_timeout_sec=_env_float(
            "OILSPILL_METOCEAN_FETCH_TIMEOUT_SEC",
            float(metocean_cfg.get("fetch_timeout_sec", 25.0)),
        ),
        current_nc=_optional_path(
            metocean_cfg.get("current_nc"), root
        ),
        wind_nc=_optional_path(
            metocean_cfg.get("wind_nc"), root
        ),
        environment_json=_optional_path(
            _env_str("OILSPILL_METOCEAN_JSON") or metocean_cfg.get("environment_json"), root
        ),
    )

    ais = AisSettings(
        time_window_hours=float(ais_cfg.get("time_window_hours", 6.0)),
        buffer_distance_m=float(ais_cfg.get("buffer_distance_m", 5000.0)),
        chunksize=int(ais_cfg.get("chunksize", 250_000)),
        metric_crs=str(ais_cfg.get("metric_crs", "EPSG:3857")),
        geographic_crs=str(ais_cfg.get("geographic_crs", "EPSG:4326")),
        min_ellipse_semi_major_m=float(ais_cfg.get("min_ellipse_semi_major_m", 500.0)),
        min_ellipse_semi_minor_m=float(ais_cfg.get("min_ellipse_semi_minor_m", 250.0)),
        default_csv=_optional_path(
            _env_str("OILSPILL_AIS_DEFAULT_CSV") or ais_cfg.get("default_csv"), root
        ),
    )

    briefing = BriefingSettings(
        enabled=bool(briefing_cfg.get("enabled", True)),
        model=str(briefing_cfg.get("model", "gemini-2.5-flash")),
        temperature=float(briefing_cfg.get("temperature", 0.1)),
        timeout_sec=_env_float(
            "OILSPILL_BRIEFING_TIMEOUT_SEC",
            float(briefing_cfg.get("timeout_sec", 30.0)),
        ),
        api_key_env=str(briefing_cfg.get("api_key_env", "GEMINI_API_KEY")),
    )

    cors_raw = api_cfg.get("cors_origins")
    if isinstance(cors_raw, str):
        cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()]
    elif isinstance(cors_raw, (list, tuple)):
        cors_origins = [str(item) for item in cors_raw]
    else:
        cors_origins = ["*"]
    env_cors = _env_str("OILSPILL_CORS_ORIGINS")
    if env_cors:
        cors_origins = [item.strip() for item in env_cors.split(",") if item.strip()]

    server = ServerSettings(
        title=str(api_cfg.get("title", ServerSettings.title)),
        version=str(api_cfg.get("version", ServerSettings.version)),
        description=str(api_cfg.get("description", ServerSettings.description)),
        host=str(api_cfg.get("host", "0.0.0.0")),
        port=_env_int(
            "OILSPILL_PORT", int(api_cfg.get("port", backend_cfg.get("port", 8000)))
        ),
        cors_origins=cors_origins,
        max_upload_bytes=_upload_limit_bytes(api_cfg),
    )

    return AppSettings(
        project_root=root,
        config_path=resolved_config_path,
        config=config,
        model_checkpoint=_resolve_path(
            _env_str("OILSPILL_MODEL_PATH")
            or paths_cfg.get("model_checkpoint", "models/oil_spill_segmentation_model.pth"),
            root,
        ),
        output_dir=_resolve_path(
            _env_str("OILSPILL_OUTPUT_DIR") or paths_cfg.get("output_dir", "results"), root
        ),
        ais_csv=_optional_path(
            _env_str("OILSPILL_AIS_CSV") or paths_cfg.get("ais_csv"), root
        ),
        authentic_test_root=resolve_authentic_test_root(
            root,
            _env_str("OILSPILL_AUTHENTIC_TEST_DIR") or paths_cfg.get("authentic_test_dir"),
        ),
        spill_geojson=_resolve_path(
            paths_cfg.get("spill_geojson", "results/spill_polygon.geojson"), root
        ),
        culprit_geojson=_resolve_path(
            paths_cfg.get("culprit_geojson", "results/culprit_vessel.geojson"), root
        ),
        device=(_env_str("OILSPILL_DEVICE") or "auto").lower(),
        model=model,
        inference=inference,
        detection=detection,
        lookalike_filter=lookalike_filter,
        metocean=metocean,
        ais=ais,
        briefing=briefing,
        server=server,
        attribution=attribution_cfg,
        regions=list(config.get("regions") or []),
        validation=_as_dict(config.get("validation")),
    )

