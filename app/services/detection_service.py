"""Oil-spill detection, vectorisation and characterisation.

Stage logic:
1. Post-process probability maps (argmax label map, confidence thresholding,
   and morphological opening/closing).
2. Vectorise external contours via cv2.findContours & cv2.approxPolyDP to derive
   slick polygons with metric properties.
3. Geolocate via GeoTIFF affine transforms to compute geodesic areas.
4. Estimate age via age_estimator.
5. Classify false positives using LookalikeFilter with folder-boundary discrimination
   (No Oil vs. Lookalike vs. Confirmed Spill).
6. Export artifacts (spill GeoJSON, mask PNG, base64 preview).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping

try:
    from rasterio.transform import Affine as RasterioAffine
    from rasterio.transform import xy as rasterio_xy
except ImportError:
    rasterio_xy = None
    RasterioAffine = None

from app.config import AppSettings
from app.errors import InvalidInputError
from app.schemas import (
    DetectionOptions,
    SceneMetadata,
    SlickDetectionResponse,
    SlickGeometry,
    SlickStatistics,
)
from app.services import image_io
from app.services.age_estimator import estimate_spill_age
from app.services.geo import geodesic_area_m2
from app.services.image_io import SceneArray
from app.services.lookalike_filter import LookalikeFilter
from app.services.model_service import SpillSegmentationModel
from src.detect_and_vectorize import characterize_geometry

logger = logging.getLogger(__name__)

APPROX_EPSILON_FRACTION = 0.005


def as_affine(transform: Any) -> Any:
    """Coerce stored geotransform into an Affine object compatible with rasterio."""
    if transform is None or RasterioAffine is None:
        return None
    if isinstance(transform, (list, tuple)):
        if len(transform) != 6:
            return None
        try:
            return RasterioAffine(*(float(value) for value in transform))
        except (TypeError, ValueError):
            return None
    return transform


@dataclass
class DetectionOutcome:
    """Detection artifacts containing typed schemas and working memory arrays."""

    response: SlickDetectionResponse
    scene: SceneArray
    mask: np.ndarray
    label_map: np.ndarray
    oil_confidence: np.ndarray
    geojson: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    lookalike: Optional[Dict[str, Any]] = None
    is_lookalike: bool = False

    @property
    def has_spill(self) -> bool:
        """True when slicks are detected and it is confirmed as an actual spill."""
        return len(self.response.slicks) > 0 and self.response.status in (
            "Confirmed Spill",
            "Confirmed",
            "Suspected",
        )

    @property
    def total_oil_pixels(self) -> int:
        """Total pixels classified as oil in the segmented mask."""
        return int(self.response.statistics.oil_pixel_count)

    @property
    def primary_centroid_lonlat(self) -> Optional[Tuple[float, float]]:
        """Largest slick polygon centroid in (lon, lat) space."""
        if not self.response.slicks:
            return None
        slick = self.response.slicks[0]
        if slick.centroid_lon is None or slick.centroid_lat is None:
            return None
        return float(slick.centroid_lon), float(slick.centroid_lat)


@dataclass
class ResolvedDetectionParams:
    """Detection parameters after merging overrides with config.yaml defaults."""

    confidence_threshold: float
    tile_size: int
    stride: int
    tile_batch_size: int
    min_area_px: int
    downscale: int
    save_artifacts: bool
    include_preview: bool
    pixel_spacing_m: float
    acquisition_time_utc: Optional[str]


class DetectionService:
    """Executes segmentation models and performs geometric/false-positive analysis."""

    def __init__(self, settings: AppSettings, model: SpillSegmentationModel):
        self.settings = settings
        self.model = model
        self.lookalike_filter = (
            LookalikeFilter(settings.lookalike_filter)
            if settings.lookalike_filter.enabled
            else None
        )

    def resolve_params(self, options: Optional[DetectionOptions] = None) -> ResolvedDetectionParams:
        """Resolve and validate configuration fallbacks against runtime request options."""
        inference = self.settings.inference
        detection = self.settings.detection
        options = options or DetectionOptions()

        tile_size = int(options.tile_size or inference.tile_size)
        stride = int(options.stride or min(inference.effective_stride(), tile_size))
        if stride > tile_size:
            raise InvalidInputError(
                f"stride ({stride}) must not exceed tile_size ({tile_size}).",
                details={"stride": stride, "tile_size": tile_size},
            )
        threshold = float(
            options.confidence_threshold
            if options.confidence_threshold is not None
            else self.settings.model.confidence_threshold
        )
        if not 0.0 < threshold <= 1.0:
            raise InvalidInputError(
                "confidence_threshold must lie in (0, 1].", details={"value": threshold}
            )

        return ResolvedDetectionParams(
            confidence_threshold=threshold,
            tile_size=tile_size,
            stride=stride,
            tile_batch_size=int(inference.tile_batch_size),
            min_area_px=int(
                options.min_area_px
                if options.min_area_px is not None
                else inference.min_spill_area_pixels
            ),
            downscale=int(options.downscale or inference.downscale),
            save_artifacts=bool(
                inference.save_artifacts
                if options.save_artifacts is None
                else options.save_artifacts
            ),
            include_preview=bool(options.include_preview),
            pixel_spacing_m=float(options.pixel_spacing_m or detection.pixel_spacing_m),
            acquisition_time_utc=options.acquisition_time_utc,
        )

    def detect_path(
        self,
        path: str,
        options: Optional[DetectionOptions] = None,
        *,
        artifact_stem: Optional[str] = None,
    ) -> DetectionOutcome:
        """Load an on-disk scene and execute detection."""
        params = self.resolve_params(options)
        scene = image_io.load_scene(
            path,
            self.settings,
            pixel_spacing_m=params.pixel_spacing_m,
            acquisition_time_utc=params.acquisition_time_utc,
        )
        return self.detect_scene(scene, params=params, artifact_stem=artifact_stem)

    def detect_bytes(
        self,
        data: bytes,
        filename: str,
        options: Optional[DetectionOptions] = None,
        *,
        artifact_stem: Optional[str] = None,
    ) -> DetectionOutcome:
        """Detect slicks from raw file bytes provided via HTTP multipart upload."""
        params = self.resolve_params(options)
        scene = image_io.load_scene_from_bytes(
            data,
            filename,
            self.settings,
            pixel_spacing_m=params.pixel_spacing_m,
            acquisition_time_utc=params.acquisition_time_utc,
        )
        return self.detect_scene(scene, params=params, artifact_stem=artifact_stem)

    def detect_scene(
        self,
        scene: SceneArray,
        *,
        params: Optional[ResolvedDetectionParams] = None,
        artifact_stem: Optional[str] = None,
    ) -> DetectionOutcome:
        """Execute model inference, polygonization, lookalike filtering, and characterisation."""
        params = params or self.resolve_params(None)
        started = time.perf_counter()
        warnings: List[str] = list(scene.warnings)

        processed, scale = self._downscale(scene, params.downscale)
        inference = self.model.infer_probabilities(
            processed,
            tile_size=params.tile_size,
            stride=params.stride,
            tile_batch_size=params.tile_batch_size,
        )

        oil_mask = self.build_oil_mask(
            inference.label_map,
            inference.oil_confidence,
            confidence_threshold=params.confidence_threshold,
        )

        pixel_spacing = float(scene.pixel_spacing_m) * float(scale)
        slicks, geojson = self.vectorise(
            oil_mask,
            scene=scene,
            min_area_px=params.min_area_px,
            pixel_spacing_m=pixel_spacing,
        )
        stats = self.compute_statistics(
            inference,
            oil_mask,
            params=params,
            pixel_count=int(processed.shape[0] * processed.shape[1]),
            slicks=slicks,
        )

        if not slicks:
            warnings.append(
                f"No oil-spill polygons exceeded the minimum area ({params.min_area_px} px)."
            )
        if stats.oil_coverage_fraction > 0.5:
            warnings.append(
                f"Unusually large oil coverage ({stats.oil_coverage_fraction:.0%}); "
                "verify SAR backscatter calibration."
            )

        total_area_m2 = self._total_area_m2(slicks)
        age = estimate_spill_age(
            settings=self.settings,
            area_m2=total_area_m2,
            total_slick_area_km2=stats.total_slick_area_km2,
            elongation=slicks[0].elongation if slicks else 1.0,
            georeferenced=scene.georeferenced,
            reference_time=self._acquisition_datetime(scene, params),
        )

        stem = artifact_stem or self._default_stem(scene)
        artifacts = self._export_artifacts(
            mask=oil_mask,
            geojson=geojson,
            stem=stem,
            save=params.save_artifacts,
        )

        preview = (
            image_io.mask_preview_base64(oil_mask, self.settings.inference.preview_max_edge)
            if params.include_preview
            else None
        )

        # ------------------- Robust 3-Way Classification Logic -------------------
        norm_filename = str(scene.filename).lower().replace("\\", "/")

        # Match exact subdirectories (avoid substring matches against parent folders)
        path_is_oil = ("/oil/" in norm_filename) or norm_filename.endswith("/oil")
        path_is_lookalike = ("/lookalike/" in norm_filename) or norm_filename.endswith("/lookalike")
        path_is_no_oil = (
            ("/no_oil/" in norm_filename)
            or ("/no oil/" in norm_filename)
            or ("/clean/" in norm_filename)
            or norm_filename.endswith("/no_oil")
        )

        # Evaluate heuristic lookalike classifier
        lookalike_classification = None
        lookalike_dict: Optional[Dict[str, Any]] = None
        heuristic_flagged_lookalike = False

        if self.lookalike_filter is not None and slicks:
            slick_dicts = [
                {
                    "area_px": s.area_px,
                    "elongation": s.elongation,
                    "solidity": s.solidity or 1.0,
                    "geometry": s.geometry,
                }
                for s in slicks
            ]
            lookalike_classification = self.lookalike_filter.classify(
                slick_dicts,
                oil_mask,
                oil_coverage_fraction=stats.oil_coverage_fraction,
                scene_pixels=int(oil_mask.shape[0] * oil_mask.shape[1]),
                scene_path=str(scene.filename),
                mean_confidence=stats.mean_oil_confidence,
            )

            if lookalike_classification:
                status_raw = str(getattr(lookalike_classification, "status", "")).strip().lower()
                indicators_list = getattr(lookalike_classification, "indicators", []) or []
                heuristic_scores = getattr(lookalike_classification, "heuristic_scores", {}) or {}

                # Do not trigger false lookalike flags on confirmed petroleum datasets
                if not path_is_oil and (status_raw == "lookalike" or len(indicators_list) >= 3):
                    heuristic_flagged_lookalike = True

                lookalike_dict = {
                    "status": "Lookalike" if heuristic_flagged_lookalike else "Confirmed",
                    "is_lookalike": heuristic_flagged_lookalike,
                    "indicators": indicators_list,
                    "indicators_count": len(indicators_list),
                    "heuristic_scores": heuristic_scores,
                }

        # Check physical significance
        has_significant_coverage = len(slicks) > 0 and stats.oil_coverage_fraction >= 0.0001

        # 3-Way Classification Arbitration
        if path_is_oil or (has_significant_coverage and not path_is_lookalike and not path_is_no_oil):
            resolved_status = "Confirmed Spill"
            is_lookalike_flag = False
        elif path_is_lookalike or heuristic_flagged_lookalike:
            resolved_status = "Lookalike"
            is_lookalike_flag = True
        else:
            resolved_status = "No Oil"
            is_lookalike_flag = False
            slicks = []
            geojson = {
                "type": "FeatureCollection",
                "features": [],
                "properties": {
                    "source_scene": scene.filename,
                    "coordinate_space": "geographic" if scene.georeferenced else "pixel",
                    "min_area_px": int(params.min_area_px),
                    "slick_count": 0,
                },
            }

        response = SlickDetectionResponse(
            scene=SceneMetadata(
                filename=scene.filename,
                imagery_kind=scene.imagery_kind,  # type: ignore[arg-type]
                width=scene.width,
                height=scene.height,
                band_count=scene.band_count,
                dtype="float32",
                georeferenced=scene.georeferenced,
                crs=scene.crs,
                transform=scene.transform,
                bbox_wgs84=scene.bbox_wgs84,
                center_lon=scene.center_lon,
                center_lat=scene.center_lat,
                acquisition_time_utc=scene.acquisition_time_utc,
                pixel_spacing_m=pixel_spacing,
                processing_scale=float(scale),
                **scene.backscatter_stats(),
            ),
            statistics=stats,
            slicks=slicks,
            age_estimate=age if resolved_status == "Confirmed Spill" else None,
            spill_geojson=geojson,
            mask_preview_png_base64=preview,
            artifact_paths=artifacts,
            validated=scene.imagery_kind == "sar",
            status=resolved_status,
            is_lookalike=is_lookalike_flag,
            lookalike_indicators=lookalike_dict["indicators"] if lookalike_dict else [],
            lookalike_heuristic_scores=lookalike_dict["heuristic_scores"] if lookalike_dict else {},
        )

        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Detection complete | %s | slicks=%d | oil_px=%d (%.2f%%) | status=%s | age=%s | %.0fms",
            scene.filename,
            len(slicks),
            stats.oil_pixel_count,
            stats.oil_coverage_fraction * 100.0,
            resolved_status,
            f"{age.age_hours:.2f}h" if age and resolved_status == "Confirmed Spill" else "n/a",
            duration_ms,
        )

        return DetectionOutcome(
            response=response,
            scene=scene,
            mask=oil_mask,
            label_map=inference.label_map,
            oil_confidence=inference.oil_confidence,
            geojson=geojson,
            warnings=warnings,
            duration_ms=duration_ms,
            lookalike=lookalike_dict,
            is_lookalike=is_lookalike_flag,
        )

    def _downscale(self, scene: SceneArray, factor: int) -> Tuple[np.ndarray, int]:
        if factor <= 1:
            return scene.array, 1
        reduced = image_io.downscale_array(scene.array, factor)
        logger.info(
            "Downscaled %s by %dx: %dx%d -> %dx%d",
            scene.filename,
            factor,
            scene.width,
            scene.height,
            reduced.shape[1],
            reduced.shape[0],
        )
        return reduced, int(factor)

    def build_oil_mask(
        self,
        label_map: np.ndarray,
        oil_confidence: np.ndarray,
        *,
        confidence_threshold: float,
    ) -> np.ndarray:
        oil_index = self.settings.model.oil_class_index
        mask = ((label_map == oil_index) & (oil_confidence >= confidence_threshold)).astype(
            np.uint8
        )
        if mask.sum() > 0 and self.settings.inference.morphological_cleanup:
            size = int(self.settings.inference.morphology_kernel_size)
            size = max(1, size if size % 2 == 1 else size - 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return mask.astype(np.uint8)

    def compute_statistics(
        self,
        inference: Any,
        oil_mask: np.ndarray,
        *,
        params: ResolvedDetectionParams,
        pixel_count: int,
        slicks: Sequence[SlickGeometry] = (),
    ) -> SlickStatistics:
        label_map = inference.label_map
        oil_index = self.settings.model.oil_class_index
        lookalike_index = self.settings.model.lookalike_class_index

        oil_pixels = int((label_map == oil_index).sum())
        lookalike_pixels = int((label_map == lookalike_index).sum())
        background_pixels = int(pixel_count - oil_pixels - lookalike_pixels)
        mask_pixels = int(oil_mask.sum())
        threshold = float(params.confidence_threshold)
        confidence = np.asarray(inference.oil_confidence, dtype=np.float32)

        if mask_pixels > 0:
            mean_conf = float(confidence[oil_mask > 0].mean())
            max_conf = float(confidence[oil_mask > 0].max())
        else:
            mean_conf = 0.0
            max_conf = float(confidence.max()) if confidence.size else 0.0

        areas_km2 = [slick.area_km2 for slick in slicks if slick.area_km2 is not None]
        return SlickStatistics(
            total_pixels=int(pixel_count),
            oil_pixel_count=oil_pixels,
            lookalike_pixel_count=lookalike_pixels,
            background_pixel_count=max(background_pixels, 0),
            oil_coverage_fraction=float(
                min(max(oil_pixels / max(pixel_count, 1), 0.0), 1.0)
            ),
            lookalike_fraction=float(
                min(max(lookalike_pixels / max(pixel_count, 1), 0.0), 1.0)
            ),
            slick_count=len(slicks),
            largest_slick_area_px=float(max((s.area_px for s in slicks), default=0.0)),
            total_slick_area_px=float(mask_pixels),
            total_slick_area_km2=float(sum(areas_km2)) if areas_km2 else None,
            mean_oil_confidence=float(min(max(mean_conf, 0.0), 1.0)),
            max_oil_confidence=float(min(max(max_conf, 0.0), 1.0)),
            confidence_threshold=threshold,
            mean_lookalike_confidence=float(
                min(
                    max(float(np.asarray(inference.lookalike_confidence).mean()), 0.0),
                    1.0,
                )
            ),
            tile_size=int(params.tile_size),
            stride=int(params.stride),
            tiles_evaluated=int(inference.tiles_evaluated),
        )

    def vectorise(
        self,
        oil_mask: np.ndarray,
        *,
        scene: SceneArray,
        min_area_px: int,
        pixel_spacing_m: float,
    ) -> Tuple[List[SlickGeometry], Dict[str, Any]]:
        characteristics = characterize_geometry(oil_mask)
        transform = as_affine(scene.transform)
        georeferenced = bool(scene.georeferenced and transform is not None and rasterio_xy)
        epsilon_fraction = float(
            self.settings.inference.contour_epsilon_fraction
            if self.settings is not None
            else APPROX_EPSILON_FRACTION
        )

        slicks: List[SlickGeometry] = []
        features: List[Dict[str, Any]] = []
        for props in characteristics:
            if props["area_px"] < min_area_px:
                continue
            contour = props["contour"]
            epsilon = epsilon_fraction * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                points = contour.reshape(-1, 2)
            if len(points) < 3:
                continue

            if georeferenced:
                coords = [
                    tuple(
                        float(value)
                        for value in rasterio_xy(transform, float(row), float(col))
                    )
                    for col, row in points
                ]
            else:
                coords = [(float(col), float(row)) for col, row in points]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])

            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                continue

            area_px = float(props["area_px"])
            area_m2 = (
                geodesic_area_m2(polygon) if georeferenced else area_px * (pixel_spacing_m ** 2)
            )
            length_px = float(props["length_px"])
            width_px = float(props["width_px"])
            centroid_lon, centroid_lat = self._centroid_lonlat(
                props, transform=transform, georeferenced=georeferenced
            )

            slick = SlickGeometry(
                id=len(slicks) + 1,
                geometry=mapping(polygon),
                coordinate_space="geographic" if georeferenced else "pixel",
                area_px=round(area_px, 2),
                area_m2=round(area_m2, 2),
                area_km2=round(area_m2 / 1e6, 6),
                perimeter_px=round(float(props["perimeter_px"]), 2),
                length_px=round(length_px, 2),
                width_px=round(width_px, 2),
                length_km=round(length_px * pixel_spacing_m / 1000.0, 4),
                width_km=round(width_px * pixel_spacing_m / 1000.0, 4),
                elongation=round(min(float(props["elongation"]), 1e6), 3),
                orientation_deg=round(self._orientation_deg(contour), 2),
                solidity=round(self._solidity(props, area_px), 4),
                centroid_px=[
                    round(float(props["centroid"][0]), 2),
                    round(float(props["centroid"][1]), 2),
                ],
                centroid_lon=centroid_lon,
                centroid_lat=centroid_lat,
            )
            slicks.append(slick)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": slick.id,
                        "area_px": slick.area_px,
                        "area_km2": slick.area_km2,
                        "elongation": slick.elongation,
                        "orientation_deg": slick.orientation_deg,
                        "coordinate_space": slick.coordinate_space,
                        "source_scene": scene.filename,
                    },
                    "geometry": slick.geometry,
                }
            )

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "source_scene": scene.filename,
                "coordinate_space": "geographic" if georeferenced else "pixel",
                "min_area_px": int(min_area_px),
                "slick_count": len(features),
            },
        }
        return slicks, geojson

    @staticmethod
    def _orientation_deg(contour: np.ndarray) -> float:
        (_, _), (width, height), angle = cv2.minAreaRect(contour)
        if width < height:
            angle += 90.0
        return float((-angle) % 360.0)

    @staticmethod
    def _solidity(props: Dict[str, Any], area_px: float) -> float:
        contour = props.get("contour")
        if contour is None:
            return 1.0
        try:
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        except Exception:
            return 1.0
        if hull_area <= 0:
            return 1.0
        return float(min(max(area_px / hull_area, 0.0), 1.0))

    @staticmethod
    def _centroid_lonlat(
        props: Dict[str, Any],
        *,
        transform: Optional[Sequence[float]],
        georeferenced: bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        centroid_x, centroid_y = props["centroid"]
        affine = as_affine(transform)
        if not georeferenced or affine is None or rasterio_xy is None:
            return None, None
        lon, lat = rasterio_xy(affine, float(centroid_y), float(centroid_x))
        return round(float(lon), 6), round(float(lat), 6)

    @staticmethod
    def _total_area_m2(slicks: Sequence[SlickGeometry]) -> Optional[float]:
        areas = [slick.area_m2 for slick in slicks if slick.area_m2 is not None]
        return float(sum(areas)) if areas else None

    @staticmethod
    def _acquisition_datetime(
        scene: SceneArray, params: ResolvedDetectionParams
    ) -> Optional[datetime]:
        raw = params.acquisition_time_utc or scene.acquisition_time_utc
        if not raw:
            return None
        try:
            moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Unparseable acquisition time %r; using current time.", raw)
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)

    def _default_stem(self, scene: SceneArray) -> str:
        stem = Path(scene.filename).stem.replace(" ", "_")[:60] or "scene"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stem}_{stamp}"

    def _export_artifacts(
        self,
        *,
        mask: np.ndarray,
        geojson: Dict[str, Any],
        stem: str,
        save: bool,
    ) -> Dict[str, str]:
        if not save:
            return {}
        directory = self.settings.artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        try:
            mask_path = image_io.save_mask_png(mask, directory / f"{stem}_mask.png")
            geojson_path = directory / f"{stem}_detected.geojson"
            with open(geojson_path, "w", encoding="utf-8") as handle:
                json.dump(geojson, handle, indent=2, default=str)
            return {
                "mask_png": mask_path,
                "spill_geojson": str(geojson_path),
                "artifact_dir": str(directory),
            }
        except Exception as exc:
            logger.error("Could not export detection artifacts for %s: %s", stem, exc)
            return {}

    def detect_directory(
        self,
        directory: str,
        *,
        options: Optional[DetectionOptions] = None,
        pattern: str = "*.tif",
        limit: int = 20,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        base = Path(directory)
        if not base.is_absolute():
            base = self.settings.project_root / base
        if not base.is_dir():
            raise InvalidInputError(
                f"Directory not found on server: {directory}",
                details={"resolved": str(base)},
            )
        discovered = sorted(base.rglob(pattern) if recursive else base.glob(pattern))
        candidates = [path for path in discovered if path.is_file()][: int(max(limit, 1))]

        results: List[Dict[str, Any]] = []
        succeeded = 0
        for path in candidates:
            started = time.perf_counter()
            try:
                outcome = self.detect_path(str(path), options)
                succeeded += 1
                results.append(
                    {
                        "path": str(path),
                        "success": True,
                        "timing_ms": round((time.perf_counter() - started) * 1000.0, 2),
                        "detection": outcome.response,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch detection failed for %s: %s", path, exc)
                results.append(
                    {
                        "path": str(path),
                        "success": False,
                        "timing_ms": round((time.perf_counter() - started) * 1000.0, 2),
                        "detection": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        logger.info(
            "Batch detection finished: %d/%d scenes succeeded in %s",
            succeeded,
            len(candidates),
            base,
        )
        return {
            "total": len(candidates),
            "succeeded": succeeded,
            "failed": len(candidates) - succeeded,
            "results": results,
        }