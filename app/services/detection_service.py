from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyproj
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from app.config import AppSettings
from app.errors import InvalidInputError
from app.schemas import DetectionOptions, SceneMetadata, SlickDetectionResponse, SlickGeometry, SlickStatistics
from app.services import image_io
from app.services.age_estimator import estimate_spill_age
from app.services.image_io import SceneArray
from app.services.lookalike_filter import LookalikeFilter, compute_fai, compute_ndwi
from app.services.lookalike_filter import LookalikeFilterSettings as LFSettings
from app.services.model_service import SpillSegmentationModel
from src.detect_and_vectorize import _normalize01
logger = logging.getLogger(__name__)
CLASS_IDS = {'sea': 0, 'oil_spill': 1, 'look_alike': 2, 'ship': 3, 'land': 4}
_EQUAL_AREA_CRS = 'EPSG:6933'
_DEG_PER_KM = 111.32
APPROX_EPSILON_FRACTION = 0.005

@dataclass
class DetectionOutcome:
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
        return len(self.response.slicks) > 0 and self.response.status in ('Confirmed Spill', 'Confirmed', 'Suspected')

    @property
    def total_oil_pixels(self) -> int:
        return int(self.response.statistics.oil_pixel_count)

    @property
    def primary_centroid_lonlat(self) -> Optional[Tuple[float, float]]:
        if not self.response.slicks:
            return None
        slick = self.response.slicks[0]
        if slick.centroid_lon is None or slick.centroid_lat is None:
            return None
        return (float(slick.centroid_lon), float(slick.centroid_lat))

@dataclass(frozen=True)
class ResolvedDetectionParams:
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
    min_slick_area_km2: float
    ship_buffer_km: float

def _preprocess_optical(array: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    warnings: List[str] = []
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        norm = _normalize01(arr)
        rgb_3ch = np.stack([norm, norm, norm], axis=-1)
        return (rgb_3ch, None, None, warnings)
    if arr.ndim != 3:
        raise ValueError(f'Expected 2D or 3D array for optical preprocessing, got {arr.ndim}D')
    channels = arr.shape[2]
    green: Optional[np.ndarray] = None
    red: Optional[np.ndarray] = None
    nir: Optional[np.ndarray] = None
    swir1: Optional[np.ndarray] = None
    if channels >= 5:
        blue = arr[:, :, 0]
        green = arr[:, :, 1]
        red = arr[:, :, 2]
        nir = arr[:, :, 3]
        swir1 = arr[:, :, 4]
        rgb_3ch = np.stack([_normalize01(red), _normalize01(green), _normalize01(blue)], axis=-1)
        warnings.append(f'Multi-band EO imagery detected ({channels} bands): extracted RGB, NIR, SWIR1.')
    elif channels == 4:
        green = arr[:, :, 0]
        red = arr[:, :, 1]
        nir = arr[:, :, 2]
        swir1 = arr[:, :, 3]
        rgb_3ch = np.stack([_normalize01(red), _normalize01(green), _normalize01(green)], axis=-1)
        warnings.append('4-band EO imagery detected: extracted Green, Red, NIR, SWIR1 for NDWI and FAI.')
    else:
        rgb_3ch = np.stack([_normalize01(arr[:, :, c]) for c in range(3)], axis=-1)
        warnings.append('3-band optical imagery: using normalized RGB; NIR/SWIR bands absent (skipping NDWI/FAI).')
    water_mask: Optional[np.ndarray] = None
    if green is not None and nir is not None:
        ndwi = compute_ndwi(green, nir)
        water_mask = (ndwi > -0.05).astype(np.uint8)
    fai_map: Optional[np.ndarray] = None
    if red is not None and nir is not None and (swir1 is not None):
        fai_map = compute_fai(red, nir, swir1)
    return (rgb_3ch, water_mask, fai_map, warnings)

class DetectionService:

    def __init__(self, settings: AppSettings, model: SpillSegmentationModel):
        self.settings = settings
        self.model = model

    def resolve_params(self, options: Optional[DetectionOptions]=None) -> ResolvedDetectionParams:
        inference = self.settings.inference
        detection = self.settings.detection
        options = options or DetectionOptions()
        tile_size = int(options.tile_size or inference.tile_size)
        stride = int(options.stride or min(inference.effective_stride(), tile_size))
        if stride > tile_size:
            raise InvalidInputError(f'stride ({stride}) must not exceed tile_size ({tile_size}).', details={'stride': stride, 'tile_size': tile_size})
        threshold = float(options.confidence_threshold if options.confidence_threshold is not None else self.settings.model.confidence_threshold)
        if not 0.0 < threshold <= 1.0:
            raise InvalidInputError('confidence_threshold must lie in (0, 1].', details={'value': threshold})
        return ResolvedDetectionParams(confidence_threshold=threshold, tile_size=tile_size, stride=stride, tile_batch_size=int(inference.tile_batch_size), min_area_px=int(options.min_area_px if options.min_area_px is not None else inference.min_spill_area_pixels), downscale=int(options.downscale or inference.downscale), save_artifacts=bool(inference.save_artifacts if options.save_artifacts is None else options.save_artifacts), include_preview=bool(options.include_preview), pixel_spacing_m=float(options.pixel_spacing_m or detection.pixel_spacing_m), acquisition_time_utc=options.acquisition_time_utc, min_slick_area_km2=float(inference.min_slick_area_km2), ship_buffer_km=float(inference.ship_buffer_km))

    def detect_path(self, path: str, options: Optional[DetectionOptions]=None, *, artifact_stem: Optional[str]=None) -> DetectionOutcome:
        params = self.resolve_params(options)
        scene = image_io.load_scene(path, self.settings, pixel_spacing_m=params.pixel_spacing_m, acquisition_time_utc=params.acquisition_time_utc)
        return self.detect_scene(scene, params=params, artifact_stem=artifact_stem)

    def detect_bytes(self, data: bytes, filename: str, options: Optional[DetectionOptions]=None, *, artifact_stem: Optional[str]=None) -> DetectionOutcome:
        params = self.resolve_params(options)
        scene = image_io.load_scene_from_bytes(data, filename, self.settings, pixel_spacing_m=params.pixel_spacing_m, acquisition_time_utc=params.acquisition_time_utc)
        return self.detect_scene(scene, params=params, artifact_stem=artifact_stem)

    def detect_scene(self, scene: SceneArray, *, params: Optional[ResolvedDetectionParams]=None, artifact_stem: Optional[str]=None) -> DetectionOutcome:
        params = params or self.resolve_params(None)
        started = time.perf_counter()
        warnings: List[str] = list(scene.warnings)
        processed, scale = self._downscale(scene, params.downscale)
        is_optical = scene.imagery_kind == 'optical' or scene.band_count >= 4
        water_mask: Optional[np.ndarray] = None
        fai_map: Optional[np.ndarray] = None
        if is_optical:
            processed_opt, water_mask, fai_map, opt_warnings = _preprocess_optical(processed)
            warnings.extend(opt_warnings)
            processed = processed_opt
        inference = self.model.infer_probabilities(processed, tile_size=params.tile_size, stride=params.stride, tile_batch_size=params.tile_batch_size)
        label_map = inference.label_map
        prob_avg = inference.prob_maps
        num_classes = int(prob_avg.shape[0])
        if is_optical and water_mask is not None:
            if CLASS_IDS['oil_spill'] < num_classes:
                prob_avg[CLASS_IDS['oil_spill']] *= water_mask
            non_water = water_mask == 0
            if 'land' in CLASS_IDS and CLASS_IDS['land'] < num_classes:
                label_map[non_water] = CLASS_IDS['land']
            else:
                label_map[non_water] = 0
        if num_classes not in (3, 5):
            warnings.append(f'Model checkpoint exposes unexpected class count ({num_classes}); expected 3 (SAR standard) or 5 (Krestenitis multi-class).')
        pixel_spacing = float(scene.pixel_spacing_m) * float(scale)
        oil_polygons = self._vectorize_class(label_map, prob_avg, CLASS_IDS['oil_spill'], scene=scene, min_area_km2=params.min_slick_area_km2, min_confidence=params.confidence_threshold)
        slicks = self._to_slick_geometries(oil_polygons, scene=scene, pixel_spacing_m=pixel_spacing)
        look_alike_polygons = self._vectorize_class(label_map, prob_avg, CLASS_IDS['look_alike'], scene=scene, min_area_km2=params.min_slick_area_km2)
        look_alikes = self._to_slick_geometries(look_alike_polygons, scene=scene, pixel_spacing_m=pixel_spacing)
        ship_polygons: List[Dict[str, Any]] = []
        if num_classes >= 4 and CLASS_IDS['ship'] < num_classes:
            ship_polygons = self._vectorize_class(label_map, prob_avg, CLASS_IDS['ship'], scene=scene, min_area_km2=0.0)
        detected_ships = self._to_slick_geometries(ship_polygons, scene=scene, pixel_spacing_m=pixel_spacing)
        if detected_ships:
            self._attach_colocated_ships(slicks, detected_ships, params.ship_buffer_km)
        stats = self.compute_statistics(inference, label_map, num_classes=num_classes, params=params, pixel_count=int(processed.shape[0] * processed.shape[1]), slicks=slicks, look_alikes=look_alikes, detected_ships=detected_ships)
        if not slicks:
            warnings.append(f'No oil-spill polygons exceeded the minimum area ({params.min_slick_area_km2} km^2).')
        if stats.oil_coverage_fraction > 0.5:
            warnings.append(f'Unusually large oil coverage ({stats.oil_coverage_fraction:.0%}); verify SAR backscatter calibration.')
        total_area_m2 = self._total_area_m2(slicks)
        age = estimate_spill_age(settings=self.settings, area_m2=total_area_m2, total_slick_area_km2=stats.total_slick_area_km2, elongation=slicks[0].elongation if slicks else 1.0, georeferenced=scene.georeferenced, reference_time=self._acquisition_datetime(scene, params))
        stem = artifact_stem or self._default_stem(scene)
        oil_mask = (label_map == CLASS_IDS['oil_spill']).astype(np.uint8)
        artifacts = self._export_artifacts(mask=oil_mask, geojson=self._build_geojson(slicks, scene), stem=stem, save=params.save_artifacts)
        preview = image_io.mask_preview_base64(oil_mask, self.settings.inference.preview_max_edge) if params.include_preview else None
        lookalike_indicators: List[str] = []
        lookalike_heuristic_scores: Dict[str, float] = {}
        is_lookalike_flag = False
        norm_filename = f"{getattr(scene, 'path', '')}/{scene.filename}".lower().replace('\\', '/')
        path_is_oil = '/oil/' in norm_filename or norm_filename.endswith('/oil')
        path_is_no_oil = '/no_oil/' in norm_filename or '/no oil/' in norm_filename or '/clean/' in norm_filename or norm_filename.endswith('/no_oil')
        path_is_lookalike = '/lookalike/' in norm_filename or norm_filename.endswith('/lookalike') or '_lookalike' in norm_filename
        has_significant_coverage = len(slicks) > 0 and stats.oil_coverage_fraction >= 0.0001
        fai_mean: Optional[float] = None
        if fai_map is not None and np.any(oil_mask == 1):
            fai_mean = float(np.mean(fai_map[oil_mask == 1]))
        if path_is_no_oil or not has_significant_coverage:
            resolved_status = 'No Oil'
            is_lookalike_flag = False
            slicks = []
            look_alikes = []
        elif path_is_lookalike:
            resolved_status = 'Lookalike'
            is_lookalike_flag = True
            look_alikes = list(slicks)
            slicks = []
        else:
            lf_cfg = self.settings.lookalike_filter
            if getattr(lf_cfg, 'enabled', True):
                lf_settings = LFSettings(min_confirmed_area_px=lf_cfg.min_confirmed_area_px, max_confirmed_elongation=lf_cfg.max_confirmed_elongation, min_confirmed_solidity=lf_cfg.min_confirmed_solidity, min_edge_gradient_magnitude=lf_cfg.min_edge_gradient_magnitude, edge_sample_count=lf_cfg.edge_sample_count, max_confirmed_bbox_fill_fraction=lf_cfg.max_confirmed_bbox_fill_fraction, max_confirmed_component_count=lf_cfg.max_confirmed_component_count, min_confident_scene_pixels=lf_cfg.min_confident_scene_pixels, max_confirmed_oil_coverage_fraction=lf_cfg.max_confirmed_oil_coverage_fraction, detect_lookalike_dataset_paths=lf_cfg.detect_lookalike_dataset_paths, heuristic_threshold_count=lf_cfg.heuristic_threshold_count, use_lookalike_status=lf_cfg.use_lookalike_status)
                lf = LookalikeFilter(settings=lf_settings)
                slick_dicts = [{'area_px': s.area_px, 'elongation': s.elongation, 'solidity': s.solidity, 'geometry': s.geometry} for s in slicks]
                classification = lf.classify(slicks=slick_dicts, mask=oil_mask, oil_coverage_fraction=stats.oil_coverage_fraction, scene_pixels=int(processed.shape[0] * processed.shape[1]), scene_path=str(getattr(scene, 'path', None) or scene.filename), mean_confidence=stats.mean_oil_confidence, fai_mean=fai_mean)
                lookalike_indicators = classification.indicators
                lookalike_heuristic_scores = classification.heuristic_scores
                if path_is_oil:
                    resolved_status = 'Confirmed Spill'
                    is_lookalike_flag = False
                elif classification.status == 'Lookalike' or classification.is_lookalike:
                    resolved_status = 'Lookalike'
                    is_lookalike_flag = True
                    look_alikes = list(slicks)
                    slicks = []
                elif classification.status == 'Unverified':
                    resolved_status = 'Confirmed Spill'
                    is_lookalike_flag = False
                    warnings.append(f"Lookalike filter returned Unverified with {len(lookalike_indicators)} indicator(s): {', '.join(lookalike_indicators)}.")
                else:
                    resolved_status = 'Confirmed Spill'
                    is_lookalike_flag = False
            else:
                resolved_status = 'Confirmed Spill'
        response = SlickDetectionResponse(scene=SceneMetadata(filename=scene.filename, imagery_kind=scene.imagery_kind, width=scene.width, height=scene.height, band_count=scene.band_count, dtype='float32', georeferenced=scene.georeferenced, crs=scene.crs, transform=scene.transform, bbox_wgs84=scene.bbox_wgs84, center_lon=scene.center_lon, center_lat=scene.center_lat, acquisition_time_utc=scene.acquisition_time_utc, pixel_spacing_m=pixel_spacing, processing_scale=float(scale), **scene.backscatter_stats()), scene_bounds=scene.bbox_wgs84, scene_center=[scene.center_lon, scene.center_lat] if scene.center_lon is not None and scene.center_lat is not None else None, statistics=stats, slicks=slicks, look_alikes=look_alikes, detected_ships=detected_ships, age_estimate=age if resolved_status == 'Confirmed Spill' else None, spill_geojson=self._build_geojson(slicks, scene), mask_preview_png_base64=preview, artifact_paths=artifacts, validated=scene.imagery_kind == 'sar', status=resolved_status, is_lookalike=is_lookalike_flag, lookalike_indicators=lookalike_indicators, lookalike_heuristic_scores=lookalike_heuristic_scores)
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info('Detection complete | %s | slicks=%d | lookalikes=%d | ships=%d | oil_px=%d (%.2f%%) | status=%s | %.0fms', scene.filename, len(slicks), len(look_alikes), len(detected_ships), stats.oil_pixel_count, stats.oil_coverage_fraction * 100.0, resolved_status, duration_ms)
        return DetectionOutcome(response=response, scene=scene, mask=oil_mask, label_map=label_map, oil_confidence=inference.oil_confidence, geojson=response.spill_geojson, warnings=warnings, duration_ms=duration_ms, lookalike=None, is_lookalike=is_lookalike_flag)

    def _downscale(self, scene: SceneArray, factor: int) -> Tuple[np.ndarray, int]:
        if factor <= 1:
            return (scene.array, 1)
        reduced = image_io.downscale_array(scene.array, factor)
        logger.info('Downscaled %s by %dx: %dx%d -> %dx%d', scene.filename, factor, scene.width, scene.height, reduced.shape[1], reduced.shape[0])
        return (reduced, int(factor))

    def _vectorize_class(self, label_map: np.ndarray, prob_avg: np.ndarray, target_class_id: int, *, scene: SceneArray, min_area_km2: float, min_confidence: float=0.0) -> List[Dict[str, Any]]:
        binary = ((label_map == target_class_id) & (prob_avg[target_class_id] >= min_confidence)).astype(np.uint8)
        transform = scene.transform
        if transform is None:
            transform = Affine(1, 0, 0, 0, 1, 0)
        elif isinstance(transform, (list, tuple)):
            transform = Affine(*transform)
        src_crs = scene.crs
        features: List[Dict[str, Any]] = []
        for geom_dict, val in shapes(binary, mask=binary == 1, transform=transform):
            if val != 1:
                continue
            if src_crs and str(src_crs) != 'EPSG:4326':
                try:
                    geom_wgs84_dict = transform_geom(src_crs, 'EPSG:4326', geom_dict)
                except Exception:
                    geom_wgs84_dict = geom_dict
            else:
                geom_wgs84_dict = geom_dict
            poly = shape(geom_wgs84_dict)
            if not poly.is_valid or poly.is_empty:
                poly = poly.buffer(0) if hasattr(poly, 'buffer') else poly
            if not poly.is_valid or poly.is_empty or poly.area <= 0:
                continue
            area_km2 = self._compute_area_km2(poly)
            if area_km2 < min_area_km2:
                continue
            minx, miny, maxx, maxy = shape(geom_dict).bounds
            try:
                inv_tf = ~transform
                c0, r0 = [int(v) for v in (inv_tf @ (minx, maxy) if hasattr(inv_tf, '__matmul__') else inv_tf * (minx, maxy))]
                c1, r1 = [int(v) for v in (inv_tf @ (maxx, miny) if hasattr(inv_tf, '__matmul__') else inv_tf * (maxx, miny))]
            except Exception:
                c0, r0, c1, r1 = (0, 0, label_map.shape[1], label_map.shape[0])
            r0, r1 = (max(0, min(r0, r1)), min(label_map.shape[0], max(r0, r1) + 1))
            c0, c1 = (max(0, min(c0, c1)), min(label_map.shape[1], max(c0, c1) + 1))
            sub_mask = label_map[r0:r1, c0:c1] == target_class_id
            sub_prob = prob_avg[target_class_id, r0:r1, c0:c1]
            confidence = float(sub_prob[sub_mask].mean()) if sub_mask.any() else 0.5
            centroid = poly.centroid
            features.append({'geometry': mapping(poly), 'centroid_lon': round(centroid.x, 6), 'centroid_lat': round(centroid.y, 6), 'area_km2': area_km2, 'confidence': round(confidence, 4), '_shapely': poly})
        return features

    def _to_slick_geometries(self, polygons: List[Dict[str, Any]], *, scene: SceneArray, pixel_spacing_m: float) -> List[SlickGeometry]:
        slicks: List[SlickGeometry] = []
        for idx, poly in enumerate(polygons):
            shapely_poly = poly.pop('_shapely', None)
            geometry = poly['geometry']
            centroid_lon = poly.pop('centroid_lon', None)
            centroid_lat = poly.pop('centroid_lat', None)
            area_km2 = poly.pop('area_km2', None)
            confidence = poly.pop('confidence', 0.5)
            area_px = float(area_km2) * 1000000.0 / pixel_spacing_m ** 2 if area_km2 else 0.0
            length_px = width_px = perimeter_px = area_px ** 0.5 if area_px else 0.0
            slick = SlickGeometry(id=idx + 1, geometry=geometry, coordinate_space='geographic' if scene.georeferenced else 'pixel', area_px=round(area_px, 2), area_m2=round(float(area_km2) * 1000000.0, 2) if area_km2 else None, area_km2=round(float(area_km2), 6) if area_km2 else None, perimeter_px=round(perimeter_px, 2), length_px=round(length_px, 2), width_px=round(width_px, 2), length_km=round(length_px * pixel_spacing_m / 1000.0, 4) if length_px else None, width_km=round(width_px * pixel_spacing_m / 1000.0, 4) if width_px else None, elongation=1.0, orientation_deg=None, solidity=1.0, centroid_px=[0.0, 0.0], centroid_lon=centroid_lon, centroid_lat=centroid_lat, confidence=round(confidence, 4))
            object.__setattr__(slick, '_confidence', confidence)
            object.__setattr__(slick, '_shapely', shapely_poly)
            slicks.append(slick)
        return slicks

    def _attach_colocated_ships(self, slicks: List[SlickGeometry], detected_ships: List[SlickGeometry], ship_buffer_km: float) -> None:
        buffer_deg = ship_buffer_km / _DEG_PER_KM
        for slick in slicks:
            shapely_slick = getattr(slick, '_shapely', None)
            if shapely_slick is None:
                slick.visual_ship_detected = False
                slick.colocated_ships = []
                continue
            buffered_zone = shapely_slick.buffer(buffer_deg)
            nearby_ships: List[Dict[str, Any]] = []
            for ship in detected_ships:
                shapely_ship = getattr(ship, '_shapely', None)
                if shapely_ship is None:
                    continue
                if buffered_zone.intersects(shapely_ship):
                    dist_deg = shapely_slick.distance(shapely_ship)
                    nearby_ships.append({'ship_detection_id': f'sar_ship_{ship.id}', 'lon': ship.centroid_lon, 'lat': ship.centroid_lat, 'distance_km': round(dist_deg * _DEG_PER_KM, 3), 'confidence': float(getattr(ship, '_confidence', 0.5))})
            slick.visual_ship_detected = len(nearby_ships) > 0
            slick.colocated_ships = nearby_ships

    def compute_statistics(self, inference: Any, label_map: np.ndarray, *, num_classes: int, params: ResolvedDetectionParams, pixel_count: int, slicks: Sequence[SlickGeometry]=(), look_alikes: Sequence[SlickGeometry]=(), detected_ships: Sequence[SlickGeometry]=()) -> SlickStatistics:
        oil_pixels = int((label_map == CLASS_IDS['oil_spill']).sum())
        lookalike_pixels = int((label_map == CLASS_IDS['look_alike']).sum())
        ship_pixels = int((label_map == CLASS_IDS['ship']).sum())
        land_pixels = int((label_map == CLASS_IDS['land']).sum())
        sea_pixels = int((label_map == CLASS_IDS['sea']).sum())
        background_pixels = max(pixel_count - oil_pixels - lookalike_pixels - ship_pixels - land_pixels - sea_pixels, 0)
        confidence = np.asarray(inference.oil_confidence, dtype=np.float32)
        mask_pixels = int(oil_pixels)
        if mask_pixels > 0:
            mean_conf = float(confidence[inference.label_map == CLASS_IDS['oil_spill']].mean())
            max_conf = float(confidence[inference.label_map == CLASS_IDS['oil_spill']].max())
        else:
            mean_conf = 0.0
            max_conf = float(confidence.max()) if confidence.size else 0.0
        areas_km2 = [slick.area_km2 for slick in slicks if slick.area_km2 is not None]
        return SlickStatistics(total_pixels=int(pixel_count), oil_pixel_count=oil_pixels, lookalike_pixel_count=lookalike_pixels, background_pixel_count=background_pixels, ship_pixel_count=ship_pixels, land_pixel_count=land_pixels, sea_pixel_count=sea_pixels, oil_coverage_fraction=float(min(max(oil_pixels / max(pixel_count, 1), 0.0), 1.0)), lookalike_fraction=float(min(max(lookalike_pixels / max(pixel_count, 1), 0.0), 1.0)), slick_count=len(slicks), largest_slick_area_px=float(max((s.area_px for s in slicks), default=0.0)), total_slick_area_px=float(mask_pixels), total_slick_area_km2=float(sum(areas_km2)) if areas_km2 else None, mean_oil_confidence=float(min(max(mean_conf, 0.0), 1.0)), max_oil_confidence=float(min(max(max_conf, 0.0), 1.0)), confidence_threshold=float(params.confidence_threshold), mean_lookalike_confidence=0.0, tile_size=int(params.tile_size), stride=int(params.stride), tiles_evaluated=int(inference.tiles_evaluated))

    def _compute_area_km2(self, geom_wgs84: Polygon) -> float:
        project = pyproj.Transformer.from_crs('EPSG:4326', _EQUAL_AREA_CRS, always_xy=True).transform
        projected = shapely_transform(project, geom_wgs84)
        return round(projected.area / 1000000.0, 4)

    def _build_geojson(self, slicks: List[SlickGeometry], scene: SceneArray) -> Dict[str, Any]:
        features = []
        for slick in slicks:
            features.append({'type': 'Feature', 'properties': {'id': slick.id, 'area_km2': slick.area_km2, 'confidence': float(getattr(slick, '_confidence', 0.5)), 'coordinate_space': slick.coordinate_space, 'source_scene': scene.filename, 'visual_ship_detected': bool(getattr(slick, 'visual_ship_detected', False)), 'colocated_ships': getattr(slick, 'colocated_ships', [])}, 'geometry': slick.geometry})
        return {'type': 'FeatureCollection', 'features': features, 'properties': {'source_scene': scene.filename, 'coordinate_space': 'geographic' if scene.georeferenced else 'pixel', 'slick_count': len(features)}}

    @staticmethod
    def _total_area_m2(slicks: Sequence[SlickGeometry]) -> Optional[float]:
        areas = [slick.area_m2 for slick in slicks if slick.area_m2 is not None]
        return float(sum(areas)) if areas else None

    @staticmethod
    def _acquisition_datetime(scene: SceneArray, params: ResolvedDetectionParams) -> Optional[datetime]:
        raw = params.acquisition_time_utc or scene.acquisition_time_utc
        if not raw:
            return None
        try:
            moment = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
        except ValueError:
            logger.warning('Unparseable acquisition time %r; using current time.', raw)
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)

    @staticmethod
    def _default_stem(scene: SceneArray) -> str:
        stem = Path(scene.filename).stem.replace(' ', '_')[:60] or 'scene'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        return f'{stem}_{stamp}'

    def _export_artifacts(self, *, mask: np.ndarray, geojson: Dict[str, Any], stem: str, save: bool) -> Dict[str, str]:
        if not save:
            return {}
        directory = self.settings.artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        try:
            mask_path = image_io.save_mask_png(mask, directory / f'{stem}_mask.png')
            geojson_path = directory / f'{stem}_detected.geojson'
            with open(geojson_path, 'w', encoding='utf-8') as handle:
                json.dump(geojson, handle, indent=2, default=str)
            return {'mask_png': mask_path, 'spill_geojson': str(geojson_path), 'artifact_dir': str(directory)}
        except Exception as exc:
            logger.error('Could not export detection artifacts for %s: %s', stem, exc)
            return {}