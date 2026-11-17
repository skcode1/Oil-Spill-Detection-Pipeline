from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from app.config import AUTHENTIC_CLASSES, AUTHENTIC_IMAGE_DIRNAME, AUTHENTIC_MASK_DIRNAME
T = TypeVar('T')
ImageryKind = Literal['sar', 'sentinel1', 'sentinel2', 'unknown']

class Envelope(BaseModel, Generic[T]):
    success: bool = Field(True, description='True when the request completed.')
    message: str = Field('ok', description='Human-readable outcome summary.')
    request_id: str = Field('', description='Correlation id for log tracing.')
    timing_ms: float = Field(0.0, description='Server-side processing time (ms).')
    warnings: List[str] = Field(default_factory=list, description='Non-fatal advisories (e.g. degraded engine).')
    data: Optional[T] = Field(None, description='Endpoint-specific payload.')

class ErrorDetail(BaseModel):
    code: str = Field(..., description="Stable error code, e.g. 'invalid_image'.")
    message: str = Field(..., description='Actionable human-readable message.')
    details: Optional[Any] = Field(None, description='Optional structured context.')

class ErrorEnvelope(BaseModel):
    success: bool = False
    message: str
    request_id: str = ''
    error: ErrorDetail

class BoundingBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @model_validator(mode='after')
    def _check_order(self) -> 'BoundingBox':
        if self.min_lon > self.max_lon or self.min_lat > self.max_lat:
            raise ValueError('bounding box min values must not exceed max values')
        return self

class GeoPoint(BaseModel):
    lon: float = Field(..., ge=-180.0, le=180.0)
    lat: float = Field(..., ge=-90.0, le=90.0)

class UncertaintyEllipse(BaseModel):
    semi_major_axis_meters: float = Field(..., gt=0)
    semi_minor_axis_meters: float = Field(..., gt=0)
    orientation_deg: float = Field(..., description='Semi-major axis bearing, degrees clockwise from True North.')
    confidence_level: float = Field(0.95, ge=0.0, le=1.0)
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    area_km2: Optional[float] = Field(None, description='Ellipse area (km^2) for dashboard display.')

def utc_iso(moment: Optional[datetime]=None) -> str:
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()

def _require_iso_timestamp(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError('timestamp must be a non-empty ISO-8601 string')
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'invalid ISO-8601 timestamp {value!r}: {exc}') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()

class DetectionOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    confidence_threshold: Optional[float] = Field(None, gt=0.0, le=1.0, description='Min oil-class probability (default from config).')
    tile_size: Optional[int] = Field(None, ge=32, le=2048)
    stride: Optional[int] = Field(None, ge=1, le=2048, description='Sliding-window step (tile_size = no overlap).')
    min_area_px: Optional[int] = Field(None, ge=0, description='Discard contours smaller than this (pixels).')
    downscale: Optional[int] = Field(None, ge=1, le=8, description='Integer pre-inference downscale (speed/accuracy trade-off).')
    save_artifacts: Optional[bool] = Field(None, description='Write mask PNG + spill GeoJSON into the results folder.')
    include_preview: bool = Field(True, description='Return a base64 PNG preview of the predicted oil mask.')
    pixel_spacing_m: Optional[float] = Field(None, gt=0.0, description='Override ground sample distance for metric conversion.')
    acquisition_time_utc: Optional[str] = Field(None, description='Satellite acquisition time (ISO-8601); inferred when omitted.')

    @field_validator('acquisition_time_utc')
    @classmethod
    def _validate_time(cls, value: Optional[str]) -> Optional[str]:
        return _require_iso_timestamp(value) if value else None

    @model_validator(mode='after')
    def _check_stride(self) -> 'DetectionOptions':
        if self.tile_size is not None and self.stride is not None:
            if self.stride > self.tile_size:
                raise ValueError('stride must not exceed tile_size')
        return self

class DetectionByPathRequest(DetectionOptions):
    path: str = Field(..., min_length=1, description='Absolute or project-relative scene path.')

class DetectionBatchRequest(DetectionOptions):
    directory: str = Field(..., min_length=1)
    pattern: str = Field('*.tif', description='Glob pattern applied inside the directory.')
    limit: int = Field(20, ge=1, le=500, description='Maximum number of scenes to process.')
    recursive: bool = False

class DetectionBatchItem(BaseModel):
    path: str
    success: bool
    timing_ms: float = 0.0
    error: Optional[str] = None
    detection: Optional['SlickDetectionResponse'] = None

class DetectionBatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[DetectionBatchItem]

class SceneMetadata(BaseModel):
    filename: str
    imagery_kind: ImageryKind = 'sar'
    width: int
    height: int
    band_count: int
    dtype: str
    georeferenced: bool = False
    crs: Optional[str] = None
    transform: Optional[List[float]] = Field(None, description='Rasterio affine transform as (a, b, c, d, e, f).')
    bbox_wgs84: Optional[List[float]] = None
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    acquisition_time_utc: Optional[str] = None
    pixel_spacing_m: float = 10.0
    processing_scale: float = Field(1.0, description='Downscale factor applied before inference (1.0 = native).')
    backscatter_min: float
    backscatter_max: float
    backscatter_mean: float
    backscatter_std: float

class SlickGeometry(BaseModel):
    id: int
    geometry: Dict[str, Any] = Field(..., description='GeoJSON Polygon (lon/lat or pixel space).')
    coordinate_space: Literal['geographic', 'pixel'] = 'pixel'
    area_px: float
    area_m2: Optional[float] = None
    area_km2: Optional[float] = None
    perimeter_px: float
    length_px: float
    width_px: float
    length_km: Optional[float] = None
    width_km: Optional[float] = None
    elongation: float = Field(..., description='Length / width aspect ratio.')
    orientation_deg: Optional[float] = Field(None, description='Major-axis bearing (degrees clockwise from North).')
    solidity: Optional[float] = Field(None, description='Contour area / convex-hull area (1.0 = perfectly convex).')
    centroid_px: List[float]
    centroid_lon: Optional[float] = None
    centroid_lat: Optional[float] = None
    confidence: Optional[float] = Field(None, description="Mean softmax confidence for the polygon's class.")
    visual_ship_detected: bool = Field(False, description='True when a co-located SAR ship was detected near this slick.')
    colocated_ships: List[Dict[str, Any]] = Field(default_factory=list, description='Ship detections near this slick (dark-vessel cue).')

    @property
    def pixel_area(self) -> float:
        return float(self.area_px)

    @property
    def perimeter_m(self) -> float:
        if self.area_m2 is not None and self.area_m2 > 0 and (self.area_px > 0):
            scale = math.sqrt(self.area_m2 / self.area_px)
            return float(self.perimeter_px * scale)
        return float(self.perimeter_px * 10.0)

class SlickStatistics(BaseModel):
    total_pixels: int
    oil_pixel_count: int
    lookalike_pixel_count: int
    background_pixel_count: int
    ship_pixel_count: int = 0
    land_pixel_count: int = 0
    sea_pixel_count: int = 0
    oil_coverage_fraction: float = Field(..., ge=0.0, le=1.0)
    lookalike_fraction: float = Field(..., ge=0.0, le=1.0)
    slick_count: int
    largest_slick_area_px: float
    total_slick_area_px: float
    total_slick_area_km2: Optional[float] = None
    mean_oil_confidence: float = Field(..., ge=0.0, le=1.0)
    max_oil_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    mean_lookalike_confidence: float = Field(0.0, ge=0.0, le=1.0)
    tile_size: int
    stride: int
    tiles_evaluated: int

class SpillAgeEstimate(BaseModel):
    age_hours: float
    age_days: float
    min_possible_hours: float
    max_possible_hours: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str
    estimated_discharge_time_utc: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)

class SlickDetectionResponse(BaseModel):
    scene: SceneMetadata
    scene_bounds: Optional[List[float]] = Field(None, description='Scene bounding box [min_lon, min_lat, max_lon, max_lat].')
    scene_center: Optional[List[float]] = Field(None, description='Scene center [lon, lat].')
    statistics: SlickStatistics
    slicks: List[SlickGeometry]
    look_alikes: List[SlickGeometry] = Field(default_factory=list, description='Look-alike polygons (biogenic films, wind slicks) excluded from drift.')
    detected_ships: List[SlickGeometry] = Field(default_factory=list, description='SAR ship detections used for visual-AIS cross-correlation.')
    age_estimate: Optional[SpillAgeEstimate] = None
    spill_geojson: Dict[str, Any] = Field(..., description='GeoJSON FeatureCollection of detected spill polygons.')
    mask_preview_png_base64: Optional[str] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    validated: bool = Field(True, description='False when radiometric plausibility checks warned.')
    status: str = Field('Confirmed', description="Resolved status: 'Confirmed', 'Lookalike', 'Suspected', or 'No Spill Detected'.")
    is_lookalike: bool = Field(False, description='True when geometric and contextual heuristics flag this detection as a false positive.')
    lookalike_indicators: List[str] = Field(default_factory=list, description='List of heuristic rule names that triggered lookalike classification.')
    lookalike_heuristic_scores: Dict[str, float] = Field(default_factory=dict, description='Numeric scores (0-1) for each lookalike heuristic evaluated.')
DetectionBatchItem.model_rebuild()

class MetoceanSample(BaseModel):
    model_config = ConfigDict(extra='forbid')
    timestamp_utc: Optional[str] = Field(None, description='Sample time; omitted for a time-invariant field.')
    u_current_mps: float = Field(..., description='Eastward surface current (m/s).')
    v_current_mps: float = Field(..., description='Northward surface current (m/s).')
    u_wind_mps: float = Field(0.0, description='Eastward 10 m wind (m/s).')
    v_wind_mps: float = Field(0.0, description='Northward 10 m wind (m/s).')
    u_stokes_mps: Optional[float] = Field(0.0, description='Eastward Stokes drift velocity (m/s).')
    v_stokes_mps: Optional[float] = Field(0.0, description='Northward Stokes drift velocity (m/s).')
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator('timestamp_utc')
    @classmethod
    def _validate_time(cls, value: Optional[str]) -> Optional[str]:
        return _require_iso_timestamp(value) if value else None

    @field_validator('u_current_mps', 'v_current_mps', 'u_wind_mps', 'v_wind_mps')
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value or abs(value) == float('inf'):
            raise ValueError('metocean components must be finite')
        if abs(value) > 20.0:
            raise ValueError('metocean component magnitude exceeds 20 m/s (implausible)')
        return value

class DriftSource(BaseModel):
    slick_geojson: Optional[Dict[str, Any]] = Field(None, description='GeoJSON Polygon/FeatureCollection used as the release area.')
    source_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    source_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    source_radius_m: float = Field(1000.0, gt=0.0, le=200000.0, description='Radius when seeding from a point.')

    @model_validator(mode='after')
    def _check_source(self) -> 'DriftSource':
        if self.slick_geojson is None and (self.source_lon is None or self.source_lat is None):
            raise ValueError('provide either slick_geojson or both source_lon and source_lat')
        return self

class DriftRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: DriftSource
    start_time_utc: str = Field(..., description='Satellite acquisition time (hindcast) or release time (forecast).')
    duration_hours: float = Field(24.0, gt=0.0, le=720.0)
    time_step_minutes: int = Field(15, ge=1, le=1440)
    particles: int = Field(1000, ge=10, le=20000)
    metocean: Optional[List[MetoceanSample]] = Field(None, description='Oceanographic/meteorological samples. Defaults to config fallbacks.')
    wind_drift_factor: Optional[float] = Field(None, gt=0.0, lt=1.0)
    horizontal_diffusivity_m2_s: Optional[float] = Field(None, ge=0.0, le=1000.0)
    confidence_level: Optional[float] = Field(None, gt=0.0, lt=1.0)
    random_seed: int = Field(42, ge=0, le=2 ** 31 - 1)
    engine: Literal['auto', 'lagrangian', 'opendrift'] = Field('auto', description='Selected drift simulation engine.')
    current_nc: Optional[str] = Field(None, description='Server-side current NetCDF path.')
    wind_nc: Optional[str] = Field(None, description='Server-side wind NetCDF path.')
    wave_nc: Optional[str] = Field(None, description='Optional wave NetCDF path.')
    oil_type: Optional[str] = None

    @field_validator('start_time_utc')
    @classmethod
    def _validate_time(cls, value: str) -> str:
        return _require_iso_timestamp(value)

    @model_validator(mode='after')
    def _check_consistency(self) -> 'DriftRequest':
        if self.metocean is not None and len(self.metocean) > 1:
            stamps = [sample.timestamp_utc for sample in self.metocean]
            if not all((stamp for stamp in stamps)):
                raise ValueError('every metocean sample needs timestamp_utc when several are supplied')
            if stamps != sorted(stamps):
                raise ValueError('metocean samples must be ordered by timestamp_utc')
        if self.engine == 'opendrift' and (not (self.current_nc and self.wind_nc)):
            raise ValueError("engine='opendrift' requires both current_nc and wind_nc")
        return self

class DriftPathPoint(BaseModel):
    timestamp_utc: str
    lon: float
    lat: float

class DriftResponse(BaseModel):
    engine: str
    direction: Literal['backward', 'forward']
    start_time_utc: str
    end_time_utc: str
    duration_hours: float
    time_step_minutes: int
    particles: int
    particles_in_domain: int
    wind_drift_factor: float
    horizontal_diffusivity_m2_s: float
    metocean_source: str = Field(..., description="'request' | 'config_fallback' | 'netcdf'")
    drift_bearing_deg: float = Field(..., ge=0.0, lt=360.0)
    mean_drift_speed_mps: float
    max_drift_speed_mps: float
    origin_lon: Optional[float] = Field(None, description='Hindcast-inferred discharge longitude.')
    origin_lat: Optional[float] = Field(None, description='Hindcast-inferred discharge latitude.')
    endpoint_lon: Optional[float] = Field(None, description='Forecast terminal longitude.')
    endpoint_lat: Optional[float] = Field(None, description='Forecast terminal latitude.')
    inferred_discharge_time_utc: Optional[str] = None
    centroid_path: List[DriftPathPoint] = Field(default_factory=list)
    particle_endpoints: List[List[float]] = Field(default_factory=list, description='Sub-sampled final positions as [lon, lat] pairs.')
    envelope_geojson: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: UncertaintyEllipse
    warnings: List[str] = Field(default_factory=list)
    wave_provenance: Optional[str] = Field(default='standard_leeway_fallback (u_current + 0.032 * u_wind)', description='Provenance of wave and Stokes drift data used in the drift simulation.')

class AttributionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    origin_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    origin_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    t0_utc: Optional[str] = Field(None, description='Inferred discharge time (ISO-8601 UTC).')
    slick_geojson: Optional[Dict[str, Any]] = Field(None, description='Detected slick polygon; its centroid is used when origin is omitted.')
    uncertainty: Optional[UncertaintyEllipse] = Field(None, description='Drift uncertainty ellipse; a config-sized default is used when omitted.')
    drift_bearing_deg: Optional[float] = Field(None, ge=0.0, lt=360.0, description='Metocean drift bearing enabling the trajectory-congruence factor.')

    @field_validator('drift_bearing_deg', mode='before')
    @classmethod
    def _sanitize_bearing(cls, value: Any) -> Any:
        if value is not None and isinstance(value, float) and math.isnan(value):
            return None
        return value
    ais_csv_path: Optional[str] = Field(None, description='Server-side AIS CSV/CSV.gz path (any schema; auto-detected).')
    ais_records: Optional[List[Dict[str, Any]]] = Field(None, description='Inline AIS records.')
    detection_result: Optional[Dict[str, Any]] = Field(None, description='Detection result dict with slicks[].colocated_ships.')
    temporal_window_hours: Optional[float] = Field(None, gt=0.0, le=720.0)
    chunksize: Optional[int] = Field(None, ge=1000, le=5000000)
    weights: Optional[Dict[str, float]] = Field(None, description='Override attribution factor weights.')
    include_traffic_reconstruction: bool = Field(True, description='Add the spatio-temporal traffic funnel summary.')
    max_candidates: int = Field(50, ge=1, le=1000, description='Maximum ranked vessels returned.')

    @field_validator('t0_utc')
    @classmethod
    def _validate_time(cls, value: Optional[str]) -> Optional[str]:
        return _require_iso_timestamp(value) if value else None

    @model_validator(mode='after')
    def _check_inputs(self) -> 'AttributionRequest':
        if (self.origin_lon is None) != (self.origin_lat is None):
            raise ValueError('origin_lon and origin_lat must be supplied together')
        if self.origin_lon is None and self.slick_geojson is None:
            raise ValueError('provide origin_lon/origin_lat or a slick_geojson')
        return self

class PruneRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    origin_lon: float = Field(..., ge=-180.0, le=180.0)
    origin_lat: float = Field(..., ge=-90.0, le=90.0)
    t0_utc: str
    uncertainty: Optional[UncertaintyEllipse] = None
    ais_csv_path: Optional[str] = None
    ais_records: Optional[List[Dict[str, Any]]] = None
    temporal_window_hours: Optional[float] = Field(None, gt=0.0, le=720.0)
    chunksize: Optional[int] = Field(None, ge=1000, le=5000000)
    include_traffic_reconstruction: bool = True

    @field_validator('t0_utc')
    @classmethod
    def _validate_time(cls, value: str) -> str:
        return _require_iso_timestamp(value)

class TrafficFilterSummary(BaseModel):
    source: str
    temporal_window_hours: float
    window_start_utc: Optional[str] = None
    window_end_utc: Optional[str] = None
    total_records_scanned: int = 0
    records_in_time_window: int = 0
    unique_vessels_in_time_window: int = 0
    candidate_pings: int = 0
    candidate_vessels: int = 0
    rejected_pings: int = Field(0, description='Records inside the time window but outside the ellipse.')
    rejected_vessels: int = 0
    vessel_type_breakdown: Dict[str, int] = Field(default_factory=dict, description='Vessel counts per type inside the envelope.')
    reconstruction_available: bool = True

class CandidateVessel(BaseModel):
    rank: int
    mmsi: int
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None
    callsign: Optional[str] = None
    nav_status: Optional[str] = None
    attribution_score: float = Field(..., ge=0.0, le=100.0, description='Weighted score on a 0-100 scale.')
    attribution_score_normalized: float = Field(..., ge=0.0, le=1.0, description='Same score expressed as a 0-1 fraction.')
    confidence: Literal['high', 'medium', 'low']
    distance_to_origin_m: float = Field(..., ge=0.0)
    temporal_offset_min: float
    cpa_lon: Optional[float] = None
    cpa_lat: Optional[float] = None
    cpa_timestamp_utc: Optional[str] = None
    mean_speed_knots: Optional[float] = None
    max_speed_knots: Optional[float] = None
    ping_count: int = 0
    first_seen_utc: Optional[str] = None
    last_seen_utc: Optional[str] = None
    factor_scores: Dict[str, float] = Field(default_factory=dict)
    factor_contributions: Dict[str, float] = Field(default_factory=dict, description='score x normalised weight (0-1 scale).')
    top_factors: List[str] = Field(default_factory=list)
    explanation: str = ''
    visually_confirmed: bool = Field(False, description='True when visually confirmed by SAR at the slick.')
    evidence_flags: List[str] = Field(default_factory=list, description='Human-readable evidence flags for this candidate.')
    sar_visual_match: Optional[Dict[str, Any]] = Field(None, description='SAR ship detection matched to this vessel.')

class AttributionResponse(BaseModel):
    origin_lon: float
    origin_lat: float
    t0_utc: str
    ellipse: UncertaintyEllipse
    weights: Dict[str, float] = Field(default_factory=dict)
    candidate_count: int
    traffic_filter: Optional[TrafficFilterSummary] = None
    candidates: List[CandidateVessel] = Field(default_factory=list)
    dark_vessel_alerts: List[Dict[str, Any]] = Field(default_factory=list, description='SAR-detected ships near the slick with no AIS match.')
    warnings: List[str] = Field(default_factory=list)
    attribution_geojson: Dict[str, Any] = Field(default_factory=dict)
    artifact_paths: Dict[str, str] = Field(default_factory=dict)

class PruneResponse(BaseModel):
    origin_lon: float
    origin_lat: float
    t0_utc: str
    ellipse: UncertaintyEllipse
    traffic_filter: Optional[TrafficFilterSummary] = None
    pruned_ping_count: int
    candidate_vessel_count: int
    candidate_mmsi: List[int] = Field(default_factory=list)
    pruned_records: List[Dict[str, Any]] = Field(default_factory=list, description='Candidate pings.')
    artifact_paths: Dict[str, str] = Field(default_factory=dict)

class PipelineOptions(DetectionOptions):
    run_drift: bool = True
    drift_duration_hours: Optional[float] = Field(None, gt=0.0, le=720.0)
    drift_time_step_minutes: Optional[int] = Field(None, ge=1, le=1440)
    drift_particles: Optional[int] = Field(None, ge=10, le=20000)
    run_forecast: bool = Field(True, description='Run forward forecast drift integration over the detected slick.')
    forecast_duration_hours: Optional[float] = Field(None, gt=0.0, le=720.0)
    metocean: Optional[List[MetoceanSample]] = None
    wind_drift_factor: Optional[float] = Field(None, gt=0.0, lt=1.0)
    horizontal_diffusivity_m2_s: Optional[float] = Field(None, ge=0.0, le=1000.0)
    engine: Literal['auto', 'lagrangian', 'opendrift'] = 'auto'
    current_nc: Optional[str] = None
    wind_nc: Optional[str] = None
    wave_nc: Optional[str] = None
    oil_type: Optional[str] = None
    run_attribution: bool = True
    ais_csv_path: Optional[str] = None
    ais_records: Optional[List[Dict[str, Any]]] = None
    temporal_window_hours: Optional[float] = Field(None, gt=0.0, le=720.0)
    weights: Optional[Dict[str, float]] = None
    include_traffic_reconstruction: bool = True
    max_candidates: int = Field(50, ge=1, le=1000)
    fetch_live_metocean: bool = Field(True, description='When true, download live Copernicus currents + ERA5 winds centered on the primary slick before drift (degrades gracefully on failure).')
    strict_metocean: bool = Field(False, description='When true, reject config_fallback metocean with HTTP 502 instead of running drift on static constants. Requires live fetch or readable NetCDF files.')
    generate_briefing: bool = Field(False, description='When true, ask Gemini to synthesise an NTRO intelligence narrative from the verified dossier (skipped when GEMINI_API_KEY is unset or the call fails).')

    @model_validator(mode='after')
    def _check_ais(self) -> 'PipelineOptions':
        if self.run_attribution and (not self.ais_csv_path) and (not self.ais_records):
            raise ValueError('run_attribution=true requires ais_csv_path or ais_records (the pipeline service falls back to settings.ais.default_csv).')
        return self

class PipelineByPathRequest(PipelineOptions):
    path: str = Field(..., min_length=1)

class PipelineStageReport(BaseModel):
    stage: str = Field(..., description="'detection' | 'drift' (hindcast) | 'forecast' | 'attribution'")
    status: Literal['completed', 'skipped', 'failed']
    duration_ms: float = 0.0
    detail: str = ''
    error: Optional[str] = None

class PipelineResponse(BaseModel):
    spill_status: Optional[str] = Field(None, description="High-level classification: 'Lookalike', 'Confirmed', 'Suspected', or 'No Spill Detected'.")
    detection: SlickDetectionResponse
    drift: Optional[DriftResponse] = Field(None, description='Backward hindcast: slick -> inferred discharge point.')
    forecast: Optional[DriftResponse] = Field(None, description='Forward forecast: slick -> predicted future extent.')
    attribution: Optional[AttributionResponse] = None
    briefing: Optional['IntelligenceBriefing'] = Field(None, description='LLM-synthesised intelligence narrative (Stage 4).')
    stages: List[PipelineStageReport] = Field(default_factory=list)
    origin_source: str = Field('drift_hindcast', description="'drift_hindcast' | 'slick_centroid_fallback' | 'no_slick'")
    warnings: List[str] = Field(default_factory=list)

class IntelligenceBriefing(BaseModel):
    executive_summary: str = Field(..., description='Plain-language incident summary.')
    primary_suspect_rationale: str = Field(..., description='Why the top candidate vessel is suspected.')
    environmental_risk_level: str = Field(..., description="Qualitative risk: 'low', 'moderate', 'high', or 'critical'.")
    recommended_next_actions: List[str] = Field(default_factory=list, description='Concrete investigative or response actions.')
    status: str = Field('completed', description="'completed' | 'skipped' when Gemini is unavailable or rate-limited.")

class MetadataRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str = Field(..., min_length=1)
    patch_row_offset: int = Field(0, ge=0)
    patch_col_offset: int = Field(0, ge=0)
    patch_size: Optional[int] = Field(None, ge=1, le=100000)

class SceneMetadataResponse(BaseModel):
    path: str
    filename: str
    georeferenced: bool
    source_crs: Optional[str] = None
    target_crs: Optional[str] = None
    timestamp_utc: Optional[str] = None
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    bbox_wgs84: Optional[List[float]] = None
    fallback_region: Optional[str] = None
    metocean_api_params: Optional[Dict[str, Any]] = None

class RegionInfo(BaseModel):
    name: str
    lat: float
    lon: float
    corridor_bearing: float = 0.0

class RegionsResponse(BaseModel):
    regions: List[RegionInfo]

class SceneEntry(BaseModel):
    path: str = Field(..., description='Project-relative path accepted by POST /pipeline.')
    filename: str
    category: str = Field(..., description="Ground-truth class folder: 'Oil' | 'No oil' | 'Lookalike'.")

class ScenesResponse(BaseModel):
    scenes: List[SceneEntry]
    root: Optional[str] = None
    truncated: bool = False

class ClassStatistics(BaseModel):
    class_name: str
    scene_count: int
    mask_count: int
    images_dir: str
    masks_dir: str
    oil_pixel_fraction_mean: float = 0.0

class EvaluationDatasetInfo(BaseModel):
    available: bool
    root: Optional[str] = None
    image_dirname: str = AUTHENTIC_IMAGE_DIRNAME
    mask_dirname: str = AUTHENTIC_MASK_DIRNAME
    classes: List[str] = Field(default_factory=lambda: list(AUTHENTIC_CLASSES))
    class_statistics: List[ClassStatistics] = Field(default_factory=list)
    total_scenes: int = 0

class ModelInfo(BaseModel):
    ready: bool
    architecture: str = 'deeplabv3plus'
    encoder: str = 'resnet34'
    in_channels: int = 3
    num_classes: int = 5
    class_names: List[str] = Field(default_factory=lambda: ['sea_surface', 'oil_spill', 'look_alike', 'ship', 'land'])
    oil_class_index: int = 1
    oil_class_name: str = 'oil_spill'
    ship_class_index: int = 3
    ship_class_name: str = 'ship'
    device: str = 'cpu'
    checkpoint_path: str = ''
    checkpoint_size_bytes: int = 0
    input_size: List[int] = Field(default_factory=lambda: [512, 512])
    confidence_threshold: float = 0.5
    tile_size: int = 512
    stride: int = 256
    tile_batch_size: int = 4
    loaded_at_utc: Optional[str] = None
    load_duration_ms: float = 0.0
    parameters: int = 0
    error: Optional[str] = None
    val_oil_iou: Optional[float] = Field(None, description='Best validation Oil Spill IoU recorded at training time.')

class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded']
    version: str
    uptime_seconds: float
    model: ModelInfo
    paths: Dict[str, Any] = Field(default_factory=dict)
    evaluation_dataset: EvaluationDatasetInfo
    checks: Dict[str, bool] = Field(default_factory=dict)
    metocean: Dict[str, Any] = Field(default_factory=dict, description='Metocean sources drift requests will use.')
    warnings: List[str] = Field(default_factory=list)

class GeoStats(BaseModel):
    location_name: str
    authority: str
    centroid_lat: float
    centroid_lon: float
    bounds: Optional[List[float]] = None
    area_km2: float
    slick_count: int

class Dossier(BaseModel):
    vessel_name: str
    mmsi: int
    vessel_type: str
    avg_speed_knots: Optional[float] = None
    min_distance_m: Optional[float] = None
    last_seen: Optional[str] = None

class IncidentReport(BaseModel):
    has_spill: bool
    geo_stats: Optional[GeoStats] = None
    dossier: Optional[Dossier] = None

class DispatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    alert_level: Literal['CRITICAL', 'HIGH', 'ADVISORY']
    channels: List[str]
    incident_data: IncidentReport

class DispatchResponse(BaseModel):
    status: Literal['queued']
    dispatch_id: str
    simulated: bool = True
    message: str = 'Alert queued for simulated transmission. No real-world messages sent.'