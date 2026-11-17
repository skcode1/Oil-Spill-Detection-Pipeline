from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import ValidationError
from app.api.deps import AppServices, get_briefing_service, get_drift_service, get_pipeline_service, get_services, get_settings, health_report
from app.config import AppSettings
from app.errors import InvalidInputError
from app.schemas import DispatchRequest, DispatchResponse, Dossier, DriftRequest, DriftResponse, GeoStats, HealthResponse, IncidentReport, IntelligenceBriefing, PipelineByPathRequest, PipelineOptions, PipelineResponse, RegionInfo, RegionsResponse, SceneEntry, ScenesResponse
from app.services import BriefingService, DriftService, PipelineService
from app.services.incident_store import get_incident_store
logger = logging.getLogger(__name__)
router = APIRouter()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
LEGACY_MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES
PIPELINE_DESCRIPTION = '\nRun detection, then drift in **both** directions over the detected slick, then\nattribution against the *hindcast* envelope:\n\n* ``detection`` - U-Net segmentation, slick geometry, statistics and age estimate.\n* ``drift`` (**hindcast**) - backward integration: slick -> inferred discharge\n  point, discharge instant, uncertainty ellipse and drift bearing.\n* ``forecast`` - forward integration over the same seed: the extent the slick is\n  expected to reach.\n* ``attribution`` - AIS pruning and multi-factor ranking.\n\nEvery stage degrades independently and records one entry in ``stages``, so a\nfailed forecast never hides a working hindcast, and a failed attribution never\nhides the drift result.\n'
SAR_ALLOWED_SUFFIXES = {'.tif', '.tiff'}
SAR_ALLOWED_MIME_TYPES = {'image/tiff', 'image/x-tiff', 'application/octet-stream'}

def _parse_options(options_json: Optional[str]) -> PipelineOptions:
    if not options_json or not options_json.strip():
        return PipelineOptions(run_attribution=False)
    try:
        return PipelineOptions.model_validate_json(options_json)
    except ValidationError as exc:
        raise InvalidInputError("The 'options' form field is not a valid PipelineOptions object.", details=exc.errors(include_url=False)) from exc

def _validate_sar_upload(file: UploadFile) -> None:
    filename = (file.filename or '').strip()
    suffix = Path(filename).suffix.lower()
    mime = (file.content_type or '').lower()
    if suffix not in SAR_ALLOWED_SUFFIXES:
        raise InvalidInputError("Only GeoTIFF SAR scenes ('.tif' or '.tiff') are accepted for upload.", details={'field': 'file', 'accepted': sorted(SAR_ALLOWED_SUFFIXES)})
    if mime and mime not in SAR_ALLOWED_MIME_TYPES:
        raise InvalidInputError('Non-SAR payload detected: upload must be a GeoTIFF SAR scene.', details={'field': 'file', 'content_type': mime, 'accepted': sorted(SAR_ALLOWED_MIME_TYPES)})

async def _read_upload(file: UploadFile, *, limit_bytes: int) -> bytes:
    _validate_sar_upload(file)
    data = await file.read()
    if not data:
        raise InvalidInputError('The uploaded file is empty.')
    if len(data) > limit_bytes:
        raise InvalidInputError(f'Uploaded scene exceeds the {limit_bytes // (1024 * 1024)} MiB limit.', details={'size_bytes': len(data), 'limit_bytes': limit_bytes})
    return data

@router.get('/health', response_model=HealthResponse, tags=['system'], summary='Liveness and readiness report')
def health(services: AppServices=Depends(get_services)) -> HealthResponse:
    return health_report(services)

@router.get('/regions', response_model=RegionsResponse, tags=['system'], summary='Configured maritime corridor anchors')
def regions(settings: AppSettings=Depends(get_settings)) -> RegionsResponse:
    entries = [RegionInfo(name=str(region.get('name') or 'unnamed'), lat=float(region['lat']), lon=float(region['lon']), corridor_bearing=float(region.get('corridor_bearing') or 0.0)) for region in settings.regions if region.get('lat') is not None and region.get('lon') is not None]
    return RegionsResponse(regions=entries)
SCENES_PER_CLASS = 30
SCENES_MAX_TOTAL = 120

@router.get('/scenes', response_model=ScenesResponse, tags=['system'], summary='Server-side SAR scenes for the dataset browser')
def scenes(settings: AppSettings=Depends(get_settings)) -> ScenesResponse:
    root = settings.authentic_test_root
    entries: list[SceneEntry] = []
    truncated = False
    if root is not None and root.is_dir():
        images_root = root / 'Images'
        class_dirs = sorted((entry for entry in images_root.iterdir() if entry.is_dir())) if images_root.is_dir() else []
        for class_dir in class_dirs:
            files = sorted((path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in {'.tif', '.tiff'}))
            for path in files[:SCENES_PER_CLASS]:
                if len(entries) >= SCENES_MAX_TOTAL:
                    truncated = True
                    break
                relative = path.relative_to(settings.project_root) if path.is_relative_to(settings.project_root) else path
                entries.append(SceneEntry(path=str(relative), filename=path.name, category=class_dir.name))
            if truncated:
                break
    return ScenesResponse(scenes=entries, root=str(root) if root is not None else None, truncated=truncated)

@router.post('/pipeline', response_model=PipelineResponse, tags=['pipeline'], summary='Detection + bidirectional drift + attribution for a server-side scene', description=PIPELINE_DESCRIPTION)
def run_pipeline(request: PipelineByPathRequest, pipeline: PipelineService=Depends(get_pipeline_service)) -> PipelineResponse:
    logger.info('Pipeline request for scene %s', request.path)
    return pipeline.run_path(request)

@router.post('/pipeline/upload', response_model=PipelineResponse, tags=['pipeline'], summary='Detection + bidirectional drift + attribution for an uploaded scene', description='\nMultipart alternative to ``POST /pipeline`` for callers that hold the scene in\nmemory (browser upload, downstream microservice).\n\n* ``file`` - the SAR scene (GeoTIFF / NPZ / PNG / JPEG).\n* ``options`` - optional JSON object carrying any :class:`PipelineOptions` field,\n  e.g. ``{"ais_csv_path": "data/ais.csv", "engine": "opendrift"}``.\n\n**Attribution is skipped unless the ``options`` JSON names an AIS source**\n(``ais_csv_path`` or ``ais_records``): ``PipelineOptions`` rejects an attribution\nrun without one, so the bare default is deliberately attribution-free.  ``stages``\nin the response always states what ran, what was skipped and why.\n')
async def run_pipeline_upload(file: UploadFile=File(..., description='SAR scene to analyse.'), options: Optional[str]=Form(None, description='PipelineOptions as JSON, e.g. {"ais_csv_path": "ais.csv"}'), pipeline: PipelineService=Depends(get_pipeline_service), settings: AppSettings=Depends(get_settings)) -> PipelineResponse:
    data = await _read_upload(file, limit_bytes=settings.server.max_upload_bytes)
    parsed = _parse_options(options)
    logger.info('Pipeline upload request: %s (%.1f MiB, attribution=%s, hindcast=%s, forecast=%s)', file.filename, len(data) / (1024 * 1024), parsed.run_attribution, parsed.run_drift, parsed.run_forecast)
    return pipeline.run_bytes(data, file.filename, options=parsed)

@router.post('/briefing', response_model=IntelligenceBriefing, tags=['briefing'], summary='Synthesise an NTRO intelligence narrative from a verified dossier.', description='\nStage 4 of the hybrid architecture.  Accepts a deterministic dossier JSON\n(detection + drift + attribution) and returns a Gemini-structured NTRO\nintelligence narrative.  The stage is read-only downstream of attribution and\nnever blocks the evidence dossier: when ``GEMINI_API_KEY`` is unset or the call\nfails, the response reports ``status="skipped"``.\n')
def generate_briefing_endpoint(dossier: Dict[str, Any], briefing: BriefingService=Depends(get_briefing_service)) -> IntelligenceBriefing:
    return briefing.generate_briefing(dossier)

@router.post('/drift/hindcast', response_model=DriftResponse, tags=['drift'], summary='Backward drift: slick -> inferred discharge point')
def drift_hindcast(request: DriftRequest, drift: DriftService=Depends(get_drift_service)) -> DriftResponse:
    return drift.run_hindcast(request)

@router.post('/drift/forecast', response_model=DriftResponse, tags=['drift'], summary='Forward drift: slick -> predicted future extent')
def drift_forecast(request: DriftRequest, drift: DriftService=Depends(get_drift_service)) -> DriftResponse:
    return drift.run_forecast(request)

@router.get('/incidents/latest', response_model=IncidentReport, tags=['incidents'], summary='Latest completed pipeline incident', responses={404: {'description': 'No incident stored yet'}})
def latest_incident() -> IncidentReport:
    store = get_incident_store()
    pipeline_response = store.latest()
    if pipeline_response is None:
        raise InvalidInputError('No incident data available. Run the pipeline first.', details={'code': 'artifact_not_found'})
    detection = pipeline_response.detection
    attribution = pipeline_response.attribution
    has_spill = len(detection.slicks) > 0
    geo_stats = None
    if has_spill and detection.statistics:
        stats = detection.statistics
        scene = detection.scene
        geolocated_slick = next((s for s in detection.slicks if s.centroid_lat is not None and s.centroid_lon is not None), None)
        location_name = scene.filename if scene else 'SAR scene'
        centroid_lat = geolocated_slick.centroid_lat if geolocated_slick else scene.center_lat
        centroid_lon = geolocated_slick.centroid_lon if geolocated_slick else scene.center_lon
        bounds = None
        if detection.spill_geojson and detection.spill_geojson.get('features'):
            coords = []
            for feature in detection.spill_geojson['features']:
                geom = feature.get('geometry')
                if geom and geom.get('type') == 'Polygon':
                    for ring in geom.get('coordinates', []):
                        for point in ring:
                            coords.append(point)
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bounds = [min(lons), min(lats), max(lons), max(lats)]
        authority = 'MRCC General Desk'
        if centroid_lat is not None and centroid_lon is not None:
            pass
        geo_stats = GeoStats(location_name=location_name, authority=authority, centroid_lat=centroid_lat or 0.0, centroid_lon=centroid_lon or 0.0, bounds=bounds, area_km2=stats.total_slick_area_km2 or 0.0, slick_count=len(detection.slicks))
    dossier = None
    if attribution and attribution.candidates:
        top = attribution.candidates[0]
        dossier = Dossier(vessel_name=top.vessel_name or f'MMSI {top.mmsi}', mmsi=top.mmsi, vessel_type=top.vessel_type or 'Unknown', avg_speed_knots=top.mean_speed_knots, min_distance_m=top.distance_to_origin_m, last_seen=top.last_seen_utc)
    return IncidentReport(has_spill=has_spill, geo_stats=geo_stats, dossier=dossier)

@router.post('/alerts/dispatch', response_model=DispatchResponse, tags=['alerts'], summary='Dispatch an alert for an incident (simulated)')
def dispatch_alert(request: DispatchRequest) -> DispatchResponse:
    import uuid
    valid_levels = {'CRITICAL', 'HIGH', 'ADVISORY'}
    if request.alert_level not in valid_levels:
        raise InvalidInputError(f'Invalid alert_level: must be one of {sorted(valid_levels)}', details={'field': 'alert_level'})
    if not request.channels:
        raise InvalidInputError('At least one dispatch channel must be selected', details={'field': 'channels'})
    if not request.incident_data.has_spill and request.incident_data.geo_stats is None:
        raise InvalidInputError('Incident data must include geo_stats when has_spill is true', details={'field': 'incident_data.geo_stats'})
    dispatch_id = f'dispatch_{uuid.uuid4().hex[:12]}'
    logger.info('SIMULATED dispatch %s: level=%s channels=%s incident_has_spill=%s', dispatch_id, request.alert_level, request.channels, request.incident_data.has_spill)
    return DispatchResponse(status='queued', dispatch_id=dispatch_id, simulated=True, message='Alert queued for simulated transmission. No real-world messages sent.')