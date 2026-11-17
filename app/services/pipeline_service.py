from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import ValidationError
from app.config import AppSettings
from app.errors import InvalidInputError, PipelineError
from app.schemas import AttributionRequest, AttributionResponse, DriftRequest, DriftResponse, DriftSource, IntelligenceBriefing, PipelineByPathRequest, PipelineOptions, PipelineResponse, PipelineStageReport, utc_iso
from app.services.attribution_service import AttributionService
from app.services.briefing_service import BriefingService
from app.services.detection_service import DetectionOutcome, DetectionService
from app.services.drift_service import DriftService
from app.services.incident_store import get_incident_store
from src.environment_fetcher import fetch_incident_metocean
logger = logging.getLogger(__name__)
ORIGIN_FROM_DRIFT = 'drift_hindcast'
ORIGIN_FROM_SLICK = 'slick_centroid_fallback'
ORIGIN_NO_SLICK = 'no_slick'
STATUS_NO_OIL = 'No Oil'
STATUS_LOOKALIKE = 'Lookalike'
STATUS_CONFIRMED = 'Confirmed Spill'
STATUS_SUSPECTED = 'Suspected'

def _milliseconds(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)

def _resolve_incident_meteocean_sync(*, slicks: list[dict[str, Any]], scene_timestamp_utc: str | None, hindcast_hours: float, forecast_hours: float, request_id: str, timeout_sec: float=25.0) -> tuple[Optional[dict[str, str]], list[str]]:
    warnings: list[str] = []
    if not slicks:
        warnings.append('No slicks detected; skipping live metocean fetch.')
        return (None, warnings)
    primary_slick = max(slicks, key=lambda s: float(s.get('area_km2') or 0.0))
    lon = primary_slick.get('centroid_lon')
    lat = primary_slick.get('centroid_lat')
    if lon is None or lat is None:
        warnings.append('Primary slick missing centroid; falling back to config metocean.')
        return (None, warnings)
    acq_time = scene_timestamp_utc or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rid = request_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path('results/api') / rid / 'metocean'
    try:
        paths = fetch_incident_metocean(lon=float(lon), lat=float(lat), detection_time_utc=acq_time, hindcast_hours=int(hindcast_hours), forecast_hours=int(forecast_hours), out_dir=str(out_dir))
        return (paths, warnings)
    except Exception as exc:
        msg = f'Live metocean fetch failed ({type(exc).__name__}: {exc}); degrading to config_fallback.'
        logger.warning(msg)
        warnings.append(msg)
        return (None, warnings)

@dataclass(frozen=True)
class DriftSeed:
    acquisition_utc: str
    slick_geojson: Dict[str, Any]

class PipelineService:

    def __init__(self, settings: AppSettings, detection: DetectionService, drift: DriftService, attribution: AttributionService, briefing: BriefingService) -> None:
        self.settings = settings
        self.detection = detection
        self.drift = drift
        self.attribution = attribution
        self.briefing = briefing

    def run_path(self, request: PipelineByPathRequest) -> PipelineResponse:
        return self._execute(lambda: self.detection.detect_path(request.path, options=request), request)

    def run_bytes(self, data: bytes, filename: Optional[str], options: Optional[PipelineOptions]=None) -> PipelineResponse:
        if options is None:
            options = PipelineOptions()

        def detect() -> DetectionOutcome:
            return self.detection.detect_bytes(data, filename, options=options)
        return self._execute(detect, options)

    @staticmethod
    def _evaluate_spill_status(outcome: DetectionOutcome) -> Tuple[str, bool]:
        resolved_status = getattr(outcome.response, 'status', None)
        if resolved_status in (STATUS_NO_OIL, STATUS_LOOKALIKE, STATUS_CONFIRMED, STATUS_SUSPECTED):
            return (resolved_status, resolved_status == STATUS_LOOKALIKE)
        slicks_count = len(getattr(outcome.response, 'slicks', []))
        if slicks_count == 0:
            return (STATUS_NO_OIL, False)
        if getattr(outcome, 'is_lookalike', False) or getattr(outcome.response, 'is_lookalike', False):
            return (STATUS_LOOKALIKE, True)
        return (STATUS_CONFIRMED, False)

    def _execute(self, detect: Callable[[], DetectionOutcome], options: PipelineOptions) -> PipelineResponse:
        warnings: List[str] = []
        started = time.perf_counter()
        outcome = detect()
        response = outcome.response
        warnings.extend(outcome.warnings)
        spill_status, is_lookalike = self._evaluate_spill_status(outcome)
        slicks = getattr(response, 'slicks', [])
        detection_detail = f'{len(slicks)} slick polygon(s), {outcome.total_oil_pixels} oil pixels | Status: {spill_status}'
        stages = [PipelineStageReport(stage='detection', status='completed', duration_ms=outcome.duration_ms or _milliseconds(started), detail=detection_detail)]
        if spill_status == STATUS_NO_OIL or len(slicks) == 0:
            warnings.append('Scene contains no petroleum slick; drift and attribution stages bypassed.')
            stages.extend((PipelineStageReport(stage=stage, status='skipped', detail='bypassed: scene verified clean (No Oil)') for stage in ('drift', 'forecast', 'attribution')))
            return PipelineResponse(spill_status=STATUS_NO_OIL, detection=response, stages=stages, origin_source=ORIGIN_NO_SLICK, warnings=warnings)
        if is_lookalike or spill_status == STATUS_LOOKALIKE:
            warnings.append('Scene classified as natural lookalike; vessel attribution bypassed to prevent false accusation.')
            stages.extend((PipelineStageReport(stage=stage, status='skipped', detail='bypassed: lookalike feature detected') for stage in ('drift', 'forecast', 'attribution')))
            return PipelineResponse(spill_status=STATUS_LOOKALIKE, detection=response, stages=stages, origin_source=ORIGIN_FROM_SLICK, warnings=warnings)
        try:
            seed = self._drift_seed(outcome, options, warnings)
        except InvalidInputError as err:
            logger.warning('Could not construct drift seed: %s', err)
            stages.extend((PipelineStageReport(stage=stage, status='failed', detail='invalid slick geometry') for stage in ('drift', 'forecast', 'attribution')))
            return PipelineResponse(spill_status=spill_status, detection=response, stages=stages, origin_source=ORIGIN_FROM_SLICK, warnings=warnings + [str(err)])
        metocean_override: Optional[Dict[str, str]] = None
        if options.fetch_live_metocean and slicks:
            metocean_override, metocean_warnings = _resolve_incident_meteocean_sync(slicks=[s.model_dump(mode='json') for s in slicks], scene_timestamp_utc=seed.acquisition_utc, hindcast_hours=self._hindcast_hours(options), forecast_hours=self._forecast_hours(options), request_id=getattr(response, 'request_id', '') or '', timeout_sec=self.settings.metocean.fetch_timeout_sec)
            warnings.extend(metocean_warnings)
        detection_result = response.model_dump(mode='json')
        drift_response: Optional[DriftResponse] = None
        origin_source = ORIGIN_FROM_SLICK
        if options.run_drift:
            drift_response, drift_stage = self._run_drift(options, seed, metocean_paths=metocean_override, strict_metocean=options.strict_metocean)
            stages.append(drift_stage)
            if drift_response is not None:
                warnings.extend(drift_response.warnings)
                if drift_response.origin_lon is not None and drift_response.origin_lat is not None:
                    origin_source = ORIGIN_FROM_DRIFT
        else:
            stages.append(PipelineStageReport(stage='drift', status='skipped', detail='run_drift=false'))
        forecast_response: Optional[DriftResponse] = None
        if options.run_forecast:
            forecast_response, forecast_stage = self._run_forecast(options, seed, metocean_paths=metocean_override, strict_metocean=options.strict_metocean)
            stages.append(forecast_stage)
            if forecast_response is not None:
                warnings.extend(forecast_response.warnings)
        else:
            stages.append(PipelineStageReport(stage='forecast', status='skipped', detail='run_forecast=false'))
        attribution_response = None
        if options.run_attribution:
            attribution_response, attribution_stage = self._run_attribution(outcome, options, drift_response, detection_result)
            stages.append(attribution_stage)
        else:
            stages.append(PipelineStageReport(stage='attribution', status='skipped', detail='run_attribution=false'))
        briefing_response = None
        if options.generate_briefing:
            briefing_response, briefing_stage = self._run_briefing(detection_result, drift_response, attribution_response)
            stages.append(briefing_stage)
        result = PipelineResponse(spill_status=spill_status, detection=response, drift=drift_response, forecast=forecast_response, attribution=attribution_response, briefing=briefing_response, stages=stages, origin_source=origin_source, warnings=warnings)
        try:
            incident_store = get_incident_store()
            incident_store.add(result)
        except Exception as exc:
            logger.warning('Could not append incident to local store: %s', exc)
        return result

    def _run_drift(self, options: PipelineOptions, seed: DriftSeed, *, metocean_paths: Optional[Dict[str, str]]=None, strict_metocean: bool=False) -> Tuple[Optional[DriftResponse], PipelineStageReport]:
        started = time.perf_counter()
        try:
            request = self._drift_request(options, seed, self._hindcast_hours(options))
            drift_response = self.drift.run_hindcast(request, metocean_paths=metocean_paths, strict_metocean=strict_metocean)
        except Exception as exc:
            logger.exception('Pipeline drift hindcast stage failed')
            return (None, PipelineStageReport(stage='drift', status='failed', duration_ms=_milliseconds(started), detail='falling back to slick centroid as attribution origin', error=self._describe(exc)))
        return (drift_response, PipelineStageReport(stage='drift', status='completed', duration_ms=_milliseconds(started), detail=f'{drift_response.engine} hindcast ({drift_response.duration_hours:.1f} h) -> ' + self._origin_text(drift_response)))

    def _run_forecast(self, options: PipelineOptions, seed: DriftSeed, *, metocean_paths: Optional[Dict[str, str]]=None, strict_metocean: bool=False) -> Tuple[Optional[DriftResponse], PipelineStageReport]:
        started = time.perf_counter()
        try:
            request = self._drift_request(options, seed, self._forecast_hours(options))
            forecast_response = self.drift.run_forecast(request, metocean_paths=metocean_paths, strict_metocean=strict_metocean)
        except Exception as exc:
            logger.exception('Pipeline drift forecast stage failed')
            return (None, PipelineStageReport(stage='forecast', status='failed', duration_ms=_milliseconds(started), detail='hindcast and attribution results remain unaffected', error=self._describe(exc)))
        return (forecast_response, PipelineStageReport(stage='forecast', status='completed', duration_ms=_milliseconds(started), detail=f'{forecast_response.engine} forecast ({forecast_response.duration_hours:.1f} h) -> ' + self._endpoint_text(forecast_response)))

    @staticmethod
    def _origin_text(drift_response: DriftResponse) -> str:
        if drift_response.origin_lon is None or drift_response.origin_lat is None:
            return 'no geographic origin (pixel-space scene)'
        return f'({drift_response.origin_lon:.5f}, {drift_response.origin_lat:.5f})'

    @staticmethod
    def _endpoint_text(forecast: DriftResponse) -> str:
        if forecast.endpoint_lon is None or forecast.endpoint_lat is None:
            return 'no geographic endpoint (pixel-space scene)'
        return f'({forecast.endpoint_lon:.5f}, {forecast.endpoint_lat:.5f}) by {forecast.end_time_utc}'

    @staticmethod
    def _drift_seed(outcome: DetectionOutcome, options: PipelineOptions, warnings: List[str]) -> DriftSeed:
        scene = outcome.response.scene
        acquisition = options.acquisition_time_utc or scene.acquisition_time_utc
        if not acquisition:
            acquisition = utc_iso()
            warnings.append('Scene carries no acquisition time: drift reference initialized from current UTC.')
        slick = outcome.geojson or outcome.response.spill_geojson
        if not slick or not slick.get('features'):
            raise InvalidInputError('detection produced no slick geometry to seed drift')
        return DriftSeed(acquisition_utc=acquisition, slick_geojson=slick)

    def _hindcast_hours(self, options: PipelineOptions) -> float:
        return float(options.drift_duration_hours or self.settings.metocean.fallback_hindcast_hours)

    def _forecast_hours(self, options: PipelineOptions) -> float:
        return float(options.forecast_duration_hours or self.settings.metocean.fallback_forecast_hours)

    def _drift_request(self, options: PipelineOptions, seed: DriftSeed, duration_hours: float) -> DriftRequest:
        try:
            return DriftRequest(source=DriftSource(slick_geojson=seed.slick_geojson), start_time_utc=seed.acquisition_utc, duration_hours=duration_hours, time_step_minutes=options.drift_time_step_minutes or int(self.settings.metocean.time_step_minutes), particles=options.drift_particles or int(self.settings.metocean.particles), metocean=options.metocean, wind_drift_factor=options.wind_drift_factor, horizontal_diffusivity_m2_s=options.horizontal_diffusivity_m2_s, engine=options.engine, current_nc=options.current_nc, wind_nc=options.wind_nc, wave_nc=options.wave_nc, oil_type=options.oil_type)
        except ValidationError as exc:
            raise InvalidInputError('Drift parameters validation failed', details={'errors': exc.errors(include_url=False)}) from exc

    def _run_attribution(self, outcome: DetectionOutcome, options: PipelineOptions, drift_response: Optional[DriftResponse], detection_result: Optional[Dict[str, Any]]=None) -> Tuple[Optional[AttributionResponse], PipelineStageReport]:
        started = time.perf_counter()
        scene = outcome.response.scene
        slick = outcome.geojson or outcome.response.spill_geojson
        origin_lon: Optional[float] = None
        origin_lat: Optional[float] = None
        t0_utc: Optional[str] = None
        uncertainty = None
        bearing: Optional[float] = None
        if drift_response is not None:
            origin_lon = drift_response.origin_lon
            origin_lat = drift_response.origin_lat
            t0_utc = drift_response.inferred_discharge_time_utc or drift_response.start_time_utc
            uncertainty = drift_response.uncertainty
            bearing = drift_response.drift_bearing_deg
        else:
            centroid = outcome.primary_centroid_lonlat
            if centroid is not None:
                origin_lon, origin_lat = centroid
            t0_utc = options.acquisition_time_utc or scene.acquisition_time_utc
        if origin_lon is None or origin_lat is None:
            return (None, PipelineStageReport(stage='attribution', status='skipped', duration_ms=_milliseconds(started), detail='scene is not georeferenced (missing coordinates)'))
        ais_csv_path = options.ais_csv_path
        ais_records = options.ais_records
        if ais_csv_path is None and ais_records is None:
            default_csv = self.settings.ais.default_csv
            if default_csv is not None and default_csv.is_file():
                ais_csv_path = str(default_csv)
                logger.info('Using configured AIS repository: %s', default_csv)
        if ais_csv_path is None and ais_records is None:
            return (None, PipelineStageReport(stage='attribution', status='skipped', duration_ms=_milliseconds(started), detail='no AIS source available (ais_csv_path or ais_records)'))
        try:
            response = self.attribution.attribute(AttributionRequest(origin_lon=origin_lon, origin_lat=origin_lat, t0_utc=t0_utc, slick_geojson=slick, uncertainty=uncertainty, drift_bearing_deg=bearing, ais_csv_path=ais_csv_path, ais_records=ais_records, temporal_window_hours=options.temporal_window_hours, weights=options.weights, include_traffic_reconstruction=options.include_traffic_reconstruction, max_candidates=options.max_candidates, detection_result=detection_result))
        except PipelineError:
            raise
        except Exception as exc:
            logger.exception('Pipeline attribution stage failed')
            return (None, PipelineStageReport(stage='attribution', status='failed', duration_ms=_milliseconds(started), detail='attribution failed; detection and drift results retained', error=self._describe(exc)))
        candidates = getattr(response, 'candidates', []) or []
        count = len(candidates)
        top = candidates[0] if count > 0 else None
        detail = f'{count} ranked candidate(s)'
        if top is not None:
            detail += f' | Top MMSI: {top.mmsi} (Score: {top.attribution_score:.1f}/100)'
        else:
            detail += ' | No vessel correlation in temporal-spatial envelope'
        return (response, PipelineStageReport(stage='attribution', status='completed', duration_ms=_milliseconds(started), detail=detail))

    @staticmethod
    def _describe(exc: Exception) -> str:
        return f'{type(exc).__name__}: {exc}'

    def _run_briefing(self, detection_result: Dict[str, Any], drift_response: Optional[DriftResponse], attribution_response: Optional[AttributionResponse]) -> Tuple[Optional[IntelligenceBriefing], PipelineStageReport]:
        started = time.perf_counter()
        dossier: Dict[str, Any] = {'detection': detection_result.get('detection') if isinstance(detection_result, dict) else detection_result, 'drift': drift_response.model_dump(mode='json') if drift_response else None, 'attribution': attribution_response.model_dump(mode='json') if attribution_response else None}
        try:
            briefing = self.briefing.generate_briefing(dossier)
            return (briefing, PipelineStageReport(stage='briefing', status='completed', duration_ms=_milliseconds(started), detail='Gemini intelligence narrative generated'))
        except Exception as exc:
            logger.warning('Briefing stage skipped: %s', exc)
            return (None, PipelineStageReport(stage='briefing', status='skipped', duration_ms=_milliseconds(started), detail=f'briefing unavailable ({type(exc).__name__})'))