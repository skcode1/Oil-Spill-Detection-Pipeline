from __future__ import annotations
import pytest
from app.schemas import DriftRequest, PipelineByPathRequest, PipelineOptions
from tests.conftest import FAST_DOWNSCALE
FORECAST_POLYGON = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Polygon', 'coordinates': [[[53.0, 26.45], [53.06, 26.45], [53.06, 26.51], [53.0, 26.51], [53.0, 26.45]]]}}]}

def _drift_request(**overrides) -> DriftRequest:
    body = {'source': {'slick_geojson': FORECAST_POLYGON}, 'start_time_utc': '2026-05-01T12:00:00Z', 'duration_hours': 6.0, 'time_step_minutes': 60, 'particles': 100, 'engine': 'lagrangian'}
    body.update(overrides)
    return DriftRequest(**body)

def test_drift_hindcast_uses_builtin_engine_without_netcdf(services):
    request = _drift_request()
    response = services.drift.run_hindcast(request)
    assert response.engine in ('lagrangian', 'analytical')
    assert response.direction == 'backward'
    assert response.origin_lon is not None and response.origin_lat is not None
    assert -180.0 <= response.origin_lon <= 180.0
    assert -90.0 <= response.origin_lat <= 90.0
    assert response.uncertainty is not None
    assert response.uncertainty.semi_major_axis_meters > 0

def test_drift_forecast_roundtrip(services):
    request = _drift_request()
    response = services.drift.run_forecast(request)
    assert response.direction == 'forward'
    assert response.endpoint_lon is not None
    assert response.centroid_path, 'expected a sampled centroid path'

def test_drift_step_cap_is_enforced(services):
    request = _drift_request(duration_hours=720.0, time_step_minutes=1, particles=10)
    from app.errors import DriftEngineError
    with pytest.raises(DriftEngineError):
        services.drift.run_hindcast(request)

def test_pipeline_end_to_end_authentic_scene(services, sample_scenes):
    image = sample_scenes['Oil'][0]
    options = PipelineOptions(run_attribution=False, run_drift=True, run_forecast=True, downscale=FAST_DOWNSCALE, drift_particles=50, drift_duration_hours=6.0, drift_time_step_minutes=60, save_artifacts=False, include_preview=False)
    result = services.pipeline.run_path(PipelineByPathRequest(path=str(image), **options.model_dump()))
    assert result.detection is not None
    assert [s.stage for s in result.stages] == ['detection', 'drift', 'forecast', 'attribution']
    assert result.stages[0].status == 'completed'
    jsonable = result.model_dump(mode='json')
    assert jsonable['detection']['scene']['filename']
    if result.origin_source != 'no_slick':
        assert result.drift is not None
        assert result.forecast is not None