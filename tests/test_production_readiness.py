from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import API_PREFIX, create_app
from app.schemas import AttributionRequest, DetectionOptions
from app.services import DetectionService
from app.services.attribution_service import AttributionService
from tests.conftest import FAST_DOWNSCALE

def _make_app(services):
    return TestClient(create_app(settings=services.settings, services=services, load_model=False), raise_server_exceptions=False)

def test_no_synthetic_ais(services):
    attribution: AttributionService = services.attribution
    req = AttributionRequest(origin_lon=-20.0, origin_lat=-30.0, t0_utc='2026-05-01T00:00:00Z', slick_geojson=None, uncertainty=None, drift_bearing_deg=None, ais_csv_path=None, ais_records=[{'mmsi': 999999999, 'lon': 0.0, 'lat': 0.0, 'timestamp': '2026-05-01T00:00:00Z', 'sog': 10.0, 'cog': 0.0, 'heading': 0, 'nav_status': 'Under way using engine'}], temporal_window_hours=6.0, weights=None, include_traffic_reconstruction=True, max_candidates=10, detection_result=None)
    result = attribution.attribute(req)
    assert result.candidates == []
    assert any(('zero_ais_candidates_in_envelope' in w for w in result.warnings or []))

def test_lookalike_suppression(services, sample_scenes):
    detection: DetectionService = services.detection
    lookalike_images = sample_scenes.get('Lookalike', [])
    if not lookalike_images:
        pytest.skip('No look-alike scenes in authentic corpus')
    image = lookalike_images[0]
    outcome = detection.detect_path(str(image), options=DetectionOptions(downscale=FAST_DOWNSCALE))
    response = outcome.response
    assert response.status in ('Lookalike', 'No Oil', 'Confirmed Spill')

def test_dynamic_scene_bounds(services, sample_scenes):
    detection: DetectionService = services.detection
    oil_images = sample_scenes.get('Oil', [])
    nooil_images = sample_scenes.get('No oil', [])
    if not oil_images or not nooil_images:
        pytest.skip('Need both Oil and No-oil scenes for bounds comparison')
    outcome1 = detection.detect_path(str(oil_images[0]), options=DetectionOptions(downscale=FAST_DOWNSCALE))
    outcome2 = detection.detect_path(str(nooil_images[0]), options=DetectionOptions(downscale=FAST_DOWNSCALE))
    bounds1 = outcome1.response.scene_bounds
    bounds2 = outcome2.response.scene_bounds
    assert bounds1 is not None, 'First scene missing scene_bounds'
    assert bounds2 is not None, 'Second scene missing scene_bounds'
    assert bounds1 != bounds2, 'Two different scenes returned identical bounds'
    assert len(bounds1) == 4 and len(bounds2) == 4

def test_health_readiness(services):
    client = _make_app(services)
    response = client.get(f'{API_PREFIX}/health')
    assert response.status_code == 200
    data = response.json()
    assert data['model']['ready'] is True
    assert data['status'] in ('ok', 'degraded')
    checks = data.get('checks', {})
    assert checks.get('output_dir_writable') is True, 'results/ not writable'
    assert checks.get('artifact_dir_writable') is True, 'artifacts/ not writable'
    metocean = data.get('metocean', {})
    assert isinstance(metocean, dict)
    assert services.settings.ais.allow_synthetic_ais is False