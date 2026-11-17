from __future__ import annotations
import io
import json
import cv2
import numpy as np
import tifffile
from fastapi.testclient import TestClient
from app.main import API_PREFIX, create_app
from tests.conftest import FAST_DOWNSCALE

def _scene_bytes(services, image_path: str, downscale: int=FAST_DOWNSCALE) -> tuple[bytes, str]:
    from app.services import image_io
    scene = image_io.load_scene(str(image_path), services.settings)
    array = image_io.downscale_array(scene.array, downscale)
    from pathlib import Path
    suffix = Path(image_path).suffix.lower()
    if suffix in ('.tif', '.tiff'):
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, np.asarray(array))
        return (buffer.getvalue(), Path(image_path).name)
    ok, encoded = cv2.imencode('.png', np.asarray(array))
    assert ok
    return (bytes(encoded), Path(image_path).name + '.png')

def test_health_reports_model_and_dataset(services):
    client = TestClient(create_app(settings=services.settings, services=services, load_model=False), raise_server_exceptions=False)
    response = client.get(f'{API_PREFIX}/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] in ('ok', 'degraded')
    assert data['model']['ready'] is True
    assert data['evaluation_dataset']['available'] is True
    assert data['evaluation_dataset']['total_scenes'] > 0
    assert 'X-Request-ID' in response.headers
    assert response.headers['X-Request-ID']
    assert float(response.headers['X-Process-Time-Ms']) >= 0.0

def test_pipeline_upload_representative_scene(services, sample_scenes, tmp_path):
    client = TestClient(create_app(settings=services.settings, services=services, load_model=False), raise_server_exceptions=False)
    image = sample_scenes['Oil'][0]
    payload, filename = _scene_bytes(services, str(image))
    options = {'run_attribution': False, 'run_drift': True, 'run_forecast': True, 'downscale': FAST_DOWNSCALE, 'drift_particles': 50, 'drift_duration_hours': 2.0, 'drift_time_step_minutes': 30, 'save_artifacts': False, 'include_preview': False}
    response = client.post(f'{API_PREFIX}/pipeline/upload', files={'file': (filename, payload, 'image/tiff')}, data={'options': json.dumps(options)})
    assert response.status_code == 200, response.text[:2000]
    data = response.json()
    assert data['detection'] is not None
    assert [s['stage'] for s in data['stages']] == ['detection', 'drift', 'forecast', 'attribution']

def test_upload_empty_file_rejected(services):
    client = TestClient(create_app(settings=services.settings, services=services, load_model=False), raise_server_exceptions=False)
    response = client.post(f'{API_PREFIX}/pipeline/upload', files={'file': ('empty.tif', b'', 'image/tiff')}, data={'options': '{"run_attribution": false}'})
    assert response.status_code in (400, 413, 422)
    assert response.json()['success'] is False

def test_regions_endpoint_lists_configured_anchors(services):
    client = TestClient(create_app(settings=services.settings, services=services, load_model=False), raise_server_exceptions=False)
    response = client.get(f'{API_PREFIX}/regions')
    assert response.status_code == 200
    body = response.json()
    assert len(body['regions']) >= 1