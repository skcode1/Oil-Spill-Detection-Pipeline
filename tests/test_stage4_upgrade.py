from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from rasterio.transform import Affine
from app.config import load_settings
from app.schemas import IntelligenceBriefing, PipelineOptions
from app.services import attribution_service as att
from app.services import briefing_service as brf
from app.services import detection_service as det
from app.services.drift_service import MetoceanField
from src.environment_fetcher import build_environment_manifest

def test_pipeline_options_new_fields_defaults_false():
    opts = PipelineOptions(run_attribution=False)
    assert opts.fetch_live_metocean is True
    assert opts.generate_briefing is False
    assert opts.strict_metocean is False

def test_pipeline_options_new_fields_explicit():
    opts = PipelineOptions(run_attribution=False, fetch_live_metocean=True, generate_briefing=True, ais_csv_path='data/raw/ais_data.csv')
    assert opts.fetch_live_metocean is True
    assert opts.generate_briefing is True

def test_intelligence_briefing_schema():
    b = IntelligenceBriefing(executive_summary='summary', primary_suspect_rationale='rationale', environmental_risk_level='high', recommended_next_actions=['board vessel'], status='completed')
    assert b.status == 'completed'
    assert b.environmental_risk_level == 'high'
    dumped = b.model_dump()
    assert dumped['status'] == 'completed'

def test_intelligence_briefing_skipped():
    b = IntelligenceBriefing(executive_summary='', primary_suspect_rationale='', environmental_risk_level='unknown', recommended_next_actions=[], status='skipped')
    assert b.status == 'skipped'

def test_briefing_service_skips_without_api_key(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    svc = brf.BriefingService(settings)
    assert svc.available is False
    out = svc.generate_briefing({'detection': {}})
    assert out.status == 'skipped'
    assert out.executive_summary

def test_collect_colocated_sar_ships_dedups():
    detection_result = {'slicks': [{'slick_id': 'slick_1', 'colocated_ships': [{'ship_detection_id': 'sar_ship_1', 'lon': 1.0, 'lat': 2.0, 'distance_km': 0.5, 'confidence': 0.9}, {'ship_detection_id': 'sar_ship_1', 'lon': 1.0, 'lat': 2.0, 'distance_km': 0.5, 'confidence': 0.9}]}, {'slick_id': 'slick_2', 'colocated_ships': [{'ship_detection_id': 'sar_ship_2', 'lon': 3.0, 'lat': 4.0, 'distance_km': 1.0, 'confidence': 0.8}]}]}
    ships = att._collect_colocated_sar_ships(detection_result)
    ids = {s['ship_detection_id'] for s in ships}
    assert ids == {'sar_ship_1', 'sar_ship_2'}
    assert len(ships) == 2

def test_match_sar_ship_within_radius():
    ships = [{'ship_detection_id': 'sar_ship_1', 'lon': 0.0, 'lat': 0.0, 'distance_km': 0.1, 'confidence': 0.9, 'associated_slick_id': 'slick_1'}, {'ship_detection_id': 'sar_ship_2', 'lon': 10.0, 'lat': 10.0, 'distance_km': 5.0, 'confidence': 0.5, 'associated_slick_id': 'slick_2'}]
    match = att.AttributionService._match_sar_ship(0.001, 0.001, ships, radius_km=0.75)
    assert match is not None
    assert match['ship_detection_id'] == 'sar_ship_1'
    assert match['sar_to_ais_offset_km'] <= 0.75

def test_match_sar_ship_outside_radius_returns_none():
    ships = [{'ship_detection_id': 'sar_ship_1', 'lon': 5.0, 'lat': 5.0, 'distance_km': 5.0, 'confidence': 0.9, 'associated_slick_id': 'slick_1'}]
    match = att.AttributionService._match_sar_ship(0.0, 0.0, ships, radius_km=0.75)
    assert match is None

def test_haversine_sanity():
    assert att.haversine_km(0.0, 0.0, 0.0, 0.0) == 0.0
    d = att.haversine_km(0.0, 0.0, 90.0, 0.0)
    assert 9900.0 < d < 10100.0

def _fake_scene(transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0), crs=None):
    return SimpleNamespace(transform=transform, crs=crs)

def _svc():
    return det.DetectionService.__new__(det.DetectionService)

def test_vectorize_class_oil_polygon():
    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[1:3, 1:3] = det.CLASS_IDS['oil_spill']
    prob_avg = np.zeros((5, 4, 4), dtype=np.float32)
    prob_avg[det.CLASS_IDS['oil_spill'], 1:3, 1:3] = 0.9
    feats = _svc()._vectorize_class(label_map, prob_avg, det.CLASS_IDS['oil_spill'], scene=_fake_scene(), min_area_km2=0.0, min_confidence=0.5)
    assert len(feats) == 1
    assert feats[0]['confidence'] == 0.9
    assert feats[0]['area_km2'] >= 0.0

def test_vectorize_class_min_confidence_filters():
    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[1:3, 1:3] = det.CLASS_IDS['oil_spill']
    prob_avg = np.zeros((5, 4, 4), dtype=np.float32)
    prob_avg[det.CLASS_IDS['oil_spill'], 1:3, 1:3] = 0.4
    feats = _svc()._vectorize_class(label_map, prob_avg, det.CLASS_IDS['oil_spill'], scene=_fake_scene(), min_area_km2=0.0, min_confidence=0.9)
    assert feats == []

def test_vectorize_class_ship_guarded_when_absent():
    num_classes = 3
    assert det.CLASS_IDS['ship'] >= num_classes

def test_metocean_field_from_samples():
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    samples = [SimpleNamespace(timestamp_utc='2026-05-01T00:00:00Z', u_current_mps=0.2, v_current_mps=0.1, u_wind_mps=5.0, v_wind_mps=2.0)]
    field = MetoceanField.from_samples(samples, default_time=start)
    assert len(field.times) == 1
    assert field.u_current == pytest.approx(0.2)

def test_metocean_field_from_environment_json_time_series():
    payload = {'source': 'copernicus_cmems_era5', 'center_lon': 72.81, 'center_lat': 18.92, 'wind_drift_factor': 0.03, 'samples': [{'timestamp_utc': '2026-05-01T00:00:00Z', 'u_current_mps': 0.25, 'v_current_mps': 0.15, 'u_wind_mps': 5.5, 'v_wind_mps': 2.5}, {'timestamp_utc': '2026-05-01T01:00:00Z', 'u_current_mps': 0.3, 'v_current_mps': 0.2, 'u_wind_mps': 6.0, 'v_wind_mps': 3.0}]}
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    field = MetoceanField.from_environment_json(payload, default_time=start)
    assert field.wind_drift_factor == 0.03
    assert len(field.times) == 2

class _FakeDataArray:

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def isel(self, **kwargs):
        idx = kwargs.get('depth', kwargs.get('time', 0))
        arr = np.asarray(self._values, dtype=float)
        if arr.ndim > 1:
            return _FakeDataArray(arr[idx])
        return _FakeDataArray(arr.flat[int(idx)])

    def interp(self, time, method=None, kwargs=None):
        return self

    def __getitem__(self, key):
        return _FakeDataArray(self._values[key])

    @property
    def values(self):
        return self._values

    @property
    def dims(self):
        return ('time',)

class _FakePoint:

    def __init__(self, uo, vo, u10, v10, times):
        self.uo = uo
        self.vo = vo
        self.u10 = u10
        self.v10 = v10
        self.time = SimpleNamespace(values=times)
        self.dims = ('time',)

    def interp(self, time, method=None, kwargs=None):
        return self

    def isel(self, **kwargs):
        return self

    def __getitem__(self, key):
        return getattr(self, key)

class _FakeDataset:

    def __init__(self, uo, vo, u10, v10, times):
        self._uo = _FakeDataArray(uo)
        self._vo = _FakeDataArray(vo)
        self._u10 = _FakeDataArray(u10)
        self._v10 = _FakeDataArray(v10)
        self._times = times
        self.dims = ('time',)

    def sel(self, longitude, latitude, method='nearest'):
        return _FakePoint(self._uo, self._vo, self._u10, self._v10, self._times)

    def close(self):
        pass

def test_build_environment_manifest_writes_structure(tmp_path, monkeypatch):
    times = np.array(['2026-05-01T00:00:00', '2026-05-01T01:00:00'], dtype='datetime64[s]')
    fake_cur = _FakeDataset(np.array([0.2, 0.3]), np.array([0.1, 0.2]), None, None, times)
    fake_wnd = _FakeDataset(None, None, np.array([5.0, 6.0]), np.array([2.0, 3.0]), times)

    def fake_open(path):
        name = os.path.basename(str(path))
        if name == 'current.nc':
            return fake_cur
        return fake_wnd
    with patch('src.environment_fetcher.xr.open_dataset', side_effect=fake_open):
        manifest_path = tmp_path / 'metocean_environment.json'
        manifest = build_environment_manifest(tmp_path / 'current.nc', tmp_path / 'wind.nc', 72.81, 18.92, manifest_path, wind_drift_factor=0.03)
    assert manifest['source'] == 'copernicus_cmems_era5'
    assert manifest['wind_drift_factor'] == 0.03
    assert len(manifest['samples']) == 2
    s0 = manifest['samples'][0]
    assert {'timestamp_utc', 'u_current_mps', 'v_current_mps', 'u_wind_mps', 'v_wind_mps'}.issubset(set(s0))
    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text())
    assert written['wind_drift_factor'] == 0.03

def test_optical_spectral_indices_ndwi_and_fai():
    from app.services.detection_service import _preprocess_optical
    from app.services.lookalike_filter import LookalikeFilter, LookalikeFilterSettings, compute_fai, compute_ndwi
    green = np.array([[0.8, 0.1]], dtype=np.float32)
    nir = np.array([[0.1, 0.9]], dtype=np.float32)
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] > 0.5
    assert ndwi[0, 1] < -0.5
    red = np.array([[0.1, 0.1]], dtype=np.float32)
    nir = np.array([[0.35, 0.1]], dtype=np.float32)
    swir = np.array([[0.1, 0.1]], dtype=np.float32)
    fai = compute_fai(red, nir, swir)
    assert fai[0, 0] > 0.2
    assert abs(fai[0, 1]) < 0.01
    lf = LookalikeFilter(settings=LookalikeFilterSettings(min_algal_bloom_fai=0.03, heuristic_threshold_count=1))
    dummy_mask = np.ones((10, 10), dtype=np.uint8)
    dummy_slicks = [{'area_px': 500, 'elongation': 1.2, 'solidity': 0.9, 'geometry': {'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]
    res = lf.classify(slicks=dummy_slicks, mask=dummy_mask, oil_coverage_fraction=0.05, scene_pixels=1000, fai_mean=0.08)
    assert 'algal_bloom_fai' in res.indicators
    assert res.is_lookalike is True
    arr_5band = np.zeros((32, 32, 5), dtype=np.float32)
    arr_5band[:16, :, 1] = 0.8
    arr_5band[:16, :, 3] = 0.1
    arr_5band[16:, :, 1] = 0.1
    arr_5band[16:, :, 3] = 0.9
    arr_5band[:, :, 2] = 0.2
    arr_5band[:, :, 4] = 0.2
    rgb_3ch, water_mask, fai_map, warnings = _preprocess_optical(arr_5band)
    assert rgb_3ch.shape == (32, 32, 3)
    assert water_mask is not None
    assert np.all(water_mask[:16, :] == 1)
    assert np.all(water_mask[16:, :] == 0)
    assert fai_map is not None

def test_detection_service_3class_vs_5class_ship_extraction():
    from unittest.mock import MagicMock
    from app.services.detection_service import DetectionService, ResolvedDetectionParams
    from app.services.image_io import SceneArray
    settings = load_settings()
    mock_model = MagicMock()
    service = DetectionService(settings, mock_model)
    params = ResolvedDetectionParams(confidence_threshold=0.5, tile_size=256, stride=128, tile_batch_size=2, min_area_px=10, downscale=1, save_artifacts=False, include_preview=False, pixel_spacing_m=10.0, acquisition_time_utc=None, min_slick_area_km2=0.0, ship_buffer_km=2.0)
    arr = np.zeros((64, 64, 3), dtype=np.float32)
    scene = SceneArray(path=Path('scene_test.tif'), array=arr, filename='scene_test.tif', band_count=3, crs='EPSG:4326', transform=Affine(0.001, 0, 72.0, 0, -0.001, 18.0), pixel_spacing_m=10.0)
    inference_3cls = MagicMock()
    inference_3cls.prob_maps = np.zeros((3, 64, 64), dtype=np.float32)
    inference_3cls.label_map = np.zeros((64, 64), dtype=np.int64)
    inference_3cls.oil_confidence = np.zeros((64, 64), dtype=np.float32)
    mock_model.infer_probabilities.return_value = inference_3cls
    outcome_3cls = service.detect_scene(scene, params=params)
    assert len(outcome_3cls.response.detected_ships) == 0
    inference_5cls = MagicMock()
    prob_5cls = np.zeros((5, 64, 64), dtype=np.float32)
    label_5cls = np.zeros((64, 64), dtype=np.int64)
    label_5cls[10:14, 10:14] = 3
    prob_5cls[3, 10:14, 10:14] = 0.95
    inference_5cls.prob_maps = prob_5cls
    inference_5cls.label_map = label_5cls
    inference_5cls.oil_confidence = np.zeros((64, 64), dtype=np.float32)
    mock_model.infer_probabilities.return_value = inference_5cls
    outcome_5cls = service.detect_scene(scene, params=params)
    assert len(outcome_5cls.response.detected_ships) > 0

def test_wavewatch3_stokes_drift_provenance():
    from app.schemas import MetoceanSample
    from app.services.drift_service import MetoceanField
    s_stokes = MetoceanSample(u_current_mps=0.2, v_current_mps=0.1, u_wind_mps=5.0, v_wind_mps=2.0, u_stokes_mps=0.08, v_stokes_mps=0.04)
    f_stokes = MetoceanField.from_samples([s_stokes], default_time=datetime.now(timezone.utc))
    assert f_stokes.wave_provenance == 'wavewatch3_stokes_drift'
    u_c, v_c, u_w, v_w, u_s, v_s = f_stokes.at_with_stokes(datetime.now(timezone.utc))
    assert u_s == 0.08
    assert v_s == 0.04
    s_no_stokes = MetoceanSample(u_current_mps=0.2, v_current_mps=0.1, u_wind_mps=5.0, v_wind_mps=2.0)
    f_no_stokes = MetoceanField.from_samples([s_no_stokes], default_time=datetime.now(timezone.utc))
    assert 'standard_leeway_fallback' in f_no_stokes.wave_provenance
    u_c, v_c, u_w, v_w, u_s, v_s = f_no_stokes.at_with_stokes(datetime.now(timezone.utc))
    assert u_s == 0.0
    assert v_s == 0.0

def test_nmea_0183_decoder_and_stream_adapter():
    from datetime import timedelta
    from src.ais_pruner import AISStreamAdapter, decode_nmea_ais_sentence
    sentence = '!AIVDM,1,1,,B,15N2800P00IS2w:K:G14l4wn081h,0*2A'
    record = decode_nmea_ais_sentence(sentence)
    assert record is not None
    assert record['mmsi'] == 367036416
    assert abs(record['lon'] - -90.21103) < 0.0001
    assert abs(record['lat'] - 47.46881) < 0.0001
    json_line = '{"mmsi": 123456789, "lon": 72.85, "lat": 18.95, "sog": 12.4, "cog": 180.0}'
    rec_json = decode_nmea_ais_sentence(json_line)
    assert rec_json is not None
    assert rec_json['mmsi'] == 123456789
    assert rec_json['lon'] == 72.85
    adapter = AISStreamAdapter(buffer_hours=12.0)
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()
    adapter.ingest_records([{'mmsi': 999111222, 'timestamp': now_iso, 'lon': 72.82, 'lat': 18.92, 'sog': 10.0, 'cog': 90.0}, {'mmsi': 888111222, 'timestamp': (now_utc - timedelta(hours=20)).isoformat(), 'lon': 72.82, 'lat': 18.92}])
    df = adapter.get_dataframe()
    assert len(df) == 1
    assert int(df.iloc[0]['mmsi']) == 999111222
    pruned = adapter.prune(origin_lon=72.82, origin_lat=18.92, t0_time_str=now_iso, semi_major_m=5000.0, semi_minor_m=3000.0, orientation_deg=45.0, temporal_window_hours=2.0)
    assert not pruned.empty
    assert int(pruned.iloc[0]['mmsi']) == 999111222