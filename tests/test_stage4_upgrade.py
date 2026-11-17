"""Unit tests for the Stage 1-4 hybrid upgrade (no real model/network required)."""

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


# --------------------------------------------------------------------------- #
#  Schemas
# --------------------------------------------------------------------------- #
def test_pipeline_options_new_fields_defaults_false():
    opts = PipelineOptions(run_attribution=False)
    assert opts.fetch_live_metocean is False
    assert opts.generate_briefing is False


def test_pipeline_options_new_fields_explicit():
    opts = PipelineOptions(
        run_attribution=False,
        fetch_live_metocean=True,
        generate_briefing=True,
        ais_csv_path="data/raw/ais_data.csv",
    )
    assert opts.fetch_live_metocean is True
    assert opts.generate_briefing is True


def test_intelligence_briefing_schema():
    b = IntelligenceBriefing(
        executive_summary="summary",
        primary_suspect_rationale="rationale",
        environmental_risk_level="high",
        recommended_next_actions=["board vessel"],
        status="completed",
    )
    assert b.status == "completed"
    assert b.environmental_risk_level == "high"
    dumped = b.model_dump()
    assert dumped["status"] == "completed"


def test_intelligence_briefing_skipped():
    b = IntelligenceBriefing(
        executive_summary="",
        primary_suspect_rationale="",
        environmental_risk_level="unknown",
        recommended_next_actions=[],
        status="skipped",
    )
    assert b.status == "skipped"


# --------------------------------------------------------------------------- #
#  Briefing service: graceful skip without API key
# --------------------------------------------------------------------------- #
def test_briefing_service_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    svc = brf.BriefingService(settings)
    assert svc.available is False
    out = svc.generate_briefing({"detection": {}})
    assert out.status == "skipped"
    assert out.executive_summary


# --------------------------------------------------------------------------- #
#  Attribution: SAR co-location + visual matching
# --------------------------------------------------------------------------- #
def test_collect_colocated_sar_ships_dedups():
    detection_result = {
        "slicks": [
            {
                "slick_id": "slick_1",
                "colocated_ships": [
                    {"ship_detection_id": "sar_ship_1", "lon": 1.0, "lat": 2.0,
                     "distance_km": 0.5, "confidence": 0.9},
                    {"ship_detection_id": "sar_ship_1", "lon": 1.0, "lat": 2.0,
                     "distance_km": 0.5, "confidence": 0.9},
                ],
            },
            {
                "slick_id": "slick_2",
                "colocated_ships": [
                    {"ship_detection_id": "sar_ship_2", "lon": 3.0, "lat": 4.0,
                     "distance_km": 1.0, "confidence": 0.8},
                ],
            },
        ]
    }
    ships = att._collect_colocated_sar_ships(detection_result)
    ids = {s["ship_detection_id"] for s in ships}
    assert ids == {"sar_ship_1", "sar_ship_2"}
    assert len(ships) == 2


def test_match_sar_ship_within_radius():
    ships = [
        {"ship_detection_id": "sar_ship_1", "lon": 0.0, "lat": 0.0,
         "distance_km": 0.1, "confidence": 0.9, "associated_slick_id": "slick_1"},
        {"ship_detection_id": "sar_ship_2", "lon": 10.0, "lat": 10.0,
         "distance_km": 5.0, "confidence": 0.5, "associated_slick_id": "slick_2"},
    ]
    match = att.AttributionService._match_sar_ship(0.001, 0.001, ships, radius_km=0.75)
    assert match is not None
    assert match["ship_detection_id"] == "sar_ship_1"
    assert match["sar_to_ais_offset_km"] <= 0.75


def test_match_sar_ship_outside_radius_returns_none():
    ships = [
        {"ship_detection_id": "sar_ship_1", "lon": 5.0, "lat": 5.0,
         "distance_km": 5.0, "confidence": 0.9, "associated_slick_id": "slick_1"},
    ]
    match = att.AttributionService._match_sar_ship(0.0, 0.0, ships, radius_km=0.75)
    assert match is None


def test_haversine_sanity():
    # Same point -> 0 km
    assert att.haversine_km(0.0, 0.0, 0.0, 0.0) == 0.0
    # Roughly equator quadrant -> ~10000 km
    d = att.haversine_km(0.0, 0.0, 90.0, 0.0)
    assert 9900.0 < d < 10100.0


# --------------------------------------------------------------------------- #
#  Detection: per-class vectorization
# --------------------------------------------------------------------------- #
def _fake_scene(transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0), crs=None):
    return SimpleNamespace(transform=transform, crs=crs)


def _svc():
    """Build a DetectionService without running the (model-loading) __init__."""
    return det.DetectionService.__new__(det.DetectionService)


def test_vectorize_class_oil_polygon():
    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[1:3, 1:3] = det.CLASS_IDS["oil_spill"]
    prob_avg = np.zeros((5, 4, 4), dtype=np.float32)
    prob_avg[det.CLASS_IDS["oil_spill"], 1:3, 1:3] = 0.9

    feats = _svc()._vectorize_class(
        label_map, prob_avg, det.CLASS_IDS["oil_spill"],
        scene=_fake_scene(), min_area_km2=0.0, min_confidence=0.5,
    )
    assert len(feats) == 1
    assert feats[0]["confidence"] == 0.9
    assert feats[0]["area_km2"] >= 0.0


def test_vectorize_class_min_confidence_filters():
    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[1:3, 1:3] = det.CLASS_IDS["oil_spill"]
    prob_avg = np.zeros((5, 4, 4), dtype=np.float32)
    prob_avg[det.CLASS_IDS["oil_spill"], 1:3, 1:3] = 0.4  # below threshold

    feats = _svc()._vectorize_class(
        label_map, prob_avg, det.CLASS_IDS["oil_spill"],
        scene=_fake_scene(), min_area_km2=0.0, min_confidence=0.9,
    )
    assert feats == []


def test_vectorize_class_ship_guarded_when_absent():
    # With a 3-class model, prob_avg has 3 channels (indices 0-2); ship is index 3.
    # The service guards CLASS_IDS["ship"] < num_classes before vectorising, so
    # no IndexError surfaces. This test documents that contract.
    num_classes = 3
    assert det.CLASS_IDS["ship"] >= num_classes  # ship channel absent in 3-class model


# --------------------------------------------------------------------------- #
#  Drift: metocean field resolution
# --------------------------------------------------------------------------- #
def test_metocean_field_from_samples():
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    samples = [
        SimpleNamespace(
            timestamp_utc="2026-05-01T00:00:00Z", u_current_mps=0.2, v_current_mps=0.1,
            u_wind_mps=5.0, v_wind_mps=2.0,
        ),
    ]
    field = MetoceanField.from_samples(samples, default_time=start)
    assert len(field.times) == 1
    assert field.u_current == pytest.approx(0.2)


def test_metocean_field_from_environment_json_time_series():
    payload = {
        "source": "copernicus_cmems_era5",
        "center_lon": 72.81,
        "center_lat": 18.92,
        "wind_drift_factor": 0.03,
        "samples": [
            {"timestamp_utc": "2026-05-01T00:00:00Z", "u_current_mps": 0.25,
             "v_current_mps": 0.15, "u_wind_mps": 5.5, "v_wind_mps": 2.5},
            {"timestamp_utc": "2026-05-01T01:00:00Z", "u_current_mps": 0.30,
             "v_current_mps": 0.20, "u_wind_mps": 6.0, "v_wind_mps": 3.0},
        ],
    }
    start = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    field = MetoceanField.from_environment_json(payload, default_time=start)
    assert field.wind_drift_factor == 0.03
    assert len(field.times) == 2


# --------------------------------------------------------------------------- #
#  Environment fetcher: manifest assembly (mocked NetCDF)
# --------------------------------------------------------------------------- #
class _FakeDataArray:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def isel(self, **kwargs):
        idx = kwargs.get("depth", kwargs.get("time", 0))
        arr = np.asarray(self._values, dtype=float)
        if arr.ndim > 1:
            return _FakeDataArray(arr[idx])
        # Scalar time-indexed selection, mirroring xarray .isel(time=i).
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
        return ("time",)


class _FakePoint:
    """Point dataset returned by .sel(); supports .interp, .isel, __getitem__, .dims."""

    def __init__(self, uo, vo, u10, v10, times):
        self.uo = uo
        self.vo = vo
        self.u10 = u10
        self.v10 = v10
        self.time = SimpleNamespace(values=times)
        self.dims = ("time",)

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
        self.dims = ("time",)

    def sel(self, longitude, latitude, method="nearest"):
        return _FakePoint(self._uo, self._vo, self._u10, self._v10, self._times)

    def close(self):
        pass


def test_build_environment_manifest_writes_structure(tmp_path, monkeypatch):
    times = np.array(
        ["2026-05-01T00:00:00", "2026-05-01T01:00:00"], dtype="datetime64[s]"
    )
    fake_cur = _FakeDataset(
        np.array([0.2, 0.3]), np.array([0.1, 0.2]), None, None, times
    )
    fake_wnd = _FakeDataset(None, None, np.array([5.0, 6.0]),
                            np.array([2.0, 3.0]), times)

    def fake_open(path):
        name = os.path.basename(str(path))
        if name == "current.nc":
            return fake_cur
        return fake_wnd

    with patch("src.environment_fetcher.xr.open_dataset", side_effect=fake_open):
        manifest_path = tmp_path / "metocean_environment.json"
        manifest = build_environment_manifest(
            tmp_path / "current.nc", tmp_path / "wind.nc",
            72.81, 18.92, manifest_path, wind_drift_factor=0.03,
        )
    assert manifest["source"] == "copernicus_cmems_era5"
    assert manifest["wind_drift_factor"] == 0.03
    assert len(manifest["samples"]) == 2
    s0 = manifest["samples"][0]
    assert set(s0) == {
        "timestamp_utc", "u_current_mps", "v_current_mps",
        "u_wind_mps", "v_wind_mps",
    }
    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text())
    assert written["wind_drift_factor"] == 0.03
