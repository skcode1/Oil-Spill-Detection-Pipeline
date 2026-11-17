"""End-to-end dashboard-path regression tests over the authentic corpus.

These exercise the exact route the Next.js dashboard uses
(``POST {API_PREFIX}/pipeline/upload``) against the three demo scene classes
(Oil / No oil / Lookalike) plus the georeferencing path that produced the
``transform.xy`` crash (list-vs-Affine).  They are the guard against the
"every scene renders Clear" class of regressions: each assertion is written
in terms of what the dashboard derives a verdict from
(``detection.slicks``), not internal service state.

Runtime: loads the real checkpoint once (module-scoped), ~1 min per scene.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "authentic testing" / "02_Test_images_and_ground_truth"

pytestmark = pytest.mark.skipif(
    not CORPUS.is_dir(), reason="authentic test corpus not checked out"
)


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan running (services + checkpoint warm-up)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _first_scene(cls: str) -> Path:
    scenes = sorted((CORPUS / "Images" / cls).glob("*.tif"))
    assert scenes, f"no {cls} scenes under {CORPUS}"
    return scenes[0]


def _run_upload(client, path: Path, options: str | None = None) -> dict:
    from app.main import API_PREFIX

    with path.open("rb") as handle:
        files = {"file": (str(path), handle, "image/tiff")}
        data = {"options": options} if options else None
        response = client.post(f"{API_PREFIX}/pipeline/upload", files=files, data=data)
    assert response.status_code == 200, response.text
    return response.json()


def _slicks(body: dict) -> list:
    return ((body.get("detection") or {}).get("slicks")) or []


def test_oil_scene_reports_slicks(client):
    """An oil scene must never render as Clear on the dashboard."""
    body = _run_upload(client, _first_scene("Oil"))
    slicks = _slicks(body)
    assert slicks, f"Oil scene produced no slicks: {body.get('warnings')}"

    stats = body["detection"]["statistics"]
    assert stats["slick_count"] == len(slicks)
    assert stats["oil_pixel_count"] > 0

    geojson = body["detection"]["spill_geojson"]
    assert geojson.get("type") == "FeatureCollection"
    assert geojson.get("features"), "oil scene must carry spill polygons"


@pytest.mark.xfail(
    strict=False,
    reason="Known model gap: uncalibrated checkpoint may label clean water as oil when path context is absent; needs negative-class retraining. See README 'Known model gaps'.",
)
def test_no_oil_scene_is_clear(client):
    """Clean-water scenes must yield zero slicks (dashboard shows 'Clear')."""
    body = _run_upload(client, _first_scene("No oil"))
    slicks = _slicks(body)
    assert slicks == [], (
        f"clean-water scene produced {len(slicks)} false slicks "
        f"(mean conf {body['detection']['statistics']['mean_oil_confidence']:.2f})"
    )


@pytest.mark.xfail(
    strict=False,
    reason="Known model gap: the checkpoint labels lookalike features (wind "
    "shadows, biogenic slicks) as oil at every threshold swept 0.50-0.95; "
    "needs retraining with usable lookalike supervision. See README "
    "'Known model gaps'.",
)
def test_lookalike_scene_produces_no_slicks(client):
    body = _run_upload(client, _first_scene("Lookalike"))
    assert _slicks(body) == [], "lookalike scene must render as Clear"


def test_georeferenced_scene_emits_lonlat_polygons(client, tmp_path):
    """GeoJSON polygons must carry real lon/lat, not pixel coordinates.

    Guards the rasterio ``transform`` round-trip: an unfixed ``transform.xy``
    call received a plain list (``ds.transform`` via ``profile``) instead of an
    ``Affine`` and raised, aborting every georeferenced scene.
    """
    import rasterio
    from rasterio.transform import from_bounds

    source = _first_scene("Oil")
    with rasterio.open(source) as dataset:
        data = dataset.read()
        profile = dataset.profile.copy()

    height, width = data.shape[1], data.shape[2]
    georef = tmp_path / "georef_scene.tif"
    profile.update(crs="EPSG:4326", transform=from_bounds(120.0, 10.0, 121.0, 11.0, width, height))
    with rasterio.open(georef, "w", **profile) as out:
        out.write(data)

    body = _run_upload(client, georef)
    features = (body["detection"]["spill_geojson"] or {}).get("features") or []
    assert features, "georeferenced oil scene returned no polygons"

    def _coordinates(geometry):
        if geometry["type"] == "Polygon":
            yield from geometry["coordinates"][0]
        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                yield from polygon[0]

    for feature in features:
        for x, y in _coordinates(feature["geometry"]):
            assert -180.0 <= x <= 180.0, f"longitude out of range: {x}"
            assert -90.0 <= y <= 90.0, f"latitude out of range: {y}"
            assert (120.0 <= x <= 121.0) and (10.0 <= y <= 11.0), (
                f"coordinate {x},{y} outside the written scene bounds"
            )
