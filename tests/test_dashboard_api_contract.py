from pathlib import Path


def _page_text() -> str:
    page = Path(__file__).resolve().parents[1] / "maritime-surveillance" / "app" / "page.tsx"
    return page.read_text(encoding="utf-8")


def test_dashboard_uses_current_backend_upload_contract():
    text = _page_text()

    # The dashboard must target the live FastAPI service (/api/v1), not the
    # legacy backend routes.  The upload URL is composed from API_BASE_URL.
    assert '"http://localhost:8000/api/v1"' in text
    assert "`${API_BASE_URL}/pipeline/upload`" in text
    assert '"http://localhost:8000/api/run-pipeline"' not in text
    assert '"http://localhost:8000/api/status"' not in text


def test_dashboard_maps_pipeline_response_fields():
    """The dashboard must derive its cards from fields the API really returns.

    The legacy backend exposed top-level ``has_spill``/``geo_stats``/``dossier``
    keys; the live ``PipelineResponse`` nests everything under ``detection`` and
    ``attribution``.  Reading the legacy keys made every run render "Clear"
    regardless of the detection result, so this pins the mapping.
    """
    text = _page_text()

    # Legacy top-level reads must be gone.
    for legacy in (
        "pipelineState?.has_spill",
        "pipelineState?.geo_stats",
        "pipelineState?.dossier",
        "pipelineState?.spill_geojson",
        "pipelineState?.culprit_geojson",
        "pipelineState?.mask_url",
        "pipelineState?.target_scene_url",
    ):
        assert legacy not in text, f"dashboard still reads removed field {legacy}"

    # The mapping must be driven by the real contract.
    for needle in (
        "pipelineState?.detection",
        "detection?.slicks",
        "detection?.spill_geojson",
        "detection?.mask_preview_png_base64",
        "pipelineState?.attribution",
        "attribution?.candidates",
    ):
        assert needle in text, f"dashboard no longer maps {needle}"

    # Request-side contract: dataset mode posts JSON path, threshold rides in options.
    assert "confidence_threshold" in text
    assert "run_attribution" in text
    assert "path: selectedScene" in text


def test_pipeline_response_exposes_dashboard_roots():
    """Schema-side half of the dashboard contract."""
    from app.schemas import PipelineResponse, SceneEntry, ScenesResponse

    response_fields = set(PipelineResponse.model_fields)
    assert {"detection", "attribution"} <= response_fields

    scene_fields = set(SceneEntry.model_fields)
    assert {"path", "filename", "category"} == scene_fields

    scenes_fields = set(ScenesResponse.model_fields)
    assert "scenes" in scenes_fields


def test_scenes_endpoint_lists_authentic_corpus():
    """``GET /scenes`` must return project-relative paths /pipeline can load."""
    from fastapi.testclient import TestClient

    from app.api.deps import build_services
    from app.config import load_settings
    from app.main import create_app

    settings = load_settings(project_root=Path(__file__).resolve().parents[1])
    if settings.authentic_test_root is None:
        import pytest

        pytest.skip("authentic testing corpus not available")
    services = build_services(settings, load_model=False)
    client = TestClient(create_app(services=services))
    try:
        response = client.get("/api/v1/scenes")
        assert response.status_code == 200
        payload = response.json()
        assert payload["scenes"], "expected at least one dataset scene"
        first = payload["scenes"][0]
        assert first["path"] and first["filename"] and first["category"]
        resolved = settings.project_root / first["path"]
        assert resolved.is_file(), f"scene path not resolvable: {first['path']}"
    finally:
        services.close()
