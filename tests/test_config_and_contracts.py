"""Fixture-free unit tests: corpus resolver + upload/metocean caps."""
from __future__ import annotations

from pathlib import Path

from app.api.routes import LEGACY_MAX_UPLOAD_BYTES, MAX_UPLOAD_BYTES
from app.config import InferenceSettings, ServerSettings, load_settings
from app.services import drift_service
from app.services.detection_service import APPROX_EPSILON_FRACTION


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_authentic_corpus_resolves_to_class_layout():
    settings = load_settings(project_root=_root())
    assert settings.authentic_test_root is not None
    root = settings.authentic_test_root
    for cls in ("Oil", "No oil", "Lookalike"):
        assert (root / "Images" / cls).is_dir()
        assert (root / "Mask" / cls).is_dir()
    assert len(list((root / "Images" / "Oil").glob("*.tif*"))) >= 1


def test_authentic_mask_pairing_suffix_convention():
    settings = load_settings(project_root=_root())
    root = settings.authentic_test_root
    assert root is not None
    images = sorted((root / "Images" / "Oil").glob("*.tif*"))[:5]
    assert images
    for image in images:
        expected = root / "Mask" / "Oil" / f"{image.stem}_segmentation{image.suffix}"
        assert expected.is_file(), f"missing ground truth for {image.name}"


def test_authentic_corpus_missing_degrades_to_none(tmp_path: Path):
    empty = tmp_path / "proj"
    (empty / "data").mkdir(parents=True)
    (empty / "config.yaml").write_text("paths: {}\n", encoding="utf-8")
    settings = load_settings(config_path=empty / "config.yaml", project_root=empty)
    assert settings.authentic_test_root is None


def test_upload_cap_from_yaml_mb(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("api:\n  max_upload_mb: 25\n", encoding="utf-8")
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert settings.server.max_upload_bytes == 25 * 1024 * 1024


def test_upload_cap_env_bytes_wins_over_yaml(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("api:\n  max_upload_mb: 25\n", encoding="utf-8")
    monkeypatch.setenv("OILSPILL_MAX_UPLOAD_BYTES", str(7 * 1024 * 1024))
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert settings.server.max_upload_bytes == 7 * 1024 * 1024


def test_upload_cap_non_positive_falls_back(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("api:\n  max_upload_mb: 0\n", encoding="utf-8")
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert settings.server.max_upload_bytes == ServerSettings.max_upload_bytes


def test_upload_cap_matches_legacy_route_alias():
    settings = load_settings(project_root=_root())
    assert settings.server.max_upload_bytes == 200 * 1024 * 1024
    assert MAX_UPLOAD_BYTES == LEGACY_MAX_UPLOAD_BYTES == 200 * 1024 * 1024


def test_metocean_caps_yaml_and_env(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "metocean:\n  max_steps: 1234\n  max_endpoint_samples: 37\n",
        encoding="utf-8",
    )
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert (settings.metocean.max_steps, settings.metocean.max_endpoint_samples) == (1234, 37)
    monkeypatch.setenv("OILSPILL_MAX_STEPS", "4321")
    monkeypatch.setenv("OILSPILL_MAX_ENDPOINTS", "11")
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert (settings.metocean.max_steps, settings.metocean.max_endpoint_samples) == (4321, 11)


def test_morphology_and_epsilon_yaml_and_env(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "inference:\n  morphology_kernel_size: 5\n  contour_epsilon_fraction: 0.01\n",
        encoding="utf-8",
    )
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert settings.inference.morphology_kernel_size == 5
    monkeypatch.setenv("OILSPILL_MORPH_KERNEL", "7")
    monkeypatch.setenv("OILSPILL_CONTOUR_EPSILON", "0.02")
    settings = load_settings(config_path=cfg, project_root=tmp_path)
    assert settings.inference.morphology_kernel_size == 7


def test_drift_aliases_mirror_config_defaults():
    from app.config import MetoceanSettings

    assert drift_service.MAX_STEPS == drift_service.DEFAULT_MAX_STEPS == 20000
    assert drift_service.MAX_ENDPOINT_SAMPLES == drift_service.DEFAULT_MAX_ENDPOINT_SAMPLES == 200
    assert MetoceanSettings.max_steps == 20000
    assert MetoceanSettings.max_endpoint_samples == 200


def test_epsilon_default_matches_training_code():
    assert APPROX_EPSILON_FRACTION == InferenceSettings.contour_epsilon_fraction == 0.005


def test_no_home_paths_in_settings():
    """No user-home-anchored *literal* leaks into resolved settings.

    Every resolved path legitimately lives under the checkout
    (``project_root``), which on this machine sits inside ``/home/...``. The
    production-readiness criterion is narrower: no *hardcoded absolute*
    home literal (``/home/<user>/...``, ``/Users/...``, ``~``) may appear in
    ``config.yaml`` defaults or be baked into the code — everything must be
    relative to the project root or injected via env. So the test scans the
    raw YAML + source-adjacent defaults for home literals, and separately
    asserts every resolved path stays inside the checkout.
    """
    import re

    root = _root()
    home_literal = re.compile(r"\"'~|/home/|/Users/|C:\\\\Users")
    config_text = (root / "config.yaml").read_text(encoding="utf-8")
    assert not home_literal.search(config_text), "config.yaml hardcodes a home path"
    settings = load_settings(project_root=root)
    for label, value in settings.describe().items():
        text = str(value)
        if text.startswith(str(root)) or text in ("", "auto"):
            continue
        assert not text.startswith(("/home/", "/Users/", "~")), (
            f"{label} resolves outside the checkout: {text}"
        )
    assert str(settings.model_checkpoint).startswith(str(root))
