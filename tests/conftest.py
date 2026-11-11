"""Shared pytest fixtures: resolved settings, warmed services, corpus helpers.

The suite validates the model on the shipped ``authentic_testing`` corpus
through the same service wrapper the API uses
(``AppSettings.resolve_authentic_test_root`` -> ``build_services`` ->
``DetectionService``), so tests exercise the production tiling, channel
layout and normalisation — never a reimplementation.

Speed strategy: 2048x2048 scenes take ~53 s at native resolution on CPU.
Fixtures default to ``downscale=8`` (256x256 effective, ~1 s/scene) and the
slow native-resolution test is marked ``slow`` and skipped unless
``RUN_SLOW_TESTS=1`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from app.api.deps import AppServices, build_services
from app.config import AppSettings, load_settings

#: Downscale used by the fast suite (8 -> 2048px scenes infer at ~256px).
FAST_DOWNSCALE = int(os.environ.get("OILSPILL_TEST_DOWNSCALE", "8"))

#: Max scenes per class for the fast suite (keeps CI under a few minutes).
SCENES_PER_CLASS = int(os.environ.get("OILSPILL_TEST_SCENES_PER_CLASS", "3"))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _project_root()


@pytest.fixture(scope="session")
def settings(project_root: Path) -> AppSettings:
    """Resolved ``config.yaml`` settings for the checked-out project."""
    loaded = load_settings(project_root=project_root)
    assert loaded.authentic_test_root is not None, (
        "authentic_testing corpus not found: tests need "
        "data/authentic testing/02_Test_images_and_ground_truth "
        f"(resolved from {project_root})"
    )
    return loaded


@pytest.fixture(scope="session")
def services(settings: AppSettings) -> AppServices:
    """Fully wired services with the real checkpoint loaded (session-scoped).

    The model warms once per session (~seconds for torch import + checkpoint
    load); health is asserted here so every downstream failure message shows
    the checkpoint path instead of a cryptic 503.
    """
    built = build_services(settings, load_model=True)
    assert built.model.ready, (
        f"segmentation checkpoint failed to load from {settings.model_checkpoint}: "
        f"{built.model.info().error}"
    )
    yield built
    built.close()


@pytest.fixture(scope="session")
def corpus_index(settings: AppSettings) -> Dict[str, Dict[str, List[Path]]]:
    """Map class name -> {\"images\": [...], \"masks\": [...]} sorted by stem.

    Mask filenames carry a ``_segmentation`` suffix
    (``00000_segmentation.tif`` for image ``00000.tif``); the index resolves
    that pairing explicitly instead of assuming parallel ordering.
    """
    root = settings.authentic_test_root
    assert root is not None
    index: Dict[str, Dict[str, List[Path]]] = {}
    for cls in ("Oil", "No oil", "Lookalike"):
        images = sorted((root / "Images" / cls).glob("*.tif*"))
        masks_by_stem = {
            p.name.replace("_segmentation", ""): p
            for p in sorted((root / "Mask" / cls).glob("*.tif*"))
        }
        paired_masks = [masks_by_stem[n.name] for n in images if n.name in masks_by_stem]
        index[cls] = {"images": images, "masks": paired_masks}
    return index


@pytest.fixture(scope="session")
def sample_scenes(corpus_index) -> Dict[str, List[Path]]:
    """Small per-class scene subset for the fast suite."""
    return {
        cls: entry["images"][:SCENES_PER_CLASS] for cls, entry in corpus_index.items()
    }


def sample_mask_for(corpus_index, image_path: Path) -> Optional[Path]:
    """Return the ground-truth mask paired with ``image_path`` (or None)."""
    for entry in corpus_index.values():
        for img, mask in zip(entry["images"], entry["masks"]):
            if img == image_path:
                return mask
    stem = image_path.stem + "_segmentation"
    for entry in corpus_index.values():
        for mask in entry["masks"]:
            if mask.stem == stem:
                return mask
    return None
