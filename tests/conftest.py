from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional
import pytest
from app.api.deps import AppServices, build_services
from app.config import AppSettings, load_settings
FAST_DOWNSCALE = int(os.environ.get('OILSPILL_TEST_DOWNSCALE', '8'))
SCENES_PER_CLASS = int(os.environ.get('OILSPILL_TEST_SCENES_PER_CLASS', '3'))

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def project_root() -> Path:
    return _project_root()

@pytest.fixture(scope='session')
def settings(project_root: Path) -> AppSettings:
    loaded = load_settings(project_root=project_root)
    assert loaded.authentic_test_root is not None, f'authentic_testing corpus not found: tests need data/authentic testing/02_Test_images_and_ground_truth (resolved from {project_root})'
    return loaded

@pytest.fixture(scope='session')
def services(settings: AppSettings) -> AppServices:
    built = build_services(settings, load_model=True)
    assert built.model.ready, f'segmentation checkpoint failed to load from {settings.model_checkpoint}: {built.model.info().error}'
    yield built
    built.close()

@pytest.fixture(scope='session')
def corpus_index(settings: AppSettings) -> Dict[str, Dict[str, List[Path]]]:
    root = settings.authentic_test_root
    assert root is not None
    index: Dict[str, Dict[str, List[Path]]] = {}
    for cls in ('Oil', 'No oil', 'Lookalike'):
        images = sorted((root / 'Images' / cls).glob('*.tif*'))
        masks_by_stem = {p.name.replace('_segmentation', ''): p for p in sorted((root / 'Mask' / cls).glob('*.tif*'))}
        paired_masks = [masks_by_stem[n.name] for n in images if n.name in masks_by_stem]
        index[cls] = {'images': images, 'masks': paired_masks}
    return index

@pytest.fixture(scope='session')
def sample_scenes(corpus_index) -> Dict[str, List[Path]]:
    return {cls: entry['images'][:SCENES_PER_CLASS] for cls, entry in corpus_index.items()}

def sample_mask_for(corpus_index, image_path: Path) -> Optional[Path]:
    for entry in corpus_index.values():
        for img, mask in zip(entry['images'], entry['masks']):
            if img == image_path:
                return mask
    stem = image_path.stem + '_segmentation'
    for entry in corpus_index.values():
        for mask in entry['masks']:
            if mask.stem == stem:
                return mask
    return None