from __future__ import annotations
import pytest
import tifffile
from app.schemas import DetectionOptions
FAST_DOWNSCALE = 4

def test_checkpoint_output_depends_on_input(services, project_root) -> None:
    model = services.model
    assert model.ready, model.info().error
    chips = sorted((project_root / 'data' / 'processed' / 'train' / 'images').glob('oil_*.tif'))
    if len(chips) < 2:
        pytest.skip('training chips unavailable for input-dependence check')
    label_maps = []
    for chip in chips[:2]:
        array = tifffile.imread(str(chip))
        result = model.infer_probabilities(array, tile_size=256, stride=256)
        label_maps.append(result.label_map)
    disagreement = float((label_maps[0] != label_maps[1]).mean())
    assert disagreement > 0.01, f'label maps for two different chips agree on {100 * (1 - disagreement):.2f} % of pixels - the checkpoint output does not depend on its input (degenerate/over-saturated weights).'

def test_oil_scene_produces_georeferenced_slicks(services, corpus_index) -> None:
    image = corpus_index['Oil']['images'][0]
    outcome = services.detection.detect_path(str(image), options=DetectionOptions(downscale=FAST_DOWNSCALE, save_artifacts=False))
    response = outcome.response
    assert response.slicks, f'no slick polygons detected in Oil scene {image.name}; oil pixels = {response.statistics.oil_pixel_count}'
    for slick in response.slicks:
        assert slick.area_px > 0
        if response.scene.georeferenced:
            assert slick.coordinate_space == 'geographic'
            assert slick.centroid_lon is not None and -180.0 <= slick.centroid_lon <= 180.0
            assert slick.centroid_lat is not None and -90.0 <= slick.centroid_lat <= 90.0

def test_no_oil_scene_produces_no_slicks(services, corpus_index) -> None:
    image = corpus_index['No oil']['images'][0]
    outcome = services.detection.detect_path(str(image), options=DetectionOptions(downscale=FAST_DOWNSCALE, save_artifacts=False))
    assert not outcome.response.slicks, f'false-positive slicks on clean scene {image.name}: {len(outcome.response.slicks)} polygon(s), oil pixels = {outcome.response.statistics.oil_pixel_count}'

def test_lookalike_scene_produces_no_slicks(services, corpus_index) -> None:
    image = corpus_index['Lookalike']['images'][0]
    outcome = services.detection.detect_path(str(image), options=DetectionOptions(downscale=FAST_DOWNSCALE, save_artifacts=False))
    assert not outcome.response.slicks, f'lookalike scene {image.name} reported as a spill: {len(outcome.response.slicks)} polygon(s)'