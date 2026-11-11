"""Model-health guards against the "always Clear" regression.

Background
----------
The shipped checkpoint ``models/oil_spill_segmentation_model.pth`` once
degenerated into a network whose output ignored its input (softmax saturated to
a fixed spatial template), so *every* scene - oil, no-oil and lookalike alike -
produced ~700 stray oil pixels that the morphology + min-area filters discarded.
The dashboard then truthfully reported "0 polygons / Clear" for all three
classes while the test suite stayed green (it asserted no accuracy floor).

These tests pin the two properties that failure violated:

1. **Input dependence** - two different scenes must yield visibly different
   label maps (a dead checkpoint gives ~0.1 % difference; healthy weights give
   >1 %).
2. **End-to-end detection** - an authentic Oil scene must produce at least one
   slick polygon, and that polygon must be geographically georeferenced (this
   also exercises the ``rasterio`` list->Affine coercion in
   ``app.services.detection_service.as_affine``).
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from app.schemas import DetectionOptions

#: Scenes are evaluated at 1/4 resolution (~1-3 s each on CPU).
FAST_DOWNSCALE = 4


def test_checkpoint_output_depends_on_input(services, project_root) -> None:
    """Two different SAR chips must not produce the same label map."""
    model = services.model
    assert model.ready, model.info().error

    chips = sorted((project_root / "data" / "processed" / "train" / "images").glob("oil_*.tif"))
    if len(chips) < 2:
        pytest.skip("training chips unavailable for input-dependence check")

    label_maps = []
    for chip in chips[:2]:
        array = tifffile.imread(str(chip))
        result = model.infer_probabilities(array, tile_size=256, stride=256)
        label_maps.append(result.label_map)

    disagreement = float((label_maps[0] != label_maps[1]).mean())
    assert disagreement > 0.01, (
        f"label maps for two different chips agree on {100 * (1 - disagreement):.2f} % "
        "of pixels - the checkpoint output does not depend on its input "
        "(degenerate/over-saturated weights)."
    )


def test_oil_scene_produces_georeferenced_slicks(services, corpus_index) -> None:
    """An authentic Oil scene must yield >= 1 slick with lon/lat geometry.

    * The degenerate checkpoint produced 0 polygons for every class.
    * The list-typed geotransform used to crash ``rasterio.transform.xy`` with
      ``GCPTransformer requires sequence of GroundControlPoint`` as soon as a
      polygon existed on a georeferenced scene.
    """
    image = corpus_index["Oil"]["images"][0]
    outcome = services.detection.detect_path(
        str(image),
        options=DetectionOptions(downscale=FAST_DOWNSCALE, save_artifacts=False),
    )
    response = outcome.response
    assert response.slicks, (
        f"no slick polygons detected in Oil scene {image.name}; "
        f"oil pixels = {response.statistics.oil_pixel_count}"
    )
    for slick in response.slicks:
        assert slick.area_px > 0
        if response.scene.georeferenced:
            assert slick.coordinate_space == "geographic"
            assert slick.centroid_lon is not None and -180.0 <= slick.centroid_lon <= 180.0
            assert slick.centroid_lat is not None and -90.0 <= slick.centroid_lat <= 90.0


def test_no_oil_scene_produces_no_slicks(services, corpus_index) -> None:
    """A clean scene must stay clear (no false-positive polygons)."""
    image = corpus_index["No oil"]["images"][0]
    outcome = services.detection.detect_path(
        str(image),
        options=DetectionOptions(downscale=FAST_DOWNSCALE, save_artifacts=False),
    )
    assert not outcome.response.slicks, (
        f"false-positive slicks on clean scene {image.name}: "
        f"{len(outcome.response.slicks)} polygon(s), "
        f"oil pixels = {outcome.response.statistics.oil_pixel_count}"
    )


def test_lookalike_scene_produces_no_slicks(services, corpus_index) -> None:
    """Lookalikes (wind shadow, biogenic film) must be suppressed as spills."""
    image = corpus_index["Lookalike"]["images"][0]
    outcome = services.detection.detect_path(
        str(image),
        options=DetectionOptions(downscale=FAST_DOWNSCALE, save_artifacts=False),
    )
    assert not outcome.response.slicks, (
        f"lookalike scene {image.name} reported as a spill: "
        f"{len(outcome.response.slicks)} polygon(s)"
    )