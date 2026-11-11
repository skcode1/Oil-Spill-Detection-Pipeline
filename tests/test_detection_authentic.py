"""Detection + pixel metrics on authentic_testing (real checkpoint).

What is asserted
----------------
* response schema shape (slick list, statistics, geometry fields);
* confidence values inside [0, 1];
* geometry sanity (area/perimeter > 0, valid lon/lat ranges when present);
* pixel-level precision/recall/F1/dice/IoU against the paired
  ``<stem>_segmentation.tif`` ground truth (resized to the inference grid).

What is NOT asserted
--------------------
No minimum F1/recall threshold: the shipped prototype checkpoint under-
segments the 2048px authentic scenes (see README "Known limitations"), so the
suite checks that the metric harness runs and reports sane values rather
than gating on an accuracy number the current weights cannot meet.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
import tifffile

from app.schemas import DetectionOptions
from tests.conftest import FAST_DOWNSCALE, sample_mask_for

RUN_SLOW = os.environ.get("RUN_SLOW_TESTS", "0") == "1"


def _read_mask(mask_path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(str(mask_path)))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def _pixel_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    truth = (np.asarray(truth) > 0).astype(np.uint8)
    if truth.shape != pred.shape:
        truth = cv2.resize(
            truth, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    tp = int(np.logical_and(pred == 1, truth == 1).sum())
    fp = int(np.logical_and(pred == 1, truth == 0).sum())
    fn = int(np.logical_and(pred == 0, truth == 1).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = int(np.logical_or(pred == 1, truth == 1).sum())
    iou = tp / union if union else 1.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": f1, "dice": f1, "iou": iou,
    }


def _check_response_shape(outcome) -> None:
    response = outcome.response
    assert response.scene is not None
    stats = response.statistics
    assert stats.oil_pixel_count >= 0
    assert 0.0 <= stats.mean_oil_confidence <= 1.0
    assert 0.0 <= stats.max_oil_confidence <= 1.0
    assert stats.max_oil_confidence + 1e-9 >= stats.mean_oil_confidence
    for slick in response.slicks:
        assert slick.area_m2 > 0
        assert slick.perimeter_m > 0
        assert slick.pixel_area >= 0
        assert 0.0 <= slick.confidence <= 1.0
        if slick.centroid_lon is not None:
            assert -180.0 <= slick.centroid_lon <= 180.0
        if slick.centroid_lat is not None:
            assert -90.0 <= slick.centroid_lat <= 90.0
    assert outcome.mask.shape == outcome.oil_confidence.shape
    assert set(np.unique(outcome.mask).tolist()) <= {0, 1}


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="native-resolution inference (~53 s/scene); set RUN_SLOW_TESTS=1")
def test_detection_oil_scene_native_resolution(services, corpus_index):
    """One Oil scene end-to-end at native resolution via the service wrapper."""
    image = corpus_index["Oil"]["images"][0]
    mask_path = sample_mask_for(corpus_index, image)
    assert mask_path is not None
    outcome = services.detection.detect_path(
        str(image),
        DetectionOptions(save_artifacts=False, include_preview=False),
    )
    _check_response_shape(outcome)
    scores = _pixel_scores(outcome.mask, _read_mask(mask_path))
    for key in ("precision", "recall", "f1", "dice", "iou"):
        assert 0.0 <= scores[key] <= 1.0
    print(f"\nnative {image.name}: oil_px={outcome.total_oil_pixels} " + str(scores))


def test_detection_sample_subset_fast(services, corpus_index, sample_scenes):
    """Per-class subset at downscale=8: schema + confidence + geometry sanity."""
    options = DetectionOptions(
        downscale=FAST_DOWNSCALE, save_artifacts=False, include_preview=False
    )
    seen = 0
    for cls, images in sample_scenes.items():
        assert images, f"no scenes sampled for class {cls}"
        for image in images:
            outcome = services.detection.detect_path(str(image), options)
            _check_response_shape(outcome)
            seen += 1
    assert seen >= 3


def test_pixel_metrics_against_ground_truth(services, corpus_index, sample_scenes):
    """Precision/recall/F1/Dice/IoU are computed and bounded for Oil scenes."""
    options = DetectionOptions(
        downscale=FAST_DOWNSCALE, save_artifacts=False, include_preview=False
    )
    images = sample_scenes["Oil"]
    assert images
    for image in images:
        mask_path = sample_mask_for(corpus_index, image)
        assert mask_path is not None, f"no ground truth for {image.name}"
        outcome = services.detection.detect_path(str(image), options)
        scores = _pixel_scores(outcome.mask, _read_mask(mask_path))
        for key in ("precision", "recall", "f1", "dice", "iou"):
            assert 0.0 <= scores[key] <= 1.0, f"{image.name}: {key}={scores[key]}"
        print(f"\n{image.name}: oil_px={outcome.total_oil_pixels} " + str(
            {k: round(scores[k], 4) for k in ("precision", "recall", "f1", "iou")}))
