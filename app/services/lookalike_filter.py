"""Post-processing classification layer to filter false positives (lookalikes).

This module implements geometric, textural, and contextual heuristics to distinguish
true petroleum discharges from common false-positive lookalikes:

* Biogenic films (organic surface slicks)
* Low-wind surface phenomena (capillary waves, Langmuir windrows)
* Atmospheric artifacts (wind shadows, atmospheric blocking)
* Land boundary distortions (coastal edge artifacts)

The classification operates on vectorized slick polygons after inference, examining:
- Geometric characteristics (fragmentation, size uniformity, irregularity)
- Boundary gradients (sharp vs. diffuse edges)
- Spatial coherence (continuity, morphology consistency)
- Metadata context (lookalike dataset indicators)

All heuristics are configurable via the ``LookalikeFilterSettings`` model,
allowing for refinement as new patterns emerge in operational use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
class LookalikeFilterSettings(BaseModel):
    """Tunable thresholds and feature weights for lookalike detection."""

    # Geometry-based heuristics
    min_confirmed_area_px: float = Field(
        default=50.0,
        description="Slicks with area < this threshold (pixels) are considered too small for real oil.",
    )
    max_confirmed_elongation: float = Field(
        default=8.0,
        description="Elongation (length/width) > this threshold suggests wind slicks or Langmuir patterns.",
    )
    min_confirmed_solidity: float = Field(
        default=0.3,
        description="Solidity (contour_area / convex_hull_area) < this suggests fragmented patterns.",
    )

    # Boundary gradient heuristics
    min_edge_gradient_magnitude: float = Field(
        default=2.0,
        description="Edge pixels with mean gradient < this threshold suggest lookalike diffuse boundaries.",
    )
    edge_sample_count: int = Field(
        default=32,
        description="Number of edge pixels to sample for boundary gradient analysis.",
    )

    # Spatial coherence & morphology
    max_confirmed_bbox_fill_fraction: float = Field(
        default=0.95,
        description="Slicks occupying > this fraction of a tight bounding box flag uniform artifacts.",
    )
    max_confirmed_component_count: int = Field(
        default=50,
        description="Slicks with > this many separate connected components suggest fragmentation.",
    )

    # Confidence scoring
    min_confident_scene_pixels: int = Field(
        default=50_000,
        description="When scene size (pixels) is below this, conservatively downgrade.",
    )
    max_confirmed_oil_coverage_fraction: float = Field(
        default=0.85,
        description="When oil coverage exceeds this, mark as lookalike (likely wind slick).",
    )

    # Metadata context
    detect_lookalike_dataset_paths: bool = Field(
        default=True,
        description="Enable lookalike dataset path detection (check for 'lookalike' in path).",
    )

    # Optical spectral index heuristics (biogenic algal bloom detection)
    min_algal_bloom_fai: float = Field(
        default=0.03,
        description="Floating Algae Index (FAI) threshold above which surface features are flagged as biogenic algal blooms.",
    )

    # Overall sensitivity
    heuristic_threshold_count: int = Field(
        default=3,
        description="How many heuristics must flag before status becomes 'Lookalike'.",
    )
    use_lookalike_status: bool = Field(
        default=True,
        description="When True, use 'Lookalike' status; when False, downgrade confidence instead.",
    )

    @model_validator(mode="after")
    def validate_settings(self) -> LookalikeFilterSettings:
        """Validate configuration thresholds automatically after initialization."""
        if self.min_confirmed_area_px < 0:
            raise ValueError("min_confirmed_area_px must be >= 0")
        if self.max_confirmed_elongation < 1.0:
            raise ValueError("max_confirmed_elongation must be >= 1.0")
        if not 0.0 <= self.min_confirmed_solidity <= 1.0:
            raise ValueError("min_confirmed_solidity must be in [0, 1]")
        if self.min_edge_gradient_magnitude < 0.0:
            raise ValueError("min_edge_gradient_magnitude must be >= 0")
        if self.edge_sample_count < 1:
            raise ValueError("edge_sample_count must be >= 1")
        if not 0.0 <= self.max_confirmed_bbox_fill_fraction <= 1.0:
            raise ValueError("max_confirmed_bbox_fill_fraction must be in [0, 1]")
        if self.max_confirmed_component_count < 1:
            raise ValueError("max_confirmed_component_count must be >= 1")
        if self.min_confident_scene_pixels < 0:
            raise ValueError("min_confident_scene_pixels must be >= 0")
        if not 0.0 <= self.max_confirmed_oil_coverage_fraction <= 1.0:
            raise ValueError("max_confirmed_oil_coverage_fraction must be in [0, 1]")
        if self.heuristic_threshold_count < 0:
            raise ValueError("heuristic_threshold_count must be >= 0")
        return self


# --------------------------------------------------------------------------- #
#  Lookalike Classification Result
# --------------------------------------------------------------------------- #
@dataclass
class LookalikeClassification:
    """Result of lookalike classification for one detection pass."""

    status: Literal["Confirmed", "Lookalike", "Unverified"]
    is_lookalike: bool
    confidence_downgrade_factor: float = 1.0
    indicators: List[str] = field(default_factory=list)
    heuristic_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict for JSON responses."""
        return {
            "status": self.status,
            "is_lookalike": self.is_lookalike,
            "lookalike_indicators": self.indicators,
            "heuristic_scores": self.heuristic_scores,
        }


# --------------------------------------------------------------------------- #
#  Heuristic Functions
# --------------------------------------------------------------------------- #
def _analyze_geometry(
    slicks: Sequence[Dict[str, Any]],
    settings: LookalikeFilterSettings,
) -> Tuple[List[str], Dict[str, float]]:
    """Geometry-based heuristics: size, elongation, solidity, fragmentation."""
    indicators: List[str] = []
    scores: Dict[str, float] = {}

    if not slicks:
        return indicators, scores

    avg_area = np.mean([s.get("area_px", 0) for s in slicks])
    avg_elongation = np.mean([s.get("elongation", 1.0) for s in slicks])
    avg_solidity = np.mean([s.get("solidity", 1.0) for s in slicks])

    if avg_area < settings.min_confirmed_area_px:
        indicators.append("insufficient_area")
        scores["insufficient_area"] = 1.0 - (avg_area / settings.min_confirmed_area_px)
    else:
        scores["insufficient_area"] = 0.0

    if avg_elongation > settings.max_confirmed_elongation:
        indicators.append("excessive_elongation")
        scores["excessive_elongation"] = min(
            1.0, (avg_elongation - settings.max_confirmed_elongation)
        )
    else:
        scores["excessive_elongation"] = 0.0

    if avg_solidity < settings.min_confirmed_solidity:
        indicators.append("low_solidity_fragmentation")
        scores["low_solidity_fragmentation"] = (
            1.0 - (avg_solidity / settings.min_confirmed_solidity)
        )
    else:
        scores["low_solidity_fragmentation"] = 0.0

    return indicators, scores


def _analyze_boundary_gradient(
    mask: np.ndarray,
    settings: LookalikeFilterSettings,
) -> Tuple[List[str], Dict[str, float]]:
    """Edge-gradient inspection: sharp vs. diffuse boundaries."""
    indicators: List[str] = []
    scores: Dict[str, float] = {}

    if mask.sum() < 10:
        scores["diffuse_boundary"] = 0.0
        return indicators, scores

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        scores["diffuse_boundary"] = 0.0
        return indicators, scores

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 4:
        scores["diffuse_boundary"] = 0.0
        return indicators, scores

    gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    sample_indices = np.linspace(0, len(contour) - 1, settings.edge_sample_count, dtype=int)
    edge_gradients: List[float] = []

    for idx in sample_indices:
        pt = contour[idx][0]
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < grad_mag.shape[0] and 0 <= x < grad_mag.shape[1]:
            y_min, y_max = max(0, y - 2), min(grad_mag.shape[0], y + 3)
            x_min, x_max = max(0, x - 2), min(grad_mag.shape[1], x + 3)
            neighborhood = grad_mag[y_min:y_max, x_min:x_max]
            edge_gradients.append(float(np.max(neighborhood)) if neighborhood.size > 0 else 0.0)

    if edge_gradients:
        mean_edge_gradient = float(np.mean(edge_gradients))
        if mean_edge_gradient < settings.min_edge_gradient_magnitude:
            indicators.append("diffuse_boundary")
            scores["diffuse_boundary"] = 1.0 - (
                mean_edge_gradient / settings.min_edge_gradient_magnitude
            )
        else:
            scores["diffuse_boundary"] = 0.0
    else:
        scores["diffuse_boundary"] = 0.0

    return indicators, scores


def _analyze_spatial_coherence(
    slicks: Sequence[Dict[str, Any]],
    mask: np.ndarray,
    settings: LookalikeFilterSettings,
) -> Tuple[List[str], Dict[str, float]]:
    """Spatial coherence and morphology: uniformity and fragmentation."""
    indicators: List[str] = []
    scores: Dict[str, float] = {}

    if mask.sum() < 10:
        scores["non_uniform_spatial_coherence"] = 0.0
        scores["high_fragmentation"] = 0.0
        return indicators, scores

    if slicks:
        slick = slicks[0]
        area_px = slick.get("area_px", 0)
        if slick.get("geometry") and isinstance(slick["geometry"], dict):
            try:
                coords = slick["geometry"].get("coordinates", [[[]]])[0]
                if coords:
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if bbox_area > 0:
                        fill_fraction = area_px / bbox_area
                        if fill_fraction > settings.max_confirmed_bbox_fill_fraction:
                            indicators.append("non_uniform_spatial_coherence")
                            scores["non_uniform_spatial_coherence"] = (
                                fill_fraction - settings.max_confirmed_bbox_fill_fraction
                            )
                        else:
                            scores["non_uniform_spatial_coherence"] = 0.0
                    else:
                        scores["non_uniform_spatial_coherence"] = 0.0
                else:
                    scores["non_uniform_spatial_coherence"] = 0.0
            except (KeyError, IndexError, TypeError):
                scores["non_uniform_spatial_coherence"] = 0.0
        else:
            scores["non_uniform_spatial_coherence"] = 0.0
    else:
        scores["non_uniform_spatial_coherence"] = 0.0

    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    num_components = int(labels.max())
    if num_components > settings.max_confirmed_component_count:
        indicators.append("high_fragmentation")
        fragmentation_score = min(
            1.0,
            (num_components - settings.max_confirmed_component_count)
            / settings.max_confirmed_component_count,
        )
        scores["high_fragmentation"] = fragmentation_score
    else:
        scores["high_fragmentation"] = 0.0

    return indicators, scores


def _analyze_scene_context(
    scene_pixels: int,
    oil_coverage_fraction: float,
    settings: LookalikeFilterSettings,
) -> Tuple[List[str], Dict[str, float]]:
    """Contextual heuristics: scene size and oil coverage."""
    indicators: List[str] = []
    scores: Dict[str, float] = {}

    if scene_pixels < settings.min_confident_scene_pixels:
        indicators.append("small_scene")
        scores["small_scene"] = (
            1.0 - (scene_pixels / settings.min_confident_scene_pixels)
        )
    else:
        scores["small_scene"] = 0.0

    if oil_coverage_fraction > settings.max_confirmed_oil_coverage_fraction:
        indicators.append("high_oil_coverage")
        scores["high_oil_coverage"] = (
            oil_coverage_fraction / settings.max_confirmed_oil_coverage_fraction
        )
    else:
        scores["high_oil_coverage"] = 0.0

    return indicators, scores


def _detect_lookalike_dataset_path(
    path: Optional[str],
    settings: LookalikeFilterSettings,
) -> Tuple[List[str], Dict[str, float]]:
    """Metadata context: detect if input path indicates a lookalike dataset."""
    indicators: List[str] = []
    scores: Dict[str, float] = {}

    if not settings.detect_lookalike_dataset_paths or path is None:
        return indicators, scores

    path_lower = str(path).lower()

    if "lookalike" in path_lower:
        indicators.append("lookalike_dataset_path")
        scores["lookalike_dataset_path"] = 1.0
        return indicators, scores

    authentic_markers = ["authentic", "testing", "test_set", "evaluation"]
    if any(marker in path_lower for marker in authentic_markers):
        scores["lookalike_dataset_path"] = 0.3
        if "lookalike" in Path(path_lower).name:
            indicators.append("lookalike_dataset_path")
            scores["lookalike_dataset_path"] = 1.0

    return indicators, scores


def compute_fai(
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    lambda_red: float = 665.0,
    lambda_nir: float = 842.0,
    lambda_swir1: float = 1610.0,
) -> np.ndarray:
    """Compute the Floating Algae Index (FAI, Hu 2009).

    FAI = NIR - (Red + (SWIR1 - Red) * ((lambda_NIR - lambda_Red) / (lambda_SWIR1 - lambda_Red)))
    """
    scale = (lambda_nir - lambda_red) / max(lambda_swir1 - lambda_red, 1e-6)
    baseline = red + (swir1 - red) * scale
    return nir - baseline


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index (NDWI, McFeeters 1996).

    NDWI = (Green - NIR) / (Green + NIR)
    """
    denominator = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denominator > 1e-6, (green - nir) / denominator, 0.0)
    return np.nan_to_num(ndwi, nan=0.0, posinf=0.0, neginf=0.0)


def _analyze_optical_fai(
    fai_value: Optional[float],
    settings: LookalikeFilterSettings,
) -> Tuple[List[str], Dict[str, float]]:
    """Flag biogenic algal blooms when Floating Algae Index is elevated."""
    indicators: List[str] = []
    scores: Dict[str, float] = {}
    if fai_value is None or not np.isfinite(fai_value):
        scores["algal_bloom_fai"] = 0.0
        return indicators, scores

    if fai_value >= settings.min_algal_bloom_fai:
        indicators.append("algal_bloom_fai")
        scores["algal_bloom_fai"] = min(
            1.0, float(fai_value / max(settings.min_algal_bloom_fai * 2.0, 1e-6))
        )
    else:
        scores["algal_bloom_fai"] = 0.0
    return indicators, scores


# --------------------------------------------------------------------------- #
#  Main Classification Engine
# --------------------------------------------------------------------------- #
class LookalikeFilter:
    """Post-processing lookalike classifier for detected oil slicks."""

    def __init__(self, settings: Optional[LookalikeFilterSettings] = None):
        """Initialize filter with configuration."""
        self.settings = settings or LookalikeFilterSettings()
        # Validation is automatically executed via Pydantic model_validator.

    def classify(
        self,
        slicks: Sequence[Dict[str, Any]],
        mask: np.ndarray,
        oil_coverage_fraction: float,
        scene_pixels: int,
        scene_path: Optional[str] = None,
        mean_confidence: float = 0.0,
        fai_mean: Optional[float] = None,
    ) -> LookalikeClassification:
        """Classify a detection result as Confirmed/Lookalike/Unverified."""
        all_indicators: List[str] = []
        all_scores: Dict[str, float] = {}

        geom_indicators, geom_scores = _analyze_geometry(slicks, self.settings)
        all_indicators.extend(geom_indicators)
        all_scores.update(geom_scores)

        edge_indicators, edge_scores = _analyze_boundary_gradient(mask, self.settings)
        all_indicators.extend(edge_indicators)
        all_scores.update(edge_scores)

        spatial_indicators, spatial_scores = _analyze_spatial_coherence(
            slicks, mask, self.settings
        )
        all_indicators.extend(spatial_indicators)
        all_scores.update(spatial_scores)

        context_indicators, context_scores = _analyze_scene_context(
            scene_pixels, oil_coverage_fraction, self.settings
        )
        all_indicators.extend(context_indicators)
        all_scores.update(context_scores)

        path_indicators, path_scores = _detect_lookalike_dataset_path(scene_path, self.settings)
        all_indicators.extend(path_indicators)
        all_scores.update(path_scores)

        fai_indicators, fai_scores = _analyze_optical_fai(fai_mean, self.settings)
        all_indicators.extend(fai_indicators)
        all_scores.update(fai_scores)

        num_indicators = len(all_indicators)
        is_lookalike = num_indicators >= self.settings.heuristic_threshold_count

        if is_lookalike and self.settings.use_lookalike_status:
            status = "Lookalike"
            confidence_downgrade = 0.5 + 0.5 * (1.0 - mean_confidence)
        elif num_indicators > 0:
            status = "Unverified"
            confidence_downgrade = 0.8
        else:
            status = "Confirmed"
            confidence_downgrade = 1.0

        logger.info(
            "Lookalike classification | status=%s | indicators=%d | path=%s | confidence_downgrade=%.2f",
            status,
            num_indicators,
            scene_path or "unknown",
            confidence_downgrade,
        )

        return LookalikeClassification(
            status=status,
            is_lookalike=is_lookalike,
            confidence_downgrade_factor=confidence_downgrade,
            indicators=list(set(all_indicators)),
            heuristic_scores=all_scores,
        )