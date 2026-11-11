"""Heuristic spill-age estimation from detected slick extent.

Method
------
A surface slick's horizontal extent grows through turbulent (eddy) diffusion.
For a passive scalar patch in the diffusive regime the second moment grows
linearly in time, giving the classic area-growth law

.. math::  A(t) ~= 4 * pi * D * t

where ``A`` is the slick area (m^2) and ``D`` the horizontal eddy diffusivity
(m^2/s).  Inverting it yields an order-of-magnitude age:

.. math::  t ~= A / (4 * pi * D)

``D`` defaults to ``config.yaml -> metocean.horizontal_diffusivity`` so the age
estimate and the drift simulation share one physical parameter; the result is
clamped to ``detection.age.min_age_hours .. max_age_hours``.

This is deliberately a *heuristic*, not a calibrated inversion: it assumes the
slick is a passively spreading patch in steady turbulent conditions.  The
returned ``confidence`` therefore encodes how much of the required context was
actually available (georeferencing, metocean drift speed, plausible slick
shape), and ``assumptions`` lists every simplification applied.  When a
backward hindcast is available its ``t0`` supersedes the heuristic entirely.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app.config import AppSettings
from app.schemas import SpillAgeEstimate

logger = logging.getLogger(__name__)

#: Slope of the turbulent area-growth law ``A(t) = AREA_GROWTH_SLOPE * D * t``.
AREA_GROWTH_SLOPE: float = 4.0 * math.pi

BASE_ASSUMPTIONS: List[str] = [
    "Passive slick spreading by horizontal turbulent diffusion, A(t) = 4*pi*D*t.",
    "Constant eddy diffusivity over the slick lifetime (no wind/current history).",
    "Slick thickness has not yet fallen below the SAR detection limit.",
]


def _as_utc(moment: datetime) -> datetime:
    """Return ``moment`` as a timezone-aware UTC datetime."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def estimate_spill_age(
    *,
    settings: AppSettings,
    area_m2: Optional[float],
    total_slick_area_km2: Optional[float] = None,
    elongation: float = 1.0,
    georeferenced: bool = True,
    diffusivity_m2_s: Optional[float] = None,
    drift_speed_mps: Optional[float] = None,
    reference_time: Optional[datetime] = None,
    hindcast_origin_time: Optional[datetime] = None,
) -> Optional[SpillAgeEstimate]:
    """Estimate the age of a detected slick.

    Args:
        settings: Resolved settings (``detection.age`` holds the bounds).
        area_m2: Total slick area in square metres (preferred input).
        total_slick_area_km2: Alternative area input when ``area_m2`` is absent.
        elongation: Length/width aspect ratio of the largest slick (used only to
            modulate confidence - drift-stretched slicks are more plausible).
        georeferenced: Whether the scene carried a CRS; pixel-space areas rely on
            the assumed ground sample distance and therefore lower confidence.
        diffusivity_m2_s: Override for the eddy diffusivity.
        drift_speed_mps: Metocean drift speed when available (raises confidence).
        reference_time: Satellite observation time (defaults to "now").
        hindcast_origin_time: Authoritative discharge time from a backward
            hindcast; when given, the heuristic is bypassed.

    Returns:
        A :class:`~app.schemas.SpillAgeEstimate`, or ``None`` when the age model
        is disabled or no usable area could be derived.
    """
    age_cfg = settings.detection.age
    if not age_cfg.enabled:
        return None

    reference = _as_utc(reference_time or datetime.now(timezone.utc))

    if hindcast_origin_time is not None:
        origin = _as_utc(hindcast_origin_time)
        hours = max((reference - origin).total_seconds() / 3600.0, 0.0)
        return SpillAgeEstimate(
            age_hours=round(hours, 3),
            age_days=round(hours / 24.0, 3),
            min_possible_hours=0.0,
            max_possible_hours=round(age_cfg.max_age_hours, 3),
            confidence=round(max(age_cfg.min_confidence, 0.85), 3),
            method="hindcast_lagrangian_backtrack",
            estimated_discharge_time_utc=origin.isoformat(),
            assumptions=[
                "Age derived from the Lagrangian backward-hindcast origin time.",
                "Accuracy depends on the quality of the supplied metocean fields.",
            ],
        )

    effective_area = area_m2
    if (effective_area is None or effective_area <= 0) and total_slick_area_km2:
        effective_area = float(total_slick_area_km2) * 1e6
    if effective_area is None or effective_area <= 0 or not math.isfinite(effective_area):
        logger.debug("No usable slick area; spill age not estimated.")
        return None

    diffusivity = float(diffusivity_m2_s or age_cfg.diffusivity_m2_s)
    diffusivity = diffusivity if diffusivity > 0 else 1.0

    raw_hours = effective_area / (AREA_GROWTH_SLOPE * diffusivity)
    age_hours = min(max(raw_hours, age_cfg.min_age_hours), age_cfg.max_age_hours)
    clamped_low = raw_hours < age_cfg.min_age_hours
    clamped_high = raw_hours > age_cfg.max_age_hours

    # ---- confidence: rewards real context, penalises assumptions ----------
    confidence = 0.45
    confidence += 0.20 if georeferenced else -0.10
    if drift_speed_mps is not None and drift_speed_mps > 0:
        confidence += 0.15
    if 1.2 <= float(elongation) <= 25.0:
        # Drift-stretched, elongated slicks are typical of a spreading patch.
        confidence += 0.10
    if clamped_low or clamped_high:
        # Saturated inversion -> the true age lies outside the model's validity.
        confidence -= 0.10
    confidence = min(max(confidence, age_cfg.min_confidence), age_cfg.max_confidence)

    assumptions = list(BASE_ASSUMPTIONS)
    assumptions.append(
        f"Horizontal eddy diffusivity D = {diffusivity:.1f} m^2/s "
        f"({'request override' if diffusivity_m2_s else 'config.yaml -> metocean.horizontal_diffusivity'})."
    )
    if not georeferenced:
        assumptions.append(
            "Scene is not georeferenced: area derived from the assumed "
            f"{settings.detection.pixel_spacing_m:.1f} m ground sample distance."
        )
    if clamped_low or clamped_high:
        assumptions.append(
            "Inverted age fell outside the configured validity window "
            f"[{age_cfg.min_age_hours:.1f}, {age_cfg.max_age_hours:.1f}] h and was clamped."
        )

    discharge_time = reference - timedelta(hours=age_hours)
    return SpillAgeEstimate(
        age_hours=round(age_hours, 3),
        age_days=round(age_hours / 24.0, 3),
        min_possible_hours=round(age_cfg.min_age_hours, 3),
        max_possible_hours=round(age_cfg.max_age_hours, 3),
        confidence=round(confidence, 3),
        method="turbulent_diffusion_area_inversion",
        estimated_discharge_time_utc=discharge_time.isoformat(),
        assumptions=assumptions,
    )

