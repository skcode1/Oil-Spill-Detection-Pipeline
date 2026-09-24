"""
Dynamic maritime region / shipping-corridor registry.

Single source of truth for geographic anchor regions used by the synthetic
AIS generator, the SAR metadata fallback georeferencing, and the dashboard.
Regions come from ``config.yaml -> regions`` when present; the built-in list
below is only the documented default so the system still boots without a
config file. Any user region can be added purely through configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

#: Default corridor anchors (name, centroid, dominant TSS bearing). These are
#: fallbacks, not runtime constants - config.yaml -> regions overrides them.
DEFAULT_REGIONS: List[Dict[str, Any]] = [
    {"name": "Strait of Malacca", "lat": 2.5000, "lon": 101.5000, "corridor_bearing": 310.0},
    {"name": "Persian Gulf TSS", "lat": 26.5000, "lon": 53.0000, "corridor_bearing": 290.0},
    {"name": "Singapore Strait", "lat": 1.2500, "lon": 103.9000, "corridor_bearing": 70.0},
    {"name": "English Channel TSS", "lat": 50.2000, "lon": -0.5000, "corridor_bearing": 65.0},
    {"name": "Gulf of Guinea", "lat": 4.5000, "lon": 6.0000, "corridor_bearing": 90.0},
]


def load_regions(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """Load the region registry, preferring config.yaml -> regions over defaults.

    Each region entry must provide ``name``, ``lat``, ``lon`` and optionally
    ``corridor_bearing`` (degrees clockwise from True North).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        List of region dicts (validated).
    """
    regions: Optional[List[Dict[str, Any]]] = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
            regions = cfg.get("regions") or None
        except Exception as exc:  # config is advisory, never fatal
            logger.warning("Could not read regions from %s (%s); using defaults.", config_path, exc)

    if not regions:
        return [dict(region) for region in DEFAULT_REGIONS]

    validated: List[Dict[str, Any]] = []
    for entry in regions:
        if not isinstance(entry, dict) or "name" not in entry or "lat" not in entry or "lon" not in entry:
            raise ValueError(f"Invalid region entry (needs name/lat/lon): {entry!r}")
        region = {
            "name": str(entry["name"]),
            "lat": float(entry["lat"]),
            "lon": float(entry["lon"]),
            "corridor_bearing": float(entry.get("corridor_bearing", 0.0)) % 360.0,
        }
        if not -90.0 <= region["lat"] <= 90.0 or not -180.0 <= region["lon"] <= 180.0:
            raise ValueError(f"Region coordinates out of range: {region!r}")
        validated.append(region)
    return validated


def get_region(
    index: int = 0,
    name: Optional[str] = None,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Fetch a region by index or name (name wins; index wraps around)."""
    regions = load_regions(config_path)
    if name:
        for region in regions:
            if region["name"].casefold() == str(name).casefold():
                return region
        raise KeyError(f"Region {name!r} not found in registry.")
    return regions[index % len(regions)]
