"""
Dynamic metocean data acquisition.

Builds dataset URLs for any user-specified bounding box + temporal window
(typically the output of ``metadata_extractor.extract_spatial_metadata``) and
optionally downloads them so the OpenDrift engine consumes fresh, incident-
specific metocean fields instead of cached ones.

Dataset endpoints are deployment-specific OPeNDAP/HTTP resources; they live in
``config.yaml -> metocean.datasets`` as templates with the placeholders
``{min_lat} {max_lat} {min_lon} {max_lon} {start} {end}`` (``start``/``end``
accept ``strftime`` formats such as ``{start:%Y-%m-%d}``). The built-in
defaults below are illustrative examples - production deployments override
them in configuration, not in code.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["MetoceanDataFetcher"]

DEFAULT_DATASET_TEMPLATES: Dict[str, str] = {
    # Copernicus Marine current field (OPeNDAP constraint expression example).
    "copernicus_currents": (
        "https://nrt.cmems-du.eu/thredds/dodsC/global-analysis-forecast-phy-001-024"
        "?uo[({start}):({end})][({min_lat}):({max_lat})][({min_lon}):({max_lon})]"
    ),
    # Generic wind dataset example (OPeNDAP constraint expression).
    "era5_wind": (
        "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface"
        "/uwnd.sig995.nc?uwnd[({start}):({end})][({min_lat}):({max_lat})][({min_lon}):({max_lon})]"
    ),
}


class MetoceanDataFetcher:
    """Constructs and optionally downloads incident-specific metocean fields.

    Args:
        datasets: Template overrides merged over ``DEFAULT_DATASET_TEMPLATES``
            (typically ``config.yaml -> metocean.datasets``).
        download_dir: Destination directory for fetched files.
        timeout_seconds: Network timeout for downloads.
    """

    def __init__(
        self,
        datasets: Optional[Mapping[str, str]] = None,
        download_dir: str = "data/metocean",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.templates: Dict[str, str] = dict(DEFAULT_DATASET_TEMPLATES)
        if datasets:
            self.templates.update(dict(datasets))
        self.download_dir = download_dir
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def normalize_window(
        start: datetime, end: Optional[datetime] = None, window_hours: float = 24.0
    ) -> Tuple[datetime, datetime]:
        """Resolves the temporal window; a missing end defaults to start + window."""
        if end is None:
            end = start + timedelta(hours=float(window_hours))
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def normalize_bbox(bbox: Sequence[float]) -> Dict[str, float]:
        """Validates and expands a bbox into named bounds.

        Accepts either ``[min_lon, min_lat, max_lon, max_lat]`` (GeoJSON order)
        or a mapping with ``min_lat``/``max_lat``/``min_lon``/``max_lon`` keys
        (the ``metadata_extractor`` layout).
        """
        if isinstance(bbox, Mapping):
            bounds = {
                "min_lat": float(bbox["min_lat"]),
                "max_lat": float(bbox["max_lat"]),
                "min_lon": float(bbox["min_lon"]),
                "max_lon": float(bbox["max_lon"]),
            }
        else:
            min_lon, min_lat, max_lon, max_lat = (float(value) for value in bbox)
            bounds = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}
        if not (-90.0 <= bounds["min_lat"] <= bounds["max_lat"] <= 90.0):
            raise ValueError(f"Invalid latitude bounds: {bounds}")
        if not (-180.0 <= bounds["min_lon"] <= bounds["max_lon"] <= 180.0):
            raise ValueError(f"Invalid longitude bounds: {bounds}")
        return bounds

    def build_query(
        self,
        bbox: Sequence[float] | Mapping[str, float],
        start: datetime,
        end: Optional[datetime] = None,
        window_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """Normalized incident query: named bounds + ISO window."""
        start, end = self.normalize_window(start, end, window_hours)
        return {"bbox": self.normalize_bbox(bbox), "start": start, "end": end}

    def build_url(
        self,
        dataset_key: str,
        bbox: Sequence[float] | Mapping[str, float],
        start: datetime,
        end: Optional[datetime] = None,
        window_hours: float = 24.0,
    ) -> str:
        """Expands a configured dataset template for the requested incident.

        Raises:
            KeyError: If ``dataset_key`` is not present in the template set.
        """
        if dataset_key not in self.templates:
            raise KeyError(f"Unknown metocean dataset {dataset_key!r}; available: {sorted(self.templates)}")
        query = self.build_query(bbox, start, end, window_hours)

        def _strftime_replace(match: "re.Match[str]") -> str:
            timestamp = query["start"] if match.group(1) == "start" else query["end"]
            return timestamp.strftime(match.group(2))

        expanded = template
        for name, value in fields.items():
            if isinstance(value, str):
                expanded = expanded.replace("{" + name + "}", urllib.parse.quote(value, safe=""))
            else:
                expanded = expanded.replace("{" + name + "}", str(value))
        # strftime-style placeholders, e.g. {start:%Y-%m-%d}
        expanded = re.sub(r"\{(start|end):([^}]+)\}", _strftime_replace, expanded)
        # DAP constraint expressions keep [ ( ) : characters; encode the rest.
        expanded = expanded.replace("%28", "(").replace("%29", ")").replace("%3A", ":").replace("%5B", "[").replace("%5D", "]")
        return expanded

    def fetch(
        self,
        dataset_key: str,
        bbox: Sequence[float] | Mapping[str, float],
        start: datetime,
        end: Optional[datetime] = None,
        window_hours: float = 24.0,
        dest_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Builds the dataset URL and optionally downloads it.

        Args:
            dataset_key: Template key (see class docstring).
            bbox: Incident bounding box (GeoJSON order or named mapping).
            start: Window start (usually the satellite acquisition time).
            end: Window end; defaults to ``start + window_hours``.
            window_hours: Default window length when ``end`` is omitted.
            dest_path: Explicit destination file; defaults to
                ``<download_dir>/<dataset_key>_<timestamp>.nc``.
            dry_run: When True (default) only the URL is returned - useful for
                validation and for passing OPeNDAP URLs straight to OpenDrift,
                which streams remote datasets natively.

        Returns:
            The constructed URL, or the destination path when downloaded.
        """
        url = self.build_url(dataset_key, bbox, start, end, window_hours)
        if dry_run:
            logger.info("[dry-run] %s -> %s", dataset_key, url)
            return url

        os.makedirs(self.download_dir, exist_ok=True)
        if dest_path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest_path = os.path.join(self.download_dir, f"{dataset_key}_{stamp}.nc")
        logger.info("Downloading %s -> %s", url, dest_path)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response, open(dest_path, "wb") as handle:
                handle.write(response.read())
        except Exception as exc:
            logger.error("Metocean download failed for %s: %s", dataset_key, exc)
            raise
        return dest_path
