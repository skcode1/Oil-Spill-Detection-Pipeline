"""
Core Engine: AIS Spatiotemporal Pruning and Elliptical Uncertainty Filtering.

Production-grade, schema-agnostic AIS ingestion for the spill-attribution
pipeline. Every formerly hard-coded operational assumption is now dynamic
and validated:

    * Column schema         -> alias-resolved per feed (MarineCadastre, JSON
                               exports, synthetic simulators...) through
                               DEFAULT_SCHEMA_ALIASES plus user overrides,
                               with delimiter sniffing and compression
                               inference.
    * Temporal window       -> asymmetric (before/after t0) or symmetric,
                               config-driven instead of a fixed +-3h literal.
    * Coarse WGS84 envelope -> derived geodesically (pyproj.Geod) from the
                               uncertainty ellipse - no meters-per-degree
                               magic numbers - and antimeridian-safe.
    * Streaming profile     -> validated chunksize, bounded periodic GC,
                               progress accounting, structured summaries.
    * Ellipse geometry      -> axis/orientation normalization, configurable
                               boundary-inclusive containment.
    * Data hygiene          -> dirty-row accounting (unparseable timestamps,
                               coordinates, MMSI), optional invalid-MMSI and
                               duplicate-ping rejection, typed empty result.

Downstream output contract (canonical columns, plus every non-canonical
source column forwarded unchanged):

    mmsi                  vessel identifier (original feed representation)
    timestamp             tz-aware UTC ping timestamp
    lon, lat              float64 WGS84 coordinates (lon normalized to
                          [-180, 180) when normalize_longitude is enabled)
    distance_to_origin_m  radial metric distance to the hindcast origin
    temporal_offset_min   signed minutes between ping time and inferred t0
"""

from __future__ import annotations

import bz2
import csv
import gc
import gzip
import logging
import lzma
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.affinity import rotate, scale
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

__all__ = [
    "AISPrunerConfig",
    "AISPrunerError",
    "AISSchemaError",
    "DEFAULT_SCHEMA_ALIASES",
    "REQUIRED_AIS_FIELDS",
    "build_aeqd_crs",
    "compute_geodesic_bounding_box",
    "generate_error_ellipse_polygon",
    "prune_ais_stream",
    "prune_marinecadastre_ais",
    "resolve_ais_schema",
    "sniff_delimiter",
    "unwrap_longitude_deg",
]


class AISPrunerError(RuntimeError):
    """Base class for AIS pruner failures."""


class AISSchemaError(AISPrunerError):
    """Raised when an AIS source cannot be mapped onto the canonical schema."""


REQUIRED_AIS_FIELDS: Tuple[str, ...] = ("mmsi", "timestamp", "lon", "lat")

MMSI_MIN: int = 1
MMSI_MAX: int = 999_999_999  # IMO 9-digit MMSI range

DEFAULT_CHUNKSIZE: int = 250_000
DEFAULT_TEMPORAL_WINDOW_HOURS: float = 3.0
DEFAULT_BOUNDING_BOX_MARGIN: float = 0.15
DEFAULT_ELLIPSE_RESOLUTION: int = 64

#: Alias registry used by :func:`resolve_ais_schema`. Keys are canonical
#: fields (required by the pruner) plus optional enrichment fields; values
#: are accepted header spellings, matched case/space/underscore-insensitively.
DEFAULT_SCHEMA_ALIASES: Dict[str, Tuple[str, ...]] = {
    # Required canonical fields
    "mmsi": ("mmsi", "mmsi_id", "mmsi_nbr", "mmsi_number", "vessel_mmsi", "user_id"),
    "timestamp": (
        "timestamp", "basedatetime", "base_date_time", "datetime", "date_time",
        "time", "time_utc", "position_time", "msg_time", "record_time", "date",
    ),
    "lon": ("lon", "longitude", "long", "lng"),
    "lat": ("lat", "latitude"),
    # Optional enrichment aliases (forwarded when present; never required)
    "sog": ("sog", "speed_over_ground", "speed_knots", "speed"),
    "cog": ("cog", "course_over_ground", "course"),
    "heading": ("heading", "true_heading"),
    "vessel_name": ("vessel_name", "vesselname", "ship_name", "shipname", "name", "vessel"),
    "vessel_type": ("vessel_type", "vesseltype", "ship_type", "shiptype"),
    "nav_status": ("nav_status", "navigational_status", "status"),
    "callsign": ("callsign", "call_sign"),
    "draught": ("draught", "draft"),
    "rot": ("rot", "rate_of_turn"),
    "imo": ("imo", "imo_number"),
    "destination": ("destination",),
}


@dataclass(frozen=True)
class AISPrunerConfig:
    """Tunable operational profile for :func:`prune_ais_stream`.

    Any field can be supplied programmatically, from an environment-specific
    settings layer, or from a YAML config section; the pruner never relies
    on undocumented literals.
    """

    chunksize: int = DEFAULT_CHUNKSIZE
    window_before_hours: float = DEFAULT_TEMPORAL_WINDOW_HOURS
    window_after_hours: float = DEFAULT_TEMPORAL_WINDOW_HOURS
    #: Extra/overriding alias rules merged over DEFAULT_SCHEMA_ALIASES.
    schema_aliases: Optional[Mapping[str, Sequence[str]]] = None
    #: Explicit field delimiter; None triggers automatic sniffing.
    delimiter: Optional[str] = None
    encoding: str = "utf-8"
    compression: str = "infer"  # "infer"|"gzip"|"bz2"|"xz"|"zip"|"zstd"|None
    #: strptime-style hint, "mixed" for heterogeneous ISO feeds, or None to
    #: let pandas infer. Use ``timestamp_unit`` for numeric epoch feeds.
    timestamp_format: Optional[str] = "mixed"
    #: pandas unit ("s", "ms", "us", "ns") for numeric epoch timestamps.
    timestamp_unit: Optional[str] = None
    normalize_longitude: bool = True
    drop_invalid_mmsi: bool = False
    drop_duplicate_pings: bool = False
    #: read_csv bad-line policy: "error" | "warn" | "skip".
    on_bad_lines: str = "error"
    #: Safety margin applied to the geodesic coarse envelope (fraction).
    bounding_box_margin_fraction: float = DEFAULT_BOUNDING_BOX_MARGIN
    ellipse_resolution: int = DEFAULT_ELLIPSE_RESOLUTION
    #: True keeps pings that fall exactly on the ellipse boundary.
    boundary_inclusive: bool = True
    rounding_decimals: int = 2
    temporal_rounding_decimals: int = 3
    #: Periodic gc.collect() cadence (chunks); None disables explicit GC.
    gc_interval_chunks: Optional[int] = 25
    progress_log_interval_chunks: int = 50
    datum: str = "WGS84"

    def __post_init__(self) -> None:
        if self.chunksize < 1:
            raise ValueError(f"chunksize must be >= 1, got {self.chunksize}")
        if self.window_before_hours < 0 or self.window_after_hours < 0:
            raise ValueError("Temporal windows must be non-negative.")
        if not 0.0 <= self.bounding_box_margin_fraction <= 1.0:
            raise ValueError("bounding_box_margin_fraction must lie in [0, 1].")
        if self.ellipse_resolution < 8:
            raise ValueError("ellipse_resolution must be >= 8 boundary segments.")
        if self.rounding_decimals < 0 or self.temporal_rounding_decimals < 0:
            raise ValueError("Rounding precisions must be non-negative.")
        if self.on_bad_lines not in ("error", "warn", "skip"):
            raise ValueError("on_bad_lines must be 'error', 'warn' or 'skip'.")
        if self.gc_interval_chunks is not None and self.gc_interval_chunks < 1:
            raise ValueError("gc_interval_chunks must be >= 1 or None.")
        if self.progress_log_interval_chunks < 1:
            raise ValueError("progress_log_interval_chunks must be >= 1.")


def unwrap_longitude_deg(lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize longitude(s) into the [-180, 180) principal branch."""
    return (np.asarray(lon, dtype="float64") + 180.0) % 360.0 - 180.0


def build_aeqd_crs(origin_lat: float, origin_lon: float, datum: str = "WGS84") -> str:
    """Build a local Azimuthal Equidistant Proj4 string centered on the origin.

    AEQD preserves true metric distance radially from the origin, which makes
    it the natural planar frame for uncertainty-ellipse containment tests
    without UTM zone-border artifacts.

    Args:
        origin_lat: Geodetic latitude of the projection origin (WGS84).
        origin_lon: Geodetic longitude of the projection origin (WGS84).
        datum: Geodetic datum identifier (default ``WGS84``).

    Returns:
        Proj4 projection string.

    Raises:
        ValueError: If the supplied coordinates fall outside valid ranges.
    """
    if not -90.0 <= origin_lat <= 90.0:
        raise ValueError(f"origin_lat out of range [-90, 90]: {origin_lat}")
    if not -180.0 <= origin_lon <= 180.0:
        raise ValueError(f"origin_lon out of range [-180, 180]: {origin_lon}")
    return (
        f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
        f"+x_0=0 +y_0=0 +datum={datum} +units=m +no_defs"
    )


def generate_error_ellipse_polygon(
    origin_lon: float,
    origin_lat: float,
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    aeqd_crs: Optional[str] = None,
    resolution: int = DEFAULT_ELLIPSE_RESOLUTION,
) -> Polygon:
    """Construct a metric-accurate, rotated 2D uncertainty ellipse.

    The ellipse is centered at (0, 0) of the local AEQD projection plane:
    the semi-major axis initially points North and the semi-minor axis East,
    then the whole figure is rotated clockwise from True North by
    ``orientation_deg``.

    Args:
        origin_lon: Spill origin longitude (WGS84); retained for interface
            compatibility and signature introspection by callers.
        origin_lat: Spill origin latitude (WGS84).
        semi_major_m: Length of the semi-major axis in meters.
        semi_minor_m: Length of the semi-minor axis in meters.
        orientation_deg: Heading of the semi-major axis, degrees clockwise
            from True North.
        aeqd_crs: Optional Proj4 projection string centered on the origin.
            Accepted for interface compatibility; containment tests are
            performed in the metric plane where the polygon already lives.
        resolution: Boundary vertex count used when approximating the unit
            circle before scaling.

    Returns:
        Shapely Polygon representing the rotated uncertainty perimeter.

    Raises:
        ValueError: On non-finite or non-positive axis magnitudes.
    """
    if not np.isfinite(semi_major_m) or not np.isfinite(semi_minor_m):
        raise ValueError(f"Ellipse axes must be finite (got {semi_major_m}, {semi_minor_m}).")
    if semi_major_m <= 0 or semi_minor_m <= 0:
        raise ValueError(
            f"Ellipse axes must be strictly positive (got major={semi_major_m}, "
            f"minor={semi_minor_m})."
        )
    if not np.isfinite(orientation_deg):
        raise ValueError(f"orientation_deg must be finite, got {orientation_deg}.")

    # Axes are order-agnostic: the major axis always drives the long dimension
    # so callers may pass (major, minor) or (a, b) from an upstream covariance
    # decomposition without silently producing a degenerate/inverted ellipse.
    major_m = max(semi_major_m, semi_minor_m)
    minor_m = min(semi_major_m, semi_minor_m)

    # Unit circle centered at the local planar origin (0, 0).
    circle = Point(0, 0).buffer(1.0, resolution=int(resolution))

    # Scale: major along Y (North), minor along X (East).
    ellipse = scale(circle, xfact=minor_m, yfact=major_m)

    # Shapely rotates counter-clockwise (math convention, CCW from +X/East).
    # The unscaled major axis already points along +Y (math angle 90 deg).
    # A compass bearing B (clockwise from North) corresponds to the math
    # angle (90 - B), so the required CCW rotation is theta = -B: the major
    # axis lands on (sin B, cos B), i.e. azimuth B clockwise from North.
    rotation_angle = -float(orientation_deg)
    return rotate(ellipse, rotation_angle, origin=(0, 0))


def compute_geodesic_bounding_box(
    origin_lat: float,
    origin_lon: float,
    radius_m: float,
    margin_fraction: float = DEFAULT_BOUNDING_BOX_MARGIN,
    geod: Optional[pyproj.Geod] = None,
) -> Tuple[float, float, float, float]:
    """Compute a WGS84 pre-filter envelope around the origin geodesically.

    Projects ``radius_m`` (inflated by ``margin_fraction``) along the four
    cardinal directions on the WGS84 ellipsoid via :class:`pyproj.Geod`,
    avoiding the meters-per-degree equirectangular approximation that degrades
    toward the poles.

    Args:
        origin_lat: Envelope center latitude (WGS84).
        origin_lon: Envelope center longitude (WGS84).
        radius_m: Characteristic search radius in meters (use the ellipse
            semi-major axis as the upper bound).
        margin_fraction: Relative safety inflation in [0, 1].
        geod: Optional shared :class:`pyproj.Geod` instance.

    Returns:
        Tuple ``(lon_min, lat_min, lon_max, lat_max)`` in degrees. Longitudes
        are expressed in the unwrapped branch anchored at ``origin_lon``
        (``lon_min``/``lon_max`` may leave the [-180, 180) principal branch
        when the envelope crosses the antimeridian); callers must compare
        feed longitudes under the same convention (see
        :func:`unwrap_around_anchor`).
    """
    if radius_m <= 0 or not np.isfinite(radius_m):
        raise ValueError(f"radius_m must be a positive finite number, got {radius_m}")
    if not -90.0 <= origin_lat <= 90.0:
        raise ValueError(f"origin_lat out of range [-90, 90]: {origin_lat}")
    if not -180.0 <= origin_lon <= 180.0:
        raise ValueError(f"origin_lon out of range [-180, 180]: {origin_lon}")
    geod = geod or pyproj.Geod(ellps="WGS84")
    search_radius = radius_m * (1.0 + float(margin_fraction))

    cardinal_azimuths = (0.0, 90.0, 180.0, 270.0)  # N, E, S, W
    extremal: Dict[str, float] = {"lat_min": origin_lat, "lat_max": origin_lat,
                                  "lon_min": origin_lon, "lon_max": origin_lon}
    for azimuth in cardinal_azimuths:
        end_lon, end_lat, _ = geod.fwd(origin_lon, origin_lat, azimuth, search_radius)
        # pyproj normalizes returned longitudes to [-180, 180); re-express
        # each endpoint in the unwrapped branch anchored at origin_lon so an
        # antimeridian-crossing envelope stays a monotonic interval.
        unwrapped_lon = float(unwrap_around_anchor(end_lon, origin_lon))
        extremal["lat_min"] = min(extremal["lat_min"], end_lat)
        extremal["lat_max"] = max(extremal["lat_max"], end_lat)
        extremal["lon_min"] = min(extremal["lon_min"], unwrapped_lon)
        extremal["lon_max"] = max(extremal["lon_max"], unwrapped_lon)

    return (
        float(extremal["lon_min"]),
        float(extremal["lat_min"]),
        float(extremal["lon_max"]),
        float(extremal["lat_max"]),
    )


def unwrap_around_anchor(
    lon: Union[float, np.ndarray], anchor_lon: float
) -> Union[float, np.ndarray]:
    """Re-express longitude(s) in the 360-degree branch anchored at a center.

    Enables plain interval comparisons across the antimeridian: for an origin
    at 179.95 E, a ping at -179.9 E unwraps to 180.1 instead of crossing the
    principal-branch discontinuity.

    Args:
        lon: Longitude(s) in degrees, any branch.
        anchor_lon: Reference longitude anchoring the output branch.

    Returns:
        Unwrapped longitude(s) within (anchor_lon - 180, anchor_lon + 180).
    """
    return anchor_lon + unwrap_longitude_deg(np.asarray(lon, dtype="float64") - anchor_lon)


def resolve_ais_schema(
    header: Sequence[str],
    aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, str]:
    """Map raw feed columns onto canonical AIS fields via alias lookup.

    Matching is exact after normalization (casefold + removal of every
    non-alphanumeric character), so ``BaseDateTime``, ``base_date_time`` and
    ``Base Date Time`` resolve identically. The first header column claiming a
    canonical field wins; later duplicates are ignored, which keeps
    MarineCadastre exports, simulated feeds and vendor variants working
    without caller-side column renaming.

    Args:
        header: Raw column names exactly as present in the source.
        aliases: Optional alias overrides merged on top of
            DEFAULT_SCHEMA_ALIASES (a field supplied here replaces its
            default alias tuple entirely).

    Returns:
        Mapping of canonical field -> actual source column name, covering
        REQUIRED_AIS_FIELDS plus every recognized optional enrichment field.

    Raises:
        AISSchemaError: If any required canonical field cannot be resolved.
    """
    merged: Dict[str, Tuple[str, ...]] = dict(DEFAULT_SCHEMA_ALIASES)
    if aliases:
        merged.update({name: tuple(tuple_of_aliases) for name, tuple_of_aliases in aliases.items()})

    alias_lookup: Dict[str, str] = {}
    for canonical_field, alias_tuple in merged.items():
        for alias in alias_tuple:
            alias_lookup.setdefault(_normalize_header_token(alias), canonical_field)

    resolved: Dict[str, str] = {}
    for raw_column in header:
        canonical_field = alias_lookup.get(_normalize_header_token(raw_column))
        if canonical_field is None:
            continue
        if canonical_field in resolved:
            logger.debug(
                "Duplicate candidate column %r ignored for canonical field %r.",
                raw_column,
                canonical_field,
            )
            continue
        resolved[canonical_field] = raw_column

    missing = [name for name in REQUIRED_AIS_FIELDS if name not in resolved]
    if missing:
        guidance = "; ".join(
            f"{name!r} accepts any of {merged.get(name, ())}" for name in missing
        )
        raise AISSchemaError(
            f"AIS source columns {list(header)} do not satisfy required fields "
            f"{list(REQUIRED_AIS_FIELDS)}. {guidance}"
        )
    return resolved


def _normalize_header_token(column: str) -> str:
    """Case/space/underscore/hyphen-insensitive token for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(column).strip().casefold())


def _compression_opener(path: str, compression: str) -> Callable:
    """Return a text-mode opener honoring inferred archive compression."""
    if compression != "infer":
        return open
    suffix_openers = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
    return suffix_openers.get(os.path.splitext(path)[1].lower(), open)


def sniff_delimiter(
    path: str,
    encoding: str = "utf-8",
    compression: str = "infer",
    sample_lines: int = 5,
) -> str:
    """Detect the field delimiter of a delimited AIS source.

    Samples the first ``sample_lines`` records (transparently decompressing
    gz/bz2/xz archives) and falls back to ',' when the dialect is ambiguous,
    so single-column probes never abort ingestion.

    Args:
        path: CSV path to probe.
        encoding: Text encoding of the source.
        compression: "infer" to pick the opener from the file extension.
        sample_lines: Number of leading records used for detection.

    Returns:
        The detected single-character delimiter.
    """
    opener = _compression_opener(path, compression)
    try:
        with opener(path, "rt", encoding=encoding, newline="") as handle:
            sample = "".join(handle.readline() for _ in range(sample_lines))
    except (OSError, UnicodeDecodeError, EOFError, lzma.LZMAError):
        return ","
    if not sample.strip():
        return ","
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _parse_ping_timestamps(series: pd.Series, config: AISPrunerConfig) -> pd.Series:
    """Vectorized UTC timestamp coercion; unparseable rows become NaT."""
    try:
        if config.timestamp_unit:
            return pd.to_datetime(series, unit=config.timestamp_unit, utc=True, errors="coerce")
        return pd.to_datetime(series, utc=True, format=config.timestamp_format, errors="coerce")
    except (TypeError, ValueError):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")


def _parse_t0(t0_time_str: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Parse the discharge instant; naive inputs are interpreted as UTC."""
    try:
        t0 = pd.to_datetime(t0_time_str, utc=True)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid t0 timestamp {t0_time_str!r}: {err}") from err
    if pd.isna(t0):
        raise ValueError(f"t0 timestamp {t0_time_str!r} resolved to NaT.")
    return t0


def _empty_result_frame() -> pd.DataFrame:
    """Typed, schema-stable empty frame for zero-match runs."""
    return pd.DataFrame(
        {
            "mmsi": pd.Series(dtype="int64"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "lon": pd.Series(dtype="float64"),
            "lat": pd.Series(dtype="float64"),
            "distance_to_origin_m": pd.Series(dtype="float64"),
            "temporal_offset_min": pd.Series(dtype="float64"),
        }
    )


def prune_ais_stream(
    ais_csv_path: str,
    origin_lon: float,
    origin_lat: float,
    t0_time_str: Union[str, datetime, pd.Timestamp],
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    temporal_window_hours: Optional[float] = None,
    chunksize: Optional[int] = None,
    config: Optional[AISPrunerConfig] = None,
    window_before_hours: Optional[float] = None,
    window_after_hours: Optional[float] = None,
    schema_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    delimiter: Optional[str] = None,
    sort_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter a streamed AIS archive against an elliptical uncertainty envelope.

    Records are consumed chunk-by-chunk (working memory stays O(chunksize))
    through three vectorized stages: temporal windowing, a geodesic coarse
    WGS84 envelope, and fine point-in-ellipse containment inside a local
    AEQD metric plane. The feed schema, delimiters, compression and all
    operational tolerances are resolved dynamically from
    :class:`AISPrunerConfig` and the alias registry.

    Args:
        ais_csv_path: AIS source path (comma/semicolon/tab/pipe delimited;
            gzip/bz2/xz archives supported via compression inference).
        origin_lon: WGS84 longitude of the hindcast discharge origin.
        origin_lat: WGS84 latitude of the hindcast discharge origin.
        t0_time_str: Inferred discharge time (ISO 8601 string, ``datetime``
            or ``pd.Timestamp``); naive values are interpreted as UTC.
        semi_major_m: Uncertainty ellipse semi-major axis (meters). Axis
            roles are normalized, so either order is accepted.
        semi_minor_m: Uncertainty ellipse semi-minor axis (meters).
        orientation_deg: Bearing of the semi-major axis, degrees clockwise
            from True North.
        temporal_window_hours: Symmetric search window around t0. Mutually
            exclusive with the explicit asymmetric pair below; when omitted,
            ``config.window_before_hours``/``window_after_hours`` apply.
        chunksize: Streaming batch size; overrides ``config.chunksize``.
        config: Optional :class:`AISPrunerConfig` operational profile.
        window_before_hours: Explicit lookback window before t0 (hours);
            an omitted side defaults to 0.0 when the pair is used.
        window_after_hours: Explicit lookahead window after t0 (hours).
        schema_aliases: Extra alias rules merged over ``config.schema_aliases``
            and DEFAULT_SCHEMA_ALIASES (per-field overrides win).
        delimiter: Explicit field delimiter; skips sniffing.
        sort_by: Output ordering keys; defaults to
            ("distance_to_origin_m", "timestamp").

    Returns:
        Canonical DataFrame described in the module docstring (an empty but
        typed frame when nothing matches).

    Raises:
        FileNotFoundError: If ``ais_csv_path`` does not exist.
        AISSchemaError: If required columns cannot be resolved.
        ValueError: On invalid geometry/window/config parameters.
        AISPrunerError: On structurally invalid streams or schema drift.
    """
    effective = config or AISPrunerConfig()
    if chunksize is not None:
        effective = replace(effective, chunksize=int(chunksize))
    if delimiter is not None:
        effective = replace(effective, delimiter=delimiter)

    merged_aliases: Dict[str, Sequence[str]] = dict(effective.schema_aliases or {})
    if schema_aliases:
        merged_aliases.update(schema_aliases)

    if not -90.0 <= origin_lat <= 90.0:
        raise ValueError(f"origin_lat out of range [-90, 90]: {origin_lat}")
    if not -180.0 <= origin_lon <= 180.0:
        raise ValueError(f"origin_lon out of range [-180, 180]: {origin_lon}")
    if not (np.isfinite(semi_major_m) and np.isfinite(semi_minor_m)):
        raise ValueError("Ellipse axes must be finite.")
    if semi_major_m <= 0 or semi_minor_m <= 0:
        raise ValueError(
            f"Ellipse axes must be strictly positive (got major={semi_major_m}, "
            f"minor={semi_minor_m})."
        )
    if not np.isfinite(orientation_deg):
        raise ValueError(f"orientation_deg must be finite, got {orientation_deg}.")

    if temporal_window_hours is not None and (
        window_before_hours is not None or window_after_hours is not None
    ):
        raise ValueError(
            "Provide either temporal_window_hours or explicit "
            "window_before/after_hours, not both."
        )
    if temporal_window_hours is not None:
        if temporal_window_hours < 0 or not np.isfinite(temporal_window_hours):
            raise ValueError("temporal_window_hours must be a non-negative number.")
        window_before = float(temporal_window_hours)
        window_after = window_before
    else:
        window_before = float(
            window_before_hours
            if window_before_hours is not None
            else effective.window_before_hours
        )
        window_after = float(
            window_after_hours
            if window_after_hours is not None
            else effective.window_after_hours
        )
        if window_before < 0 or window_after < 0:
            raise ValueError("Temporal windows must be non-negative.")

    t0 = _parse_t0(t0_time_str)
    t_min = t0 - pd.Timedelta(hours=window_before)
    t_max = t0 + pd.Timedelta(hours=window_after)

    if not os.path.isfile(ais_csv_path):
        raise FileNotFoundError(f"AIS record source does not exist: {ais_csv_path}")

    resolved_delimiter = effective.delimiter or sniff_delimiter(
        ais_csv_path, encoding=effective.encoding, compression=effective.compression
    )
    header_frame = pd.read_csv(
        ais_csv_path,
        sep=resolved_delimiter,
        nrows=0,
        encoding=effective.encoding,
        compression=effective.compression,
    )
    header = [str(column) for column in header_frame.columns]
    column_map = resolve_ais_schema(header, aliases=merged_aliases or None)
    rename_map = {
        column_map[canonical]: canonical for canonical in REQUIRED_AIS_FIELDS
    }

    max_radius = max(semi_major_m, semi_minor_m)
    lon_min, lat_min, lon_max, lat_max = compute_geodesic_bounding_box(
        origin_lat,
        origin_lon,
        max_radius,
        margin_fraction=effective.bounding_box_margin_fraction,
    )
    aeqd_crs = build_aeqd_crs(origin_lat, origin_lon, datum=effective.datum)
    uncertainty_polygon = generate_error_ellipse_polygon(
        origin_lon,
        origin_lat,
        semi_major_m,
        semi_minor_m,
        orientation_deg,
        resolution=effective.ellipse_resolution,
    )
    to_metric = pyproj.Transformer.from_crs("EPSG:4326", aeqd_crs, always_xy=True)
    containment = shapely.covers if effective.boundary_inclusive else shapely.within

    logger.info("Ingesting AIS traffic: %s (delimiter=%r)", ais_csv_path, resolved_delimiter)
    logger.info("Resolved schema mapping: %s", column_map)
    logger.info(
        "Target temporal boundary: [%s -> %s] (-%.2fh before, +%.2fh after t0)",
        t_min.isoformat(),
        t_max.isoformat(),
        window_before,
        window_after,
    )
    logger.info("Configured planar projection: %s", aeqd_crs)
    logger.info(
        "Geodesic coarse envelope: Lat [%.4f, %.4f], Lon [%.4f, %.4f] "
        "(unwrapped around origin; margin +%.0f%%)",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        effective.bounding_box_margin_fraction * 100.0,
    )

    read_kwargs: Dict[str, Any] = dict(
        sep=resolved_delimiter,
        chunksize=int(effective.chunksize),
        encoding=effective.encoding,
        compression=effective.compression,
        on_bad_lines=effective.on_bad_lines,
    )

    accepted: List[pd.DataFrame] = []
    stats = {
        "chunks": 0,
        "scanned": 0,
        "time_ok": 0,
        "spatial_ok": 0,
        "ellipse_ok": 0,
        "invalid_mmsi": 0,
        "duplicates": 0,
        "parse_failures": 0,
    }

    try:
        for chunk in pd.read_csv(ais_csv_path, **read_kwargs):
            stats["chunks"] += 1
            stats["scanned"] += len(chunk)
            chunk = chunk.rename(columns=rename_map)

            # --- Stage 1: vectorized temporal window filter -----------------
            ping_ts = _parse_ping_timestamps(chunk["timestamp"], effective)
            stats["parse_failures"] += int(ping_ts.isna().sum())
            time_mask = (ping_ts >= t_min) & (ping_ts <= t_max)
            if not time_mask.any():
                continue
            windowed = chunk.assign(timestamp=ping_ts).loc[time_mask]
            stats["time_ok"] += len(windowed)

            # --- Stage 2: geodesic coarse envelope (antimeridian-safe) ------
            lon_values = pd.to_numeric(windowed["lon"], errors="coerce").to_numpy(dtype="float64")
            lat_values = pd.to_numeric(windowed["lat"], errors="coerce").to_numpy(dtype="float64")
            if effective.normalize_longitude:
                lon_values = unwrap_longitude_deg(lon_values)
            lon_unwrapped = unwrap_around_anchor(lon_values, origin_lon)
            finite_coords = np.isfinite(lon_values) & np.isfinite(lat_values)
            spatial_mask = (
                finite_coords
                & (lat_values >= lat_min)
                & (lat_values <= lat_max)
                & (lon_unwrapped >= lon_min)
                & (lon_unwrapped <= lon_max)
            )
            if not spatial_mask.any():
                continue
            candidates = windowed.loc[spatial_mask]
            stats["spatial_ok"] += len(candidates)

            # Optional MMSI hygiene (IMO 9-digit plausibility window).
            mmsi_values = pd.to_numeric(candidates["mmsi"], errors="coerce").to_numpy(dtype="float64")
            invalid_mmsi = (
                ~np.isfinite(mmsi_values)
                | (mmsi_values < MMSI_MIN)
                | (mmsi_values > MMSI_MAX)
            )
            stats["invalid_mmsi"] += int(invalid_mmsi.sum())
            if effective.drop_invalid_mmsi and invalid_mmsi.any():
                candidates = candidates.iloc[np.flatnonzero(~invalid_mmsi)]
                if candidates.empty:
                    continue

            # --- Stage 3: fine containment in the AEQD metric plane ---------
            cand_lon = pd.to_numeric(candidates["lon"], errors="coerce").to_numpy(dtype="float64")
            cand_lat = pd.to_numeric(candidates["lat"], errors="coerce").to_numpy(dtype="float64")
            if effective.normalize_longitude:
                cand_lon = unwrap_longitude_deg(cand_lon)
            x_m, y_m = to_metric.transform(cand_lon, cand_lat)
            inside_flags = np.asarray(
                containment(uncertainty_polygon, shapely.points(x_m, y_m)),
                dtype=bool,
            )
            inside_positions = np.flatnonzero(inside_flags)
            if inside_positions.size == 0:
                continue

            selected = candidates.iloc[inside_positions].copy()
            stats["ellipse_ok"] += len(selected)
            selected["distance_to_origin_m"] = np.round(
                np.hypot(
                    np.asarray(x_m)[inside_positions],
                    np.asarray(y_m)[inside_positions],
                ),
                effective.rounding_decimals,
            )
            selected["temporal_offset_min"] = np.round(
                ((selected["timestamp"] - t0).dt.total_seconds() / 60.0).to_numpy(
                    dtype="float64"
                ),
                effective.temporal_rounding_decimals,
            )

            if effective.drop_duplicate_pings:
                before_dedup = len(selected)
                selected = selected.drop_duplicates(
                    subset=list(REQUIRED_AIS_FIELDS), keep="first"
                )
                stats["duplicates"] += before_dedup - len(selected)
                if selected.empty:
                    continue

            accepted.append(selected)

            if (
                effective.gc_interval_chunks
                and stats["chunks"] % effective.gc_interval_chunks == 0
            ):
                gc.collect()
            if stats["chunks"] % effective.progress_log_interval_chunks == 0:
                logger.info(
                    "AIS pruning progress: chunk=%d scanned=%d candidates=%d",
                    stats["chunks"],
                    stats["scanned"],
                    stats["ellipse_ok"],
                )
    except pd.errors.ParserError as err:
        raise AISPrunerError(
            f"AIS CSV is structurally invalid (chunk {stats['chunks']}): {err}"
        ) from err
    except KeyError as err:
        raise AISPrunerError(
            f"Schema drift inside chunk {stats['chunks']}: missing column {err}"
        ) from err

    gc.collect()
    logger.info(
        "AIS pruning summary: chunks=%d scanned=%d time_ok=%d spatial_ok=%d "
        "ellipse_ok=%d invalid_mmsi=%d duplicates=%d parse_failures=%d",
        stats["chunks"],
        stats["scanned"],
        stats["time_ok"],
        stats["spatial_ok"],
        stats["ellipse_ok"],
        stats["invalid_mmsi"],
        stats["duplicates"],
        stats["parse_failures"],
    )

    if not accepted:
        logger.warning(
            "Zero candidate pings matched the spatiotemporal uncertainty envelope."
        )
        return _empty_result_frame()

    results_df = pd.concat(accepted, ignore_index=True)
    default_sorting: Tuple[str, ...] = ("distance_to_origin_m", "timestamp")
    ordering = [
        column
        for column in (tuple(sort_by) if sort_by else default_sorting)
        if column in results_df.columns
    ]
    if ordering:
        results_df = results_df.sort_values(
            by=ordering, ascending=[True] * len(ordering), kind="stable"
        ).reset_index(drop=True)

    logger.info(
        "Pruned traffic: %d pings across %d unique vessels.",
        len(results_df),
        results_df["mmsi"].nunique(),
    )
    return results_df


def prune_marinecadastre_ais(
    ais_csv_path: str,
    origin_lon: float,
    origin_lat: float,
    t0_time_str: str,
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    temporal_window_hours: float = DEFAULT_TEMPORAL_WINDOW_HOURS,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Backwards-compatible entry point for existing call sites.

    Retained for ``run_pipeline.execute_stage2_pruning`` and any external
    consumers of the historical name. Delegates to :func:`prune_ais_stream`,
    whose schema auto-detection means the source no longer has to be a
    MarineCadastre export despite the historical function name.
    """
    return prune_ais_stream(
        ais_csv_path=ais_csv_path,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        t0_time_str=t0_time_str,
        semi_major_m=semi_major_m,
        semi_minor_m=semi_minor_m,
        orientation_deg=orientation_deg,
        temporal_window_hours=temporal_window_hours,
        chunksize=chunksize,
    )
