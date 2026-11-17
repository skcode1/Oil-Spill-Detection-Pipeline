from __future__ import annotations
import bz2
import collections
import csv
import gc
import gzip
import json
import logging
import lzma
import os
import re
import socket
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.affinity import rotate, scale
from shapely.geometry import Point, Polygon
logger = logging.getLogger(__name__)
__all__ = ['AISPrunerConfig', 'AISPrunerError', 'AISSchemaError', 'AISStreamAdapter', 'DEFAULT_SCHEMA_ALIASES', 'REQUIRED_AIS_FIELDS', 'build_aeqd_crs', 'compute_geodesic_bounding_box', 'decode_nmea_ais_sentence', 'generate_error_ellipse_polygon', 'prune_ais_stream', 'prune_dataframe', 'prune_marinecadastre_ais', 'resolve_ais_schema', 'sniff_delimiter', 'unwrap_longitude_deg']

class AISPrunerError(RuntimeError):
    pass


class AISSchemaError(AISPrunerError):
    pass
REQUIRED_AIS_FIELDS: Tuple[str, ...] = ('mmsi', 'timestamp', 'lon', 'lat')
MMSI_MIN: int = 1
MMSI_MAX: int = 999999999
DEFAULT_CHUNKSIZE: int = 250000
DEFAULT_TEMPORAL_WINDOW_HOURS: float = 3.0
DEFAULT_BOUNDING_BOX_MARGIN: float = 0.15
DEFAULT_ELLIPSE_RESOLUTION: int = 64
DEFAULT_SCHEMA_ALIASES: Dict[str, Tuple[str, ...]] = {'mmsi': ('mmsi', 'mmsi_id', 'mmsi_nbr', 'mmsi_number', 'vessel_mmsi', 'user_id'), 'timestamp': ('timestamp', 'basedatetime', 'base_date_time', 'datetime', 'date_time', 'time', 'time_utc', 'position_time', 'msg_time', 'record_time', 'date'), 'lon': ('lon', 'longitude', 'long', 'lng'), 'lat': ('lat', 'latitude'), 'sog': ('sog', 'speed_over_ground', 'speed_knots', 'speed'), 'cog': ('cog', 'course_over_ground', 'course'), 'heading': ('heading', 'true_heading'), 'vessel_name': ('vessel_name', 'vesselname', 'ship_name', 'shipname', 'name', 'vessel'), 'vessel_type': ('vessel_type', 'vesseltype', 'ship_type', 'shiptype'), 'nav_status': ('nav_status', 'navigational_status', 'status'), 'callsign': ('callsign', 'call_sign'), 'draught': ('draught', 'draft'), 'rot': ('rot', 'rate_of_turn'), 'imo': ('imo', 'imo_number'), 'destination': ('destination',)}

@dataclass(frozen=True)
class AISPrunerConfig:
    chunksize: int = DEFAULT_CHUNKSIZE
    window_before_hours: float = DEFAULT_TEMPORAL_WINDOW_HOURS
    window_after_hours: float = DEFAULT_TEMPORAL_WINDOW_HOURS
    schema_aliases: Optional[Mapping[str, Sequence[str]]] = None
    delimiter: Optional[str] = None
    encoding: str = 'utf-8'
    compression: str = 'infer'
    timestamp_format: Optional[str] = 'mixed'
    timestamp_unit: Optional[str] = None
    normalize_longitude: bool = True
    drop_invalid_mmsi: bool = False
    drop_duplicate_pings: bool = False
    on_bad_lines: str = 'error'
    bounding_box_margin_fraction: float = DEFAULT_BOUNDING_BOX_MARGIN
    ellipse_resolution: int = DEFAULT_ELLIPSE_RESOLUTION
    boundary_inclusive: bool = True
    rounding_decimals: int = 2
    temporal_rounding_decimals: int = 3
    gc_interval_chunks: Optional[int] = 25
    progress_log_interval_chunks: int = 50
    datum: str = 'WGS84'

    def __post_init__(self) -> None:
        if self.chunksize < 1:
            raise ValueError(f'chunksize must be >= 1, got {self.chunksize}')
        if self.window_before_hours < 0 or self.window_after_hours < 0:
            raise ValueError('Temporal windows must be non-negative.')
        if not 0.0 <= self.bounding_box_margin_fraction <= 1.0:
            raise ValueError('bounding_box_margin_fraction must lie in [0, 1].')
        if self.ellipse_resolution < 8:
            raise ValueError('ellipse_resolution must be >= 8 boundary segments.')
        if self.rounding_decimals < 0 or self.temporal_rounding_decimals < 0:
            raise ValueError('Rounding precisions must be non-negative.')
        if self.on_bad_lines not in ('error', 'warn', 'skip'):
            raise ValueError("on_bad_lines must be 'error', 'warn' or 'skip'.")
        if self.gc_interval_chunks is not None and self.gc_interval_chunks < 1:
            raise ValueError('gc_interval_chunks must be >= 1 or None.')
        if self.progress_log_interval_chunks < 1:
            raise ValueError('progress_log_interval_chunks must be >= 1.')

def unwrap_longitude_deg(lon: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.asarray(lon, dtype='float64') + 180.0) % 360.0 - 180.0

def build_aeqd_crs(origin_lat: float, origin_lon: float, datum: str='WGS84') -> str:
    if not -90.0 <= origin_lat <= 90.0:
        raise ValueError(f'origin_lat out of range [-90, 90]: {origin_lat}')
    if not -180.0 <= origin_lon <= 180.0:
        raise ValueError(f'origin_lon out of range [-180, 180]: {origin_lon}')
    return f'+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} +x_0=0 +y_0=0 +datum={datum} +units=m +no_defs'

def generate_error_ellipse_polygon(origin_lon: float, origin_lat: float, semi_major_m: float, semi_minor_m: float, orientation_deg: float, aeqd_crs: Optional[str]=None, resolution: int=DEFAULT_ELLIPSE_RESOLUTION) -> Polygon:
    if not np.isfinite(semi_major_m) or not np.isfinite(semi_minor_m):
        raise ValueError(f'Ellipse axes must be finite (got {semi_major_m}, {semi_minor_m}).')
    if semi_major_m <= 0 or semi_minor_m <= 0:
        raise ValueError(f'Ellipse axes must be strictly positive (got major={semi_major_m}, minor={semi_minor_m}).')
    if not np.isfinite(orientation_deg):
        raise ValueError(f'orientation_deg must be finite, got {orientation_deg}.')
    major_m = max(semi_major_m, semi_minor_m)
    minor_m = min(semi_major_m, semi_minor_m)
    circle = Point(0, 0).buffer(1.0, resolution=int(resolution))
    ellipse = scale(circle, xfact=minor_m, yfact=major_m)
    rotation_angle = -float(orientation_deg)
    return rotate(ellipse, rotation_angle, origin=(0, 0))

def compute_geodesic_bounding_box(origin_lat: float, origin_lon: float, radius_m: float, margin_fraction: float=DEFAULT_BOUNDING_BOX_MARGIN, geod: Optional[pyproj.Geod]=None) -> Tuple[float, float, float, float]:
    if radius_m <= 0 or not np.isfinite(radius_m):
        raise ValueError(f'radius_m must be a positive finite number, got {radius_m}')
    if not -90.0 <= origin_lat <= 90.0:
        raise ValueError(f'origin_lat out of range [-90, 90]: {origin_lat}')
    if not -180.0 <= origin_lon <= 180.0:
        raise ValueError(f'origin_lon out of range [-180, 180]: {origin_lon}')
    geod = geod or pyproj.Geod(ellps='WGS84')
    search_radius = radius_m * (1.0 + float(margin_fraction))
    cardinal_azimuths = (0.0, 90.0, 180.0, 270.0)
    extremal: Dict[str, float] = {'lat_min': origin_lat, 'lat_max': origin_lat, 'lon_min': origin_lon, 'lon_max': origin_lon}
    for azimuth in cardinal_azimuths:
        end_lon, end_lat, _ = geod.fwd(origin_lon, origin_lat, azimuth, search_radius)
        unwrapped_lon = float(unwrap_around_anchor(end_lon, origin_lon))
        extremal['lat_min'] = min(extremal['lat_min'], end_lat)
        extremal['lat_max'] = max(extremal['lat_max'], end_lat)
        extremal['lon_min'] = min(extremal['lon_min'], unwrapped_lon)
        extremal['lon_max'] = max(extremal['lon_max'], unwrapped_lon)
    return (float(extremal['lon_min']), float(extremal['lat_min']), float(extremal['lon_max']), float(extremal['lat_max']))

def unwrap_around_anchor(lon: Union[float, np.ndarray], anchor_lon: float) -> Union[float, np.ndarray]:
    return anchor_lon + unwrap_longitude_deg(np.asarray(lon, dtype='float64') - anchor_lon)

def resolve_ais_schema(header: Sequence[str], aliases: Optional[Mapping[str, Sequence[str]]]=None) -> Dict[str, str]:
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
            logger.debug('Duplicate candidate column %r ignored for canonical field %r.', raw_column, canonical_field)
            continue
        resolved[canonical_field] = raw_column
    missing = [name for name in REQUIRED_AIS_FIELDS if name not in resolved]
    if missing:
        guidance = '; '.join((f'{name!r} accepts any of {merged.get(name, ())}' for name in missing))
        raise AISSchemaError(f'AIS source columns {list(header)} do not satisfy required fields {list(REQUIRED_AIS_FIELDS)}. {guidance}')
    return resolved

def _normalize_header_token(column: str) -> str:
    return re.sub('[^a-z0-9]', '', str(column).strip().casefold())

def _compression_opener(path: str, compression: str) -> Callable:
    if compression != 'infer':
        return open
    suffix_openers = {'.gz': gzip.open, '.bz2': bz2.open, '.xz': lzma.open}
    return suffix_openers.get(os.path.splitext(path)[1].lower(), open)

def sniff_delimiter(path: str, encoding: str='utf-8', compression: str='infer', sample_lines: int=5) -> str:
    opener = _compression_opener(path, compression)
    try:
        with opener(path, 'rt', encoding=encoding, newline='') as handle:
            sample = ''.join((handle.readline() for _ in range(sample_lines)))
    except (OSError, UnicodeDecodeError, EOFError, lzma.LZMAError):
        return ','
    if not sample.strip():
        return ','
    try:
        return csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
    except csv.Error:
        return ','

def _parse_ping_timestamps(series: pd.Series, config: AISPrunerConfig) -> pd.Series:
    try:
        if config.timestamp_unit:
            return pd.to_datetime(series, unit=config.timestamp_unit, utc=True, errors='coerce')
        return pd.to_datetime(series, utc=True, format=config.timestamp_format, errors='coerce')
    except (TypeError, ValueError):
        return pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns, UTC]')

def _parse_t0(t0_time_str: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    try:
        t0 = pd.to_datetime(t0_time_str, utc=True)
    except (TypeError, ValueError) as err:
        raise ValueError(f'Invalid t0 timestamp {t0_time_str!r}: {err}') from err
    if pd.isna(t0):
        raise ValueError(f't0 timestamp {t0_time_str!r} resolved to NaT.')
    return t0

def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame({'mmsi': pd.Series(dtype='int64'), 'timestamp': pd.Series(dtype='datetime64[ns, UTC]'), 'lon': pd.Series(dtype='float64'), 'lat': pd.Series(dtype='float64'), 'distance_to_origin_m': pd.Series(dtype='float64'), 'temporal_offset_min': pd.Series(dtype='float64')})

def prune_ais_stream(ais_csv_path: str, origin_lon: float, origin_lat: float, t0_time_str: Union[str, datetime, pd.Timestamp], semi_major_m: float, semi_minor_m: float, orientation_deg: float, temporal_window_hours: Optional[float]=None, chunksize: Optional[int]=None, config: Optional[AISPrunerConfig]=None, window_before_hours: Optional[float]=None, window_after_hours: Optional[float]=None, schema_aliases: Optional[Mapping[str, Sequence[str]]]=None, delimiter: Optional[str]=None, sort_by: Optional[Sequence[str]]=None) -> pd.DataFrame:
    effective = config or AISPrunerConfig()
    if chunksize is not None:
        effective = replace(effective, chunksize=int(chunksize))
    if delimiter is not None:
        effective = replace(effective, delimiter=delimiter)
    merged_aliases: Dict[str, Sequence[str]] = dict(effective.schema_aliases or {})
    if schema_aliases:
        merged_aliases.update(schema_aliases)
    if not -90.0 <= origin_lat <= 90.0:
        raise ValueError(f'origin_lat out of range [-90, 90]: {origin_lat}')
    if not -180.0 <= origin_lon <= 180.0:
        raise ValueError(f'origin_lon out of range [-180, 180]: {origin_lon}')
    if not (np.isfinite(semi_major_m) and np.isfinite(semi_minor_m)):
        raise ValueError('Ellipse axes must be finite.')
    if semi_major_m <= 0 or semi_minor_m <= 0:
        raise ValueError(f'Ellipse axes must be strictly positive (got major={semi_major_m}, minor={semi_minor_m}).')
    if not np.isfinite(orientation_deg):
        raise ValueError(f'orientation_deg must be finite, got {orientation_deg}.')
    if temporal_window_hours is not None and (window_before_hours is not None or window_after_hours is not None):
        raise ValueError('Provide either temporal_window_hours or explicit window_before/after_hours, not both.')
    if temporal_window_hours is not None:
        if temporal_window_hours < 0 or not np.isfinite(temporal_window_hours):
            raise ValueError('temporal_window_hours must be a non-negative number.')
        window_before = float(temporal_window_hours)
        window_after = window_before
    else:
        window_before = float(window_before_hours if window_before_hours is not None else effective.window_before_hours)
        window_after = float(window_after_hours if window_after_hours is not None else effective.window_after_hours)
        if window_before < 0 or window_after < 0:
            raise ValueError('Temporal windows must be non-negative.')
    t0 = _parse_t0(t0_time_str)
    t_min = t0 - pd.Timedelta(hours=window_before)
    t_max = t0 + pd.Timedelta(hours=window_after)
    if not os.path.isfile(ais_csv_path):
        raise FileNotFoundError(f'AIS record source does not exist: {ais_csv_path}')
    resolved_delimiter = effective.delimiter or sniff_delimiter(ais_csv_path, encoding=effective.encoding, compression=effective.compression)
    header_frame = pd.read_csv(ais_csv_path, sep=resolved_delimiter, nrows=0, encoding=effective.encoding, compression=effective.compression)
    header = [str(column) for column in header_frame.columns]
    column_map = resolve_ais_schema(header, aliases=merged_aliases or None)
    rename_map = {column_map[canonical]: canonical for canonical in REQUIRED_AIS_FIELDS}
    max_radius = max(semi_major_m, semi_minor_m)
    lon_min, lat_min, lon_max, lat_max = compute_geodesic_bounding_box(origin_lat, origin_lon, max_radius, margin_fraction=effective.bounding_box_margin_fraction)
    aeqd_crs = build_aeqd_crs(origin_lat, origin_lon, datum=effective.datum)
    uncertainty_polygon = generate_error_ellipse_polygon(origin_lon, origin_lat, semi_major_m, semi_minor_m, orientation_deg, resolution=effective.ellipse_resolution)
    to_metric = pyproj.Transformer.from_crs('EPSG:4326', aeqd_crs, always_xy=True)
    containment = shapely.covers if effective.boundary_inclusive else shapely.within
    logger.info('Ingesting AIS traffic: %s (delimiter=%r)', ais_csv_path, resolved_delimiter)
    logger.info('Resolved schema mapping: %s', column_map)
    logger.info('Target temporal boundary: [%s -> %s] (-%.2fh before, +%.2fh after t0)', t_min.isoformat(), t_max.isoformat(), window_before, window_after)
    logger.info('Configured planar projection: %s', aeqd_crs)
    logger.info('Geodesic coarse envelope: Lat [%.4f, %.4f], Lon [%.4f, %.4f] (unwrapped around origin; margin +%.0f%%)', lat_min, lat_max, lon_min, lon_max, effective.bounding_box_margin_fraction * 100.0)
    read_kwargs: Dict[str, Any] = dict(sep=resolved_delimiter, chunksize=int(effective.chunksize), encoding=effective.encoding, compression=effective.compression, on_bad_lines=effective.on_bad_lines)
    accepted: List[pd.DataFrame] = []
    stats = {'chunks': 0, 'scanned': 0, 'time_ok': 0, 'spatial_ok': 0, 'ellipse_ok': 0, 'invalid_mmsi': 0, 'duplicates': 0, 'parse_failures': 0}
    try:
        for chunk in pd.read_csv(ais_csv_path, **read_kwargs):
            stats['chunks'] += 1
            stats['scanned'] += len(chunk)
            chunk = chunk.rename(columns=rename_map)
            ping_ts = _parse_ping_timestamps(chunk['timestamp'], effective)
            stats['parse_failures'] += int(ping_ts.isna().sum())
            time_mask = (ping_ts >= t_min) & (ping_ts <= t_max)
            if not time_mask.any():
                continue
            windowed = chunk.assign(timestamp=ping_ts).loc[time_mask]
            stats['time_ok'] += len(windowed)
            lon_values = pd.to_numeric(windowed['lon'], errors='coerce').to_numpy(dtype='float64')
            lat_values = pd.to_numeric(windowed['lat'], errors='coerce').to_numpy(dtype='float64')
            if effective.normalize_longitude:
                lon_values = unwrap_longitude_deg(lon_values)
            lon_unwrapped = unwrap_around_anchor(lon_values, origin_lon)
            finite_coords = np.isfinite(lon_values) & np.isfinite(lat_values)
            spatial_mask = finite_coords & (lat_values >= lat_min) & (lat_values <= lat_max) & (lon_unwrapped >= lon_min) & (lon_unwrapped <= lon_max)
            if not spatial_mask.any():
                continue
            candidates = windowed.loc[spatial_mask]
            stats['spatial_ok'] += len(candidates)
            mmsi_values = pd.to_numeric(candidates['mmsi'], errors='coerce').to_numpy(dtype='float64')
            invalid_mmsi = ~np.isfinite(mmsi_values) | (mmsi_values < MMSI_MIN) | (mmsi_values > MMSI_MAX)
            stats['invalid_mmsi'] += int(invalid_mmsi.sum())
            if effective.drop_invalid_mmsi and invalid_mmsi.any():
                candidates = candidates.iloc[np.flatnonzero(~invalid_mmsi)]
                if candidates.empty:
                    continue
            cand_lon = pd.to_numeric(candidates['lon'], errors='coerce').to_numpy(dtype='float64')
            cand_lat = pd.to_numeric(candidates['lat'], errors='coerce').to_numpy(dtype='float64')
            if effective.normalize_longitude:
                cand_lon = unwrap_longitude_deg(cand_lon)
            x_m, y_m = to_metric.transform(cand_lon, cand_lat)
            inside_flags = np.asarray(containment(uncertainty_polygon, shapely.points(x_m, y_m)), dtype=bool)
            inside_positions = np.flatnonzero(inside_flags)
            if inside_positions.size == 0:
                continue
            selected = candidates.iloc[inside_positions].copy()
            stats['ellipse_ok'] += len(selected)
            selected['distance_to_origin_m'] = np.round(np.hypot(np.asarray(x_m)[inside_positions], np.asarray(y_m)[inside_positions]), effective.rounding_decimals)
            selected['temporal_offset_min'] = np.round(((selected['timestamp'] - t0).dt.total_seconds() / 60.0).to_numpy(dtype='float64'), effective.temporal_rounding_decimals)
            if effective.drop_duplicate_pings:
                before_dedup = len(selected)
                selected = selected.drop_duplicates(subset=list(REQUIRED_AIS_FIELDS), keep='first')
                stats['duplicates'] += before_dedup - len(selected)
                if selected.empty:
                    continue
            accepted.append(selected)
            if effective.gc_interval_chunks and stats['chunks'] % effective.gc_interval_chunks == 0:
                gc.collect()
            if stats['chunks'] % effective.progress_log_interval_chunks == 0:
                logger.info('AIS pruning progress: chunk=%d scanned=%d candidates=%d', stats['chunks'], stats['scanned'], stats['ellipse_ok'])
    except pd.errors.ParserError as err:
        raise AISPrunerError(f"AIS CSV is structurally invalid (chunk {stats['chunks']}): {err}") from err
    except KeyError as err:
        raise AISPrunerError(f"Schema drift inside chunk {stats['chunks']}: missing column {err}") from err
    gc.collect()
    logger.info('AIS pruning summary: chunks=%d scanned=%d time_ok=%d spatial_ok=%d ellipse_ok=%d invalid_mmsi=%d duplicates=%d parse_failures=%d', stats['chunks'], stats['scanned'], stats['time_ok'], stats['spatial_ok'], stats['ellipse_ok'], stats['invalid_mmsi'], stats['duplicates'], stats['parse_failures'])
    if not accepted:
        logger.warning('Zero candidate pings matched the spatiotemporal uncertainty envelope.')
        return _empty_result_frame()
    results_df = pd.concat(accepted, ignore_index=True)
    default_sorting: Tuple[str, ...] = ('distance_to_origin_m', 'timestamp')
    ordering = [column for column in (tuple(sort_by) if sort_by else default_sorting) if column in results_df.columns]
    if ordering:
        results_df = results_df.sort_values(by=ordering, ascending=[True] * len(ordering), kind='stable').reset_index(drop=True)
    logger.info('Pruned traffic: %d pings across %d unique vessels.', len(results_df), results_df['mmsi'].nunique())
    return results_df

def prune_marinecadastre_ais(ais_csv_path: str, origin_lon: float, origin_lat: float, t0_time_str: str, semi_major_m: float, semi_minor_m: float, orientation_deg: float, temporal_window_hours: float=DEFAULT_TEMPORAL_WINDOW_HOURS, chunksize: int=DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    return prune_ais_stream(ais_csv_path=ais_csv_path, origin_lon=origin_lon, origin_lat=origin_lat, t0_time_str=t0_time_str, semi_major_m=semi_major_m, semi_minor_m=semi_minor_m, orientation_deg=orientation_deg, temporal_window_hours=temporal_window_hours, chunksize=chunksize)

def prune_dataframe(df: pd.DataFrame, origin_lon: float, origin_lat: float, t0_time_str: Union[str, datetime, pd.Timestamp], semi_major_m: float, semi_minor_m: float, orientation_deg: float, temporal_window_hours: Optional[float]=None, config: Optional[AISPrunerConfig]=None, window_before_hours: Optional[float]=None, window_after_hours: Optional[float]=None, schema_aliases: Optional[Mapping[str, Sequence[str]]]=None, sort_by: Optional[Sequence[str]]=None) -> pd.DataFrame:
    if df.empty:
        return _empty_result_frame()
    effective = config or AISPrunerConfig()
    t0 = _parse_t0(t0_time_str)
    if temporal_window_hours is not None:
        if window_before_hours is not None or window_after_hours is not None:
            raise AISPrunerError('Specify either temporal_window_hours or the explicit (window_before_hours, window_after_hours) pair, not both.')
        window_before = float(temporal_window_hours)
        window_after = float(temporal_window_hours)
    elif window_before_hours is not None or window_after_hours is not None:
        window_before = float(0.0 if window_before_hours is None else window_before_hours)
        window_after = float(0.0 if window_after_hours is None else window_after_hours)
    else:
        window_before = float(effective.window_before_hours)
        window_after = float(effective.window_after_hours)
    t_min = t0 - pd.Timedelta(hours=window_before)
    t_max = t0 + pd.Timedelta(hours=window_after)
    merged_aliases = dict(effective.schema_aliases or {})
    if schema_aliases:
        merged_aliases.update(schema_aliases)
    header = [str(column) for column in df.columns]
    column_map = resolve_ais_schema(header, aliases=merged_aliases or None)
    rename_map = {column_map[canonical]: canonical for canonical in REQUIRED_AIS_FIELDS if canonical in column_map}
    chunk = df.rename(columns=rename_map)
    ping_ts = _parse_ping_timestamps(chunk['timestamp'], effective)
    time_mask = (ping_ts >= t_min) & (ping_ts <= t_max)
    if not time_mask.any():
        return _empty_result_frame()
    windowed = chunk.assign(timestamp=ping_ts).loc[time_mask]
    max_radius = max(semi_major_m, semi_minor_m)
    lon_min, lat_min, lon_max, lat_max = compute_geodesic_bounding_box(origin_lat, origin_lon, max_radius, margin_fraction=effective.bounding_box_margin_fraction)
    lon_values = pd.to_numeric(windowed['lon'], errors='coerce').to_numpy(dtype='float64')
    lat_values = pd.to_numeric(windowed['lat'], errors='coerce').to_numpy(dtype='float64')
    if effective.normalize_longitude:
        lon_values = unwrap_longitude_deg(lon_values)
    lon_unwrapped = unwrap_around_anchor(lon_values, origin_lon)
    finite_coords = np.isfinite(lon_values) & np.isfinite(lat_values)
    spatial_mask = finite_coords & (lat_values >= lat_min) & (lat_values <= lat_max) & (lon_unwrapped >= lon_min) & (lon_unwrapped <= lon_max)
    if not spatial_mask.any():
        return _empty_result_frame()
    candidates = windowed.loc[spatial_mask]
    mmsi_values = pd.to_numeric(candidates['mmsi'], errors='coerce').to_numpy(dtype='float64')
    invalid_mmsi = ~np.isfinite(mmsi_values) | (mmsi_values < MMSI_MIN) | (mmsi_values > MMSI_MAX)
    if effective.drop_invalid_mmsi and invalid_mmsi.any():
        candidates = candidates.iloc[np.flatnonzero(~invalid_mmsi)]
        if candidates.empty:
            return _empty_result_frame()
    aeqd_crs = build_aeqd_crs(origin_lat, origin_lon, datum=effective.datum)
    uncertainty_polygon = generate_error_ellipse_polygon(origin_lon, origin_lat, semi_major_m, semi_minor_m, orientation_deg, resolution=effective.ellipse_resolution)
    to_metric = pyproj.Transformer.from_crs('EPSG:4326', aeqd_crs, always_xy=True)
    containment = shapely.covers if effective.boundary_inclusive else shapely.within
    cand_lon = pd.to_numeric(candidates['lon'], errors='coerce').to_numpy(dtype='float64')
    cand_lat = pd.to_numeric(candidates['lat'], errors='coerce').to_numpy(dtype='float64')
    if effective.normalize_longitude:
        cand_lon = unwrap_longitude_deg(cand_lon)
    x_m, y_m = to_metric.transform(cand_lon, cand_lat)
    inside_flags = np.asarray(containment(uncertainty_polygon, shapely.points(x_m, y_m)), dtype=bool)
    inside_positions = np.flatnonzero(inside_flags)
    if inside_positions.size == 0:
        return _empty_result_frame()
    selected = candidates.iloc[inside_positions].copy()
    selected['distance_to_origin_m'] = np.round(np.hypot(np.asarray(x_m)[inside_positions], np.asarray(y_m)[inside_positions]), effective.rounding_decimals)
    selected['temporal_offset_min'] = np.round(((selected['timestamp'] - t0).dt.total_seconds() / 60.0).to_numpy(dtype='float64'), effective.temporal_rounding_decimals)
    if effective.drop_duplicate_pings:
        selected = selected.drop_duplicates(subset=[col for col in REQUIRED_AIS_FIELDS if col in selected.columns], keep='first')
        if selected.empty:
            return _empty_result_frame()
    default_sorting: Tuple[str, ...] = ('distance_to_origin_m', 'timestamp')
    ordering = [column for column in (tuple(sort_by) if sort_by else default_sorting) if column in selected.columns]
    if ordering:
        selected = selected.sort_values(by=ordering, ascending=[True] * len(ordering), kind='stable').reset_index(drop=True)
    return selected

def decode_nmea_ais_sentence(sentence: str, timestamp: Optional[Union[datetime, str]]=None) -> Optional[Dict[str, Any]]:
    sentence = sentence.strip()
    if not sentence:
        return None
    if sentence.startswith('{') and sentence.endswith('}'):
        try:
            data = json.loads(sentence)
            if 'mmsi' in data and ('lat' in data or 'latitude' in data) and ('lon' in data or 'longitude' in data):
                ts = data.get('timestamp') or (timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)) if timestamp else datetime.now(timezone.utc).isoformat()
                return {'mmsi': int(data['mmsi']), 'timestamp': str(ts), 'lon': float(data.get('lon', data.get('longitude'))), 'lat': float(data.get('lat', data.get('latitude'))), 'sog': float(data.get('sog', data.get('speed', 0.0))), 'cog': float(data.get('cog', data.get('course', 0.0)))}
        except Exception:
            return None
    parts = sentence.split(',')
    if len(parts) < 6 or not parts[0].endswith(('AIVDM', 'AIVDO')):
        return None
    payload = parts[5]
    bits = ''
    try:
        for c in payload:
            v = ord(c) - 48
            if v > 40:
                v -= 8
            if not 0 <= v <= 63:
                return None
            bits += f'{v:06b}'
    except Exception:
        return None
    if len(bits) < 38:
        return None
    msg_type = int(bits[0:6], 2)
    mmsi = int(bits[8:38], 2)
    ts_iso = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp or datetime.now(timezone.utc).isoformat())
    if msg_type in (1, 2, 3) and len(bits) >= 128:
        sog_raw = int(bits[50:60], 2)
        sog = round(sog_raw / 10.0, 1) if sog_raw != 1023 else 0.0
        lon_raw = bits[61:89]
        lon_int = int(lon_raw, 2)
        if lon_raw[0] == '1':
            lon_int -= 1 << 28
        lon = round(lon_int / 600000.0, 6)
        lat_raw = bits[89:116]
        lat_int = int(lat_raw, 2)
        if lat_raw[0] == '1':
            lat_int -= 1 << 27
        lat = round(lat_int / 600000.0, 6)
        cog_raw = int(bits[116:128], 2)
        cog = round(cog_raw / 10.0, 1) if cog_raw != 3600 else 0.0
        if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0 and (lon != 181.0 and lat != 91.0):
            return {'mmsi': mmsi, 'timestamp': ts_iso, 'lon': lon, 'lat': lat, 'sog': sog, 'cog': cog}
    elif msg_type in (18, 19) and len(bits) >= 124:
        sog_raw = int(bits[46:56], 2)
        sog = round(sog_raw / 10.0, 1) if sog_raw != 1023 else 0.0
        lon_raw = bits[57:85]
        lon_int = int(lon_raw, 2)
        if lon_raw[0] == '1':
            lon_int -= 1 << 28
        lon = round(lon_int / 600000.0, 6)
        lat_raw = bits[85:112]
        lat_int = int(lat_raw, 2)
        if lat_raw[0] == '1':
            lat_int -= 1 << 27
        lat = round(lat_int / 600000.0, 6)
        cog_raw = int(bits[112:124], 2)
        cog = round(cog_raw / 10.0, 1) if cog_raw != 3600 else 0.0
        if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0 and (lon != 181.0 and lat != 91.0):
            return {'mmsi': mmsi, 'timestamp': ts_iso, 'lon': lon, 'lat': lat, 'sog': sog, 'cog': cog}
    return None

class AISStreamAdapter:

    def __init__(self, buffer_hours: float=24.0, max_buffer_size: int=500000, config: Optional[AISPrunerConfig]=None):
        self.buffer_hours = float(buffer_hours)
        self.max_buffer_size = int(max_buffer_size)
        self.config = config or AISPrunerConfig()
        self._lock = threading.RLock()
        self._buffer: collections.deque = collections.deque(maxlen=self.max_buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._server_socket: Optional[socket.socket] = None

    def ingest_line(self, line: str, timestamp: Optional[Union[datetime, str]]=None) -> Optional[Dict[str, Any]]:
        record = decode_nmea_ais_sentence(line, timestamp=timestamp)
        if record is not None:
            with self._lock:
                self._buffer.append(record)
        return record

    def ingest_records(self, records: Sequence[Dict[str, Any]]) -> int:
        count = 0
        with self._lock:
            for r in records:
                self._buffer.append(dict(r))
                count += 1
        return count

    def get_dataframe(self) -> pd.DataFrame:
        with self._lock:
            if not self._buffer:
                return _empty_result_frame()
            df = pd.DataFrame(list(self._buffer))
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.buffer_hours)
            valid = ts >= cutoff
            df = df.loc[valid].reset_index(drop=True)
        return df

    def prune(self, origin_lon: float, origin_lat: float, t0_time_str: Union[str, datetime, pd.Timestamp], semi_major_m: float, semi_minor_m: float, orientation_deg: float, temporal_window_hours: Optional[float]=None, **kwargs) -> pd.DataFrame:
        df = self.get_dataframe()
        if df.empty:
            return _empty_result_frame()
        return prune_dataframe(df, origin_lon=origin_lon, origin_lat=origin_lat, t0_time_str=t0_time_str, semi_major_m=semi_major_m, semi_minor_m=semi_minor_m, orientation_deg=orientation_deg, temporal_window_hours=temporal_window_hours, config=self.config, **kwargs)

    def start_listener(self, host: str='127.0.0.1', port: int=10110, is_server: bool=True):
        with self._lock:
            if self._running:
                return
            self._running = True

        def _worker():
            if is_server:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    srv.bind((host, port))
                    srv.listen(5)
                    srv.settimeout(1.0)
                    self._server_socket = srv
                except Exception as exc:
                    logger.warning('Could not bind AIS TCP listener on %s:%d: %s', host, port, exc)
                    self._running = False
                    return
                while self._running:
                    try:
                        conn, _ = srv.accept()
                        with conn:
                            conn.settimeout(1.0)
                            buf = ''
                            while self._running:
                                try:
                                    data = conn.recv(4096).decode('ascii', errors='ignore')
                                    if not data:
                                        break
                                    buf += data
                                    lines = buf.split('\n')
                                    buf = lines.pop()
                                    for line in lines:
                                        self.ingest_line(line)
                                except socket.timeout:
                                    continue
                    except socket.timeout:
                        continue
                    except Exception:
                        break
                try:
                    srv.close()
                except Exception:
                    pass
            else:
                while self._running:
                    try:
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect((host, port))
                        s.settimeout(2.0)
                        buf = ''
                        while self._running:
                            try:
                                data = s.recv(4096).decode('ascii', errors='ignore')
                                if not data:
                                    break
                                buf += data
                                lines = buf.split('\n')
                                buf = lines.pop()
                                for line in lines:
                                    self.ingest_line(line)
                            except socket.timeout:
                                continue
                        s.close()
                    except Exception:
                        if self._running:
                            import time
                            time.sleep(2.0)
        t = threading.Thread(target=_worker, daemon=True, name='AISStreamListener')
        self._thread = t
        t.start()

    def stop(self):
        with self._lock:
            self._running = False
            if self._server_socket:
                try:
                    self._server_socket.close()
                except Exception:
                    pass
                self._server_socket = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None