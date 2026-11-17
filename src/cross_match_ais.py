from __future__ import annotations
import argparse
import json
import logging
import math
import os
import sys
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import geopandas as gpd
import pandas as pd
import yaml
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.ais_pruner import prune_ais_stream
from src.attribution_engine import AttributionConfig, MultiFactorAttributionScorer
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('CrossMatchCLI')

def _load_config(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle) or {}
    logger.warning('Config %s not found; using built-in defaults.', path)
    return {}

def _geodesic_step(lat: float, lon: float, distance_m: float, bearing_deg: float):
    radius = 6371000.0
    delta = distance_m / radius
    theta = math.radians(bearing_deg)
    phi1, lam0 = (math.radians(lat), math.radians(lon))
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lam2 = lam0 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(phi1), math.cos(delta) - math.sin(phi1) * math.sin(phi2))
    return (math.degrees(phi2), math.degrees(lam2))

def _analytical_origin(cfg, lat_sat, lon_sat, t_sat, hindcast_hours):
    metocean_cfg = cfg.get('metocean') or {}
    u_current = float(metocean_cfg.get('fallback_current_u_mps', 0.22))
    v_current = float(metocean_cfg.get('fallback_current_v_mps', 0.14))
    u_wind = float(metocean_cfg.get('fallback_wind_u_mps', 5.2))
    v_wind = float(metocean_cfg.get('fallback_wind_v_mps', 3.1))
    leeway = float(metocean_cfg.get('wind_drift_factor', 0.032))
    drift_x = u_current + leeway * u_wind
    drift_y = v_current + leeway * v_wind
    drift_speed = math.hypot(drift_x, drift_y)
    drift_bearing = (math.degrees(math.atan2(drift_x, drift_y)) + 360.0) % 360.0
    drift_distance = drift_speed * hindcast_hours * 3600.0
    lat0, lon0 = _geodesic_step(lat_sat, lon_sat, drift_distance, (drift_bearing + 180.0) % 360.0)
    t0 = t_sat - timedelta(hours=hindcast_hours)
    profile = {'timestamp_sat': t_sat.isoformat(), 'u_current_mps': u_current, 'v_current_mps': v_current, 'u_wind_mps': u_wind, 'v_wind_mps': v_wind, 'wind_drift_factor': leeway, 'hindcast_hours': hindcast_hours, 'origin_estimate': {'origin_latitude': round(lat0, 6), 'origin_longitude': round(lon0, 6), 'origin_timestamp': t0.isoformat(), 'net_drift_meters': round(drift_distance, 2), 'drift_bearing_deg': round(drift_bearing, 2)}}
    return (lat0, lon0, t0, profile)

def _resolve_incident(cfg, spill_geojson, hindcast_hours):
    processed_dir = (cfg.get('paths') or {}).get('processed_dir', 'data/processed')
    metocean_json = os.path.join(processed_dir, 'metocean_environment.json')
    if not spill_geojson or not os.path.exists(spill_geojson):
        raise FileNotFoundError(f'Spill polygon GeoJSON not found: {spill_geojson}')
    slick_gdf = gpd.read_file(spill_geojson)
    if slick_gdf.empty:
        raise ValueError(f'Spill polygon GeoJSON is empty: {spill_geojson}')
    union = slick_gdf.union_all() if hasattr(slick_gdf, 'union_all') else slick_gdf.unary_union
    centroid = union.centroid
    lat_sat, lon_sat = (float(centroid.y), float(centroid.x))
    t_sat = datetime.now(timezone.utc)
    if os.path.exists(metocean_json):
        try:
            with open(metocean_json, 'r', encoding='utf-8') as handle:
                profile = json.load(handle)
            origin = profile.get('origin_estimate', {})
            lat0 = float(origin['origin_latitude'])
            lon0 = float(origin['origin_longitude'])
            t0 = pd.to_datetime(origin['origin_timestamp']).to_pydatetime()
            logger.info('Resolved origin from metocean manifest: (%.5f, %.5f) @ %s', lat0, lon0, t0)
            return (lat0, lon0, t0, profile)
        except Exception as exc:
            logger.warning('Metocean manifest unreadable (%s); using analytical fallback.', exc)
    logger.warning('Metocean manifest missing; using config-driven analytical fallback.')
    return _analytical_origin(cfg, lat_sat, lon_sat, t_sat, hindcast_hours)

def run_cross_matching(config_path: str='config.yaml', ais_csv_path: Optional[str]=None, output_dir: Optional[str]=None, spill_geojson: Optional[str]=None) -> pd.DataFrame:
    cfg = _load_config(config_path)
    paths_cfg = cfg.get('paths') or {}
    ais_cfg = cfg.get('ais') or {}
    metocean_cfg = cfg.get('metocean') or {}
    output_dir = output_dir or paths_cfg.get('output_dir', 'results')
    ais_csv_path = ais_csv_path or paths_cfg.get('ais_csv', 'data/raw/ais_data.csv')
    spill_geojson = spill_geojson or paths_cfg.get('spill_geojson', os.path.join(output_dir, 'spill_polygon.geojson'))
    os.makedirs(output_dir, exist_ok=True)
    fallback_hours = float(metocean_cfg.get('fallback_hindcast_hours', 4.0))
    lat0, lon0, t0, metocean_profile = _resolve_incident(cfg, spill_geojson, fallback_hours)
    radius_m = float(ais_cfg.get('buffer_distance_m', 5000.0))
    pruned = prune_ais_stream(ais_csv_path=ais_csv_path, origin_lon=lon0, origin_lat=lat0, t0_time_str=t0, semi_major_m=radius_m, semi_minor_m=radius_m, orientation_deg=0.0, temporal_window_hours=ais_cfg.get('time_window_hours'))
    raw_vessel_count = int(pd.read_csv(ais_csv_path, usecols=['mmsi']).shape[0]) if os.path.exists(ais_csv_path) else 0
    if pruned.empty:
        logger.warning('Screened %d AIS rows; zero candidates inside the spatiotemporal envelope.', raw_vessel_count)
        return pd.DataFrame()
    attribution_cfg = dict(cfg.get('attribution') or {})
    attribution_cfg.setdefault('sigma_r_meters', radius_m)
    known_fields = {field_info.name for field_info in fields(AttributionConfig)}
    scorer_cfg = AttributionConfig(**{key: value for key, value in attribution_cfg.items() if key in known_fields})
    scorer = MultiFactorAttributionScorer(config=scorer_cfg)
    board = scorer.rank_suspects(pruned_ais_df=pruned, origin_lon=lon0, origin_lat=lat0, t0_timestamp_iso=t0.isoformat(), slick_drift_bearing=(metocean_profile.get('origin_estimate') or {}).get('drift_bearing_deg'))
    rankings_csv = os.path.join(output_dir, 'ranked_suspects.csv')
    board.to_csv(rankings_csv, index=False)
    top = board.iloc[0]
    properties = {}
    for key, value in top.items():
        if key in ('lon', 'lat'):
            continue
        if hasattr(value, 'item'):
            value = value.item()
        elif hasattr(value, 'isoformat'):
            value = value.isoformat()
        properties[key] = value
    properties.setdefault('min_distance_m', properties.get('distance_to_origin_m'))
    properties.setdefault('last_seen', properties.get('cpa_timestamp', properties.get('timestamp')))
    culprit_feature = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': properties, 'geometry': {'type': 'Point', 'coordinates': [float(top['lon']), float(top['lat'])]}}]}
    culprit_path = os.path.join(output_dir, 'culprit_vessel.geojson')
    with open(culprit_path, 'w', encoding='utf-8') as handle:
        json.dump(culprit_feature, handle, indent=2, default=str)
    logger.info('[✓] Screened %s AIS rows -> %d suspect vessels. Ranked leaderboard: %s', f'{raw_vessel_count:,}', len(board), rankings_csv)
    logger.info('🎯 Top suspect: MMSI %s (%s) score %.2f/100 - CPA %.1f m, dt %.1f min', top['mmsi'], properties.get('vessel_name', 'UNKNOWN'), top['attribution_score'], top['distance_to_origin_m'], top['temporal_offset_min'])
    logger.info('Culprit dossier exported: %s', culprit_path)
    return board

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Dynamic AIS cross-matching (legacy-compatible CLI)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to pipeline configuration YAML')
    parser.add_argument('--ais-csv', type=str, default=None, help='AIS source path (default: config paths.ais_csv)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: config paths.output_dir)')
    parser.add_argument('--spill', type=str, default=None, help='Spill polygon GeoJSON (default: config paths.spill_geojson)')
    return parser
if __name__ == '__main__':
    args = build_cli().parse_args()
    run_cross_matching(config_path=args.config, ais_csv_path=args.ais_csv, output_dir=args.output_dir, spill_geojson=args.spill)