import os
import json
import math
import random
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from datetime import datetime, timezone, timedelta
from typing import Optional
from shapely.geometry import Point, Polygon
from shapely.affinity import scale, translate

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.region_registry import load_regions


# ==========================================
# 1. GEODETIC UTILITIES & CONFIG
# ==========================================
KNOTS_TO_MPS = 0.514444


def _load_config(path: str = "config.yaml") -> dict:
    """Loads config.yaml; absence/malformation degrades to built-in defaults."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def geodesic_step(lat: float, lon: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    """Calculates destination latitude and longitude given distance in meters and bearing."""
    R = 6371000.0  # Earth radius in meters
    delta = distance_m / R
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda0 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) +
        math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda0 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )
    return math.degrees(phi2), math.degrees(lambda2)


def apply_sensor_jitter(lat: float, lon: float, sog: float, sigma_m: float = 8.0) -> tuple[float, float, float]:
    """Simulates realistic GNSS position jitter and speed sensor noise."""
    jitter_dist = random.gauss(0, sigma_m)
    jitter_bearing = random.uniform(0, 360)
    j_lat, j_lon = geodesic_step(lat, lon, jitter_dist, jitter_bearing)
    j_sog = max(0.0, sog + random.gauss(0, 0.25))
    return round(j_lat, 6), round(j_lon, 6), round(j_sog, 2)


# ==========================================
# 2. SIMULATION GENERATOR
# ==========================================
def generate_mock_environment(
    slick_geojson_path: str,
    output_dir: str = "data/processed",
    hindcast_hours: Optional[float] = None,
    corridor_idx: int = 0,
    random_seed: Optional[int] = None,
    num_background_vessels: Optional[int] = None,
    observation_time: Optional[datetime] = None,
    config_path: str = "config.yaml",
    pipeline_input_path: Optional[str] = None,
) -> dict:
    """
    Constructs a synchronized MetOcean field and multi-vessel AIS traffic
    simulation for any region and temporal window.

    Every formerly hard-coded generator constant (hindcast duration, seed,
    fleet size, sensor jitter, sampling cadence, scenario-vessel kinematics,
    metocean fallback field, pixel-space anchor region) is resolved from
    ``config.yaml -> synthetic`` / ``metocean`` / ``regions`` with the
    historical values as documented defaults; explicit arguments always win.
    """
    cfg = _load_config(config_path)
    synth_cfg = cfg.get("synthetic") or {}
    metocean_cfg = cfg.get("metocean") or {}

    hindcast_hours = float(
        hindcast_hours if hindcast_hours is not None else synth_cfg.get("hindcast_hours", 4.0)
    )
    random_seed = int(random_seed if random_seed is not None else synth_cfg.get("random_seed", 42))
    num_background_vessels = int(
        num_background_vessels
        if num_background_vessels is not None
        else synth_cfg.get("background_vessels", 26)
    )
    jitter_sigma_m = float(synth_cfg.get("sensor_jitter_m", 10.0))
    sample_before_min = int(synth_cfg.get("sample_before_minutes", 60))
    sample_after_min = int(synth_cfg.get("sample_after_minutes", 60))
    sample_step_min = int(synth_cfg.get("sample_step_minutes", 5))
    scenario_cfg = dict(synth_cfg.get("scenario") or {})
    if pipeline_input_path is None:
        pipeline_input_path = synth_cfg.get("pipeline_ais_output", "data/raw/ais_data.csv")

    random.seed(random_seed)
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(slick_geojson_path):
        raise FileNotFoundError(f"Slick GeoJSON not found at: {slick_geojson_path}")

    slick_gdf = gpd.read_file(slick_geojson_path)
    if slick_gdf.empty:
        raise ValueError("Provided slick GeoJSON is empty.")

    # 1. Resolve Geographic Bounds (Pixel Space vs. WGS84)
    bounds = slick_gdf.total_bounds
    has_no_crs = slick_gdf.crs is None
    exceeds_wgs84 = bounds[0] < -180.0 or bounds[2] > 180.0 or bounds[1] < -90.0 or bounds[3] > 90.0
    is_pixel_dimensions = (0.0 <= bounds[0] and bounds[2] <= 1024.0) and (0.0 <= bounds[1] and bounds[3] <= 1024.0) and has_no_crs

    is_pixel_space = has_no_crs or exceeds_wgs84 or is_pixel_dimensions
    regions = load_regions(config_path)
    selected_corridor = regions[corridor_idx % len(regions)]

    if is_pixel_space:
        print(f"[*] Detected pixel coordinates. Anchoring slick to corridor: {selected_corridor['name']}")
        ref_lat = selected_corridor["lat"]
        ref_lon = selected_corridor["lon"]
        scale_deg = 0.04 / 256.0  # ~4.4 km extent for 256px crop

        slick_gdf["geometry"] = slick_gdf["geometry"].apply(
            lambda geom: translate(scale(geom, xfact=scale_deg, yfact=scale_deg, origin=(0, 0)), xoff=ref_lon, yoff=ref_lat)
        )
        slick_gdf.set_crs(epsg=4326, allow_override=True, inplace=True)
        slick_gdf.to_file(slick_geojson_path, driver="GeoJSON")

    centroid = (slick_gdf.union_all() if hasattr(slick_gdf, "union_all") else slick_gdf.unary_union).centroid
    lat_sat, lon_sat = centroid.y, centroid.x
    t_sat = observation_time or datetime.now(timezone.utc)

    # 2. Dynamic MetOcean Environment Field (config-driven fallback physics)
    metocean_profile = {
        "timestamp_sat": t_sat.isoformat(),
        "u_current_mps": float(metocean_cfg.get("fallback_current_u_mps", 0.22)),
        "v_current_mps": float(metocean_cfg.get("fallback_current_v_mps", 0.14)),
        "u_wind_mps": float(metocean_cfg.get("fallback_wind_u_mps", 5.20)),
        "v_wind_mps": float(metocean_cfg.get("fallback_wind_v_mps", 3.10)),
        "wind_drift_factor": float(metocean_cfg.get("wind_drift_factor", 0.032)),
        "hindcast_hours": hindcast_hours
    }

    # Net advection velocity: V_drift = U_curr + (alpha * W_wind)
    v_drift_x = metocean_profile["u_current_mps"] + (metocean_profile["wind_drift_factor"] * metocean_profile["u_wind_mps"])
    v_drift_y = metocean_profile["v_current_mps"] + (metocean_profile["wind_drift_factor"] * metocean_profile["v_wind_mps"])

    total_drift_speed = math.hypot(v_drift_x, v_drift_y)
    drift_bearing = (math.degrees(math.atan2(v_drift_x, v_drift_y)) + 360) % 360

    # 3. Compute Hindcasted Release Origin (x0, y0, t0)
    duration_sec = hindcast_hours * 3600.0
    total_drift_dist_m = total_drift_speed * duration_sec
    reverse_bearing = (drift_bearing + 180.0) % 360.0

    y0, x0 = geodesic_step(lat_sat, lon_sat, total_drift_dist_m, reverse_bearing)
    t0 = t_sat - timedelta(seconds=duration_sec)

    metocean_profile["origin_estimate"] = {
        "origin_latitude": round(y0, 6),
        "origin_longitude": round(x0, 6),
        "origin_timestamp": t0.isoformat(),
        "net_drift_meters": round(total_drift_dist_m, 2),
        "drift_bearing_deg": round(drift_bearing, 2)
    }

    # 4. Synthesize Comprehensive AIS Traffic Matrix
    ais_records = []
    base_bearing = selected_corridor["corridor_bearing"]
    sample_intervals_min = list(range(-sample_before_min, sample_after_min + 1, sample_step_min))

    # --- VESSEL 1: PRIME CULPRIT (Crude Tanker) ---
    # Intersects (x0, y0) at t0; decelerates and executes dynamic course yaw during dumping
    culprit = dict(scenario_cfg.get("culprit") or {})
    mmsi_culprit = int(culprit.get("mmsi", 477901234))
    culprit_anomaly_halfwindow_min = float(culprit.get("anomaly_halfwindow_minutes", 15))
    culprit_sog_discharge = float(culprit.get("sog_discharge_knots", 8.5))
    culprit_sog_cruise = float(culprit.get("sog_cruise_knots", 13.8))
    culprit_zigzag_amplitude_deg = float(culprit.get("zigzag_amplitude_deg", 14.0))
    culprit_zigzag_period_min = float(culprit.get("zigzag_period_minutes", 5.0))
    culprit_rot_amplitude = float(culprit.get("rot_amplitude", 12.0))
    for dt_min in sample_intervals_min:
        t_current = t0 + timedelta(minutes=dt_min)

        # Operational anomaly: speed drop + zigzag yaw inside the anomaly window
        if abs(dt_min) <= culprit_anomaly_halfwindow_min:
            nominal_sog = culprit_sog_discharge
            heading_offset = math.sin(dt_min / culprit_zigzag_period_min) * culprit_zigzag_amplitude_deg
            rot_val = round(math.cos(dt_min / culprit_zigzag_period_min) * culprit_rot_amplitude, 1)
        else:
            nominal_sog = culprit_sog_cruise
            heading_offset = 0.0
            rot_val = 0.0

        current_heading = (base_bearing + heading_offset + 360) % 360
        dist_from_origin = (dt_min * 60.0) * (nominal_sog * KNOTS_TO_MPS)
        ship_lat, ship_lon = geodesic_step(y0, x0, dist_from_origin, current_heading)
        j_lat, j_lon, j_sog = apply_sensor_jitter(ship_lat, ship_lon, nominal_sog)

        ais_records.append({
            "mmsi": mmsi_culprit,
            "vessel_name": "VALIANT_CARRIER",
            "vessel_type": "Tanker",
            "callsign": "V3LK9",
            "nav_status": "Under way using engine",
            "timestamp": t_current.isoformat(),
            "latitude": j_lat,
            "longitude": j_lon,
            "speed_knots": j_sog,
            "heading": round(current_heading, 1),
            "rot": rot_val,
            "draught": 14.8,
            "destination": "SINGAPORE"
        })

    # --- VESSEL 2: TEMPORAL DISTRACTOR (Cargo) ---
    # Sailed through the exact same corridor, but 3.5 hours earlier
    mmsi_early = 211874550
    t_offset_early = timedelta(hours=3.5)
    for dt_min in sample_intervals_min:
        t_current = t0 + timedelta(minutes=dt_min) - t_offset_early
        nominal_sog = 14.5
        dist_from_origin = (dt_min * 60.0) * (nominal_sog * KNOTS_TO_MPS)
        ship_lat, ship_lon = geodesic_step(y0, x0, dist_from_origin, base_bearing)
        j_lat, j_lon, j_sog = apply_sensor_jitter(ship_lat, ship_lon, nominal_sog)

        ais_records.append({
            "mmsi": mmsi_early,
            "vessel_name": "NORDIC_TRADE",
            "vessel_type": "Cargo",
            "callsign": "LAFX2",
            "nav_status": "Under way using engine",
            "timestamp": t_current.isoformat(),
            "latitude": j_lat,
            "longitude": j_lon,
            "speed_knots": j_sog,
            "heading": round(base_bearing, 1),
            "rot": 0.0,
            "draught": 9.2,
            "destination": "ROTTERDAM"
        })

    # --- VESSEL 3: SPATIAL DISTRACTOR (Container) ---
    # Present at exact time t0, but in a parallel lane offset from the origin
    parallel = dict(scenario_cfg.get("parallel_distractor") or {})
    mmsi_parallel = int(parallel.get("mmsi", 355672109))
    parallel_sog = float(parallel.get("sog_knots", 18.2))
    parallel_offset_m = float(parallel.get("lateral_offset_m", 12000.0))
    parallel_offset_bearing = (base_bearing + float(parallel.get("lane_offset_bearing_deg", 90.0))) % 360.0
    lane_lat, lane_lon = geodesic_step(y0, x0, parallel_offset_m, parallel_offset_bearing)
    for dt_min in sample_intervals_min:
        t_current = t0 + timedelta(minutes=dt_min)
        nominal_sog = parallel_sog
        dist_from_origin = (dt_min * 60.0) * (nominal_sog * KNOTS_TO_MPS)
        ship_lat, ship_lon = geodesic_step(lane_lat, lane_lon, dist_from_origin, base_bearing)
        j_lat, j_lon, j_sog = apply_sensor_jitter(ship_lat, ship_lon, nominal_sog)

        ais_records.append({
            "mmsi": mmsi_parallel,
            "vessel_name": "EVER_HORIZON",
            "vessel_type": "Container",
            "callsign": "3FTC4",
            "nav_status": "Under way using engine",
            "timestamp": t_current.isoformat(),
            "latitude": j_lat,
            "longitude": j_lon,
            "speed_knots": j_sog,
            "heading": round(base_bearing, 1),
            "rot": 0.0,
            "draught": 12.5,
            "destination": "JEBEL ALI"
        })

    # --- VESSEL 4: DARK VESSEL SUSPECT (Bulk Carrier) ---
    # Cuts off its AIS transmitter before the origin window, reappears afterwards
    dark = dict(scenario_cfg.get("dark_vessel") or {})
    mmsi_dark = int(dark.get("mmsi", 636019888))
    dark_sog = float(dark.get("sog_knots", 12.0))
    dark_gap_start_min = float(dark.get("gap_start_minutes", -20))
    dark_gap_end_min = float(dark.get("gap_end_minutes", 25))
    for dt_min in sample_intervals_min:
        if dark_gap_start_min <= dt_min <= dark_gap_end_min:
            continue  # Blackout gap
        t_current = t0 + timedelta(minutes=dt_min)
        nominal_sog = dark_sog
        dist_from_origin = (dt_min * 60.0) * (nominal_sog * KNOTS_TO_MPS)
        ship_lat, ship_lon = geodesic_step(y0, x0, dist_from_origin, base_bearing)
        j_lat, j_lon, j_sog = apply_sensor_jitter(ship_lat, ship_lon, nominal_sog)

        ais_records.append({
            "mmsi": mmsi_dark,
            "vessel_name": "OCEAN_PHANTOM",
            "vessel_type": "Bulk Carrier",
            "callsign": "A8XQ7",
            "nav_status": "Under way using engine",
            "timestamp": t_current.isoformat(),
            "latitude": j_lat,
            "longitude": j_lon,
            "speed_knots": j_sog,
            "heading": round(base_bearing, 1),
            "rot": 0.0,
            "draught": 10.1,
            "destination": "FUJAIRAH"
        })

    # --- BACKGROUND CLUTTER GENERATOR ---
    # Generates anchored ships, slow trawlers, tugs, and crossing cargo vessels to test noise filters.
    # The default fleet profile below can be replaced wholesale via config (synthetic.background_fleet).
    clutter_types = synth_cfg.get("background_fleet") or [
        {"type": "Fishing", "nav": "Engaged in fishing", "sog": (1.5, 4.0), "draught": 3.8, "dist": (6000, 22000)},
        {"type": "Tug", "nav": "Restricted maneuverability", "sog": (3.0, 7.5), "draught": 4.5, "dist": (4000, 18000)},
        {"type": "Cargo", "nav": "At anchor", "sog": (0.0, 0.4), "draught": 8.5, "dist": (10000, 25000)},
        {"type": "Tanker", "nav": "At anchor", "sog": (0.0, 0.3), "draught": 13.0, "dist": (12000, 28000)},
        {"type": "Container", "nav": "Under way using engine", "sog": (14.0, 21.0), "draught": 11.2, "dist": (15000, 35000)},
        {"type": "Pleasure Craft", "nav": "Under way using engine", "sog": (6.0, 11.0), "draught": 2.2, "dist": (5000, 16000)}
    ]

    for idx in range(num_background_vessels):
        cfg = clutter_types[idx % len(clutter_types)]
        mmsi_bg = 200000000 + idx * 11739 + random.randint(100, 999)
        name_bg = f"MARITIME_BG_{idx+1:02d}_{cfg['type'].upper()[:4]}"
        callsign_bg = f"BG{idx+1:02d}{random.choice('ABCDEF')}"
        dest_bg = random.choice(["OFFSHORE ANCHORAGE", "PORT SULTAN", "LOCAL FISHING", "COASTAL PATROL", "AL HAMRIYA"])

        # Spatial distribution across 5 to 35 km away from origin
        radial_dist = random.uniform(cfg["dist"][0], cfg["dist"][1])
        radial_bearing = random.uniform(0, 360)
        c_lat, c_lon = geodesic_step(y0, x0, radial_dist, radial_bearing)

        course = random.choice([base_bearing, (base_bearing + 180) % 360, (base_bearing + 90) % 360])
        nominal_sog = random.uniform(cfg["sog"][0], cfg["sog"][1])
        is_stationary = cfg["nav"] == "At anchor"

        for dt_min in sample_intervals_min:
            t_current = t0 + timedelta(minutes=dt_min)
            if is_stationary:
                ship_lat, ship_lon = c_lat, c_lon
                rot_val = 0.0
            else:
                dist = (dt_min * 60.0) * (nominal_sog * KNOTS_TO_MPS)
                ship_lat, ship_lon = geodesic_step(c_lat, c_lon, dist, course)
                rot_val = round(random.gauss(0, 1.5), 1)

            j_lat, j_lon, j_sog = apply_sensor_jitter(
                ship_lat, ship_lon, 0.1 if is_stationary else nominal_sog,
                sigma_m=jitter_sigma_m * 0.4 if is_stationary else jitter_sigma_m
            )

            ais_records.append({
                "mmsi": mmsi_bg,
                "vessel_name": name_bg,
                "vessel_type": cfg["type"],
                "callsign": callsign_bg,
                "nav_status": cfg["nav"],
                "timestamp": t_current.isoformat(),
                "latitude": j_lat,
                "longitude": j_lon,
                "speed_knots": j_sog,
                "heading": round(course, 1),
                "rot": rot_val,
                "draught": cfg["draught"],
                "destination": dest_bg
            })

    # 5. Export Datasets
    ais_df = pd.DataFrame(ais_records).sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    ais_csv_out = os.path.join(output_dir, "simulated_ais_traffic.csv")
    metocean_json_out = os.path.join(output_dir, "metocean_environment.json")

    ais_df.to_csv(ais_csv_out, index=False)
    with open(metocean_json_out, "w") as f:
        json.dump(metocean_profile, f, indent=4)

    if pipeline_input_path:
        os.makedirs(os.path.dirname(pipeline_input_path) or ".", exist_ok=True)
        ais_df.to_csv(pipeline_input_path, index=False)

    print("\n" + "=" * 60)
    print(" COMPREHENSIVE SYNTHETIC MARITIME MATRIX COMPILED")
    print("=" * 60)
    print(f"Observation Point (t_sat) : Lat {lat_sat:.5f}, Lon {lon_sat:.5f}")
    print(f"Calculated Origin (t_0)   : Lat {y0:.5f}, Lon {x0:.5f}")
    print(f"Total Vessels Simulated   : {ais_df['mmsi'].nunique()} ships ({len(ais_df)} AIS pings)")
    print(f"Primary Culprit MMSI      : {mmsi_culprit} (VALIANT_CARRIER)")
    print(f"Simulated AIS Archive     : {ais_csv_out}")
    print(f"Metocean Matrix Export    : {metocean_json_out}")
    print(f"Synced Pipeline Input     : {pipeline_input_path or '(disabled)'}")
    print("=" * 60 + "\n")

    return metocean_profile


def _build_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic synthetic AIS/metocan scenario generator")
    parser.add_argument("--slick", type=str, default="results/spill_polygon.geojson", help="Spill polygon GeoJSON (pixel-space or WGS84)")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--corridor", type=int, default=0, help="Region registry index for pixel-space anchoring")
    parser.add_argument("--hindcast-hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--background-vessels", type=int, default=None)
    parser.add_argument("--no-pipeline-sync", action="store_true", help="Do not mirror the AIS feed into the pipeline input path")
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    if os.path.exists(args.slick):
        generate_mock_environment(
            args.slick,
            output_dir=args.output_dir,
            corridor_idx=args.corridor,
            hindcast_hours=args.hindcast_hours,
            random_seed=args.seed,
            num_background_vessels=args.background_vessels,
            config_path=args.config,
            pipeline_input_path="" if args.no_pipeline_sync else None,
        )
    else:
        print(f"[!] Target test polygon not found. Place a valid GeoJSON at: {args.slick}")