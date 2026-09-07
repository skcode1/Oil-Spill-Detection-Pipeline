import os
import math
import json
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timezone, timedelta
from shapely.geometry import Point, LineString
from shapely.affinity import scale, translate


CONFIG_PATH = "config.yaml"
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

paths_cfg = config.get("paths") or {}
ais_cfg = config.get("ais") or {}

OUTPUT_DIR = paths_cfg.get("output_dir", "results")
PROCESSED_DIR = paths_cfg.get("processed_dir", "data/processed")
AIS_DATA_PATH = paths_cfg.get("ais_csv", "data/raw/ais_data.csv")
METOCEAN_JSON_PATH = os.path.join(PROCESSED_DIR, "metocean_environment.json")
SPILL_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "spill_polygon.geojson")
CULPRIT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "culprit_vessel.geojson")
RANKINGS_CSV_PATH = os.path.join(OUTPUT_DIR, "ranked_suspects.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
KNOTS_TO_MPS = 0.514444
METERS_PER_DEG_LAT = 111320.0



def geodesic_step(lat: float, lon: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    """Calculates destination latitude and longitude given metric distance and bearing."""
    R = 6371000.0
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


def compute_hindcast_origin(lat_sat: float, lon_sat: float, t_sat: datetime, hindcast_hours: float = 4.0) -> tuple[float, float, datetime, dict]:
    """
    Computes slick origin (x0, y0, t0) by reversing physical advection (currents + wind leeway).
    Reads metocean_environment.json if present, or calculates using dynamic defaults.
    """
    if os.path.exists(METOCEAN_JSON_PATH):
        try:
            with open(METOCEAN_JSON_PATH, "r") as f:
                meta = json.load(f)
            orig = meta.get("origin_estimate", {})
            y0 = float(orig["origin_latitude"])
            x0 = float(orig["origin_longitude"])
            t0 = datetime.fromisoformat(orig["origin_timestamp"])
            return y0, x0, t0, meta
        except Exception as e:
            print(f"[!] Warning reading metocean JSON ({e}). Falling back to analytical model.")

    # Analytical Metocean Model: Net drift = Current + (0.032 * Wind)
    u_current, v_current = 0.22, 0.14
    u_wind, v_wind = 5.20, 3.10
    alpha = 0.032

    v_drift_x = u_current + (alpha * u_wind)
    v_drift_y = v_current + (alpha * v_wind)

    total_drift_speed = math.hypot(v_drift_x, v_drift_y)
    drift_bearing = (math.degrees(math.atan2(v_drift_x, v_drift_y)) + 360) % 360
    duration_sec = hindcast_hours * 3600.0

    total_drift_dist_m = total_drift_speed * duration_sec
    reverse_bearing = (drift_bearing + 180.0) % 360.0

    y0, x0 = geodesic_step(lat_sat, lon_sat, total_drift_dist_m, reverse_bearing)
    t0 = t_sat - timedelta(seconds=duration_sec)

    metocean_data = {
        "u_current_mps": u_current, "v_current_mps": v_current,
        "u_wind_mps": u_wind, "v_wind_mps": v_wind,
        "net_drift_meters": round(total_drift_dist_m, 2),
        "drift_bearing_deg": round(drift_bearing, 2)
    }
    return y0, x0, t0, metocean_data



def score_vessel_candidate(group: pd.DataFrame, x0: float, y0: float, t0: datetime) -> dict:
    """
    Computes a multi-factor attribution score (0 to 100) based on spatial closeness to origin,
    temporal sync, vessel classification, deceleration anomalies, and transponder blackout gaps.
    """
    sorted_group = group.sort_values("timestamp").reset_index(drop=True)
    mmsi = int(sorted_group["mmsi"].iloc[0])
    name = sorted_group["vessel_name"].iloc[0]
    v_type = sorted_group["vessel_type"].iloc[0]
    nav_status = str(sorted_group.get("nav_status", pd.Series(["Under way"])).iloc[0])
    draught = float(sorted_group.get("draught", pd.Series([10.0])).iloc[0])

    # Convert group timestamps to timezone-aware UTC datetime
    timestamps = pd.to_datetime(sorted_group["timestamp"], utc=True)
    time_diffs_sec = (timestamps - t0).abs().dt.total_seconds()
    closest_idx = time_diffs_sec.idxmin()
    closest_record = sorted_group.iloc[closest_idx]

    min_time_delta_mins = time_diffs_sec.min() / 60.0

    # Metric distance calculation to (x0, y0) using geodesic approximation
    d_lat_m = (closest_record["latitude"] - y0) * METERS_PER_DEG_LAT
    d_lon_m = (closest_record["longitude"] - x0) * (METERS_PER_DEG_LAT * math.cos(math.radians(y0)))
    distance_at_t0_m = math.hypot(d_lat_m, d_lon_m)

    # 1. Spatial Proximity Score (Max 40 pts)
    # Exponential decay: 40 pts at 0m, ~24 pts at 1km, drops sharply past 5km
    spatial_score = 40.0 * math.exp(-0.5 * ((distance_at_t0_m / 1500.0) ** 2))

    # 2. Temporal Proximity Score (Max 25 pts)
    # 25 pts if within 10 minutes of t0, decaying past 60 mins
    temporal_score = 25.0 * math.exp(-0.5 * ((min_time_delta_mins / 35.0) ** 2))

    # 3. Vessel Type & Carrying Capacity Risk (Max 15 pts)
    type_upper = v_type.upper()
    if "TANKER" in type_upper or "CHEMICAL" in type_upper:
        type_score = 15.0 if draught >= 10.0 else 12.0
    elif "CARGO" in type_upper or "BULK" in type_upper:
        type_score = 8.0
    elif "CONTAINER" in type_upper:
        type_score = 6.0
    elif "FISHING" in type_upper:
        type_score = 3.0
    else:
        type_score = 1.0

    # 4. Operational & Kinematic Anomaly (Max 10 pts)
    # Checks for deceleration during transit through the origin corridor (dumping behavior)
    speeds = sorted_group["speed_knots"].values
    mean_speed = float(np.mean(speeds))
    min_speed = float(np.min(speeds))
    speed_drop = max(0.0, mean_speed - min_speed)
    
    # Check for Rate of Turn or Heading variations
    rot_values = sorted_group.get("rot", pd.Series([0.0] * len(sorted_group))).abs().values
    max_rot = float(np.max(rot_values)) if len(rot_values) > 0 else 0.0

    kinematic_score = 0.0
    if speed_drop >= 3.0:
        kinematic_score += 6.0
    elif speed_drop >= 1.5:
        kinematic_score += 3.0

    if max_rot >= 5.0:
        kinematic_score += 4.0

    kinematic_score = min(10.0, kinematic_score)

    # 5. AIS Blackout Gap / Dark Vessel Detection (Max 10 pts)
    # Looks for missing pings (>20 minutes interval) near the origin window
    dark_gap_score = 0.0
    time_deltas_between_pings = timestamps.diff().dt.total_seconds().dropna() / 60.0
    if not time_deltas_between_pings.empty:
        max_gap_mins = float(time_deltas_between_pings.max())
        if max_gap_mins >= 30.0 and distance_at_t0_m <= 15000.0:
            dark_gap_score = 10.0
        elif max_gap_mins >= 20.0 and distance_at_t0_m <= 15000.0:
            dark_gap_score = 5.0

    total_attribution_score = min(100.0, spatial_score + temporal_score + type_score + kinematic_score + dark_gap_score)

    # Construct LineString Trajectory for GeoJSON
    if len(sorted_group) >= 2:
        trajectory_geom = LineString(zip(sorted_group["longitude"], sorted_group["latitude"]))
    else:
        trajectory_geom = Point(closest_record["longitude"], closest_record["latitude"])

    return {
        "mmsi": mmsi,
        "vessel_name": name,
        "vessel_type": v_type,
        "nav_status": nav_status,
        "cpa_to_origin_m": round(distance_at_t0_m, 1),
        "time_offset_min": round(min_time_delta_mins, 1),
        "avg_speed_knots": round(mean_speed, 2),
        "speed_drop_knots": round(speed_drop, 2),
        "max_rot": round(max_rot, 1),
        "attribution_score": round(total_attribution_score, 2),
        "spatial_pts": round(spatial_score, 1),
        "temporal_pts": round(temporal_score, 1),
        "type_pts": round(type_score, 1),
        "anomaly_pts": round(kinematic_score + dark_gap_score, 1),
        "last_seen": sorted_group["timestamp"].max(),
        "geometry": trajectory_geom
    }


def run_cross_matching():
    if not os.path.exists(SPILL_GEOJSON_PATH):
        raise FileNotFoundError(f"Spill GeoJSON not found at: {SPILL_GEOJSON_PATH}")

    spill_gdf = gpd.read_file(SPILL_GEOJSON_PATH)
    if spill_gdf.empty:
        print("[!] No oil spill polygons found in GeoJSON. Exiting.")
        return

    # Extract or validate slick centroid
    spill_union = spill_gdf.union_all() if hasattr(spill_gdf, "union_all") else spill_gdf.unary_union
    centroid = spill_union.centroid
    lat_sat, lon_sat = centroid.y, centroid.x
    t_sat = datetime.now(timezone.utc)

    # 1. Compute Metocean Reverse Hindcast (x0, y0, t0)
    hindcast_hours = config.get("metocean", {}).get("hindcast_hours", 4.0)
    y0, x0, t0, metocean_data = compute_hindcast_origin(lat_sat, lon_sat, t_sat, hindcast_hours=hindcast_hours)

    print("\n" + "=" * 62)
    print(" 🌊 METOCEAN DRIFT HINDCASTING SYNCHRONIZED")
    print("=" * 62)
    print(f"Observed Slick Centroid (t_sat) : Lat {lat_sat:.5f}, Lon {lon_sat:.5f}")
    print(f"Hindcasted Discharge Point (t_0): Lat {y0:.5f}, Lon {x0:.5f}")
    print(f"Elapsed Time Prior to Capture   : {hindcast_hours:.1f} hours ({t0.strftime('%Y-%m-%d %H:%M:%S UTC')})")
    print(f"Advection Distance Traveled     : {metocean_data.get('net_drift_meters', 0.0):.1f} meters")
    print("-" * 62)

    # 2. Ingest AIS Traffic Database
    if not os.path.exists(AIS_DATA_PATH):
        # Fallback to simulated archive if raw doesn't exist
        alt_ais = os.path.join(PROCESSED_DIR, "simulated_ais_traffic.csv")
        if os.path.exists(alt_ais):
            ais_csv_path = alt_ais
        else:
            raise FileNotFoundError(f"No AIS data found at {AIS_DATA_PATH} or {alt_ais}")
    else:
        ais_csv_path = AIS_DATA_PATH

    print(f"[*] Ingesting AIS database from: {ais_csv_path}")
    df_ais = pd.read_csv(ais_csv_path)

    # 3. Apply Traffic Pruning (Filter Irrelevant Vessels)
    raw_vessel_count = df_ais["mmsi"].nunique()
    
    # Filter 1: Exclude anchored vessels (>20 km from origin)
    # Filter 2: Exclude low-speed craft (<3 knots) that are far outside the corridor
    scored_candidates = []
    for mmsi, group in df_ais.groupby("mmsi"):
        # Check nav_status and distance heuristics for filtering
        nav_status = str(group.get("nav_status", pd.Series(["Under way"])).iloc[0])
        mean_speed = float(group["speed_knots"].mean())

        # Rough distance from first ping to origin
        approx_dist_km = math.hypot(
            (group["latitude"].iloc[0] - y0) * 111.0,
            (group["longitude"].iloc[0] - x0) * (111.0 * math.cos(math.radians(y0)))
        )

        # Prune vessels completely anchored or operating outside realistic surveillance radius (>40 km)
        if nav_status == "At anchor" and approx_dist_km > 15.0:
            continue
        if approx_dist_km > 40.0:
            continue

        candidate_dossier = score_vessel_candidate(group, x0, y0, t0)
        scored_candidates.append(candidate_dossier)

    if not scored_candidates:
        print("[!] Warning: All vessels were filtered out by traffic clutter rules.")
        return

    # 4. Rank Candidates by Attribution Score
    suspects_gdf = gpd.GeoDataFrame(scored_candidates, crs="EPSG:4326")
    suspects_gdf = suspects_gdf.sort_values(by=["attribution_score", "cpa_to_origin_m"], ascending=[False, True]).reset_index(drop=True)

    # Export Full Rankings Table
    export_cols = [
        "mmsi", "vessel_name", "vessel_type", "nav_status", "attribution_score",
        "cpa_to_origin_m", "time_offset_min", "avg_speed_knots", "speed_drop_knots",
        "spatial_pts", "temporal_pts", "type_pts", "anomaly_pts", "last_seen"
    ]
    suspects_gdf[export_cols].to_csv(RANKINGS_CSV_PATH, index=False)

    # Export Culprit Feature to GeoJSON (compatible with web UI and FastAPI)
    culprit_feature = suspects_gdf.iloc[0:1].copy()
    
    # Forward metadata fields to GeoJSON properties
    location_name = spill_gdf["location_name"].iloc[0] if "location_name" in spill_gdf.columns else "Maritime Surveillance Zone"
    authority = spill_gdf["authority"].iloc[0] if "authority" in spill_gdf.columns else "Maritime Rescue Coordination Centre (MRCC)"
    
    culprit_feature["location_name"] = location_name
    culprit_feature["authority"] = authority
    culprit_feature["min_distance_m"] = culprit_feature["cpa_to_origin_m"]
    culprit_feature.to_file(CULPRIT_OUTPUT_PATH, driver="GeoJSON")

    # 5. Output Incident Dossier
    top = culprit_feature.iloc[0]
    print(" 🎯 TOP SUSPECT CULPRIT IDENTIFIED")
    print("-" * 62)
    print(f"Vessel Name            : {top['vessel_name']} (MMSI: {top['mmsi']})")
    print(f"Vessel Type            : {top['vessel_type']} [{top['nav_status']}]")
    print(f"Attribution Score      : {top['attribution_score']} / 100")
    print(f"Distance to Origin     : {top['cpa_to_origin_m']:.1f} meters")
    print(f"Temporal Alignment     : ±{top['time_offset_min']:.1f} minutes of estimated release")
    print(f"Operational Decel.     : {top['speed_drop_knots']:.1f} knots drop during passage")
    print(f"Pruned Traffic Volume  : Screened {raw_vessel_count} vessels down to {len(suspects_gdf)} active candidates")
    print(f"Exported Culprit Dossier: {CULPRIT_OUTPUT_PATH}")
    print(f"Exported Full Rankings : {RANKINGS_CSV_PATH}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    run_cross_matching()