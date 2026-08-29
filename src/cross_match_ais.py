import os
import yaml
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.affinity import scale, translate

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
CONFIG_PATH = "config.yaml"
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

paths_cfg = config.get("paths") or {}
ais_cfg = config.get("ais") or {}

OUTPUT_DIR = paths_cfg.get("output_dir", "results")
AIS_DATA_PATH = paths_cfg.get("ais_csv", "data/raw/ais_data.csv")
SPILL_GEOJSON_PATH = os.path.join(OUTPUT_DIR, "spill_polygon.geojson")
CULPRIT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "culprit_vessel.geojson")

os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_WINDOW_HOURS = ais_cfg.get("time_window_hours", 6)


# ==========================================
# 2. DYNAMIC AIS DATA GENERATOR
# ==========================================
def generate_dynamic_ais_traffic(centroid, sar_time, seed_val=42):
    """Generates realistic multi-vessel AIS traffic around the SAR scene footprint."""
    np.random.seed(seed_val)
    vessels = [
        {"name": "PACIFIC_VOYAGER", "type": "Oil Tanker", "mmsi": 211894000, "base_spd": 13.2},
        {"name": "EVER_GLORY", "type": "Container Ship", "mmsi": 477123900, "base_spd": 18.5},
        {"name": "OCEAN_PROVIDENCE", "type": "Chemical Tanker", "mmsi": 355678120, "base_spd": 11.8},
        {"name": "SEABOURN_TRADER", "type": "Bulk Carrier", "mmsi": 636019870, "base_spd": 10.4},
        {"name": "ALBATROSS_VII", "type": "Fishing Vessel", "mmsi": 412338900, "base_spd": 6.5}
    ]

    records = []
    # Assign one vessel to pass close/through the spill, and others further away
    target_idx = np.random.randint(0, len(vessels))

    for idx, v in enumerate(vessels):
        num_pings = 6
        time_offsets = [timedelta(hours=h) for h in np.linspace(-4, 1, num_pings)]

        if idx == target_idx:
            # Trajectory passes through or near the spill centroid
            lat_start = centroid.y - 0.04 + np.random.uniform(-0.005, 0.005)
            lon_start = centroid.x - 0.04 + np.random.uniform(-0.005, 0.005)
            lat_step = 0.016
            lon_step = 0.016
        else:
            # Trajectory passes several kilometers away
            offset_dist = np.random.uniform(0.08, 0.25) * np.random.choice([-1, 1])
            lat_start = centroid.y + offset_dist
            lon_start = centroid.x + offset_dist
            lat_step = np.random.uniform(-0.01, 0.01)
            lon_step = np.random.uniform(0.005, 0.02)

        for step, t_offset in enumerate(time_offsets):
            lat = lat_start + (step * lat_step) + np.random.normal(0, 0.0005)
            lon = lon_start + (step * lon_step) + np.random.normal(0, 0.0005)
            spd = v["base_spd"] + np.random.uniform(-0.4, 0.4)
            heading = (np.degrees(np.arctan2(lon_step, lat_step)) + 360) % 360

            records.append({
                "mmsi": v["mmsi"],
                "vessel_name": v["name"],
                "vessel_type": v["type"],
                "timestamp": sar_time + t_offset,
                "latitude": lat,
                "longitude": lon,
                "speed_knots": round(spd, 2),
                "heading": round(heading, 1)
            })

    return pd.DataFrame(records)


# ==========================================
# 3. CROSS-MATCHING ENGINE
# ==========================================
def run_cross_matching():
    if not os.path.exists(SPILL_GEOJSON_PATH):
        raise FileNotFoundError(f"Spill GeoJSON not found at: {SPILL_GEOJSON_PATH}")

    spill_gdf = gpd.read_file(SPILL_GEOJSON_PATH)
    if spill_gdf.empty:
        print("[!] No oil spill polygons found in GeoJSON. Exiting.")
        return

    # Normalize pixel coordinates into geographic bounding box
    bounds = spill_gdf.total_bounds
    ref_lat, ref_lon = 13.0827, 80.2707
    if bounds[2] > 180 or bounds[3] > 90 or (bounds[2] <= 512 and bounds[3] <= 512):
        norm_factor = 0.05 / 256.0
        spill_gdf["geometry"] = spill_gdf["geometry"].apply(
            lambda geom: translate(scale(geom, xfact=norm_factor, yfact=norm_factor, origin=(0, 0)), xoff=ref_lon, yoff=ref_lat)
        )
    spill_gdf.set_crs(epsg=4326, allow_override=True, inplace=True)
    spill_gdf.to_file(SPILL_GEOJSON_PATH, driver="GeoJSON")

    sar_time = datetime.now(timezone.utc)
    spill_union = spill_gdf.union_all() if hasattr(spill_gdf, "union_all") else spill_gdf.unary_union
    spill_centroid = spill_union.centroid

    # Load actual CSV or generate scene-derived dynamic AIS traffic
    if os.path.exists(AIS_DATA_PATH):
        ais_df = pd.read_csv(AIS_DATA_PATH)
        ais_df["timestamp"] = pd.to_datetime(ais_df["timestamp"])
    else:
        # Use spill polygon area & coordinates to seed variation
        seed = int(abs(spill_centroid.x * 1000 + spill_centroid.y * 1000 + len(spill_gdf))) % 10000
        print(f"[*] Simulating multi-vessel maritime traffic (Seed: {seed})...")
        ais_df = generate_dynamic_ais_traffic(spill_centroid, sar_time, seed_val=seed)

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(ais_df["longitude"], ais_df["latitude"])]
    ais_gdf = gpd.GeoDataFrame(ais_df, geometry=geometry, crs="EPSG:4326")

    # Reproject to metric CRS (World Mercator) for exact Euclidean distance
    spill_metric = spill_gdf.to_crs(epsg=3857)
    ais_metric = ais_gdf.to_crs(epsg=3857)
    spill_geom_metric = spill_metric.union_all() if hasattr(spill_metric, "union_all") else spill_metric.unary_union

    suspects = []
    for mmsi, group in ais_metric.groupby("mmsi"):
        if len(group) < 2:
            trajectory = group.geometry.iloc[0]
        else:
            sorted_pts = group.sort_values("timestamp")
            trajectory = LineString(sorted_pts.geometry.tolist())

        dist_meters = trajectory.distance(spill_geom_metric.centroid)
        suspects.append({
            "mmsi": int(mmsi),
            "vessel_name": group["vessel_name"].iloc[0],
            "vessel_type": group["vessel_type"].iloc[0],
            "min_distance_m": round(float(dist_meters), 2),
            "avg_speed_knots": round(float(group["speed_knots"].mean()), 2),
            "last_seen": group["timestamp"].max().isoformat(),
            "geometry": trajectory
        })

    suspects_gdf = gpd.GeoDataFrame(suspects, crs="EPSG:3857").to_crs(epsg=4326)
    suspects_gdf = suspects_gdf.sort_values("min_distance_m").reset_index(drop=True)
    culprit = suspects_gdf.iloc[0:1]

    culprit.to_file(CULPRIT_OUTPUT_PATH, driver="GeoJSON")

    print("\n" + "=" * 50)
    print(" AIS CROSS-MATCHING RESULT")
    print("=" * 50)
    print(f"Top Suspect Identified : {culprit['vessel_name'].values[0]} (MMSI: {culprit['mmsi'].values[0]})")
    print(f"Vessel Type            : {culprit['vessel_type'].values[0]}")
    print(f"Minimum Distance       : {culprit['min_distance_m'].values[0]} meters")
    print(f"Average Speed          : {culprit['avg_speed_knots'].values[0]} knots")
    print(f"GeoJSON Saved To       : {CULPRIT_OUTPUT_PATH}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_cross_matching()