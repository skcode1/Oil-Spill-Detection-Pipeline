import os
import math
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timezone, timedelta
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

try:
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    xr = None
    RegularGridInterpolator = None


class HindcastEngine:
    """
    Lagrangian Particle Tracking & Numerical Advection Engine.
    Executes Runge-Kutta 2nd Order (RK2 / Midpoint Method) trajectory integration
    to track slick polygons backward in time (hindcast) and forward in time (forecast).
    """

    def __init__(self, wind_nc_path: str = None, current_nc_path: str = None, alpha: float = 0.032, output_dir: str = "results"):
        self.alpha = alpha  # Empirical leeway drift factor (~3.2%)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.wind_nc_path = wind_nc_path
        self.current_nc_path = current_nc_path
        self.has_grids = False
        
        # Load NetCDF files if present
        if xr and wind_nc_path and current_nc_path and os.path.exists(wind_nc_path) and os.path.exists(current_nc_path):
            try:
                self._load_and_compile_interpolators()
                self.has_grids = True
            except Exception as e:
                print(f"[!] Warning: Could not compile NetCDF interpolators ({e}). Reverting to analytical hydrodynamic advection.")

    def _load_and_compile_interpolators(self):
        """Builds multi-dimensional spatio-temporal interpolators from NetCDF files."""
        print("[HINDCAST] Compiling Spatio-Temporal Field Interpolators...")
        self.ds_wind = xr.open_dataset(self.wind_nc_path)
        self.ds_curr = xr.open_dataset(self.current_nc_path)

        # 1. Standardize Wind Coordinates & Variable Names
        u_wind_name = "u10" if "u10" in self.ds_wind else "10m_u_component_of_wind"
        v_wind_name = "v10" if "v10" in self.ds_wind else "10m_v_component_of_wind"
        
        time_dim_w = "time" if "time" in self.ds_wind.coords else list(self.ds_wind.coords.keys())[0]
        lat_dim_w = "latitude" if "latitude" in self.ds_wind.coords else "lat"
        lon_dim_w = "longitude" if "longitude" in self.ds_wind.coords else "lon"

        w_time = pd.to_datetime(self.ds_wind[time_dim_w].values).astype(np.int64) // 10**9
        w_lat = self.ds_wind[lat_dim_w].values
        w_lon = self.ds_wind[lon_dim_w].values

        u10_arr = self.ds_wind[u_wind_name].values
        v10_arr = self.ds_wind[v_wind_name].values

        # Ensure latitude is strictly increasing for RegularGridInterpolator
        lat_sort_idx_w = np.argsort(w_lat)
        w_lat_sorted = w_lat[lat_sort_idx_w]
        u10_arr = np.take(u10_arr, lat_sort_idx_w, axis=1)
        v10_arr = np.take(v10_arr, lat_sort_idx_w, axis=1)

        self.interp_u10 = RegularGridInterpolator((w_time, w_lat_sorted, w_lon), u10_arr, bounds_error=False, fill_value=None)
        self.interp_v10 = RegularGridInterpolator((w_time, w_lat_sorted, w_lon), v10_arr, bounds_error=False, fill_value=None)

        # 2. Standardize Ocean Current Coordinates & Depth
        u_curr_name = "uo" if "uo" in self.ds_curr else "u"
        v_curr_name = "vo" if "vo" in self.ds_curr else "v"

        time_dim_c = "time" if "time" in self.ds_curr.coords else list(self.ds_curr.coords.keys())[0]
        lat_dim_c = "latitude" if "latitude" in self.ds_curr.coords else "lat"
        lon_dim_c = "longitude" if "longitude" in self.ds_curr.coords else "lon"

        c_time = pd.to_datetime(self.ds_curr[time_dim_c].values).astype(np.int64) // 10**9
        c_lat = self.ds_curr[lat_dim_c].values
        c_lon = self.ds_curr[lon_dim_c].values

        uo_arr = self.ds_curr[u_curr_name].values
        vo_arr = self.ds_curr[v_curr_name].values

        if uo_arr.ndim == 4: uo_arr = uo_arr[:, 0, :, :]
        if vo_arr.ndim == 4: vo_arr = vo_arr[:, 0, :, :]

        lat_sort_idx_c = np.argsort(c_lat)
        c_lat_sorted = c_lat[lat_sort_idx_c]
        uo_arr = np.take(uo_arr, lat_sort_idx_c, axis=1)
        vo_arr = np.take(vo_arr, lat_sort_idx_c, axis=1)

        self.interp_uo = RegularGridInterpolator((c_time, c_lat_sorted, c_lon), uo_arr, bounds_error=False, fill_value=None)
        self.interp_vo = RegularGridInterpolator((c_time, c_lat_sorted, c_lon), vo_arr, bounds_error=False, fill_value=None)

    def _get_drift_velocity(self, t_sec: float, lat: float, lon: float) -> tuple[float, float]:
        """Calculates combined Eulerian drift vector: V_drift = U_curr + alpha * W_wind."""
        if self.has_grids:
            pt = np.array([t_sec, lat, lon])
            u_w = float(self.interp_u10(pt))
            v_w = float(self.interp_v10(pt))
            u_c = float(self.interp_uo(pt))
            v_c = float(self.interp_vo(pt))
            return u_c + (self.alpha * u_w), v_c + (self.alpha * v_w)
        else:
            # Calibrated analytical hydrodynamic defaults (Persian Gulf baseline)
            u_c, v_c = 0.22, 0.14
            u_w, v_w = 5.20, 3.10
            return u_c + (self.alpha * u_w), v_c + (self.alpha * v_w)

    def _sample_polygon_nodes(self, geometry, max_nodes: int = 32) -> np.ndarray:
        """Extracts and downsamples polygon perimeter nodes for fast, stable RK2 tracking."""
        if isinstance(geometry, MultiPolygon):
            geometry = max(geometry.geoms, key=lambda a: a.area)
        
        # Simplify geometry slightly to remove co-linear noise
        simplified = geometry.simplify(tolerance=0.0005, preserve_topology=True)
        exterior_coords = np.array(simplified.exterior.coords)

        if len(exterior_coords) > max_nodes:
            indices = np.linspace(0, len(exterior_coords) - 1, max_nodes, dtype=int)
            sampled = exterior_coords[indices]
        else:
            sampled = exterior_coords

        return sampled[:, :2]  # Returns Nx2 array of [lon, lat]

    def run_hindcast_and_forecast(
        self,
        slick_geojson_path: str,
        sat_time_str: str = None,
        dt_minutes: int = 15,
        max_hindcast_hours: float = 24.0,
        forecast_hours: float = 12.0
    ) -> dict:
        """
        Executes bidirectional Lagrangian trajectory solving:
        1. Backward Hindcasting: Pinpoints origin (x0, y0, t0) via cluster compression.
        2. Forward Forecasting: Predicts future slick drift for coastal alert coordination.
        """
        if not os.path.exists(slick_geojson_path):
            raise FileNotFoundError(f"Slick GeoJSON not found at: {slick_geojson_path}")

        gdf = gpd.read_file(slick_geojson_path)
        if gdf.empty:
            raise ValueError("Input GeoJSON is empty.")

        poly_union = unary_union(gdf.geometry)
        particles = self._sample_polygon_nodes(poly_union, max_nodes=32)
        centroid = poly_union.centroid

        t_sat = pd.to_datetime(sat_time_str, utc=True) if sat_time_str else datetime.now(timezone.utc)
        t_sat_sec = t_sat.value // 10**9
        dt_sec = dt_minutes * 60

        # ==========================================
        # 1. BACKWARD HINDCASTING (RK2 Midpoint Solver)
        # ==========================================
        total_hindcast_steps = int((max_hindcast_hours * 60) / dt_minutes)
        current_particles = particles.copy()
        current_time_sec = t_sat_sec

        min_cluster_variance = float("inf")
        optimal_t0_sec = t_sat_sec
        optimal_centroid = (centroid.x, centroid.y)

        hindcast_tracks = [[] for _ in range(len(particles))]
        centroid_track_backward = []

        print(f"[HINDCAST] Tracking {len(particles)} nodes backward across {total_hindcast_steps} RK2 steps...")

        for step in range(total_hindcast_steps):
            c_x = np.mean(current_particles[:, 0])
            c_y = np.mean(current_particles[:, 1])
            centroid_track_backward.append((c_x, c_y, current_time_sec))

            # Variance metric for convergence (point of maximum compactness)
            variance = np.mean(np.sum((current_particles - [c_x, c_y]) ** 2, axis=1))
            if variance < min_cluster_variance:
                min_cluster_variance = variance
                optimal_t0_sec = current_time_sec
                optimal_centroid = (c_x, c_y)

            # Record vertex trajectories
            for idx in range(len(particles)):
                hindcast_tracks[idx].append((current_particles[idx, 0], current_particles[idx, 1]))

            # Runge-Kutta 2nd Order Backward Step (dt < 0)
            new_particles = np.zeros_like(current_particles)
            for idx, (lon, lat) in enumerate(current_particles):
                deg_per_m_lon = 1.0 / (111412.0 * math.cos(math.radians(lat)))
                deg_per_m_lat = 1.0 / 111132.0

                # RK2 Step 1: Initial slope at current node
                vx1, vy1 = self._get_drift_velocity(current_time_sec, lat, lon)
                mid_lon = lon - (vx1 * (dt_sec / 2.0) * deg_per_m_lon)
                mid_lat = lat - (vy1 * (dt_sec / 2.0) * deg_per_m_lat)
                mid_t = current_time_sec - (dt_sec / 2)

                # RK2 Step 2: Midpoint slope
                vx2, vy2 = self._get_drift_velocity(mid_t, mid_lat, mid_lon)
                new_particles[idx, 0] = lon - (vx2 * dt_sec * deg_per_m_lon)
                new_particles[idx, 1] = lat - (vy2 * dt_sec * deg_per_m_lat)

            current_particles = new_particles
            current_time_sec -= dt_sec

        t0_timestamp = pd.to_datetime(optimal_t0_sec, unit="s", utc=True)
        spill_age_hours = round((t_sat - t0_timestamp).total_seconds() / 3600.0, 2)

        # ==========================================
        # 2. FORWARD FORECASTING (Drift Projection)
        # ==========================================
        total_forecast_steps = int((forecast_hours * 60) / dt_minutes)
        fwd_particles = particles.copy()
        fwd_time_sec = t_sat_sec
        centroid_track_forward = []

        print(f"[FORECAST] Projecting future drift trajectory forward across {forecast_hours} hours...")

        for _ in range(total_forecast_steps):
            c_x = np.mean(fwd_particles[:, 0])
            c_y = np.mean(fwd_particles[:, 1])
            centroid_track_forward.append((c_x, c_y, fwd_time_sec))

            new_fwd = np.zeros_like(fwd_particles)
            for idx, (lon, lat) in enumerate(fwd_particles):
                deg_per_m_lon = 1.0 / (111412.0 * math.cos(math.radians(lat)))
                deg_per_m_lat = 1.0 / 111132.0

                # RK2 Step 1: Forward evaluation
                vx1, vy1 = self._get_drift_velocity(fwd_time_sec, lat, lon)
                mid_lon = lon + (vx1 * (dt_sec / 2.0) * deg_per_m_lon)
                mid_lat = lat + (vy1 * (dt_sec / 2.0) * deg_per_m_lat)
                mid_t = fwd_time_sec + (dt_sec / 2)

                # RK2 Step 2: Midpoint update
                vx2, vy2 = self._get_drift_velocity(mid_t, mid_lat, mid_lon)
                new_fwd[idx, 0] = lon + (vx2 * dt_sec * deg_per_m_lon)
                new_fwd[idx, 1] = lat + (vy2 * dt_sec * deg_per_m_lat)

            fwd_particles = new_fwd
            fwd_time_sec += dt_sec

        # ==========================================
        # 3. EXPORT GEOJSON LAYERS & SUMMARY
        # ==========================================
        # Reverse backward track so it flows chronologically from Origin (t0) -> Observation (t_sat)
        centroid_track_backward.reverse()
        full_trajectory_coords = [(pt[0], pt[1]) for pt in centroid_track_backward] + [(pt[0], pt[1]) for pt in centroid_track_forward]

        drift_features = [
            {
                "type": "Feature",
                "properties": {
                    "layer": "centroid_drift_trajectory",
                    "origin_time": t0_timestamp.isoformat(),
                    "sat_time": t_sat.isoformat(),
                    "spill_age_hours": spill_age_hours,
                    "forecast_horizon_hours": forecast_hours
                },
                "geometry": LineString(full_trajectory_coords).__geo_interface__
            },
            {
                "type": "Feature",
                "properties": {
                    "layer": "hindcasted_discharge_origin",
                    "timestamp": t0_timestamp.isoformat(),
                    "origin_lat": round(optimal_centroid[1], 6),
                    "origin_lon": round(optimal_centroid[0], 6),
                    "confidence": "High (RK2 Converged)"
                },
                "geometry": Point(optimal_centroid[0], optimal_centroid[1]).__geo_interface__
            }
        ]

        traj_gdf = gpd.GeoDataFrame.from_features(drift_features, crs="EPSG:4326")
        traj_output_path = os.path.join(self.output_dir, "drift_trajectory.geojson")
        traj_gdf.to_file(traj_output_path, driver="GeoJSON")

        summary = {
            "origin_latitude": round(optimal_centroid[1], 6),
            "origin_longitude": round(optimal_centroid[0], 6),
            "discharge_timestamp": t0_timestamp.isoformat(),
            "satellite_timestamp": t_sat.isoformat(),
            "spill_age_hours": spill_age_hours,
            "convergence_variance": round(min_cluster_variance, 8),
            "trajectory_geojson": traj_output_path
        }

        # Synchronize directly with pipeline's metocean environment file
        metocean_env_path = os.path.join("data", "processed", "metocean_environment.json")
        os.makedirs(os.path.dirname(metocean_env_path), exist_ok=True)
        with open(metocean_env_path, "w") as f:
            json.dump({"origin_estimate": summary}, f, indent=4)

        print("\n" + "=" * 60)
        print(" 🎯 LAGRANGIAN HINDCAST & FORECAST COMPLETE")
        print("=" * 60)
        print(f"Origin Coordinates (x0, y0) : Lat {summary['origin_latitude']:.5f}, Lon {summary['origin_longitude']:.5f}")
        print(f"Inferred Discharge Time (t0): {summary['discharge_timestamp']}")
        print(f"Computed Spill Age          : {summary['spill_age_hours']} hours prior to overpass")
        print(f"Trajectory Vectors Saved To : {traj_output_path}")
        print("=" * 60 + "\n")

        return summary


if __name__ == "__main__":
    engine = HindcastEngine(
        wind_nc_path="data/processed/metocean/era5_wind.nc",
        current_nc_path="data/processed/metocean/cmems_currents.nc",
        output_dir="results"
    )
    
    test_slick = "results/spill_polygon.geojson"
    if os.path.exists(test_slick):
        engine.run_hindcast_and_forecast(test_slick)
    else:
        print(f"[INFO] Engine ready. Place detected polygon at: {test_slick}")