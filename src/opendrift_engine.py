import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import geopandas as gpd
import numpy as np
import pyproj
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('OpenDriftEngine')

class OpenDriftHindcastEngine:

    def __init__(self, current_data_path: str, wind_data_path: str, wave_data_path: Optional[str]=None, oil_type: str='GENERIC MEDIUM CRUDE', horizontal_diffusivity: float=10.0, wind_drift_factor: float=0.032, weathering_model: str='adios', release_seed_buffer_deg: float=0.01, confidence_level: float=0.95, output_dir: str='results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_data_path = current_data_path
        self.wind_data_path = wind_data_path
        self.wave_data_path = wave_data_path
        self.oil_type = oil_type
        self.horizontal_diffusivity = float(horizontal_diffusivity)
        if not 0.0 < float(wind_drift_factor) < 1.0:
            raise ValueError(f'wind_drift_factor must lie in (0, 1), got {wind_drift_factor}')
        self.wind_drift_factor = float(wind_drift_factor)
        self.weathering_model = str(weathering_model)
        if float(release_seed_buffer_deg) <= 0:
            raise ValueError('release_seed_buffer_deg must be positive.')
        self.release_seed_buffer_deg = float(release_seed_buffer_deg)
        if not 0.0 < float(confidence_level) < 1.0:
            raise ValueError(f'confidence_level must lie in (0, 1), got {confidence_level}')
        self.confidence_level = float(confidence_level)
        self._validate_sources()

    def _validate_sources(self):
        for path, name in [(self.current_data_path, 'Ocean Current'), (self.wind_data_path, 'Atmospheric Wind')]:
            if not path.startswith('http') and (not os.path.exists(path)):
                raise FileNotFoundError(f'[!] {name} data source does not exist: {path}')

    def _setup_simulation(self) -> OpenOil:
        o = OpenOil(weathering_model=self.weathering_model)
        o.set_config('drift:horizontal_diffusivity', self.horizontal_diffusivity)
        o.set_config('drift:wind_drift_factor', self.wind_drift_factor)
        o.set_config('processes:evaporation', False)
        o.set_config('processes:dispersion', False)
        o.set_config('processes:emulsification', False)
        logger.info('Instantiating Metocean NetCDF / OPeNDAP readers...')
        reader_curr = reader_netCDF_CF_generic.Reader(self.current_data_path)
        reader_wind = reader_netCDF_CF_generic.Reader(self.wind_data_path)
        readers = [reader_curr, reader_wind]
        self.stokes_drift_active = False
        if self.wave_data_path and (self.wave_data_path.startswith('http') or os.path.exists(self.wave_data_path)):
            try:
                logger.info('Wiring WaveWatch III Stokes drift reader: %s', self.wave_data_path)
                reader_wave = reader_netCDF_CF_generic.Reader(self.wave_data_path)
                readers.append(reader_wave)
                o.set_config('drift:stokes_drift', True)
                self.stokes_drift_active = True
            except Exception as exc:
                logger.warning('Failed to register Stokes drift reader (%s), falling back cleanly to standard leeway: %s', self.wave_data_path, exc)
                o.set_config('drift:stokes_drift', False)
        else:
            o.set_config('drift:stokes_drift', False)
        o.add_readers(readers)
        return o

    @staticmethod
    def _extract_polygon(geojson_path: str) -> Polygon:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            raise ValueError(f'Empty GeoJSON provided: {geojson_path}')
        geom = unary_union(gdf.geometry)
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda a: a.area)
        return geom

    def _compute_error_ellipse(self, lons: np.ndarray, lats: np.ndarray, confidence: Optional[float]=None) -> Dict[str, float]:
        confidence = float(self.confidence_level if confidence is None else confidence)
        if not 0.0 < confidence < 1.0:
            raise ValueError(f'confidence must lie in (0, 1), got {confidence}')
        chi2_2 = -2.0 * math.log(1.0 - confidence)
        scale_factor = math.sqrt(chi2_2)
        mean_lon = float(np.mean(lons))
        mean_lat = float(np.mean(lats))
        geod = pyproj.Geod(ellps='WGS84')
        _, _, meters_per_deg_lat = geod.inv(mean_lon, mean_lat, mean_lon, mean_lat + 1.0)
        _, _, meters_per_deg_lon = geod.inv(mean_lon, mean_lat, mean_lon + 1.0, mean_lat)
        x_east_m = (np.asarray(lons, dtype='float64') - mean_lon) * meters_per_deg_lon
        y_north_m = (np.asarray(lats, dtype='float64') - mean_lat) * meters_per_deg_lat
        cov_metric = np.cov(np.column_stack((x_east_m, y_north_m)), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_metric)
        semi_major = float(np.sqrt(max(eigenvalues[0], eigenvalues[1])) * scale_factor)
        semi_minor = float(np.sqrt(min(eigenvalues[0], eigenvalues[1])) * scale_factor)
        major_axis_bearing = (90.0 - math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))) % 360.0
        return {'semi_major_axis_meters': semi_major, 'semi_minor_axis_meters': semi_minor, 'orientation_deg': float(major_axis_bearing), 'covariance_matrix_m2': cov_metric.tolist(), 'confidence_level': confidence}

    def run_backward_hindcast(self, slick_geojson_path: str, satellite_time: datetime, hindcast_hours: float=24.0, time_step_minutes: int=15, num_particles: int=2000) -> Dict[str, Any]:
        logger.info(f'Initiating OpenOil backward hindcast for {hindcast_hours}h window...')
        slick_poly = self._extract_polygon(slick_geojson_path)
        sim = self._setup_simulation()
        sim.seed_within_polygon(slick_poly, number=num_particles, time=satellite_time, oil_type=self.oil_type)
        nc_output = os.path.join(self.output_dir, 'hindcast_simulation.nc')
        sim.run(duration=timedelta(hours=-hindcast_hours), time_step=-time_step_minutes * 60, time_step_output=time_step_minutes * 60, outfile=nc_output)
        final_lons = sim.get_property('lon')[0][-1, :]
        final_lats = sim.get_property('lat')[0][-1, :]
        valid_mask = ~np.isnan(final_lons) & ~np.isnan(final_lats)
        clean_lons = final_lons[valid_mask]
        clean_lats = final_lats[valid_mask]
        if len(clean_lons) == 0:
            raise RuntimeError('All Lagrangian particles drifted out of domain or stranded.')
        origin_lon = float(np.median(clean_lons))
        origin_lat = float(np.median(clean_lats))
        discharge_time = satellite_time - timedelta(hours=hindcast_hours)
        uncertainty_ellipse = self._compute_error_ellipse(clean_lons, clean_lats)
        origin_feature = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'layer': 'hindcasted_discharge_origin', 't0_discharge': discharge_time.isoformat(), 't_satellite': satellite_time.isoformat(), 'search_radius_m': uncertainty_ellipse['semi_major_axis_meters'], 'orientation_deg': uncertainty_ellipse['orientation_deg']}, 'geometry': Point(origin_lon, origin_lat).__geo_interface__}]}
        origin_geojson_path = os.path.join(self.output_dir, 'origin_search_window.geojson')
        with open(origin_geojson_path, 'w') as f:
            json.dump(origin_feature, f, indent=4)
        slick_centroid = slick_poly.centroid
        geod = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, _, _ = geod.inv(slick_centroid.x, slick_centroid.y, origin_lon, origin_lat)
        drift_bearing_deg = float(fwd_azimuth) % 360.0
        result_summary = {'origin_lon': origin_lon, 'origin_lat': origin_lat, 'inferred_t0': discharge_time.isoformat(), 'satellite_timestamp': satellite_time.isoformat(), 'hindcast_duration_hours': hindcast_hours, 'particles_tracked': len(clean_lons), 'drift_bearing_deg': drift_bearing_deg, 'uncertainty': uncertainty_ellipse, 'origin_geojson': origin_geojson_path, 'simulation_raw_nc': nc_output, 'stokes_drift_active': getattr(self, 'stokes_drift_active', False), 'drift_provenance': 'opendrift_with_wavewatch3_stokes_drift' if getattr(self, 'stokes_drift_active', False) else 'standard_leeway_fallback (u_current + 0.032 * u_wind)'}
        logger.info(f"Hindcast complete. Origin candidate: ({origin_lat:.5f}, {origin_lon:.5f}) ± {uncertainty_ellipse['semi_major_axis_meters']:.1f}m")
        return result_summary

    def run_forward_forecast(self, release_point: Optional[Tuple[float, float]]=None, release_geojson_path: Optional[str]=None, release_time: Optional[datetime]=None, forecast_hours: float=48.0, time_step_minutes: int=15, num_particles: int=2000) -> Dict[str, Any]:
        if release_time is None:
            raise ValueError('release_time is required for forward forecast')
        if release_point is None and release_geojson_path is None:
            raise ValueError('Either release_point or release_geojson_path must be provided')
        if release_geojson_path:
            release_poly = self._extract_polygon(release_geojson_path)
        else:
            lon, lat = release_point
            release_poly = Point(lon, lat).buffer(self.release_seed_buffer_deg)
        logger.info(f'Initiating OpenOil forward forecast for {forecast_hours}h starting {release_time.isoformat()}...')
        sim = self._setup_simulation()
        sim.set_config('processes:evaporation', True)
        sim.set_config('processes:dispersion', True)
        sim.set_config('processes:emulsification', True)
        sim.set_config('drift:wind_drift_factor', self.wind_drift_factor)
        sim.seed_within_polygon(release_poly, number=num_particles, time=release_time, oil_type=self.oil_type)
        nc_output = os.path.join(self.output_dir, 'forecast_simulation.nc')
        sim.run(duration=timedelta(hours=forecast_hours), time_step=time_step_minutes * 60, time_step_output=time_step_minutes * 60, outfile=nc_output)
        final_lons = sim.get_property('lon')[0][-1, :]
        final_lats = sim.get_property('lat')[0][-1, :]
        valid_mask = ~np.isnan(final_lons) & ~np.isnan(final_lats)
        clean_lons = final_lons[valid_mask]
        clean_lats = final_lats[valid_mask]
        if len(clean_lons) == 0:
            raise RuntimeError('All Lagrangian particles drifted out of domain or stranded during forecast.')
        points = [Point(xy) for xy in zip(clean_lons, clean_lats)]
        mpoints = MultiPoint(points)
        forecast_poly = mpoints.convex_hull
        forecast_feature = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'layer': 'forecast_slick_extent', 't_release': release_time.isoformat(), 't_end': (release_time + timedelta(hours=forecast_hours)).isoformat(), 'particles': len(clean_lons)}, 'geometry': forecast_poly.__geo_interface__}]}
        forecast_geojson_path = os.path.join(self.output_dir, 'forecast_slick_extent.geojson')
        with open(forecast_geojson_path, 'w') as f:
            json.dump(forecast_feature, f, indent=4)
        uncertainty_ellipse = self._compute_error_ellipse(clean_lons, clean_lats)
        result = {'forecast_geojson': forecast_geojson_path, 'forecast_nc': nc_output, 'forecast_lon': float(np.median(clean_lons)), 'forecast_lat': float(np.median(clean_lats)), 'particles_tracked': len(clean_lons), 'uncertainty': uncertainty_ellipse, 'forecast_end_time': (release_time + timedelta(hours=forecast_hours)).isoformat(), 'stokes_drift_active': getattr(self, 'stokes_drift_active', False), 'drift_provenance': 'opendrift_with_wavewatch3_stokes_drift' if getattr(self, 'stokes_drift_active', False) else 'standard_leeway_fallback (u_current + 0.032 * u_wind)'}
        logger.info(f'Forecast complete. Slick centroid approx ({np.median(clean_lats):.5f}, {np.median(clean_lons):.5f})')
        return result