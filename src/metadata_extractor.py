import os
import re
import sys
from datetime import datetime, timezone
from typing import Optional
import rasterio
from rasterio.warp import transform_bounds
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.region_registry import get_region

def _load_config(path: str='config.yaml') -> dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as handle:
            import yaml
            return yaml.safe_load(handle) or {}
    return {}

def extract_acquisition_time(dataset: rasterio.DatasetReader, filename: str) -> str:
    tags = dataset.tags()
    time_keys = ['ACQUISITION_DATETIME', 'TIFFTAG_DATETIME', 'START_TIME', 'TIME_OF_ACQUISITION']
    for k in time_keys:
        if k in tags:
            try:
                clean_time = tags[k].replace(':', '-', 2).strip()
                dt = datetime.fromisoformat(clean_time)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    match = re.search('(\\d{8}T\\d{6})', filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), '%Y%m%dT%H%M%S')
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()

def extract_spatial_metadata(geotiff_path: str, patch_row_offset: int=0, patch_col_offset: int=0, patch_size: int=None, config_path: str='config.yaml', fallback_region: Optional[str]=None, fallback_pad_deg: Optional[float]=None) -> dict:
    filename = os.path.basename(geotiff_path)
    meta_cfg = _load_config(config_path).get('metadata') or {}
    fallback_name = fallback_region or meta_cfg.get('fallback_region', 'Persian Gulf TSS')
    pad_deg = float(fallback_pad_deg if fallback_pad_deg is not None else meta_cfg.get('fallback_pad_deg', 0.02))
    default_patch_size = int(meta_cfg.get('default_patch_size', 512))
    if patch_row_offset == 0 and patch_col_offset == 0:
        match = re.search('_r(\\d+)_c(\\d+)', filename)
        if match:
            patch_row_offset = int(match.group(1))
            patch_col_offset = int(match.group(2))
            if patch_size is None:
                patch_size = default_patch_size
    with rasterio.open(geotiff_path) as dataset:
        crs = dataset.crs
        transform_matrix = dataset.transform
        if crs is None or dataset.transform.is_identity:
            fallback = get_region(name=fallback_name, config_path=config_path)
            print(f"[!] Warning: No CRS found for {filename}. Defaulting to corridor anchor: {fallback['name']}")
            return {'georeferenced': False, 'crs': None, 'fallback_region': fallback['name'], 'timestamp_utc': extract_acquisition_time(dataset, filename), 'center': (fallback['lat'], fallback['lon']), 'bbox_wgs84': [fallback['lon'] - pad_deg, fallback['lat'] - pad_deg, fallback['lon'] + pad_deg, fallback['lat'] + pad_deg], 'copernicus_query': {'min_lat': fallback['lat'] - pad_deg, 'max_lat': fallback['lat'] + pad_deg, 'min_lon': fallback['lon'] - pad_deg, 'max_lon': fallback['lon'] + pad_deg}}
        if patch_size is not None and (patch_row_offset > 0 or patch_col_offset > 0):
            px_left, py_top = transform_matrix * (patch_col_offset, patch_row_offset)
            px_right, py_bottom = transform_matrix * (patch_col_offset + patch_size, patch_row_offset + patch_size)
            src_min_x, src_max_x = (min(px_left, px_right), max(px_left, px_right))
            src_min_y, src_max_y = (min(py_bottom, py_top), max(py_bottom, py_top))
        else:
            src_bounds = dataset.bounds
            src_min_x, src_min_y = (src_bounds.left, src_bounds.bottom)
            src_max_x, src_max_y = (src_bounds.right, src_bounds.top)
        if crs != 'EPSG:4326':
            min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, 'EPSG:4326', src_min_x, src_min_y, src_max_x, src_max_y)
        else:
            min_lon, min_lat, max_lon, max_lat = (src_min_x, src_min_y, src_max_x, src_max_y)
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        acquisition_time = extract_acquisition_time(dataset, filename)
        return {'georeferenced': True, 'source_crs': str(crs), 'target_crs': 'EPSG:4326', 'timestamp_utc': acquisition_time, 'center': (round(center_lat, 6), round(center_lon, 6)), 'bbox_wgs84': [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)], 'metocean_api_params': {'min_lat': round(min_lat, 4), 'max_lat': round(max_lat, 4), 'min_lon': round(min_lon, 4), 'max_lon': round(max_lon, 4), 'start_datetime': acquisition_time}}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dynamic SAR scene metadata extractor')
    parser.add_argument('--input', type=str, required=True, help='GeoTIFF scene or patch path')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    if os.path.exists(args.input):
        metadata = extract_spatial_metadata(args.input, config_path=args.config)
        print('\nExtracted Spatial Metadata:')
        print(metadata)
    else:
        print(f'[!] Input not found: {args.input}')