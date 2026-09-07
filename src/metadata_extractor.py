import os
import re
from datetime import datetime, timezone
import rasterio
from rasterio.warp import transform_bounds, transform


MAJOR_SHIPPING_CORRIDORS = [
    {"name": "Strait of Malacca", "lat": 2.5000, "lon": 101.5000},
    {"name": "Persian Gulf TSS", "lat": 26.5000, "lon": 53.0000},
    {"name": "Singapore Strait", "lat": 1.2500, "lon": 103.9000},
    {"name": "English Channel TSS", "lat": 50.2000, "lon": -0.5000},
    {"name": "Gulf of Guinea", "lat": 4.5000, "lon": 6.0000},
]


def extract_acquisition_time(dataset: rasterio.DatasetReader, filename: str) -> str:
    """
    Extracts the acquisition UTC timestamp from raster metadata tags 
    or parses standard Sentinel-1 filename naming conventions.
    """
    # 1. Inspect TIFF metadata tags
    tags = dataset.tags()
    time_keys = ["ACQUISITION_DATETIME", "TIFFTAG_DATETIME", "START_TIME", "TIME_OF_ACQUISITION"]
    for k in time_keys:
        if k in tags:
            try:
                # Format: YYYY:MM:DD HH:MM:SS or ISO
                clean_time = tags[k].replace(":", "-", 2).strip()
                dt = datetime.fromisoformat(clean_time)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                pass

    # 2. Parse Sentinel-1 filename format: S1A_..._YYYYMMDDTHHMMSS_...
    match = re.search(r"(\d{8}T\d{6})", filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass

    # 3. Fallback to current UTC timestamp
    return datetime.now(timezone.utc).isoformat()


def extract_spatial_metadata(geotiff_path: str, patch_row_offset: int = 0, patch_col_offset: int = 0, patch_size: int = None) -> dict:
    """
    Extracts high-fidelity spatial bounds, center coordinates, and temporal anchors
    from full scenes or localized patch slices for hindcast/forecast physics modeling.
    """
    filename = os.path.basename(geotiff_path)

    # Auto-extract row/column offsets from patch filename if present (e.g. "..._r1024_c512.tif")
    if patch_row_offset == 0 and patch_col_offset == 0:
        match = re.search(r"_r(\d+)_c(\d+)", filename)
        if match:
            patch_row_offset = int(match.group(1))
            patch_col_offset = int(match.group(2))
            if patch_size is None:
                patch_size = 512

    with rasterio.open(geotiff_path) as dataset:
        crs = dataset.crs
        transform_matrix = dataset.transform

        # Case 1: Non-georeferenced or local pixel-space imagery
        if crs is None or dataset.transform.is_identity:
            fallback = MAJOR_SHIPPING_CORRIDORS[1]  # Persian Gulf default
            print(f"[!] Warning: No CRS found for {filename}. Defaulting to corridor anchor: {fallback['name']}")
            return {
                "georeferenced": False,
                "crs": None,
                "timestamp_utc": extract_acquisition_time(dataset, filename),
                "center": (fallback["lat"], fallback["lon"]),
                "bbox_wgs84": [fallback["lon"] - 0.02, fallback["lat"] - 0.02, fallback["lon"] + 0.02, fallback["lat"] + 0.02],
                "copernicus_query": {
                    "min_lat": fallback["lat"] - 0.02, "max_lat": fallback["lat"] + 0.02,
                    "min_lon": fallback["lon"] - 0.02, "max_lon": fallback["lon"] + 0.02
                }
            }

        # Case 2: Localized patch coordinates via Affine Transform
        if patch_size is not None and (patch_row_offset > 0 or patch_col_offset > 0):
            # Compute real projected coordinates from top-left to bottom-right of patch
            px_left, py_top = transform_matrix * (patch_col_offset, patch_row_offset)
            px_right, py_bottom = transform_matrix * (patch_col_offset + patch_size, patch_row_offset + patch_size)
            
            src_min_x, src_max_x = min(px_left, px_right), max(px_left, px_right)
            src_min_y, src_max_y = min(py_bottom, py_top), max(py_bottom, py_top)
        else:
            # Full dataset bounds
            src_bounds = dataset.bounds
            src_min_x, src_min_y = src_bounds.left, src_bounds.bottom
            src_max_x, src_max_y = src_bounds.right, src_bounds.top

        # Reproject to WGS84 (EPSG:4326) if native projection is UTM or another CRS
        if crs != "EPSG:4326":
            min_lon, min_lat, max_lon, max_lat = transform_bounds(
                crs, "EPSG:4326", src_min_x, src_min_y, src_max_x, src_max_y
            )
        else:
            min_lon, min_lat, max_lon, max_lat = src_min_x, src_min_y, src_max_x, src_max_y

        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        acquisition_time = extract_acquisition_time(dataset, filename)

        return {
            "georeferenced": True,
            "source_crs": str(crs),
            "target_crs": "EPSG:4326",
            "timestamp_utc": acquisition_time,
            "center": (round(center_lat, 6), round(center_lon, 6)),
            # Standard GeoJSON Bounding Box: [min_lon, min_lat, max_lon, max_lat]
            "bbox_wgs84": [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
            # Copernicus Marine / NOAA GFS API query parameter layout
            "metocean_api_params": {
                "min_lat": round(min_lat, 4),
                "max_lat": round(max_lat, 4),
                "min_lon": round(min_lon, 4),
                "max_lon": round(max_lon, 4),
                "start_datetime": acquisition_time
            }
        }


if __name__ == "__main__":
    sample_file = "data/raw/SAR_training/sentinel/image/sample_scene.tif"
    if os.path.exists(sample_file):
        metadata = extract_spatial_metadata(sample_file)
        print("\nExtracted Spatial Metadata:")
        print(metadata)