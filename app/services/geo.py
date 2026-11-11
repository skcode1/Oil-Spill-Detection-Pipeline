"""Geodesy, polygon and uncertainty-ellipse helpers shared by the services.

Everything here works on WGS84 lon/lat and delegates to ``pyproj``'s ellipsoidal
``Geod`` so distances/areas stay accurate away from the equator (the legacy
modules used the same approach, e.g. ``src/opendrift_engine.py``).

The error-ellipse computation intentionally mirrors
``OpenDriftHindcastEngine._compute_error_ellipse`` (local tangent plane scaled
geodesically, 2-DOF chi-square scaling, bearing clockwise from True North) so
OpenDrift-driven and built-in-engine results are directly comparable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyproj
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

GEOD: pyproj.Geod = pyproj.Geod(ellps="WGS84")

#: Vertex count used when rasterising an uncertainty ellipse to a polygon.
ELLIPSE_VERTICES = 72


def meters_per_degree(lon: float, lat: float) -> Tuple[float, float]:
    """Return ``(m_per_deg_lon, m_per_deg_lat)`` at a WGS84 location."""
    _, _, m_per_deg_lat = GEOD.inv(lon, lat, lon, lat + 1.0)
    _, _, m_per_deg_lon = GEOD.inv(lon, lat, lon + 1.0, lat)
    return float(m_per_deg_lon), float(m_per_deg_lat)


def geodesic_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Ellipsoidal distance between two WGS84 points (meters)."""
    _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance)


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from North."""
    azimuth, _, _ = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(azimuth % 360.0)


def destination_point(
    lon: float, lat: float, bearing: float, distance_m: float
) -> Tuple[float, float]:
    """Geodesic destination from ``(lon, lat)`` along ``bearing`` (degrees)."""
    end_lon, end_lat, _ = GEOD.fwd(lon, lat, bearing, distance_m)
    return float(end_lon), float(end_lat)


def offset_point(
    lon: float, lat: float, east_m: float, north_m: float
) -> Tuple[float, float]:
    """Offset a point by metric east/north components on the WGS84 ellipsoid."""
    distance = float(math.hypot(east_m, north_m))
    if distance == 0.0:
        return float(lon), float(lat)
    heading = math.degrees(math.atan2(east_m, north_m)) % 360.0
    return destination_point(lon, lat, heading, distance)


def geodesic_area_m2(geometry: BaseGeometry) -> float:
    """Geodesic area of a WGS84 polygon/multipolygon in square meters."""
    if geometry.is_empty:
        return 0.0
    area, _ = GEOD.geometry_area_perimeter(geometry)
    return abs(float(area))


def polygon_from_geojson(payload: Any) -> BaseGeometry:
    """Coerce a GeoJSON payload into a shapely geometry.

    Accepts a ``FeatureCollection``, ``Feature``, raw geometry mapping or an
    already-built shapely geometry.

    Args:
        payload: GeoJSON mapping (or shapely geometry) to convert.

    Returns:
        The parsed :class:`shapely.geometry.base.BaseGeometry`.

    Raises:
        ValueError: If ``payload`` carries no usable geometry.
    """
    if payload is None:
        raise ValueError("no GeoJSON payload supplied")
    if isinstance(payload, BaseGeometry):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported GeoJSON payload type {type(payload).__name__}")

    kind = str(payload.get("type", ""))
    if kind == "FeatureCollection":
        geometries = [
            polygon_from_geojson(feature) for feature in (payload.get("features") or [])
        ]
        geometries = [g for g in geometries if g is not None and not g.is_empty]
        if not geometries:
            raise ValueError("FeatureCollection contains no geometries")
        return unary_union(geometries)
    if kind == "Feature":
        return polygon_from_geojson(payload.get("geometry"))
    if kind in (
        "Polygon",
        "MultiPolygon",
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "GeometryCollection",
    ):
        return shape(payload)
    raise ValueError(f"unsupported GeoJSON geometry type {kind!r}")


def polygon_centroid_lonlat(geometry: BaseGeometry) -> Tuple[float, float]:
    """Return the centroid of a WGS84 geometry as ``(lon, lat)``."""
    if geometry.is_empty:
        raise ValueError("cannot compute a centroid for an empty geometry")
    point = geometry.representative_point() if geometry.area <= 0 else geometry.centroid
    return float(point.x), float(point.y)


def polygon_bbox_wgs84(geometry: BaseGeometry) -> List[float]:
    """Return ``[min_lon, min_lat, max_lon, max_lat]`` for a geometry."""
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


def feature_collection(features: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap already-built GeoJSON features in a FeatureCollection."""
    return {"type": "FeatureCollection", "features": [f for f in features if f]}


def compute_error_ellipse(
    lons: Sequence[float],
    lats: Sequence[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Fit a rotated confidence ellipse to a particle cloud (meters).

    Mirrors ``src/opendrift_engine.py::_compute_error_ellipse``: the cloud is
    projected onto a local tangent plane centred on its mean position, scaled
    geodesically, then eigen-decomposed.  The 2-DOF chi-square quantile
    ``x = -2 ln(1 - p)`` converts the 1-sigma axes to the requested confidence.

    Args:
        lons: Particle longitudes (degrees).
        lats: Particle latitudes (degrees).
        confidence: Coverage probability in ``(0, 1)``.

    Returns:
        Mapping with ``semi_major_axis_meters``, ``semi_minor_axis_meters``,
        ``orientation_deg`` (clockwise from True North), ``confidence_level``,
        ``covariance_matrix_m2`` and the cloud centroid.

    Raises:
        ValueError: If the cloud is empty or the confidence is out of range.
    """
    lon_arr = np.asarray(lons, dtype="float64").ravel()
    lat_arr = np.asarray(lats, dtype="float64").ravel()
    if lon_arr.size == 0 or lat_arr.size == 0:
        raise ValueError("cannot fit an error ellipse to an empty particle cloud")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")

    scale_factor = math.sqrt(-2.0 * math.log(1.0 - float(confidence)))
    mean_lon, mean_lat = float(lon_arr.mean()), float(lat_arr.mean())
    m_per_deg_lon, m_per_deg_lat = meters_per_degree(mean_lon, mean_lat)

    x_east = (lon_arr - mean_lon) * m_per_deg_lon
    y_north = (lat_arr - mean_lat) * m_per_deg_lat
    cov = np.atleast_2d(np.cov(np.column_stack((x_east, y_north)), rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    largest = max(float(eigenvalues[0]), float(eigenvalues[1]), 0.0)
    smallest = max(min(float(eigenvalues[0]), float(eigenvalues[1])), 0.0)
    # A degenerate cloud (identical particles) still needs strictly positive axes.
    semi_major = max(math.sqrt(largest) * scale_factor, 1e-6)
    semi_minor = max(math.sqrt(smallest) * scale_factor, 1e-6)
    if semi_minor > semi_major:  # pragma: no cover - guarded above
        semi_major, semi_minor = semi_minor, semi_major

    major_bearing = (
        90.0 - math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    ) % 360.0

    return {
        "semi_major_axis_meters": semi_major,
        "semi_minor_axis_meters": semi_minor,
        "orientation_deg": float(major_bearing),
        "confidence_level": float(confidence),
        "covariance_matrix_m2": cov.tolist(),
        "center_lon": mean_lon,
        "center_lat": mean_lat,
    }


def ellipse_frame_offset(
    along_major_m: float, across_minor_m: float, orientation_deg: float
) -> Tuple[float, float]:
    """Rotate in-frame ellipse components into metric ``(east, north)`` offsets.

    ``along_major_m`` is measured along the axis whose compass bearing is
    ``orientation_deg`` (clockwise from North); ``across_minor_m`` is measured
    90 degrees clockwise from that axis.
    """
    bearing = math.radians(float(orientation_deg))
    east = along_major_m * math.sin(bearing) + across_minor_m * math.cos(bearing)
    north = along_major_m * math.cos(bearing) - across_minor_m * math.sin(bearing)
    return float(east), float(north)


def ellipse_polygon_lonlat(
    center_lon: float,
    center_lat: float,
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    vertices: int = ELLIPSE_VERTICES,
) -> Polygon:
    """Rasterise a rotated uncertainty ellipse into a WGS84 polygon."""
    major = max(float(semi_major_m), float(semi_minor_m))
    minor = min(float(semi_major_m), float(semi_minor_m))
    angles = np.linspace(0.0, 2.0 * math.pi, int(max(vertices, 8)), endpoint=False)
    coords: List[Tuple[float, float]] = []
    for theta in angles:
        east, north = ellipse_frame_offset(
            major * math.cos(theta), minor * math.sin(theta), orientation_deg
        )
        coords.append(offset_point(center_lon, center_lat, east, north))
    if coords:
        coords.append(coords[0])
    polygon = Polygon(coords)
    return polygon if polygon.is_valid else polygon.buffer(0)


def convex_hull_of_points(
    lons: Sequence[float], lats: Sequence[float]
) -> Optional[BaseGeometry]:
    """Convex hull of a point cloud (``None`` when no points are given)."""
    lon_arr = np.asarray(lons, dtype="float64").ravel()
    lat_arr = np.asarray(lats, dtype="float64").ravel()
    if lon_arr.size == 0:
        return None
    if lon_arr.size == 1:
        return Point(float(lon_arr[0]), float(lat_arr[0]))
    return MultiPoint(list(zip(lon_arr.tolist(), lat_arr.tolist()))).convex_hull


def convex_hull_geojson(
    lons: Sequence[float], lats: Sequence[float], layer: str = "particle_envelope"
) -> Dict[str, Any]:
    """GeoJSON FeatureCollection holding the convex hull of a point cloud."""
    lon_arr = np.asarray(lons, dtype="float64").ravel()
    lat_arr = np.asarray(lats, dtype="float64").ravel()
    hull = convex_hull_of_points(lon_arr, lat_arr)
    if hull is None:
        return feature_collection([])
    if isinstance(hull, Point):
        hull = hull.buffer(0.01)
    return feature_collection(
        [
            {
                "type": "Feature",
                "properties": {"layer": layer, "particles": int(lon_arr.size)},
                "geometry": mapping(hull),
            }
        ]
    )



def sample_points_in_geometry(
    geometry: BaseGeometry,
    count: int,
    rng: np.random.Generator,
    buffer_deg: float = 0.0,
) -> List[Tuple[float, float]]:
    """Uniformly sample ``count`` lon/lat points inside a WGS84 geometry.

    Bounding-box rejection sampling with a representative-point fallback, so the
    seeder always returns exactly ``count`` points even for sliver polygons.

    Args:
        geometry: Seed region (polygon, multipolygon or point).
        count: Number of particles required (>= 1).
        rng: Seeded NumPy generator (determinism is a tested contract).
        buffer_deg: Dilation applied when the seed degenerates to a point.

    Returns:
        List of ``(lon, lat)`` tuples of length ``count``.
    """
    count = int(max(count, 1))
    region = geometry
    if region.geom_type in ("Point", "MultiPoint") or region.area == 0.0:
        radius = max(float(buffer_deg), 1e-4)
        region = region.buffer(radius)
    if isinstance(region, GeometryCollection):  # pragma: no cover - defensive
        parts = [
            g for g in region.geoms if g.geom_type in ("Polygon", "MultiPolygon")
        ]
        if parts:
            region = unary_union(parts)

    min_lon, min_lat, max_lon, max_lat = region.bounds
    if max_lon - min_lon <= 0 or max_lat - min_lat <= 0:
        point = region.representative_point()
        return [(float(point.x), float(point.y))] * count

    samples: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = 60 * count
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        lons = rng.uniform(min_lon, max_lon, size=count)
        lats = rng.uniform(min_lat, max_lat, size=count)
        for lon, lat in zip(lons.tolist(), lats.tolist()):
            if len(samples) >= count:
                break
            if region.contains(Point(lon, lat)):
                samples.append((float(lon), float(lat)))

    if len(samples) < count:
        point = region.representative_point()
        while len(samples) < count:
            samples.append((float(point.x), float(point.y)))
    return samples


def normalize_ellipse(
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    confidence: float,
    center_lon: Optional[float] = None,
    center_lat: Optional[float] = None,
) -> Dict[str, Any]:
    """Clamp/repair an ellipse description into a schema-safe dictionary."""
    major = max(float(semi_major_m), float(semi_minor_m), 1e-6)
    minor = max(min(float(semi_major_m), float(semi_minor_m)), 1e-6)
    return {
        "semi_major_axis_meters": major,
        "semi_minor_axis_meters": minor,
        "orientation_deg": float(orientation_deg) % 360.0,
        "confidence_level": min(max(float(confidence), 0.0), 1.0),
        "center_lon": center_lon,
        "center_lat": center_lat,
        "area_km2": math.pi * major * minor / 1e6,
    }


def to_geographic(
    lon: float, lat: float, min_lon: float = -180.0, max_lon: float = 180.0
) -> Tuple[float, float]:
    """Wrap a longitude into ``[min_lon, max_lon)`` and clamp latitude to +-90."""
    span = max_lon - min_lon
    wrapped = ((float(lon) - min_lon) % span) + min_lon
    return float(wrapped), float(min(max(float(lat), -90.0), 90.0))

