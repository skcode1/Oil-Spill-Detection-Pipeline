"""SAR/Sentinel scene loading, validation and artifact export.

The readers deliberately accept the same inputs the training pipeline used:

* ``tifffile`` for GeoTIFF/plain TIFF (1-channel VV, 2-channel VV+VH, or
  3-channel composites) - this is what ``src/dataset.py`` consumed.
* ``cv2.imread(..., IMREAD_UNCHANGED)`` for PNG/JPEG quick-looks, matching the
  legacy ``backend/main.py`` upload path.

Georeferencing is read opportunistically through ``rasterio``; when a scene has
no CRS (as is the case for the shipped ``authentic testing`` corpus) the loader
falls back to pixel-space geometry plus a configurable ground sample distance so
metric quantities (area, age) can still be produced - the response flags this
via ``georeferenced=False`` and ``coordinate_space="pixel"``.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # tifffile is a declared dependency; the guard keeps imports harmless.
    import tifffile as tiff
except ImportError:  # pragma: no cover
    tiff = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.warp import transform_bounds
except ImportError:  # pragma: no cover - rasterio is a declared dependency
    rasterio = None  # type: ignore[assignment]
    transform_bounds = None  # type: ignore[assignment]

from app.config import AppSettings
from app.errors import SceneNotFoundError, UnsupportedImageryError
from app.services.geo import meters_per_degree

logger = logging.getLogger(__name__)

#: Raster extensions accepted by every endpoint.
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)

#: Regex for the Sentinel-1 acquisition stamp inside a product name.
_SENTINEL_TIME_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})")


@dataclass
class SceneArray:
    """A loaded scene plus everything the detector needs to geolocate it."""

    path: str
    filename: str
    array: np.ndarray
    band_count: int
    georeferenced: bool = False
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bbox_wgs84: Optional[List[float]] = None
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    acquisition_time_utc: Optional[str] = None
    pixel_spacing_m: float = 10.0
    imagery_kind: str = "sar"
    warnings: List[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        """Scene height in pixels."""
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        """Scene width in pixels."""
        return int(self.array.shape[1])

    def backscatter_stats(self) -> Dict[str, float]:
        """Min/max/mean/std of the raw backscatter values."""
        values = np.asarray(self.array, dtype=np.float32)
        return {
            "backscatter_min": float(values.min()),
            "backscatter_max": float(values.max()),
            "backscatter_mean": float(values.mean()),
            "backscatter_std": float(values.std()),
        }


def resolve_scene_path(path: str, settings: AppSettings) -> Path:
    """Resolve a user-supplied path and assert that it is a readable raster.

    Relative paths are resolved against the project root so a frontend can send
    ``data/raw/scene.tif`` or an absolute path interchangeably.

    Raises:
        SceneNotFoundError: When the path does not resolve to a file.
        UnsupportedImageryError: When the extension is not a raster format.
    """
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = settings.project_root / candidate
    if not candidate.is_file():
        raise SceneNotFoundError(
            f"Scene not found on the server: {path}",
            details={"resolved": str(candidate)},
        )
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedImageryError(
            f"Unsupported raster extension '{candidate.suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            details={"path": str(candidate)},
        )
    return candidate


def read_raster(path: Path) -> np.ndarray:
    """Read a raster into a NumPy array, dispatching on the file extension.

    Raises:
        UnsupportedImageryError: When the file cannot be decoded.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            if tiff is None:  # pragma: no cover - dependency guard
                raise UnsupportedImageryError("tifffile is not installed")
            array = tiff.imread(str(path))
        else:
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except UnsupportedImageryError:
        raise
    except Exception as exc:
        raise UnsupportedImageryError(
            f"Could not decode raster {path.name}: {exc}",
            details={"path": str(path)},
        ) from exc

    if array is None:
        raise UnsupportedImageryError(
            f"Could not decode raster {path.name} (empty or corrupt file).",
            details={"path": str(path)},
        )
    return np.asarray(array)


def _looks_like_optical(array: np.ndarray) -> bool:
    """Heuristic: true when three bands look like an optical RGB composite.

    SAR dual-pol channels are strongly correlated (both measure the same
    surface), whereas optical RGB bands are largely decorrelated.  The metric is
    the mean absolute difference of the normalised channels, thresholded at 0.15
    (a value no SAR composite in the shipped corpus approaches).
    """
    if array.ndim != 3 or array.shape[2] < 3:
        return False
    bands = []
    for index in range(3):
        band = np.asarray(array[:, :, index], dtype=np.float32)
        span = float(band.max() - band.min())
        bands.append(np.zeros_like(band) if span < 1e-8 else (band - band.min()) / span)
    diffs = []
    for first, second in ((0, 1), (1, 2)):
        diffs.append(float(np.mean(np.abs(bands[first] - bands[second]))))
    return max(diffs) > 0.15


def scene_from_array(
    array: np.ndarray,
    *,
    path: str,
    filename: str,
    settings: AppSettings,
    pixel_spacing_m: Optional[float] = None,
    acquisition_time_utc: Optional[str] = None,
) -> SceneArray:
    """Build a validated :class:`SceneArray` from an in-memory raster.

    Applies the domain sanity checks (minimum size, finite values, non-constant
    backscatter, RGB-look detection) required before the segmentation model is
    invoked.

    Raises:
        UnsupportedImageryError: When the raster cannot be used as SAR input.
    """
    detection = settings.detection
    arr = np.asarray(array)

    # --- shape normalisation (CHW exports are transposed to HWC) ------------
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3) and arr.shape[2] not in (1, 2, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim not in (2, 3):
        raise UnsupportedImageryError(
            f"Unsupported raster dimensionality {arr.ndim}D (expected 2D or 3D).",
            details={"shape": list(arr.shape)},
        )
    if arr.ndim == 3 and arr.shape[2] not in (1, 2, 3):
        raise UnsupportedImageryError(
            f"Unsupported band count {arr.shape[2]}; expected 1 (VV), 2 (VV+VH) or 3.",
            details={"shape": list(arr.shape)},
        )

    band_count = 1 if arr.ndim == 2 else int(arr.shape[2])
    height, width = int(arr.shape[0]), int(arr.shape[1])
    if min(height, width) < detection.min_scene_dim:
        raise UnsupportedImageryError(
            f"Scene is too small ({width}x{height}px); minimum is "
            f"{detection.min_scene_dim}px per side.",
            details={"width": width, "height": height},
        )

    arr = arr.astype(np.float32)
    if not np.all(np.isfinite(arr)):
        finite = np.isfinite(arr)
        if not finite.any():
            raise UnsupportedImageryError("Scene contains no finite pixel values.")
        arr = np.where(finite, arr, float(arr[finite].min()))

    dynamic_range = float(arr.max() - arr.min())
    if dynamic_range < detection.min_dynamic_range:
        raise UnsupportedImageryError(
            "Scene has no dynamic range (constant/blank raster); nothing to detect.",
            details={"dynamic_range": dynamic_range},
        )

    warnings: List[str] = []
    imagery_kind = "sar"
    if band_count == 3 and _looks_like_optical(arr):
        imagery_kind = "unknown"
        message = (
            "Scene looks like an optical RGB composite (bands are strongly "
            "decorrelated). The segmentation model was trained on SAR "
            "backscatter; results may be unreliable."
        )
        if detection.strict_radiometric_validation:
            raise UnsupportedImageryError(message)
        warnings.append(message)

    return SceneArray(
        path=path,
        filename=filename,
        array=arr,
        band_count=band_count,
        pixel_spacing_m=float(pixel_spacing_m or detection.pixel_spacing_m),
        imagery_kind=imagery_kind,
        warnings=warnings,
        acquisition_time_utc=acquisition_time_utc,
    )



def _sentinel_time_from_name(filename: str) -> Optional[str]:
    """Extract ``YYYYMMDDTHHMMSS`` from a Sentinel-1 product name (UTC ISO)."""
    match = _SENTINEL_TIME_RE.search(filename)
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}+00:00"


def _tiff_datetime_tag(path: Path) -> Optional[str]:
    """Best-effort acquisition stamp from GeoTIFF tags (``DateTime`` etc.)."""
    if tiff is None or path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        with tiff.TiffFile(str(path)) as handle:
            pages = getattr(handle, "pages", None)
            if not pages:
                return None
            tags = dict(pages[0].tags)
            for key in ("DateTime", "DATETIME", "ACQUISITION_DATETIME", "START_TIME"):
                if key in tags:
                    raw = str(tags[key].value).strip()
                    cleaned = raw.replace(":", "-", 2)
                    for candidate in (cleaned, cleaned.replace(" ", "T")):
                        if "T" in candidate:
                            return candidate if "+" in candidate else f"{candidate}+00:00"
    except Exception as exc:  # pragma: no cover - malformed TIFF tags
        logger.debug("Could not read TIFF datetime tag for %s: %s", path.name, exc)
    return None


def _pixel_spacing_meters(dataset: "rasterio.DatasetReader", center_lat: float) -> float:
    """Ground sample distance in metres derived from the affine transform."""
    transform = dataset.transform
    x_spacing = abs(float(transform.a))
    y_spacing = abs(float(transform.e))
    spacing = max(x_spacing, y_spacing)
    if dataset.crs is not None and dataset.crs.is_geographic:
        m_per_deg_lon, m_per_deg_lat = meters_per_degree(0.0, center_lat)
        spacing = max(x_spacing * m_per_deg_lon, y_spacing * m_per_deg_lat)
    return float(spacing) if spacing > 0 else 10.0


def read_geo_metadata(
    path: Path,
    *,
    pixel_spacing_m: Optional[float] = None,
) -> Dict[str, object]:
    """Read georeferencing metadata with ``rasterio`` (never fatal).

    Returns:
        Mapping with ``georeferenced``, ``crs``, ``transform``, ``bbox_wgs84``,
        ``center_lon``, ``center_lat``, ``pixel_spacing_m`` and ``warnings``.
        Non-georeferenced rasters (and rasterio failures) yield
        ``georeferenced=False`` plus an explanatory warning.
    """
    result: Dict[str, object] = {
        "georeferenced": False,
        "crs": None,
        "transform": None,
        "bbox_wgs84": None,
        "center_lon": None,
        "center_lat": None,
        "pixel_spacing_m": pixel_spacing_m,
        "warnings": [],
    }
    if rasterio is None:  # pragma: no cover - dependency guard
        result["warnings"] = ["rasterio unavailable; geometry will be pixel-space."]
        return result
    try:
        with rasterio.open(str(path)) as dataset:
            if dataset.crs is None or dataset.transform is None:
                result["warnings"] = [
                    "Scene carries no CRS; geometry reported in pixel space."
                ]
                return result
            transform = dataset.transform
            bounds = dataset.bounds
            crs = dataset.crs
            if str(crs).upper() == "EPSG:4326":
                bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            else:
                bbox = [
                    float(value)
                    for value in transform_bounds(
                        crs,
                        "EPSG:4326",
                        bounds.left,
                        bounds.bottom,
                        bounds.right,
                        bounds.top,
                    )
                ]
            center_lat = (bbox[1] + bbox[3]) / 2.0
            result.update(
                {
                    "georeferenced": True,
                    "crs": str(crs),
                    "transform": [
                        float(transform.a),
                        float(transform.b),
                        float(transform.c),
                        float(transform.d),
                        float(transform.e),
                        float(transform.f),
                    ],
                    "bbox_wgs84": [float(value) for value in bbox],
                    "center_lon": (bbox[0] + bbox[2]) / 2.0,
                    "center_lat": center_lat,
                    "pixel_spacing_m": pixel_spacing_m
                    or _pixel_spacing_meters(dataset, center_lat),
                }
            )
    except Exception as exc:  # unsupported by GDAL -> pixel space
        logger.debug("rasterio could not georeference %s: %s", path.name, exc)
        result["warnings"] = [
            "Scene carries no readable CRS/transform; geometry reported in pixel space."
        ]
    return result



def load_scene(
    path: str | Path,
    settings: AppSettings,
    *,
    pixel_spacing_m: Optional[float] = None,
    acquisition_time_utc: Optional[str] = None,
) -> SceneArray:
    """Load, validate and geolocate a scene from disk.

    Args:
        path: Scene path (absolute, or relative to the project root).
        settings: Resolved application settings.
        pixel_spacing_m: Ground sample distance override (metres).
        acquisition_time_utc: Acquisition timestamp override (ISO-8601 UTC).

    Returns:
        A validated :class:`SceneArray`.

    Raises:
        SceneNotFoundError: When the file is missing.
        UnsupportedImageryError: When the raster is unusable.
    """
    resolved = resolve_scene_path(str(path), settings)
    array = read_raster(resolved)

    geo = read_geo_metadata(resolved, pixel_spacing_m=pixel_spacing_m)
    geo_warnings = list(geo.get("warnings") or [])  # type: ignore[arg-type]

    stamp = (
        acquisition_time_utc
        or _sentinel_time_from_name(resolved.name)
        or _tiff_datetime_tag(resolved)
    )

    scene = scene_from_array(
        array,
        path=str(resolved),
        filename=resolved.name,
        settings=settings,
        pixel_spacing_m=geo.get("pixel_spacing_m") or pixel_spacing_m,  # type: ignore[arg-type]
        acquisition_time_utc=stamp,
    )
    scene.georeferenced = bool(geo.get("georeferenced"))
    scene.crs = geo.get("crs")  # type: ignore[assignment]
    scene.transform = geo.get("transform")  # type: ignore[assignment]
    scene.bbox_wgs84 = geo.get("bbox_wgs84")  # type: ignore[assignment]
    scene.center_lon = geo.get("center_lon")  # type: ignore[assignment]
    scene.center_lat = geo.get("center_lat")  # type: ignore[assignment]
    scene.warnings.extend(geo_warnings)
    return scene


def load_scene_from_bytes(
    data: bytes,
    filename: str,
    settings: AppSettings,
    *,
    pixel_spacing_m: Optional[float] = None,
    acquisition_time_utc: Optional[str] = None,
) -> SceneArray:
    """Load an uploaded scene held in memory (no temporary file required).

    Georeferencing is skipped for in-memory uploads (GDAL needs a file path);
    the caller can pass ``pixel_spacing_m`` when a known ground sample distance
    should be applied.

    Raises:
        UnsupportedImageryError: When the buffer cannot be decoded/validated.
    """
    if not data:
        raise UnsupportedImageryError("Uploaded file is empty.")
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedImageryError(
            f"Unsupported upload extension '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    buffer = np.frombuffer(data, dtype=np.uint8)
    try:
        if suffix in (".tif", ".tiff") and tiff is not None:
            array = tiff.imread(io.BytesIO(bytes(buffer)))
        else:
            array = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise UnsupportedImageryError(
            f"Could not decode uploaded raster {filename}: {exc}"
        ) from exc

    if array is None:
        raise UnsupportedImageryError(
            f"Could not decode uploaded raster {filename}."
        )

    stamp = (
        acquisition_time_utc
        or _sentinel_time_from_name(filename)
        or None
    )
    scene = scene_from_array(
        array,
        path=f"upload://{filename}",
        filename=filename,
        settings=settings,
        pixel_spacing_m=pixel_spacing_m,
        acquisition_time_utc=stamp,
    )
    # Preserve embedded GeoTIFF georeferencing for in-memory uploads so that
    # vectorised polygons are emitted in lon/lat (not pixel) space.
    if suffix in (".tif", ".tiff") and rasterio is not None:
        try:
            from rasterio.io import MemoryFile

            with MemoryFile(data) as mem:
                with mem.open() as ds:
                    if ds.crs is not None and ds.transform is not None:
                        scene.georeferenced = True
                        scene.crs = str(ds.crs)
                        scene.transform = [
                            float(ds.transform.a),
                            float(ds.transform.b),
                            float(ds.transform.c),
                            float(ds.transform.d),
                            float(ds.transform.e),
                            float(ds.transform.f),
                        ]
        except Exception:  # noqa: BLE001 - georeferencing is best-effort for uploads
            pass
    if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        scene.warnings.append(
            "Uploaded 8-bit quick-look raster: SAR dynamic range is quantised, "
            "so detections are less reliable than from a 32-bit SAR GeoTIFF."
        )
    return scene


def save_mask_png(mask: np.ndarray, destination: str | Path) -> str:
    """Persist a binary mask as an 8-bit PNG (255 = oil). Returns the path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), encoded):
        raise UnsupportedImageryError(f"Could not write mask PNG to {path}")
    return str(path)


def mask_preview_base64(mask: np.ndarray, max_edge: int = 512) -> str:
    """Encode a (downscaled) mask as a base64 PNG data payload for the frontend."""
    encoded = (np.asarray(mask) > 0).astype(np.uint8) * 255
    height, width = encoded.shape[:2]
    longest = max(height, width)
    if longest > max_edge > 0:
        scale = max_edge / float(longest)
        encoded = cv2.resize(
            encoded,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_NEAREST,
        )
    ok, buffer = cv2.imencode(".png", encoded)
    if not ok:  # pragma: no cover - OpenCV encoding failure
        raise UnsupportedImageryError("Could not encode the mask preview PNG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def downscale_array(array: np.ndarray, factor: int) -> np.ndarray:
    """Integer-downscale a raster (nearest-neighbour) preserving channel layout."""
    factor = int(factor)
    if factor <= 1:
        return array
    height, width = array.shape[:2]
    target = (max(1, width // factor), max(1, height // factor))
    resized = cv2.resize(array.astype(np.float32), target, interpolation=cv2.INTER_AREA)
    if array.ndim == 3 and resized.ndim == 2:
        resized = resized[:, :, None]
    return resized

