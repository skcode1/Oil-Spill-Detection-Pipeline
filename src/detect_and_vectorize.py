"""
3-Class SAR Oil-Spill Segmentation: Inference & Vectorization.

Loads the production 3-class U-Net (Background / Oil Spill / Lookalike)
checkpoint, runs sliding-window inference on a raw SAR GeoTIFF, extracts
the **oil-spill** channel (class 1), and vectorises it into polygons with
full geometric characterisation (area, perimeter, length, width, elongation).

The lookalike channel (class 2) is **never** emitted as a spill polygon —
lookalike filtering happens at the pixel level inside the model, exactly as
specified by the production pipeline strategy.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import segmentation_models_pytorch as smp
import tifffile as tiff
import torch
from shapely.geometry import Polygon, mapping

# rasterio is optional — only needed for georeferenced outputs
try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import yaml

# --------------------------------------------------------------------------- #
#  Class-index constants (must match dataset.py)
# --------------------------------------------------------------------------- #
CLASS_BACKGROUND = 0
CLASS_OIL_SPILL = 1
CLASS_LOOKALIKE = 2
NUM_CLASSES = 3
CLASS_NAMES = ["background", "oil_spill", "lookalike"]


# --------------------------------------------------------------------------- #
#  1. MODEL INITIALIZATION & INFERENCE
# --------------------------------------------------------------------------- #
def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalisation to [0, 1] with epsilon guard."""
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _scene_normalise(image: np.ndarray) -> np.ndarray:
    """Normalise a full scene (HWC) to [0, 1] per channel."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return _minmax_norm(image)[..., None].repeat(3, axis=-1)
    c = image.shape[-1]
    if c == 2:
        vv = _minmax_norm(image[:, :, 0])
        vh = _minmax_norm(image[:, :, 1])
        comp = _minmax_norm(image[:, :, 0] + image[:, :, 1])
        return np.stack([vv, vh, comp], axis=-1)
    if c == 3:
        return np.stack([_minmax_norm(image[:, :, i]) for i in range(3)], axis=-1)
    raise ValueError(f"Unsupported SAR image shape {image.shape}")


def sar_to_three_channel(image: np.ndarray, *, normalisation: str = "per_tile") -> np.ndarray:
    """Convert a SAR image array to a 3-channel float32 array in [0, 1].

    Mirrors ``dataset._sar_to_three_channel`` so training and inference
    produce identical channel layouts:

    * 2-channel (VV, VH)  ->  [VV, VH, VV+VH]
    * 1-channel           ->  [VV, VV, VV]  (broadcast)
    * 3-channel (HWC)     ->  returned as-is (each normalised)

    Args:
        image: Input SAR array (HxW, HxWx2, or HxWx3).
        normalisation: "per_tile" (default) normalises each tile independently,
            matching the training chip distribution. "scene" assumes the caller
            has already normalised the full scene; this function then only
            rearranges channels without re-normalising.
    """
    image = np.asarray(image, dtype=np.float32)
    if normalisation == "scene":
        # Caller already normalised the full scene; just ensure 3 channels.
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        if image.ndim == 3:
            c = image.shape[-1]
            if c == 2:
                vv, vh = image[:, :, 0], image[:, :, 1]
                comp = vv + vh
                return np.stack([vv, vh, comp], axis=-1)
            if c == 3:
                return image
        raise ValueError(f"Unsupported SAR image shape {image.shape}")

    # per_tile: normalise each tile independently (original behaviour)
    if image.ndim == 2:
        channel = _minmax_norm(image)
        return np.stack([channel, channel, channel], axis=-1)

    if image.ndim == 3:
        c = image.shape[-1]
        if c == 2:
            vv = _minmax_norm(image[:, :, 0])
            vh = _minmax_norm(image[:, :, 1])
            comp = _minmax_norm(image[:, :, 0] + image[:, :, 1])
            return np.stack([vv, vh, comp], axis=-1)
        if c == 3:
            channels = [_minmax_norm(image[:, :, i]) for i in range(3)]
            return np.stack(channels, axis=-1)
    raise ValueError(f"Unsupported SAR image shape {image.shape}")


def _load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml; absence degrades to built-in defaults."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def load_trained_model(
    model_path: str,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    in_channels: int = 3,
    encoder_name: str = "resnet34",
) -> torch.nn.Module:
    """Initialise the 3-class ResNet34 U-Net and load weights.

    The checkpoint may be either:
    * A plain ``model.state_dict()`` (legacy / externally trained), or
    * A rich dict with ``"state_dict"`` and ``"config"`` keys (new format).

    When the rich format is detected the architecture parameters are
    honoured from the metadata, ensuring forward compatibility.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Unpack rich checkpoint if present
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("config", {})
        encoder_name = str(meta.get("encoder", encoder_name))
        in_channels = int(meta.get("in_channels", in_channels))
        num_classes = int(meta.get("classes", num_classes))
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # plain state_dict

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )

    # Load strictly: lenient loading silently leaves freshly initialised weights
    # in place when the checkpoint and runtime architectures drift, which is one
    # of the ways a "loaded" model can ship predicting nothing (see
    # tests/test_model_sanity.py).  Mismatches now fail loudly with the
    # architecture that was expected.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {model_path} does not match the runtime architecture "
            f"(encoder={encoder_name}, in_channels={in_channels}, "
            f"classes={num_classes}): {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    input_size: Tuple[int, int] = (256, 256),
    oil_class_index: int = CLASS_OIL_SPILL,
    confidence_threshold: float = 0.50,
    tile_size: int = 256,
    stride: int = 128,
    normalisation: str = "per_tile",
) -> Tuple[np.ndarray, Tuple[int, int], Any, Dict[str, Any]]:
    """Run 3-class inference and return the oil-spill binary mask.

    Returns
    -------
    oil_mask : np.ndarray
        Binary uint8 mask (0 = non-oil, 1 = oil spill).  Lookalike
        pixels are explicitly excluded.
    orig_size : (int, int)
        (width, height) of the original image.
    geo_meta : tuple
        (transform, crs) for georeferenced outputs, or (None, None).
    diagnostics : dict
        Per-class probability maps and the full argmax label map for
        downstream geometry characterisation.
    """
    # ---- Read raw SAR image ----
    arr = tiff.imread(image_path)

    # Normalisation mode:
    # - "per_tile": each tile is min-max normalised independently (matches training chips)
    # - "scene": whole scene is min-max normalised once, then tiled
    if normalisation == "scene":
        arr = _scene_normalise(arr)

    orig_h, orig_w = (arr.shape[0], arr.shape[1])
    orig_size = (orig_w, orig_h)

    # ---- Georeferencing metadata ----
    transform_meta = None
    crs_meta = None
    if HAS_RASTERIO and image_path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(image_path) as src:
                transform_meta = src.transform
                crs_meta = src.crs
        except Exception:
            pass

    # ---- Sliding-window inference for large scenes ----
    h, w = arr.shape[:2]
    tile_h = min(tile_size, h)
    tile_w = min(tile_size, w)
    stride_xy = min(stride, tile_h, tile_w)

    # Accumulate probability maps
    prob_sum = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    for row in range(0, h, stride_xy):
        for col in range(0, w, stride_xy):
            end_row = min(row + tile_h, h)
            end_col = min(col + tile_w, w)
            # Channel build + normalisation according to mode
            patch = sar_to_three_channel(arr[row:end_row, col:end_col], normalisation=normalisation)
            patch_h, patch_w = patch.shape[:2]

            if patch_h < tile_h or patch_w < tile_w:
                # Pad to tile_size
                patch_padded = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
                patch_padded[:patch_h, :patch_w] = patch
            else:
                patch_padded = patch

            # BGR → CHW, standardise
            chw = np.transpose(patch_padded, (2, 0, 1))
            tensor = torch.from_numpy(chw).float().unsqueeze(0).to(device)
            mean_t = torch.tensor(norm_mean, device=device).view(1, 3, 1, 1)
            std_t = torch.tensor(norm_std, device=device).view(1, 3, 1, 1)
            tensor = (tensor - mean_t) / std_t

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, th, tw)

            if patch_h < tile_h or patch_w < tile_w:
                probs = probs[:, :patch_h, :patch_w]

            prob_sum[:, row:end_row, col:end_col] += probs
            count_map[row:end_row, col:end_col] += 1.0

    # Avoid division by zero
    count_map = np.maximum(count_map, 1.0)
    prob_avg = prob_sum / count_map[None, :, :]  # (C, H, W)

    # ---- Softmax + argmax → 3-class label map ----
    label_map = np.argmax(prob_avg, axis=0).astype(np.int64)

    # ---- Extract oil-spill channel only ----
    oil_confidence = prob_avg[oil_class_index]
    oil_mask = ((label_map == oil_class_index) &
                (oil_confidence >= confidence_threshold)).astype(np.uint8)

    # Remove small speckles
    if oil_mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
        oil_mask = cv2.morphologyEx(oil_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    diagnostics = {
        "prob_maps": prob_avg,       # (C, H, W) float32
        "label_map": label_map,       # (H, W) int64
        "oil_confidence": oil_confidence,  # (H, W) float32
    }

    return oil_mask, orig_size, (transform_meta, crs_meta), diagnostics


# --------------------------------------------------------------------------- #
#  2. VECTORIZATION & GEOMETRY CHARACTERISATION
# --------------------------------------------------------------------------- #
def characterize_geometry(binary_mask: np.ndarray) -> List[Dict[str, Any]]:
    """Characterise each connected oil spill contour.

    Uses ``cv2.findContours`` to compute:
    * area            (pixel²)   — cv2.contourArea
    * perimeter       (pixel)      — cv2.arcLength
    * length          (pixel)      — major axis of min-area rectangle
    * width           (pixel)      — minor axis
    * elongation      (aspect ratio) — length / width
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    results: List[Dict[str, Any]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 1.0:
            continue
        perimeter = float(cv2.arcLength(cnt, closed=True))
        # Minimum-area bounding rectangle gives length/width
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        length = max(w, h)
        width_min = min(w, h)
        elongation = float(length / width_min) if width_min > 0 else float("inf")

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = cy = 0.0

        results.append({
            "contour": cnt,
            "area_px": area,
            "perimeter_px": perimeter,
            "length_px": float(length),
            "width_px": float(width_min),
            "elongation": elongation,
            "centroid": (float(cx), float(cy)),
        })

    # Sort by area descending (largest spill first)
    results.sort(key=lambda r: r["area_px"], reverse=True)
    return results


def mask_to_polygons(
    binary_mask: np.ndarray,
    geo_meta: Tuple[Any, Any] = (None, None),
    diagnostics: Optional[Dict[str, Any]] = None,
    min_area_px: int = 50,
    crs_output: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Extract geometric boundaries from the oil-spill binary mask.

    For GeoTIFF inputs the raster affine transform converts pixel coordinates
    to geographic (lon, lat).  Falls back to pixel-space coordinates when
    no georeferencing is available.
    """
    transform, crs = geo_meta
    georeferenced = transform is not None and crs is not None

    geometries: List[Dict[str, Any]] = []
    characteristics = characterize_geometry(binary_mask)

    # Build a lookup from contour to properties
    for props in characteristics:
        if props["area_px"] < min_area_px:
            continue

        cnt = props["contour"]
        # Approximate contour for cleaner polygons
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Build polygon coordinates
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            points = cnt.reshape(-1, 2)
        if len(points) < 3:
            continue

        if georeferenced:
            # Convert pixel (col, row) → (lon, lat)
            coords = []
            for col, row in points:
                lon, lat = rasterio_xy(transform, float(row), float(col))
                coords.append((lon, lat))
        else:
            # Pixel-space coordinates (col, row) → (x, y)
            coords = [(float(c), float(r)) for c, r in points]

        # Close the ring
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        geom = Polygon(coords)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or geom.area < 1e-10:
            continue

        properties: Dict[str, Any] = {
            "id": len(geometries) + 1,
            "pixel_area": round(props["area_px"], 2),
            "perimeter_px": round(props["perimeter_px"], 2),
            "length_px": round(props["length_px"], 2),
            "width_px": round(props["width_px"], 2),
            "elongation": round(props["elongation"], 3),
            "centroid_px": [round(props["centroid"][0], 2),
                            round(props["centroid"][1], 2)],
        }

        if georeferenced:
            properties["centroid_lon"] = round(coords[0][0], 6) if len(coords) > 0 else None

        geometries.append({"geometry": mapping(geom), "properties": properties})

    if not geometries:
        return gpd.GeoDataFrame(
            columns=["geometry", "pixel_area", "perimeter_px", "length_px",
                      "width_px", "elongation", "centroid_px"],
            geometry="geometry",
            crs=crs_output if not georeferenced else crs,
        )

    gdf = gpd.GeoDataFrame.from_features(
        geometries, crs=crs if georeferenced else crs_output
    )


    # Ensure expected columns exist and reorder
    expected_cols = ["geometry", "pixel_area", "perimeter_px", "length_px",
                     "width_px", "elongation", "centroid_px"]
    if georeferenced:
        expected_cols.append("centroid_lon")
    for col in expected_cols:
        if col not in gdf.columns:
            gdf[col] = None
    gdf = gdf[["geometry"] + [c for c in expected_cols if c != "geometry"]]

    return gdf


# --------------------------------------------------------------------------- #
#  3. CLI ENTRY POINT
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="3-Class SAR Oil-Spill Detection & Vectorization (Inference)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input SAR image (.tif/.tiff) or directory")
    parser.add_argument("--model", type=str, default=None,
                        help="3-class model checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for exported vectors and masks")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Oil-class confidence threshold")
    parser.add_argument("--min_area", type=int, default=None,
                        help="Minimum spill area in pixels")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Configuration YAML providing defaults")
    parser.add_argument("--save_mask", action="store_true",
                        help="Save predicted binary PNG alongside vectors")
    parser.add_argument("--classes", type=int, default=3,
                        help="Number of output classes (default 3)")
    args = parser.parse_args()

    # --- Resolve config ---
    config = _load_config(args.config)
    paths_cfg = config.get("paths") or {}
    model_cfg = config.get("model") or {}
    inference_cfg = config.get("inference") or {}
    # Anchor relative CLI defaults at the project root (the directory holding
    # config.yaml / models / outputs) instead of the caller's CWD, so the CLI
    # behaves identically no matter where it is invoked from. ``os.getcwd()``
    # is only the last resort when no config path is available at all (e.g.
    # ``--config ""``); the packaged CLI always passes a real config path.
    config_dir = os.path.dirname(args.config) if args.config else ""
    project_root = os.path.abspath(config_dir) if config_dir else os.getcwd()

    model_path = args.model or paths_cfg.get(
        "model_checkpoint",
        os.path.join(project_root, "models", "oil_spill_segmentation_model.pth"),
    )
    output_dir = args.output_dir or os.path.join(project_root, "outputs")
    threshold = args.threshold if args.threshold is not None else float(
        model_cfg.get("confidence_threshold", 0.50)
    )
    min_area = args.min_area if args.min_area is not None else int(
        inference_cfg.get("min_spill_area_pixels", 50)
    )
    input_size = tuple(model_cfg.get("input_size", (256, 256)))
    tile_size = int(inference_cfg.get("tile_size", input_size[0]))
    stride = int(inference_cfg.get("stride", tile_size // 2))
    oil_class_idx = int(model_cfg.get("oil_class_index", CLASS_OIL_SPILL))
    in_channels = int(model_cfg.get("in_channels", 3))
    num_classes = int(model_cfg.get("classes", 3))
    crs_output = inference_cfg.get("crs", "EPSG:4326")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Loading 3-class model on {device}...")
    print(f"    Architecture: {model_cfg.get('encoder', 'resnet34')} U-Net "
          f"| in_channels={in_channels} | classes={num_classes} "
          f"| class_names={model_cfg.get('class_names', CLASS_NAMES)}")
    print(f"    Model: {model_path}")
    model = load_trained_model(
        model_path, device,
        num_classes=num_classes,
        in_channels=in_channels,
        encoder_name=model_cfg.get("encoder", "resnet34"),
    )

    # --- Collect target files ---
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if os.path.isdir(args.input):
        image_paths = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(valid_exts)
        )
    elif os.path.isfile(args.input) and args.input.lower().endswith(valid_exts):
        image_paths = [args.input]
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    print(f"[*] Processing {len(image_paths)} target(s) for oil-spill detection...")

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n  >> {os.path.basename(img_path)}")
        print(f"     classes in model output: {CLASS_NAMES}")

        # --- 1. Prediction (3-class softmax + argmax) ---
        oil_mask, orig_size, geo_meta, diag = predict_mask(
            model=model,
            image_path=img_path,
            device=device,
            input_size=input_size,
            oil_class_index=oil_class_idx,
            confidence_threshold=threshold,
            tile_size=tile_size,
            stride=stride,
        )

        # Report per-class pixel counts
        label_map = diag["label_map"]
        for ci, cname in enumerate(CLASS_NAMES):
            count = int((label_map == ci).sum())
            print(f"     class {ci} ({cname}): {count} px "
                  f"({100*count/(label_map.size or 1):.1f}%)")

        if oil_mask.sum() == 0:
            print("     [No oil spill pixels detected] — lookalikes filtered at pixel level")

        # --- 2. Geometry characterisation ---
        geoms = characterize_geometry(oil_mask)
        print(f"     Detected {len(geoms)} spill contour(s) "
              f"(min_area={min_area}px)")

        # --- 3. Vectorise ---
        gdf = mask_to_polygons(
            oil_mask, geo_meta=geo_meta, diagnostics=diag,
            min_area_px=min_area, crs_output=crs_output,
        )

        if len(gdf) > 0:
            total_area = float(gdf["pixel_area"].sum())
            print(f"     Total spill area: {total_area:.1f} px² | "
                  f"{len(gdf)} polygon(s)")

        # --- 4. Export GeoJSON ---
        out_geojson = os.path.join(output_dir, f"{base_name}_detected.geojson")
        gdf.to_file(out_geojson, driver="GeoJSON")

        # --- 5. Optional mask export ---
        if args.save_mask:
            out_mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(out_mask_path, oil_mask * 255)

        print(f"     --> {out_geojson}" +
              (f" + {base_name}_mask.png" if args.save_mask else ""))

    print(f"\n[*] All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
