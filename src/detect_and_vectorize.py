import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
import geopandas as gpd
from shapely.geometry import Polygon
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# Attempt importing rasterio for georeferenced coordinate mapping
try:
    import rasterio
    import rasterio.features
    from rasterio.transform import xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ==========================================
# 1. MODEL INITIALIZATION & INFERENCE
# ==========================================
def load_trained_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Initializes the ResNet34 U-Net architecture and loads weights."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    input_size=(256, 256),
    threshold=0.5
) -> tuple[np.ndarray, tuple[int, int], object]:
    """
    Runs single-image inference and returns a binary mask matching original image dimensions.
    """
    # Open image (grayscale / 1-channel for SAR)
    pil_img = Image.open(image_path).convert("L")
    orig_w, orig_h = pil_img.size

    # Check for georeferencing metadata if rasterio is available
    transform_meta = None
    crs_meta = None
    if HAS_RASTERIO and image_path.lower().endswith((".tif", ".tiff")):
        try:
            with rasterio.open(image_path) as src:
                transform_meta = src.transform
                crs_meta = src.crs
        except Exception:
            pass

    # Preprocessing
    resized_img = TF.resize(pil_img, input_size)
    tensor_img = TF.to_tensor(resized_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float().squeeze().cpu().numpy()

    # Resize binary mask back to original resolution
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    full_mask = cv2.resize(
        pred_mask_uint8,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    return (full_mask > 127).astype(np.uint8), (orig_w, orig_h), (transform_meta, crs_meta)


# ==========================================
# 2. VECTORIZATION PIPELINE
# ==========================================
def mask_to_polygons(
    binary_mask: np.ndarray,
    geo_meta: tuple = (None, None),
    min_area_px: int = 25
) -> gpd.GeoDataFrame:
    """
    Extracts geometric boundaries from the binary mask and returns a GeoDataFrame.
    """
    transform, crs = geo_meta
    polygons = []
    properties = []

    # Find contours using OpenCV
    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for idx, cnt in enumerate(contours):
        # Filter small noise / speckles
        area = cv2.contourArea(cnt)
        if area < min_area_px or len(cnt) < 3:
            continue

        coords = cnt.squeeze(axis=1)
        if len(coords.shape) != 2 or coords.shape[0] < 3:
            continue

        # Convert to geospatial coordinates if transform exists
        if transform is not None:
            geo_coords = [transform * (pt[0], pt[1]) for pt in coords]
            poly = Polygon(geo_coords)
        else:
            poly = Polygon(coords)

        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
            properties.append({"id": idx + 1, "pixel_area": float(area)})

    if not polygons:
        gdf = gpd.GeoDataFrame(columns=["id", "pixel_area", "geometry"], geometry="geometry")
    else:
        gdf = gpd.GeoDataFrame(properties, geometry=polygons)

    if crs is not None:
        gdf.set_crs(crs, inplace=True)

    return gdf


# ==========================================
# 3. CLI & BATCH PROCESSING
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="SAR Feature Detection & Vectorization Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--model", type=str, default=os.path.join(os.getcwd(), "models", "sar_unet_best.pth"), help="Trained weights path")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "outputs"), help="Directory for exported vectors and masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--min_area", type=int, default=30, help="Minimum pixel area threshold to filter speckle noise")
    parser.add_argument("--save_mask", action="store_true", help="Save predicted binary PNG alongside vector files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Loading model on {device}...")
    model = load_trained_model(args.model, device)

    # Collect target files
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith(valid_exts)
        ]
    elif os.path.isfile(args.input) and args.input.lower().endswith(valid_exts):
        image_paths = [args.input]
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    print(f"[*] Processing {len(image_paths)} target(s)...")

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. Prediction
        binary_mask, _, geo_meta = predict_mask(
            model=model,
            image_path=img_path,
            device=device,
            threshold=args.threshold
        )

        # 2. Vectorization
        gdf = mask_to_polygons(binary_mask, geo_meta=geo_meta, min_area_px=args.min_area)

        # 3. Export Vector (GeoJSON)
        out_geojson = os.path.join(args.output_dir, f"{base_name}_detected.geojson")
        gdf.to_file(out_geojson, driver="GeoJSON")

        # 4. Optional Mask Export
        if args.save_mask:
            out_mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
            cv2.imwrite(out_mask_path, binary_mask * 255)

        print(f"--> [Done] {base_name}: {len(gdf)} polygon(s) exported to {out_geojson}")

    print(f"\nAll operations finished. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()