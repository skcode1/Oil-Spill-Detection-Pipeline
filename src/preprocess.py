

# NOT NECESSARY AS THE DATASET ALREADY CONTAINS PRE-CROPPED 256 X 256 IMAGE CHIPS. 
# STILL FOR SAFETY :) BECAUSE MY MIND IS FAAAAHHHH:)


import os
import glob
import cv2
import yaml
import numpy as np
from PIL import Image

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def normalize_sar(image):
    """Normalize SAR backscatter values to [0, 1] range."""
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return image

def preprocess_scene(image_path, mask_path=None, tile_size=256, stride=128, output_dir="data/processed"):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    if mask_path:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[!] Failed to read image: {image_path}")
        return

    # Apply Lee/Median filter for SAR speckle noise reduction
    img = cv2.medianBlur(img, 3)
    norm_img = normalize_sar(img)

    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)

    h, w = img.shape
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    count = 0

    # Sliding window tiling
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            img_tile = norm_img[y:y + tile_size, x:x + tile_size]
            tile_save_path = os.path.join(output_dir, "images", f"{base_name}_tile_{count}.png")
            Image.fromarray((img_tile * 255).astype(np.uint8)).save(tile_save_path)

            if mask is not None:
                mask_tile = mask[y:y + tile_size, x:x + tile_size]
                mask_save_path = os.path.join(output_dir, "masks", f"{base_name}_tile_{count}.png")
                Image.fromarray(mask_tile * 255).save(mask_save_path)

            count += 1

    print(f"[*] Processed {base_name}: Generated {count} tiles ({tile_size}x{tile_size}).")

def run_preprocessing():
    cfg = load_config()
    raw_dir = cfg.get("paths", {}).get("raw_sar_dir", "data/raw/SAR_training/sentinel")
    img_dir = os.path.join(raw_dir, "image")
    mask_dir = os.path.join(raw_dir, "label")
    output_dir = "data/processed"

    image_files = glob.glob(os.path.join(img_dir, "*.*"))
    if not image_files:
        print(f"[!] No raw SAR images found in {img_dir}")
        return

    print(f"[*] Found {len(image_files)} SAR scenes to preprocess...")
    for img_path in image_files:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename) if os.path.exists(mask_dir) else None
        preprocess_scene(img_path, mask_path, tile_size=256, stride=128, output_dir=output_dir)

    print(f"[Done] Preprocessing complete. Dataset saved in: {output_dir}")

if __name__ == "__main__":
    run_preprocessing()