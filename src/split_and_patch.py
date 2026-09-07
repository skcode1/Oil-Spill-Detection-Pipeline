import os
import glob
import random
import numpy as np
import tifffile as tiff
from pathlib import Path


def create_split_directories(base_dir: str):
    """Creates directory tree for train, val, and test partitions."""
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'masks']:
            os.makedirs(os.path.join(base_dir, 'processed', split, sub), exist_ok=True)


def stratified_scene_split(
    raw_dir: str,
    classes: list = ["oil_spill", "lookalike", "clean"],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Splits 450 full-scene images (150 per class) into balanced 70/15/15 sub-sets.
    Expects raw structure: raw_dir/{class_name}/images/ and /masks/
    """
    random.seed(seed)
    split_manifest = {'train': [], 'val': [], 'test': []}

    for cls in classes:
        cls_img_dir = os.path.join(raw_dir, cls, 'images')
        cls_mask_dir = os.path.join(raw_dir, cls, 'masks')
        
        # Match image and mask files
        img_files = sorted(glob.glob(os.path.join(cls_img_dir, "*.tif*")))
        random.shuffle(img_files)
        
        n_total = len(img_files)
        if n_total == 0:
            print(f"[WARN] No scenes found for class '{cls}' in {cls_img_dir}")
            continue

        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        
        splits = {
            'train': img_files[:n_train],
            'val': img_files[n_train:n_train + n_val],
            'test': img_files[n_train + n_val:]
        }

        for split_name, files in splits.items():
            for img_p in files:
                fname = os.path.basename(img_p)
                mask_p = os.path.join(cls_mask_dir, fname)
                
                # If clean water, synthetic blank mask fallback if mask not provided
                if not os.path.exists(mask_p) and cls == "clean":
                    mask_p = None

                split_manifest[split_name].append({
                    "image_path": img_p,
                    "mask_path": mask_p,
                    "class_label": cls,
                    "scene_id": os.path.splitext(fname)[0]
                })

    print("\n--- Stratified Split Summary ---")
    for s in ['train', 'val', 'test']:
        print(f"  {s.upper():<5}: {len(split_manifest[s])} scenes")
    return split_manifest


def slice_into_patches(
    image_path: str,
    mask_path: str,
    output_dir: str,
    split: str,
    prefix: str,
    class_label: str,
    patch_size: int = 512,
    empty_keep_ratio: float = 0.15
) -> dict:
    """
    Slices a 2048x2048 scene into 16 patches (512x512).
    Optimizes patch selection by filtering redundant non-oil ocean water.
    """
    # 1. Read SAR 2-Channel Imagery (VV + VH)
    img = tiff.imread(image_path)
    
    # Standardize shape to (H, W, Channels)
    if img.ndim == 3 and img.shape[0] in [2, 3]:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # 2. Read or Generate Ground Truth Mask
    if mask_path and os.path.exists(mask_path):
        mask = tiff.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    else:
        # Blank mask for clean water control scenes
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    h, w, c = img.shape
    stats = {"kept": 0, "dropped": 0}

    # 3. Patch Extraction Grid (4 x 4 = 16 patches for 2048x2048)
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            img_patch = img[i:i + patch_size, j:j + patch_size, :]
            mask_patch = mask[i:i + patch_size, j:j + patch_size]

            is_positive = np.any(mask_patch > 0)

            # Filtering Strategy for Training Set:
            # - Always keep patches containing oil spills
            # - Sub-sample 85% of pure-water patches from spill/lookalike scenes
            # - Keep clean water patches at empty_keep_ratio to prevent false-alarm bias
            if split == 'train' and not is_positive:
                if random.random() > empty_keep_ratio:
                    stats["dropped"] += 1
                    continue

            # Deterministic naming containing row and column pixel offsets
            patch_name = f"{class_label}_{prefix}_r{i}_c{j}.tif"
            img_out = os.path.join(output_dir, 'processed', split, 'images', patch_name)
            msk_out = os.path.join(output_dir, 'processed', split, 'masks', patch_name)

            tiff.imwrite(img_out, img_patch.astype(np.float32), compression='zlib')
            tiff.imwrite(msk_out, (mask_patch > 0).astype(np.uint8), compression='zlib')
            stats["kept"] += 1

    return stats


def run_pipeline(raw_base_dir: str, output_base_dir: str):
    """Executes the end-to-end dataset creation pipeline."""
    create_split_directories(output_base_dir)
    manifest = stratified_scene_split(raw_base_dir)

    total_patches = {'train': 0, 'val': 0, 'test': 0}
    total_dropped = 0

    print("\n[*] Commencing 512x512 Patch Slicing...")
    for split_name, items in manifest.items():
        for item in items:
            stats = slice_into_patches(
                image_path=item["image_path"],
                mask_path=item["mask_path"],
                output_dir=output_base_dir,
                split=split_name,
                prefix=item["scene_id"],
                class_label=item["class_label"],
                patch_size=512,
                empty_keep_ratio=0.15
            )
            total_patches[split_name] += stats["kept"]
            total_dropped += stats["dropped"]

    print("\n" + "=" * 55)
    print(" PREPROCESSING & PATCHING COMPLETED")
    print("=" * 55)
    print(f"  Training Patches Retained   : {total_patches['train']} (Filtered {total_dropped} empty patches)")
    print(f"  Validation Patches Retained : {total_patches['val']}")
    print(f"  Holdout Test Patches        : {total_patches['test']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    # Specify paths matching your project structure
    RAW_DATA_DIR = "data/raw/SAR_training"
    PROCESSED_DIR = "data"
    
    if os.path.exists(RAW_DATA_DIR):
        run_pipeline(RAW_DATA_DIR, PROCESSED_DIR)
    else:
        print(f"[INFO] Script ready. Provide raw scene files at: {RAW_DATA_DIR}")