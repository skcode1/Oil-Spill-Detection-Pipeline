from __future__ import annotations
import glob
import os
import random
import numpy as np
import tifffile as tiff
CLASS_BACKGROUND = 0
CLASS_OIL_SPILL = 1
CLASS_LOOKALIKE = 2
_CLASS_DIR_NORMALISE: dict[str, str] = {'oil': 'oil_spill', 'lookalike': 'lookalike', 'no oil': 'clean', 'no_oil': 'clean'}

def create_split_directories(base_dir: str):
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'masks']:
            os.makedirs(os.path.join(base_dir, 'processed', split, sub), exist_ok=True)

def _normalise_class_name(name: str) -> str:
    key = name.lower().strip()
    return _CLASS_DIR_NORMALISE.get(key, key)

def _match_mask_filename(img_name: str, mask_dir: str, mask_files: list[str]) -> str | None:
    base = os.path.splitext(img_name)[0]
    candidates = [img_name, f'{base}_segmentation.tif', f'{base}_segmentation.png', f'{base}.tif', f'{base}.png']
    for cand in candidates:
        if cand in mask_files:
            return os.path.join(mask_dir, cand)
    return None

def stratified_scene_split(raw_dir: str, classes: list=None, train_ratio: float=0.7, val_ratio: float=0.15, seed: int=42) -> dict:
    if classes is None:
        classes = ['Oil', 'Lookalike', 'No oil']
    random.seed(seed)
    split_manifest = {'train': [], 'val': [], 'test': []}
    for cls in classes:
        img_dir = None
        mask_dir = None
        for layout in ((os.path.join(raw_dir, cls, sub) for sub in ['Images', 'image', 'images']), (os.path.join(raw_dir, sub, cls) for sub in ['Images', 'image', 'images'])):
            for candidate in layout:
                if os.path.isdir(candidate):
                    img_dir = candidate
                    break
            if img_dir:
                break
        for layout in ((os.path.join(raw_dir, cls, sub) for sub in ['Mask', 'label', 'masks']), (os.path.join(raw_dir, sub, cls) for sub in ['Mask', 'label', 'masks'])):
            for candidate in layout:
                if os.path.isdir(candidate):
                    mask_dir = candidate
                    break
            if mask_dir:
                break
        if img_dir is None:
            continue
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.tif*')))
        if not img_files:
            continue
        random.shuffle(img_files)
        normalised = _normalise_class_name(cls)
        mask_files = []
        if mask_dir:
            mask_files = [os.path.basename(f) for f in os.listdir(mask_dir)]
        n_total = len(img_files)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        splits = {'train': img_files[:n_train], 'val': img_files[n_train:n_train + n_val], 'test': img_files[n_train + n_val:]}
        for split_name, files in splits.items():
            for img_p in files:
                fname = os.path.basename(img_p)
                mask_p = _match_mask_filename(fname, mask_dir, mask_files) if mask_dir else None
                split_manifest[split_name].append({'image_path': img_p, 'mask_path': mask_p, 'class_label': normalised, 'scene_id': os.path.splitext(fname)[0]})
    print('\n--- Stratified Split Summary ---')
    for s in ['train', 'val', 'test']:
        print(f'  {s.upper():<5}: {len(split_manifest[s])} scenes')
    return split_manifest

def slice_into_patches(image_path: str, mask_path: str | None, output_dir: str, split: str, prefix: str, class_label: str, patch_size: int=512, empty_keep_ratio: float=0.15, num_classes: int=3) -> dict:
    img = tiff.imread(image_path)
    if img.ndim == 3 and img.shape[0] in [2, 3]:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if mask_path and os.path.exists(mask_path):
        mask = tiff.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int64)
    else:
        mask = np.zeros(img.shape[:2], dtype=np.int64)
    if num_classes == 3:
        if class_label == 'lookalike':
            mask = np.where(mask > 0, CLASS_OIL_SPILL, CLASS_LOOKALIKE)
        elif class_label == 'clean':
            mask = np.where(mask > 0, CLASS_OIL_SPILL, CLASS_BACKGROUND)
        else:
            mask = np.where(mask > 0, CLASS_OIL_SPILL, CLASS_BACKGROUND)
    h, w, c = img.shape
    stats = {'kept': 0, 'dropped': 0}
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            img_patch = img[i:i + patch_size, j:j + patch_size, :]
            mask_patch = mask[i:i + patch_size, j:j + patch_size]
            is_positive = np.any(mask_patch == CLASS_OIL_SPILL)
            if split == 'train' and (not is_positive):
                if random.random() > empty_keep_ratio:
                    stats['dropped'] += 1
                    continue
            patch_name = f'{class_label}_{prefix}_r{i}_c{j}.tif'
            img_out = os.path.join(output_dir, 'processed', split, 'images', patch_name)
            msk_out = os.path.join(output_dir, 'processed', split, 'masks', patch_name)
            tiff.imwrite(img_out, img_patch.astype(np.float32), compression='zlib')
            tiff.imwrite(msk_out, mask_patch.astype(np.uint8), compression='zlib')
            stats['kept'] += 1
    return stats

def run_pipeline(raw_base_dir: str, output_base_dir: str):
    create_split_directories(output_base_dir)
    manifest = stratified_scene_split(raw_base_dir)
    total_patches = {'train': 0, 'val': 0, 'test': 0}
    total_dropped = 0
    print('\n[*] Commencing 512x512 Patch Slicing (3-class labels)...')
    for split_name, items in manifest.items():
        for item in items:
            stats = slice_into_patches(image_path=item['image_path'], mask_path=item['mask_path'], output_dir=output_base_dir, split=split_name, prefix=item['scene_id'], class_label=item['class_label'], patch_size=512, empty_keep_ratio=0.15, num_classes=3)
            total_patches[split_name] += stats['kept']
            total_dropped += stats['dropped']
    print('\n' + '=' * 55)
    print(' PREPROCESSING & PATCHING COMPLETED (3-class labels)')
    print('=' * 55)
    print(f"  Training Patches Retained   : {total_patches['train']} (Filtered {total_dropped} empty patches)")
    print(f"  Validation Patches Retained : {total_patches['val']}")
    print(f"  Holdout Test Patches        : {total_patches['test']}")
    print('=' * 55 + '\n')
if __name__ == '__main__':
    RAW_DATA_DIR = 'data/raw/zenodo_part3'
    PROCESSED_DIR = 'data'
    if os.path.exists(RAW_DATA_DIR):
        run_pipeline(RAW_DATA_DIR, PROCESSED_DIR)
    else:
        print(f'[INFO] Script ready. Provide raw scene files at: {RAW_DATA_DIR}')