from __future__ import annotations
import os
import random
from typing import Dict, List, Sequence, Tuple
import cv2
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
CLASS_BACKGROUND: int = 0
CLASS_OIL_SPILL: int = 1
CLASS_LOOKALIKE: int = 2
NUM_CLASSES: int = 3
CLASS_NAMES: List[str] = ['background', 'oil_spill', 'lookalike']
_PREFIX_CLASS_MAP: Dict[str, int] = {'oil_spill': CLASS_OIL_SPILL, 'lookalike': CLASS_LOOKALIKE, 'clean': CLASS_BACKGROUND, 'no oil': CLASS_BACKGROUND, 'nooil': CLASS_BACKGROUND}

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = (float(arr.min()), float(arr.max()))
    if mx - mn < 1e-08:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _sar_to_three_channel(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
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
    raise ValueError(f'Unsupported SAR image shape {image.shape}')

def _normalize_tensor(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_t) / std_t

def _add_speckle_noise(image: np.ndarray, var: float) -> np.ndarray:
    noise = np.random.normal(0.0, np.sqrt(var), size=image.shape).astype(np.float32)
    return np.clip(image * (1.0 + noise), 0.0, 1.0)

def _gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def _adjust_brightness_contrast(image: np.ndarray, brightness_limit: float, contrast_limit: float) -> np.ndarray:
    alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    beta = random.uniform(-brightness_limit, brightness_limit)
    return np.clip(image * alpha + beta, 0.0, 1.0)

class SentinelDataset(Dataset):

    def __init__(self, img_dir: str, mask_dir: str, size: Tuple[int, int]=(256, 256), num_classes: int=NUM_CLASSES, augment: bool=False, augmentation_cfg: dict | None=None):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'Image directory not found: {img_dir}')
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.num_classes = num_classes
        self.augment = augment
        self.aug_cfg: dict = augmentation_cfg or {}
        self.filenames: List[str] = sorted((f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))))
        if not self.filenames:
            raise FileNotFoundError(f'No supported image files in {img_dir}')
        self._prefixes: List[str] = []
        for fname in self.filenames:
            lower = fname.lower()
            matched = ''
            for prefix in sorted(_PREFIX_CLASS_MAP, key=len, reverse=True):
                if prefix in lower:
                    matched = prefix
                    break
            self._prefixes.append(matched)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image_np = self._read_image(img_path)
        mask_np = self._read_mask(img_name, idx)
        image_np = cv2.resize(image_np, self.size, interpolation=cv2.INTER_LINEAR)
        mask_np = cv2.resize(mask_np, self.size, interpolation=cv2.INTER_NEAREST)
        if self.augment:
            image_np, mask_np = self._augment(image_np, mask_np)
        norm = np.clip(image_np, 0.0, 1.0).astype(np.float32)
        mean = self.aug_cfg.get('norm_mean', [0.5, 0.5, 0.5])
        std = self.aug_cfg.get('norm_std', [0.5, 0.5, 0.5])
        if not isinstance(mean, (list, tuple)):
            mean = [float(mean)] * norm.shape[-1]
        if not isinstance(std, (list, tuple)):
            std = [float(std)] * norm.shape[-1]
        chw = np.transpose(norm, (2, 0, 1))
        img_tensor = torch.from_numpy(chw)
        img_tensor = _normalize_tensor(img_tensor, list(mean), list(std))
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
        return (img_tensor, mask_tensor)

    def _read_image(self, path: str) -> np.ndarray:
        arr = tiff.imread(path)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        return _sar_to_three_channel(arr)

    def _read_mask(self, img_name: str, idx: int) -> np.ndarray:
        mask_path = os.path.join(self.mask_dir, img_name)
        if os.path.exists(mask_path):
            arr = tiff.imread(mask_path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            arr = arr.astype(np.int64)
            if arr.max() >= 2:
                return arr
            prefix_class = _PREFIX_CLASS_MAP.get(self._prefixes[idx], CLASS_BACKGROUND)
            if prefix_class == CLASS_OIL_SPILL:
                return arr
            elif prefix_class == CLASS_LOOKALIKE:
                return np.where(arr > 0, CLASS_OIL_SPILL, CLASS_LOOKALIKE)
            else:
                return np.zeros_like(arr)
        else:
            prefix_class = _PREFIX_CLASS_MAP.get(self._prefixes[idx], CLASS_BACKGROUND)
            h, w = self.size
            return np.full((h, w), fill_value=prefix_class, dtype=np.int64)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.aug_cfg
        if p.get('speckle_noise', True) and random.random() < 0.5:
            image = _add_speckle_noise(image, float(p.get('speckle_noise_var', 0.02)))
        if p.get('gaussian_blur', True) and random.random() < 0.3:
            ksize = int(p.get('gaussian_blur_ksize', 3))
            image = _gaussian_blur(image, ksize)
        if p.get('brightness_contrast', True) and random.random() < 0.5:
            image = _adjust_brightness_contrast(image, float(p.get('brightness_limit', 0.15)), float(p.get('contrast_limit', 0.2)))
        hflip_p = float(p.get('horizontal_flip_p', 0.5))
        vflip_p = float(p.get('vertical_flip_p', 0.5))
        if p.get('horizontal_flip', True) and random.random() < hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if p.get('vertical_flip', True) and random.random() < vflip_p:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        if random.random() < 0.3:
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)
        return (image, mask)