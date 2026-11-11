"""
Multi-Class SAR Oil-Spill Segmentation Dataset.

Produces training samples that feed the 3-class U-Net
(Background / Oil Spill / Lookalike) defined in ``config.yaml``.

Image encoding
--------------
SAR imagery is dual-polarisation (VV + VH).  The production model was
trained on **three** input channels:

    channel 0  --  VV  backscatter
    channel 1  --  VH  backscatter
    channel 2  --  VV + VH  composite

Each channel is independently min-max normalised to [0, 1] and then
standardised with configurable ``norm_mean / norm_std`` statistics.

Mask encoding
-------------
Masks carry integer class labels (NOT one-hot):

    0  --  Background / No Oil
    1  --  Oil Spill  (the target class at inference time)
    2  --  Lookalike  (wind shadow, algal bloom, etc.)

When a patched directory (e.g. ``data/processed/``) is used the dataset
also accepts legacy *binary* masks and promotes them to three classes
using the filename class prefix (``oil_spill_``, ``lookalike_``,
``clean_``).

SAR Augmentations
-----------------
Because ``albumentations`` is not a declared dependency, all augmentations
are implemented natively with NumPy / OpenCV.  They are **only** applied
when ``augment=True`` (i.e. during training) and are left disabled for
validation and inference so that results are deterministic.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
#  Class-index constants (single source of truth)
# --------------------------------------------------------------------------- #
CLASS_BACKGROUND: int = 0   # No Oil / Clean water
CLASS_OIL_SPILL: int = 1    # Confirmed oil spill
CLASS_LOOKALIKE: int = 2    # Lookalike features (wind shadow, biogenic film ...)
NUM_CLASSES: int = 3

CLASS_NAMES: List[str] = ["background", "oil_spill", "lookalike"]

_PREFIX_CLASS_MAP: Dict[str, int] = {
    "oil_spill": CLASS_OIL_SPILL,
    "lookalike": CLASS_LOOKALIKE,
    "clean": CLASS_BACKGROUND,
    "no oil": CLASS_BACKGROUND,
    "nooil": CLASS_BACKGROUND,
}


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalisation to [0, 1] with epsilon guard."""
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _sar_to_three_channel(image: np.ndarray) -> np.ndarray:
    """Convert a SAR image array to a 3-channel float32 array in [0, 1].

    Mirrors ``dataset._sar_to_three_channel`` so training and inference
    produce identical channel layouts:

    * 2-channel (VV, VH)  ->  [VV, VH, VV+VH]
    * 1-channel           ->  [VV, VV, VV]  (broadcast)
    * 3-channel (HWC)     ->  returned as-is (each normalised)
    """
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
    raise ValueError(f"Unsupported SAR image shape {image.shape}")


def _normalize_tensor(tensor: torch.Tensor, mean: Sequence[float],
                       std: Sequence[float]) -> torch.Tensor:
    """Channel-wise standardisation: (x - mean) / std."""
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


# --------------------------------------------------------------------------- #
#  SAR-specific augmentation primitives  (no external deps)
# --------------------------------------------------------------------------- #

def _add_speckle_noise(image: np.ndarray, var: float) -> np.ndarray:
    """Multiplicative Gamma speckle noise -- emulates SAR sensor noise.

    ``I_noisy = I * (1 + sqrt(var) * N(0,1))``  clipped to [0, 1].
    Speckle is multiplicative (not additive), respecting the non-negative
    backscatter range and varying with signal magnitude.
    """
    noise = np.random.normal(0.0, np.sqrt(var), size=image.shape).astype(np.float32)
    return np.clip(image * (1.0 + noise), 0.0, 1.0)


def _gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur simulating azimuth thermal smearing / antenna sidelobes."""
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _adjust_brightness_contrast(image: np.ndarray, brightness_limit: float,
                                contrast_limit: float) -> np.ndarray:
    """Random brightness / contrast shifts (incidence-angle drift simulation)."""
    alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    beta = random.uniform(-brightness_limit, brightness_limit)
    return np.clip(image * alpha + beta, 0.0, 1.0)


class SentinelDataset(Dataset):
    """3-class SAR segmentation dataset with production-grade augmentations.

    Parameters
    ----------
    img_dir : str
        Directory containing SAR image chips (.tif / .png / .jpg).
    mask_dir : str
        Directory containing paired mask chips (same filenames).
    size : tuple[int, int]
        Target (H, W) for resizing.  Masks use nearest-neighbour.
    num_classes : int
        Number of segmentation classes (default 3).
    augment : bool
        Whether to apply SAR augmentations (training only).
    augmentation_cfg : dict | None
        Override for the ``augmentation:`` config block.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        size: Tuple[int, int] = (256, 256),
        num_classes: int = NUM_CLASSES,
        augment: bool = False,
        augmentation_cfg: dict | None = None,
    ):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.num_classes = num_classes
        self.augment = augment
        self.aug_cfg: dict = augmentation_cfg or {}
        self.filenames: List[str] = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
        )
        if not self.filenames:
            raise FileNotFoundError(f"No supported image files in {img_dir}")

        # Pre-extract the class prefix from each filename (for legacy binary masks).
        self._prefixes: List[str] = []
        for fname in self.filenames:
            lower = fname.lower()
            matched = ""
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

        # Resize to target size (image with bilinear, mask with nearest)
        image_np = cv2.resize(image_np, self.size, interpolation=cv2.INTER_LINEAR)
        mask_np = cv2.resize(mask_np, self.size, interpolation=cv2.INTER_NEAREST)

        # Augment *before* standardisation (augmentations operate on [0, 1] range)
        if self.augment:
            image_np, mask_np = self._augment(image_np, mask_np)

        # Normalise image channels to [0, 1]
        norm = np.clip(image_np, 0.0, 1.0).astype(np.float32)

        mean = self.aug_cfg.get("norm_mean", [0.5, 0.5, 0.5])
        std = self.aug_cfg.get("norm_std", [0.5, 0.5, 0.5])
        if not isinstance(mean, (list, tuple)):
            mean = [float(mean)] * norm.shape[-1]
        if not isinstance(std, (list, tuple)):
            std = [float(std)] * norm.shape[-1]

        # HWC -> CHW
        chw = np.transpose(norm, (2, 0, 1))
        img_tensor = torch.from_numpy(chw)
        img_tensor = _normalize_tensor(img_tensor, list(mean), list(std))

        mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
        return img_tensor, mask_tensor

    def _read_image(self, path: str) -> np.ndarray:
        """Read a SAR image and return a 3-channel float32 array in [0, 1]."""
        arr = tiff.imread(path)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        return _sar_to_three_channel(arr)

    def _read_mask(self, img_name: str, idx: int) -> np.ndarray:
        """Read a mask and return a 3-class integer array (0/1/2).

        If the mask directory is missing or the mask file does not exist,
        the class is inferred from the filename prefix and a uniform label
        is generated.
        """
        mask_path = os.path.join(self.mask_dir, img_name)
        if os.path.exists(mask_path):
            arr = tiff.imread(mask_path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            arr = arr.astype(np.int64)
            # If the mask is already 3-class (contains label 2), use directly.
            if arr.max() >= 2:
                return arr
            # Legacy binary mask: 0 = background, 1 = oil.
            # Promote to 3-class using the filename class prefix.
            prefix_class = _PREFIX_CLASS_MAP.get(self._prefixes[idx], CLASS_BACKGROUND)
            if prefix_class == CLASS_OIL_SPILL:
                return arr
            elif prefix_class == CLASS_LOOKALIKE:
                # Lookalike image: all non-oil pixels are lookalike
                return np.where(arr > 0, CLASS_OIL_SPILL, CLASS_LOOKALIKE)
            else:
                # Clean image: everything is background
                return np.zeros_like(arr)
        else:
            prefix_class = _PREFIX_CLASS_MAP.get(self._prefixes[idx], CLASS_BACKGROUND)
            h, w = self.size
            return np.full((h, w), fill_value=prefix_class, dtype=np.int64)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply configurable SAR augmentations (image + mask aware)."""
        p = self.aug_cfg

        # 1. Speckle noise (multiplicative Gamma)
        if p.get("speckle_noise", True) and random.random() < 0.5:
            image = _add_speckle_noise(image, float(p.get("speckle_noise_var", 0.02)))

        # 2. Gaussian blur
        if p.get("gaussian_blur", True) and random.random() < 0.3:
            ksize = int(p.get("gaussian_blur_ksize", 3))
            image = _gaussian_blur(image, ksize)

        # 3. Brightness / contrast jitter
        if p.get("brightness_contrast", True) and random.random() < 0.5:
            image = _adjust_brightness_contrast(
                image,
                float(p.get("brightness_limit", 0.15)),
                float(p.get("contrast_limit", 0.2)),
            )

        # 4. Geometric flips (both image and mask)
        hflip_p = float(p.get("horizontal_flip_p", 0.5))
        vflip_p = float(p.get("vertical_flip_p", 0.5))
        if p.get("horizontal_flip", True) and random.random() < hflip_p:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if p.get("vertical_flip", True) and random.random() < vflip_p:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # 5. Random 90-degree rotations (SAR backscatter is rotation-equivariant)
        if random.random() < 0.3:
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k)
                mask = np.rot90(mask, k)
            image = np.ascontiguousarray(image)
            mask = np.ascontiguousarray(mask)

        return image, mask
