"""5-class SAR segmentation training entry point (DeepLabV3+ / ResNet-34).

Trains the Stage-1 detector on the Krestenitis et al. SAR dataset taxonomy:
0 Sea Surface, 1 Oil Spill, 2 Look-Alike, 3 Ship, 4 Land.

Uses segmentation_models_pytorch DeepLabV3Plus with a ResNet-34 encoder and a
hybrid Cross-Entropy + Dice loss to handle severe class imbalance, tracking
per-class IoU so Class 1 (Oil Spill) converges cleanly before the checkpoint is
promoted.

All paths and hyperparameters are resolved from ``config.yaml`` (sections:
``paths``, ``model``, ``training``, ``augmentation``) with CLI overrides.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

# 5-Class SAR Oil Spill Taxonomy (Krestenitis et al. / M4D Sentinel-1 standard)
CLASS_NAMES: Dict[int, str] = {
    0: "Sea Surface",
    1: "Oil Spill",
    2: "Look-Alike",
    3: "Ship",
    4: "Land",
}
NUM_CLASSES = 5


def _load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class SAR5ClassDataset(Dataset):
    """Paired SAR GeoTIFF tiles and 5-class integer masks (values 0..4).

    Directory layout::

        root/images/*.tif
        root/masks/*.tif   (or .png)
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_paths = sorted(Path(image_dir).glob("*.tif"))
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        if not self.image_paths:
            raise FileNotFoundError(f"No .tif scenes found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_sar_tile(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        if data.shape[0] == 1:
            vv = data[0]
            data = np.stack([vv, vv, vv], axis=0)
        elif data.shape[0] == 2:
            vv, vh = data[0], data[1]
            ratio = np.divide(vv, vh + 1e-6)
            data = np.stack([vv, vh, ratio], axis=0)
        else:
            data = data[:3]
        # Normalise SAR backscatter to [0, 1] per channel
        if data.max() > 1.0 or data.min() < 0.0:
            for c in range(3):
                p2, p98 = np.percentile(data[c], (2, 98))
                data[c] = np.clip((data[c] - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        return np.transpose(data, (1, 2, 0))

    def _load_mask(self, img_path: Path) -> np.ndarray:
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = self.mask_dir / f"{img_path.stem}.png"
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)
        return np.where((mask >= 0) & (mask < NUM_CLASSES), mask, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = self._load_sar_tile(img_path)
        mask = self._load_mask(img_path)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            mask = torch.from_numpy(mask).long()
        return image, mask


# --------------------------------------------------------------------------- #
#  Transforms
# --------------------------------------------------------------------------- #
def get_transforms(img_size: int = 512):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        train_tf = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
        val_tf = A.Compose([A.Resize(img_size, img_size), ToTensorV2()])
        return train_tf, val_tf
    except ImportError:  # pragma: no cover - optional at train time only
        def identity(image=None, mask=None):
            return {"image": image, "mask": mask}

        return identity, identity


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #
class HybridSARLoss(nn.Module):
    """Weighted Cross-Entropy + multi-class Dice Loss for small oil/ship targets."""

    def __init__(self, class_weights: List[float]):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.ce.weight = self.ce.weight.to(logits.device)
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


def compute_per_class_iou(
    preds: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    ious: Dict[str, float] = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        pred_mask = preds == cls_id
        true_mask = targets == cls_id
        intersection = int((pred_mask & true_mask).sum().item())
        union = int((pred_mask | true_mask).sum().item())
        ious[cls_name] = round(intersection / union, 4) if union > 0 else float("nan")
    return ious


# --------------------------------------------------------------------------- #
#  Model factory
# --------------------------------------------------------------------------- #
def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the DeepLabV3+ segmentation model from config."""
    return smp.DeepLabV3Plus(
        encoder_name=str(model_cfg.get("encoder", "resnet34")),
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", NUM_CLASSES)),
    )


# --------------------------------------------------------------------------- #
#  Checkpoint persistence
# --------------------------------------------------------------------------- #
def _save_checkpoint(model: nn.Module, model_cfg: Dict[str, Any], path: str) -> None:
    """Persist model weights with architecture metadata inside the .pt."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": int(model_cfg.get("classes", NUM_CLASSES)),
        "class_names": list(model_cfg.get("class_names", [CLASS_NAMES[i] for i in range(NUM_CLASSES)])),
        "architecture": model_cfg.get("architecture", "deeplabv3plus"),
        "encoder": model_cfg.get("encoder", "resnet34"),
        "in_channels": int(model_cfg.get("in_channels", 3)),
    }
    torch.save(checkpoint, path)
    print(f"--> Saved checkpoint (with metadata) to {path}")


# --------------------------------------------------------------------------- #
#  Training orchestration
# --------------------------------------------------------------------------- #
def _resolve_split_dirs(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    paths_cfg = cfg.get("paths") or {}
    return paths_cfg.get("train_split"), paths_cfg.get("val_split")


def _build_loaders(
    cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    aug_cfg: dict,
    input_size: tuple,
    num_classes: int,
    batch_size: int,
    data_root: Optional[str],
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[int], Optional[int]]:
    """Construct train / val DataLoaders from config or CLI args."""
    train_size = val_size = None

    if data_root is not None:
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
    else:
        train_dir, val_dir = _resolve_split_dirs(cfg)

    train_loader: Optional[DataLoader] = None
    if train_dir and os.path.isdir(train_dir):
        train_ds = SAR5ClassDataset(
            os.path.join(train_dir, "images"),
            os.path.join(train_dir, "masks"),
            transform=get_transforms(input_size[0])[0],
        )
        train_size = len(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )

    val_loader: Optional[DataLoader] = None
    if val_dir and os.path.isdir(val_dir):
        val_ds = SAR5ClassDataset(
            os.path.join(val_dir, "images"),
            os.path.join(val_dir, "masks"),
            transform=get_transforms(input_size[0])[1],
        )
        val_size = len(val_ds)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

    return train_loader, val_loader, train_size, val_size


def run_training(config: Dict[str, Any], data_root: Optional[str] = None) -> None:
    paths_cfg = config.get("paths") or {}
    training_cfg = config.get("training") or {}
    model_cfg = config.get("model") or {}

    model_save_path = paths_cfg.get("model_checkpoint", "models/sar_5class_deeplabv3p_resnet34.pt")
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    batch_size = int(training_cfg.get("batch_size", 8))
    epochs = int(training_cfg.get("epochs", 40))
    learning_rate = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    input_size = tuple(model_cfg.get("input_size", (512, 512)))
    num_classes = int(model_cfg.get("classes", NUM_CLASSES))
    class_names = model_cfg.get(
        "class_names", [CLASS_NAMES[i] for i in range(num_classes)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_size, val_size = _build_loaders(
        config, model_cfg, {}, input_size, num_classes, batch_size, data_root
    )

    if train_loader is None:
        print("WARNING: No training data found. Exiting.")
        return

    model = build_model(model_cfg).to(device)

    # Higher weights on Oil Spill (1), Look-Alike (2), Ship (3); lighter on Sea/Land.
    default_weights = [0.5, 3.0, 2.0, 3.5, 0.5]
    class_weights = list(
        model_cfg.get("class_weights", default_weights[:num_classes])
    )
    criterion = HybridSARLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_oil_iou = 0.0
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, masks in (val_loader or []):
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.append(preds)
                all_targets.append(masks)

        if all_preds:
            preds_cat = torch.cat(all_preds, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            ious = compute_per_class_iou(preds_cat, targets_cat)
            oil_iou = ious["Oil Spill"]
            print(
                f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss / max(len(train_loader),1):.4f} | "
                f"Oil IoU: {oil_iou:.4f} | Look-Alike IoU: {ious['Look-Alike']:.4f} | "
                f"Ship IoU: {ious['Ship']:.4f}"
            )
            if not np.isnan(oil_iou) and oil_iou > best_oil_iou:
                best_oil_iou = oil_iou
                _save_checkpoint(
                    model,
                    {
                        "classes": num_classes,
                        "class_names": class_names,
                        "architecture": model_cfg.get("architecture", "deeplabv3plus"),
                        "encoder": model_cfg.get("encoder", "resnet34"),
                        "in_channels": int(model_cfg.get("in_channels", 3)),
                    },
                    model_save_path,
                )
        else:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss / max(len(train_loader),1):.4f} | no val set")

    print(
        f"Training complete. Best Oil Spill IoU: {best_oil_iou:.4f} saved to {model_save_path}"
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5-class SAR DeepLabV3+ training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root dir containing train/val subdirs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--save-to", type=str, default=None,
                        help="Override paths.model_checkpoint save target")
    return parser


if __name__ == "__main__":
    args = build_cli().parse_args()
    cfg = _load_config(args.config)
    training_cfg = cfg.setdefault("training", {})
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_cfg["learning_rate"] = args.learning_rate
    if args.save_to is not None:
        cfg.setdefault("paths", {})["model_checkpoint"] = args.save_to
    run_training(cfg, data_root=args.data_root)
