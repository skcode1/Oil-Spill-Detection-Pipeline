"""Dynamic SAR multi-class segmentation training entry point.

Trains a 3-class U-Net (Background / Oil Spill / Lookalike) using a
combined Dice + Focal loss to handle severe class imbalance, with
production-grade SAR augmentations applied via ``SentinelDataset``.

All paths and hyperparameters are resolved from ``config.yaml`` (sections:
``paths``, ``model``, ``training``, ``augmentation``) with CLI overrides;
no training constant is hard-coded.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

try:
    from dataset import (
        CLASS_NAMES,
        NUM_CLASSES,
        SentinelDataset,
    )
except ImportError:
    from src.dataset import (
        CLASS_NAMES,
        NUM_CLASSES,
        SentinelDataset,
    )


def _load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


# --------------------------------------------------------------------------- #
#  Combined multi-class loss:  Dice  +  Focal
# --------------------------------------------------------------------------- #
class CombinedDiceFocalLoss(nn.Module):
    """Dice Loss + Focal Loss for class-imbalanced multi-class segmentation.

    Oil spills occupy a tiny fraction of pixels; standard Cross-Entropy will
    bias the model towards predicting 'Background' everywhere.  Dice
    maximises the overlap of rare classes while Focal down-weights
    easy-to-classify majority pixels, forcing the model to learn oil edges
    aggressively.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dice_weight: float = 0.5,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.dice = smp.losses.DiceLoss(
            mode=smp.losses.MULTICLASS_MODE,
            classes=list(range(num_classes)),
            from_logits=True,
            ignore_index=ignore_index,
        )
        self.focal = smp.losses.FocalLoss(
            mode="multiclass",
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction="mean",
        )
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
        )


# --------------------------------------------------------------------------- #
#  Model factory
# --------------------------------------------------------------------------- #
def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate the segmentation model from config."""
    return smp.Unet(
        encoder_name=str(model_cfg.get("encoder", "resnet34")),
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", NUM_CLASSES)),
    )


# --------------------------------------------------------------------------- #
#  Checkpoint persistence (with architecture metadata inside the .pth)
# --------------------------------------------------------------------------- #
def _save_checkpoint(model: nn.Module, model_cfg: Dict[str, Any], path: str) -> None:
    """Persist model weights with architecture metadata inside the .pth."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "architecture": model_cfg.get("architecture", "unet"),
            "encoder": model_cfg.get("encoder", "resnet34"),
            "encoder_weights": model_cfg.get("encoder_weights"),
            "in_channels": int(model_cfg.get("in_channels", 3)),
            "classes": int(model_cfg.get("classes", NUM_CLASSES)),
            "class_names": list(model_cfg.get("class_names", CLASS_NAMES)),
        },
    }
    torch.save(checkpoint, path)
    print(f"--> Saved checkpoint (with metadata) to {path}")


def _load_state_dict_into(model: nn.Module, path: str) -> None:
    """Load weights from a checkpoint produced by :func:`_save_checkpoint`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)


def _checkpoint_sanity(
    model: nn.Module, model_cfg: Dict[str, Any], device: torch.device
) -> Tuple[bool, str]:
    """Reject degenerate candidates before they overwrite the checkpoint.

    Dice + Focal on raw logits can drive softmax saturation until the network
    emits a fixed spatial template that ignores its input - every scene then
    argmaxes to "background" and the whole pipeline reports ``Clear`` for oil,
    no-oil and lookalike alike.  A candidate must therefore (a) keep logits in a
    sane range and (b) classify two *different* inputs differently.

    Returns:
        ``(ok, reason)`` - ``reason`` is logged either way.
    """
    was_training = model.training
    model.eval()
    size = tuple(model_cfg.get("input_size", (256, 256)))
    in_channels = int(model_cfg.get("in_channels", 3))
    try:
        with torch.no_grad():
            first = torch.rand(1, in_channels, *size, device=device) * 2.0 - 1.0
            second = torch.rand(1, in_channels, *size, device=device) * 2.0 - 1.0
            logits_first = model(first)
            logits_second = model(second)
        if not torch.isfinite(logits_first).all() or not torch.isfinite(logits_second).all():
            return False, "non-finite logits"
        max_abs = float(
            torch.maximum(logits_first.abs(), logits_second.abs()).max()
        )
        if max_abs > 500.0:
            return False, f"exploded logits (max |logit| = {max_abs:.0f} > 500)"
        disagreement = float(
            (logits_first.argmax(1) != logits_second.argmax(1)).float().mean()
        )
        if disagreement < 0.005:
            return False, (
                "input-independent output (two random inputs disagree on only "
                f"{100.0 * disagreement:.2f} % of pixels)"
            )
        return True, (
            f"max |logit| = {max_abs:.1f}, "
            f"input disagreement = {100.0 * disagreement:.1f} %"
        )
    finally:
        if was_training:
            model.train()


# --------------------------------------------------------------------------- #
#  Training orchestration
# --------------------------------------------------------------------------- #
def _resolve_split_dirs(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (train_dir, val_dir) from config paths."""
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
    paths_cfg = cfg.get("paths") or {}
    train_size = val_size = None

    if data_root is not None:
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
    else:
        train_dir, val_dir = _resolve_split_dirs(cfg)

    # --- Training loader ---
    train_loader: Optional[DataLoader] = None
    if train_dir and os.path.isdir(train_dir):
        train_ds = SentinelDataset(
            os.path.join(train_dir, "images"),
            os.path.join(train_dir, "masks"),
            size=input_size,
            num_classes=num_classes,
            augment=True,
            augmentation_cfg=aug_cfg,
        )
        train_size = len(train_ds)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        # Fallback: single flat directory with random split
        raw_dir = paths_cfg.get("raw_sar_dir", "data/raw/SAR_training/sentinel")
        img_dir = os.path.join(raw_dir, "image")
        label_dir = os.path.join(raw_dir, "label")
        if os.path.isdir(img_dir):
            full_dataset = SentinelDataset(
                img_dir, label_dir, size=input_size, num_classes=num_classes,
                augment=True, augmentation_cfg=aug_cfg,
            )
            train_fraction = float(cfg.get("training", {}).get("train_fraction", 0.85))
            train_n = int(train_fraction * len(full_dataset))
            val_n = len(full_dataset) - train_n
            if val_n > 0:
                train_ds, val_ds = torch.utils.data.random_split(
                    full_dataset, [train_n, val_n]
                )
            else:
                train_ds = full_dataset
                val_ds = None
            train_size = len(train_ds)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            if val_ds is not None:
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                val_size = len(val_ds)

    # --- Validation loader ---
    val_loader: Optional[DataLoader] = None
    if val_dir and os.path.isdir(val_dir):
        val_ds = SentinelDataset(
            os.path.join(val_dir, "images"),
            os.path.join(val_dir, "masks"),
            size=input_size,
            num_classes=num_classes,
            augment=False,
            augmentation_cfg=aug_cfg,
        )
        val_size = len(val_ds)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_size, val_size


def run_training(config: Dict[str, Any], data_root: Optional[str] = None,
                 init_from: Optional[str] = None) -> None:
    paths_cfg = config.get("paths") or {}
    training_cfg = config.get("training") or {}
    model_cfg = config.get("model") or {}
    aug_cfg = config.get("augmentation") or {}

    paths_cfg.setdefault("model_checkpoint", "models/oil_spill_segmentation_model.pth")
    model_save_path = paths_cfg["model_checkpoint"]
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    batch_size = int(training_cfg.get("batch_size", 16))
    epochs = int(training_cfg.get("epochs", 50))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    input_size = tuple(model_cfg.get("input_size", (256, 256)))
    num_classes = int(model_cfg.get("classes", NUM_CLASSES))
    class_names = model_cfg.get("class_names", CLASS_NAMES[:num_classes])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_size, val_size = _build_loaders(
        config, model_cfg, aug_cfg, input_size, num_classes, batch_size, data_root
    )

    if train_loader is None:
        print("WARNING: No training data found. Exiting.")
        return

    model = build_model(model_cfg).to(device)
    if init_from:
        # Warm-start: fine-tuning from a known-good checkpoint converges far
        # faster than training from scratch on this dataset size.
        print(f"Loading initial weights from {init_from}")
        _load_state_dict_into(model, init_from)
    focal_gamma = float(training_cfg.get("loss", {}).get("focal_gamma", 2.0))
    criterion = CombinedDiceFocalLoss(
        num_classes=num_classes,
        dice_weight=float(training_cfg.get("loss", {}).get("dice_weight", 0.5)),
        focal_weight=float(training_cfg.get("loss", {}).get("focal_weight", 1.0)),
        focal_gamma=focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Device: {device}")
    print(f"Train dataset size: {train_size}")
    if val_size is not None:
        print(f"Val dataset size:   {val_size}")
    print(f"Train batches: {len(train_loader)}"
          f"{f' | Val batches: {len(val_loader)}' if val_loader else ''}")
    print(f"Profile: epochs={epochs}, batch={batch_size}, lr={learning_rate}, wd={weight_decay}")
    print(f"Architecture: {model_cfg.get('architecture', 'unet')} / "
          f"{model_cfg.get('encoder', 'resnet34')} | "
          f"in_channels={model_cfg.get('in_channels', 3)} | classes={num_classes}")
    print(f"Class names: {class_names}")
    print(f"Loss: Dice(MULTICLASS) + Focal(MULTICLASS, gamma={focal_gamma})")

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_loss = 0.0
        val_batches = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    val_loss += criterion(logits, masks).item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)

        train_batches = max(n_batches, 1)
        msg = f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {total_loss / train_batches:.4f}"
        if val_loader is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
            if val_loss < best_val_loss:
                sane, reason = _checkpoint_sanity(model, model_cfg, device)
                if sane:
                    best_val_loss = val_loss
                    _save_checkpoint(model, model_cfg, model_save_path)
                    msg += f" | -> Saved best ({reason})"
                else:
                    msg += f" | !! Candidate REJECTED, not saved ({reason})"
        print(msg)

    # Always save final if no validation loop
    if val_loader is None:
        sane, reason = _checkpoint_sanity(model, model_cfg, device)
        if sane:
            _save_checkpoint(model, model_cfg, model_save_path)
        else:
            print(f"[!] Final model REJECTED as degenerate, not saved ({reason})")

    print(f"\nModel trained successfully and checkpoint saved to: {model_save_path}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic 3-class SAR U-Net training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root dir containing train/val subdirs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Warm-start fine-tuning from an existing checkpoint")
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
    run_training(cfg, data_root=args.data_root, init_from=args.init_from)
