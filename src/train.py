"""Dynamic SAR segmentation training entry point.

All paths and hyperparameters are resolved from ``config.yaml`` (sections:
``paths``, ``model``, ``training``) with CLI overrides; no training constant
is fixed in code.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import yaml

from dataset import SentinelDataset


def _load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def run_training(config: dict) -> None:
    paths_cfg = config.get("paths") or {}
    training_cfg = config.get("training") or {}
    model_cfg = config.get("model") or {}

    raw_dir = paths_cfg.get("raw_sar_dir", "data/raw/SAR_training/sentinel")
    img_dir = os.path.join(raw_dir, "image")
    label_dir = os.path.join(raw_dir, "label")
    model_save_path = paths_cfg.get("model_checkpoint", "models/sar_unet_best.pth")
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    batch_size = int(training_cfg.get("batch_size", 16))
    epochs = int(training_cfg.get("epochs", 15))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    train_fraction = float(training_cfg.get("train_fraction", 0.85))
    input_size = tuple(model_cfg.get("input_size", (256, 256)))
    encoder_name = str(model_cfg.get("encoder", "resnet34"))
    in_channels = int(model_cfg.get("in_channels", 1))
    classes = int(model_cfg.get("classes", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Split (dynamic paths, sizes and split fraction)
    full_dataset = SentinelDataset(img_dir, label_dir, size=input_size)
    train_size = int(train_fraction * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model: 1 input channel for single-band SAR, 1 output channel for binary mask
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=in_channels,
        classes=classes,
    ).to(device)

    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Train/Val sizes: {train_size}/{val_size}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Profile: epochs={epochs}, batch={batch_size}, lr={learning_rate}, wd={weight_decay}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        # Avoid ZeroDivisionError if loader empty
        train_batches = len(train_loader) if len(train_loader) > 0 else 1
        val_batches = len(val_loader) if len(val_loader) > 0 else 1

        val_loss /= val_batches
        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {total_loss / train_batches:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved best model checkpoint to {model_save_path}")

    print("\nModel trained successfully!")


def build_cli():
    parser = argparse.ArgumentParser(description="Dynamic SAR U-Net training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config training.batch_size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config training.learning_rate")
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
    run_training(cfg)
