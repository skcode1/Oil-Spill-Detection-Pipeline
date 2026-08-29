import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from dataset import SentinelDataset

DATA_DIR = "data/raw/SAR_training/sentinel"
IMG_DIR = os.path.join(DATA_DIR, "image")
LABEL_DIR = os.path.join(DATA_DIR, "label")
MODEL_SAVE_PATH = "models/sar_unet_best.pth"
os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training():
    # Dataset & Split
    full_dataset = SentinelDataset(IMG_DIR, LABEL_DIR)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Model: 1 input channel for single-band SAR, 1 output channel for binary mask
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(device)

    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training Loop
    epochs = 15
    best_val_loss = float("inf")

    # Diagnostics: print basic info before training
    print(f"Device: {device}")
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Train/Val sizes: {train_size}/{val_size}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

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
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/train_batches:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved best model checkpoint to {MODEL_SAVE_PATH}")

    print("\nModel trained successfully!")


if __name__ == "__main__":
    run_training()