import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SentinelDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.filenames = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load SAR image & mask
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)  # [1, H, W], normalized to [0, 1]
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()  # Binary mask: 1 = spill, 0 = ocean

        return image_tensor, mask_tensor