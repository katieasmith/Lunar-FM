# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class LunarDEMDataset(Dataset):
    def __init__(self, img_dir, dem_dir, transform=None):
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.tif'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load optical image (RGB or Grayscale)
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        # Load DEM elevation map (Float values)
        dem_path = os.path.join(self.dem_dir, filename)
        
        if dem_path.endswith('.npy'):
            dem = np.load(dem_path).astype(np.float32)
        else:
            # Assumes 32-bit floating point TIFF/Image
            dem = np.array(Image.open(dem_path), dtype=np.float32)

        # Convert to Tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        dem_tensor = torch.tensor(dem).unsqueeze(0).float() # [1, H, W]

        return image, dem_tensor