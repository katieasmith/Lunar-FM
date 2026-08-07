# train_dem_sfs.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

from LunarDEMModel import LunarDEMModel, ShapeFromShadingLoss

# -------------------------------------------------------------
# 1. Native PyTorch ViT Matching lunar_encoder.pt
# -------------------------------------------------------------
class NativeLunarEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding layer
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2  # (224/16)^2 = 196
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # PyTorch native transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # Input shape: [B, 1, 224, 224]
        x = self.patch_embed.proj(x)                  # [B, 256, 14, 14]
        x = x.flatten(2).transpose(1, 2)              # [B, 196, 256]
        x = x + self.pos_embed
        x = self.encoder(x)                           # [B, 196, 256]
        return x

    # Add forward_features to satisfy LunarDEMModel interface
    def forward_features(self, x):
        return self.forward(x)

# -------------------------------------------------------------
# 2. NPZ Dataset Loader
# -------------------------------------------------------------
class LunarNPZDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        if 'images' in data:
            self.images = data['images']
        elif 'imgs' in data:
            self.images = data['imgs']
        else:
            first_key = list(data.keys())[0]
            self.images = data[first_key]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32)

        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 3:
            if img_tensor.shape[-1] in [1, 3]:
                img_tensor = img_tensor.permute(2, 0, 1)
            if img_tensor.shape[0] == 3:
                img_tensor = 0.2989 * img_tensor[0:1] + 0.5870 * img_tensor[1:2] + 0.1140 * img_tensor[2:3]

        return img_tensor

# -------------------------------------------------------------
# 3. Hyperparameters & Pipeline Setup
# -------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 40

NPZ_FILE_PATH = "C:/Users/ks59723/Documents/Lunar-FM/lunar_tiles_old.npz"

dataset = LunarNPZDataset(NPZ_FILE_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

import re

# -------------------------------------------------------------
# 4. Load Pre-trained Weights Cleanly
# -------------------------------------------------------------
checkpoint_path = r".\checkpoints\lunar_encoder.pt"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint.get('encoder', checkpoint)))
else:
    state_dict = checkpoint

# Strip ONLY distributed training prefix 'module.' if present (do not strip 'encoder.')
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Safely extract depth by finding layer numbers in key names
layer_indices = [int(m.group(1)) for k in state_dict.keys() if (m := re.search(r'layers\.(\d+)', k))]
depth = max(layer_indices) + 1 if layer_indices else 6

# Instantiate Native ViT Encoder
base_encoder = NativeLunarEncoder(
    img_size=224, 
    patch_size=16, 
    in_chans=1, 
    embed_dim=256, 
    depth=depth, 
    num_heads=8
)

# Load state dict
missing, unexpected = base_encoder.load_state_dict(state_dict, strict=False)
print(f"Successfully loaded foundation model encoder! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

# Attach DEM decoder head
model = LunarDEMModel(vit_encoder=base_encoder, embed_dim=256, patch_size=16, img_size=224)
model.to(DEVICE)

# -------------------------------------------------------------
# 5. Loss Function & Optimizer
# -------------------------------------------------------------
criterion = ShapeFromShadingLoss(
    sun_azimuth_deg=90.0, 
    sun_elevation_deg=2.0, 
    lambda_smooth=0.001,
    model_type="lommel_seeliger"
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# -------------------------------------------------------------
# 6. Training Loop
# -------------------------------------------------------------
# -------------------------------------------------------------
# 6. Training Loop
# -------------------------------------------------------------
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    running_photo = 0.0
    running_smooth = 0.0

    for images in dataloader:
        images = images.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        predicted_heights = model(images)

        # Photometric Loss
        loss, re_rendered_images, loss_dict = criterion(predicted_heights, images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_photo += loss_dict['l_photo']
        running_smooth += loss_dict['l_smooth']

    # Get current learning rate before stepping
    current_lr = scheduler.get_last_lr()[0]

    # Step learning rate scheduler forward one epoch
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Total Loss: {running_loss/len(dataloader):.4f} "
          f"| Photometric Loss: {running_photo/len(dataloader):.4f} "
          f"| Smoothness: {running_smooth/len(dataloader):.4f} "
          f"| LR: {current_lr:.2e}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"lunar_dem_sfs_epoch{epoch+1}.pth")

    # -------------------------------------------------------------
    # 7. Save Visual Progress Plot (3-Panel Comparison)
    # -------------------------------------------------------------
    os.makedirs("dem_samples", exist_ok=True)
    
    with torch.no_grad():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Ground Truth / Input Optical Image
        axes[0].imshow(images[0, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title("1. Input Optical Tile")
        axes[0].axis('off')
        
        # 2. Predicted Elevation Height Map (DEM)
        dem_plot = axes[1].imshow(predicted_heights[0, 0].cpu().numpy(), cmap='terrain')
        axes[1].set_title("2. Predicted DEM (Elevation)")
        axes[1].axis('off')
        fig.colorbar(dem_plot, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Re-rendered Shading (Shape-from-Shading Verification)
        axes[2].imshow(re_rendered_images[0, 0].cpu().numpy(), cmap='gray')
        axes[2].set_title("3. Re-rendered Shading")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"dem_samples/dem_epoch{epoch+1:02d}.png", dpi=150)
        plt.close(fig)
        
    print(f" Saved visual progress sample to dem_samples/dem_epoch{epoch+1:02d}.png")

    # Save model checkpoint
    torch.save(model.state_dict(), f"lunar_dem_sfs_epoch{epoch+1}.pth")