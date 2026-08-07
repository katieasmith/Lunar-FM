# train_dem.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from your modules
from LunarDEMModel import LunarDEMModel, DEMLoss
from dataset import LunarDEMDataset

# -------------------------------------------------------------
# 1. Hyperparameters & Settings
# -------------------------------------------------------------
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 50
PRETRAINED_MAE_PATH = "checkpoint_epoch94.pth" # Path to your pre-trained MAE weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------
# 2. Data Preparation
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = LunarDEMDataset(
    img_dir="data/train/images", 
    dem_dir="data/train/dems", 
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -------------------------------------------------------------
# 3. Model & Loss Initialization
# -------------------------------------------------------------
# Initialize your base ViT/MAE encoder backbone (adjust instantiation to your MAE setup)
# Example: base_encoder = YourMAEViTBackbone()

model = LunarDEMModel(vit_encoder=base_encoder, embed_dim=768, patch_size=16, img_size=224)

# Load pre-trained MAE weights into the backbone
checkpoint = torch.load(PRETRAINED_MAE_PATH, map_location=DEVICE)
# Handle state_dict key matching if necessary:
model.encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
print("Successfully loaded pre-trained MAE encoder weights!")

model.to(DEVICE)

# Instantiate composite loss & optimizer
criterion = DEMLoss(alpha=0.5, beta=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# -------------------------------------------------------------
# 4. Training Loop
# -------------------------------------------------------------
model.train()
best_loss = float('inf')

for epoch in range(EPOCHS):
    running_total_loss = 0.0
    running_height_loss = 0.0
    running_grad_loss = 0.0
    running_normal_loss = 0.0

    for step, (images, dem_targets) in enumerate(train_loader):
        images = images.to(DEVICE)
        dem_targets = dem_targets.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass: Predict continuous height map [B, 1, H, W]
        dem_preds = model(images)

        # Calculate composite loss
        loss, loss_components = criterion(dem_preds, dem_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track statistics
        running_total_loss += loss.item()
        running_height_loss += loss_components['l_height']
        running_grad_loss += loss_components['l_grad']
        running_normal_loss += loss_components['l_normal']

    # Log epoch metrics
    epoch_loss = running_total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Total Loss: {epoch_loss:.4f} "
          f"| Height: {running_height_loss/len(train_loader):.4f} "
          f"| Grad: {running_grad_loss/len(train_loader):.4f} "
          f"| Normal: {running_normal_loss/len(train_loader):.4f}")

    # Save best checkpoint
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_lunar_dem_model.pth")
        print(f"--> Saved new best model checkpoint (Loss: {best_loss:.4f})")