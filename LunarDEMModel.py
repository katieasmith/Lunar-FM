import torch
import torch.nn as nn
import torch.nn.functional as F

class LunarDEMModel(nn.Module):
    def __init__(self, vit_encoder, embed_dim=768, patch_size=16, img_size=224):
        super().__init__()
        # 1. Use the pre-trained MAE encoder backbone
        self.encoder = vit_encoder  
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g., 224 / 16 = 14
        
        # 2. Upsampling prediction head (Converts 14x14 tokens back to continuous 224x224 elevation map)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # -> 28x28
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # -> 56x56
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # -> 112x112
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # -> 224x224
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1) # 1 channel output (Height)
        )

    def forward(self, x):
        # Forward pass through pre-trained ViT (without masking)
        # Sequence output shape: [Batch, Num_Patches + 1, Embed_Dim]
        features = self.encoder.forward_features(x)
        
        # Strip CLS token if present
        if features.shape[1] == (self.grid_size ** 2) + 1:
            features = features[:, 1:, :]
            
        # Reshape token sequence back to 2D feature map: [B, N, D] -> [B, D, Grid, Grid]
        B, N, D = features.shape
        features_2d = features.permute(0, 2, 1).view(B, D, self.grid_size, self.grid_size)
        
        # Generate full-resolution elevation map [B, 1, H, W]
        dem_map = self.decoder_head(features_2d)
        return dem_map

class DEMLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2):
        """
        alpha: Weight for Spatial Gradient Loss
        beta:  Weight for Surface Normal Loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

        # Define 3x3 Sobel kernels for spatial derivative extraction (dz/dx, dz/dy)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradients(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return grad_x, grad_y

    def _compute_surface_normals(self, grad_x, grad_y):
        # Surface normal vector n = (-dz/dx, -dz/dy, 1) normalized
        normals = torch.cat([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=1)
        return F.normalize(normals, p=2, dim=1)

    def forward(self, pred, target):
        # 1. Height Loss (L1)
        l_height = self.l1_loss(pred, target)

        # 2. Gradient Loss (Preserves crisp crater rims and slope edges)
        pred_gx, pred_gy = self._compute_gradients(pred)
        target_gx, target_gy = self._compute_gradients(target)
        l_grad = self.l1_loss(pred_gx, target_gx) + self.l1_loss(pred_gy, target_gy)

        # 3. Surface Normal Loss (Forces realistic 3D orientation)
        pred_normals = self._compute_surface_normals(pred_gx, pred_gy)
        target_normals = self._compute_surface_normals(target_gx, target_gy)
        # Cosine distance (1 - cosine similarity)
        l_normal = 1.0 - F.cosine_similarity(pred_normals, target_normals, dim=1).mean()

        # Combine losses
        total_loss = l_height + (self.alpha * l_grad) + (self.beta * l_normal)
        
        return total_loss, {
            "l_height": l_height.item(), 
            "l_grad": l_grad.item(), 
            "l_normal": l_normal.item()
        }

class ShapeFromShadingLoss(nn.Module):
    def __init__(self, sun_azimuth_deg=45.0, sun_elevation_deg=30.0, lambda_smooth=0.05, model_type="lommel_seeliger"):
        """
        sun_azimuth_deg: Sun direction angle in degrees (clockwise from top)
        sun_elevation_deg: Sun height above horizon in degrees
        lambda_smooth: Weight for Total Variation (prevents noisy/spiky terrain)
        model_type: 'lommel_seeliger' (planetary regolith) or 'lambertian'
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.model_type = model_type

        # 1. Convert sun direction angles to 3D unit direction vector S = (Sx, Sy, Sz)
        az_rad = torch.tensor(sun_azimuth_deg * torch.pi / 180.0)
        el_rad = torch.tensor(sun_elevation_deg * torch.pi / 180.0)

        Sx = torch.cos(el_rad) * torch.sin(az_rad)
        Sy = torch.cos(el_rad) * torch.cos(az_rad)
        Sz = torch.sin(el_rad)

        light_vec = torch.tensor([Sx, Sy, Sz], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('light_vec', light_vec)

        # 2. Define 3x3 Sobel kernels to extract local spatial slopes (dz/dx, dz/dy)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_surface_normals(self, height_map):
        # Extract height gradients
        dz_dx = F.conv2d(height_map, self.sobel_x, padding=1)
        dz_dy = F.conv2d(height_map, self.sobel_y, padding=1)

        # Surface normal vector N = (-dz/dx, -dz/dy, 1) normalized
        normals = torch.cat([-dz_dx, -dz_dy, torch.ones_like(height_map)], dim=1)
        return F.normalize(normals, p=2, dim=1)

    def render_shading(self, normals):
        # Incidence angle cosine: cos(i) = N . S
        cos_i = torch.sum(normals * self.light_vec, dim=1, keepdim=True)
        cos_i = torch.clamp(cos_i, min=0.0)

        if self.model_type == "lommel_seeliger":
            # Nadir viewing direction V = (0, 0, 1), so cos(e) = Nz
            cos_e = normals[:, 2:3, :, :]
            cos_e = torch.clamp(cos_e, min=1e-5)
            # Lommel-Seeliger law for lunar dust: R = cos(i) / (cos(i) + cos(e))
            rendered = cos_i / (cos_i + cos_e + 1e-6)
            rendered = rendered * 2.0  # Normalize output intensity scale
        else:
            # Standard Lambertian reflection
            rendered = cos_i

        return rendered

    def total_variation_smoothness(self, height_map):
        # Total Variation penalty prevents spiky or noisy height estimates
        diff_x = torch.abs(height_map[:, :, :, 1:] - height_map[:, :, :, :-1])
        diff_y = torch.abs(height_map[:, :, 1:, :] - height_map[:, :, :-1, :])
        return torch.mean(diff_x) + torch.mean(diff_y)

    def forward(self, pred_height, input_img):
        # Convert RGB input to grayscale if needed
        if input_img.shape[1] == 3:
            gray_img = 0.2989 * input_img[:, 0:1, :, :] + 0.5870 * input_img[:, 1:2, :, :] + 0.1140 * input_img[:, 2:3, :, :]
        else:
            gray_img = input_img

        # 1. Predict 3D surface normals from predicted continuous elevation
        normals = self.compute_surface_normals(pred_height)

        # 2. Re-render photometrically under estimated sun illumination vector
        rendered_shading = self.render_shading(normals)

        # 3. Photometric loss (Compare re-rendered shading against real input image)
        l_photo = F.l1_loss(rendered_shading, gray_img)

        # 4. Terrain smoothness constraint
        l_smooth = self.total_variation_smoothness(pred_height)

        total_loss = l_photo + (self.lambda_smooth * l_smooth)

        return total_loss, rendered_shading, {"l_photo": l_photo.item(), "l_smooth": l_smooth.item()}

def forward(self, x):
        features = self.encoder.forward_features(x)
        if features.shape[1] == (self.grid_size ** 2) + 1:
            features = features[:, 1:, :]
            
        B, N, D = features.shape
        features_2d = features.permute(0, 2, 1).view(B, D, self.grid_size, self.grid_size)
        dem_map = self.decoder_head(features_2d)
        
        # ADD THIS LINE: Zero-mean detrending stops global plane tilts
        dem_map = dem_map - dem_map.mean(dim=(-2, -1), keepdim=True)
        
        return dem_map