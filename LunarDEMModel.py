import torch
import torch.nn as nn
import torch.nn.functional as F

class LunarDEMModel(nn.Module):
    def __init__(self, vit_encoder, embed_dim=256, patch_size=16, img_size=224):
        super().__init__()
        self.encoder = vit_encoder  
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        self.decoder_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        if features.shape[1] == (self.grid_size ** 2) + 1:
            features = features[:, 1:, :]
            
        B, N, D = features.shape
        features_2d = features.permute(0, 2, 1).view(B, D, self.grid_size, self.grid_size)
        
        dem_map = self.decoder_head(features_2d)
        
        # Zero-mean centering (prevents vertical offset without introducing adversarial steps)
        #dem_map = dem_map - dem_map.mean(dim=(-2, -1), keepdim=True)
        
        return dem_map


class ShapeFromShadingLoss(nn.Module):
    def __init__(self, sun_azimuth_deg=270.0, sun_elevation_deg=30.0, lambda_smooth=0.005, lambda_tilt=0.1, model_type="lommel_seeliger"):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_tilt = lambda_tilt
        self.model_type = model_type

        az_rad = torch.tensor(sun_azimuth_deg * torch.pi / 180.0)
        el_rad = torch.tensor(sun_elevation_deg * torch.pi / 180.0)

        Sx = torch.cos(el_rad) * torch.sin(az_rad)
        Sy = torch.cos(el_rad) * torch.cos(az_rad)
        Sz = torch.sin(el_rad)

        light_vec = torch.tensor([Sx, Sy, Sz], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('light_vec', light_vec)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_surface_normals(self, height_map):
        dz_dx = F.conv2d(height_map, self.sobel_x, padding=1)
        dz_dy = F.conv2d(height_map, self.sobel_y, padding=1)

        normals = torch.cat([-dz_dx, -dz_dy, torch.ones_like(height_map)], dim=1)
        return F.normalize(normals, p=2, dim=1)

    def render_shading(self, normals):
        cos_i = torch.sum(normals * self.light_vec, dim=1, keepdim=True)
        cos_i = torch.clamp(cos_i, min=0.0)

        if self.model_type == "lommel_seeliger":
            cos_e = normals[:, 2:3, :, :]
            cos_e = torch.clamp(cos_e, min=1e-5)
            rendered = cos_i / (cos_i + cos_e + 1e-6)
            rendered = rendered * 2.0
        else:
            rendered = cos_i

        return rendered

    def total_variation_smoothness(self, height_map):
        diff_x = torch.abs(height_map[:, :, :, 1:] - height_map[:, :, :, :-1])
        diff_y = torch.abs(height_map[:, :, 1:, :] - height_map[:, :, :-1, :])
        return torch.mean(diff_x) + torch.mean(diff_y)

    def compute_tilt_penalty(self, dem):
        """Penalizes global linear slopes smoothly in the loss."""
        B, C, H, W = dem.shape
        x = torch.linspace(-1, 1, W, device=dem.device, dtype=dem.dtype).view(1, 1, 1, W)
        y = torch.linspace(-1, 1, H, device=dem.device, dtype=dem.dtype).view(1, 1, H, 1)
        
        slope_x = (dem * x).mean(dim=(-2, -1))
        slope_y = (dem * y).mean(dim=(-2, -1))
        
        return torch.mean(slope_x**2 + slope_y**2)

    def forward(self, pred_height, input_img):
        if input_img.shape[1] == 3:
            gray_img = 0.2989 * input_img[:, 0:1, :, :] + 0.5870 * input_img[:, 1:2, :, :] + 0.1140 * input_img[:, 2:3, :, :]
        else:
            gray_img = input_img

        normals = self.compute_surface_normals(pred_height)
        rendered_shading = self.render_shading(normals)

        # Ignore cast shadows
        valid_mask = (gray_img > 0.05).float()
        l_photo = (torch.abs(rendered_shading - gray_img) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        l_smooth = self.total_variation_smoothness(pred_height)
        l_tilt = self.compute_tilt_penalty(pred_height)

        total_loss = l_photo + (self.lambda_smooth * l_smooth) + (self.lambda_tilt * l_tilt)

        return total_loss, rendered_shading, {"l_photo": l_photo.item(), "l_smooth": l_smooth.item()}