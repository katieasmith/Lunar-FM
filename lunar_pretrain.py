"""
lunar_pretrain.py
====================
Self-supervised MAE (Masked Autoencoder) pre-training for a lunar
foundation model, optimized for local Windows GPU training.

Output ("the foundation model"):
  - checkpoints/lunar_mae_last.pt   full training checkpoint (resume-able)
  - checkpoints/lunar_mae_best.pt   best-loss checkpoint
  - checkpoints/lunar_encoder.pt    ENCODER-ONLY weights -> fine-tune on downstream tasks
  - samples/recon_epochXX.png       composite reconstruction visualizations
  - loss_plot.png                   loss curve updated every epoch
"""

import os
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from lunar_dataloader import LunarNPZDataset


# ==========================================================================
# 1. MODEL: mini ViT encoder + linear reconstruction head
# ==========================================================================
class LunarPatchEmbedding(nn.Module):
    """Turns an image into a sequence of patch tokens (the ViT front-end)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def _encoder_stack(dim, heads, depth):
    """Build a stack of `depth` identical transformer blocks."""
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * 4,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class LunarMAE(nn.Module):
    """A proper (asymmetric) Masked Autoencoder for lunar tiles."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=192, depth=4, heads=4,
                 decoder_embed_dim=128, decoder_depth=2, decoder_heads=4,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.num_patches_side = img_size // patch_size
        self.num_patches = self.num_patches_side ** 2

        # Encoder
        self.patch_embed = LunarPatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder = _encoder_stack(embed_dim, heads, depth)
        self.enc_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim) 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        self.decoder = _encoder_stack(decoder_embed_dim, decoder_heads, decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        for p in (self.pos_embed, self.decoder_pos_embed, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)

    def _random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = max(1, int(N * (1 - mask_ratio)))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward_encoder(self, imgs, mask_ratio):
        x = self.patch_embed(imgs) + self.pos_embed
        x, mask, ids_restore = self._random_masking(x, mask_ratio)
        x = self.enc_norm(self.encoder(x))
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, _, D = x.shape
        n_masked = ids_restore.shape[1] - x.shape[1]
        
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.decoder_pos_embed
        x = self.decoder_norm(self.decoder(x))
        return self.decoder_pred(x)

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask

    @torch.no_grad()
    def forward_features(self, imgs):
        x = self.patch_embed(imgs) + self.pos_embed
        return self.enc_norm(self.encoder(x))


# ==========================================================================
# 2. MASKING + PATCH UTILITIES
# ==========================================================================
def patchify(imgs, patch_size):
    B, C, H, W = imgs.shape
    p = patch_size
    nh, nw = H // p, W // p 
    x = imgs.reshape(B, C, nh, p, nw, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(B, nh * nw, p * p * C)
    return x


def unpatchify(patches, patch_size, in_chans, num_patches_side):
    B = patches.shape[0]
    p, c, n = patch_size, in_chans, num_patches_side
    x = patches.reshape(B, n, n, p, p, c)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, c, n * p, n * p)


def mae_loss(pred_patches, target_images, mask, patch_size, norm_pix=True):
    target = patchify(target_images, patch_size)

    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        # 1. L1 Loss is computed in normalized z-score space
        target_norm = (target - mean) / torch.sqrt(var + 1e-6)
        loss_l1 = torch.abs(pred_patches - target_norm)

        # 2. Un-normalize predictions back to TRUE PIXEL space for edge detection
        pred_pix = pred_patches * torch.sqrt(var + 1e-6) + mean
    else:
        loss_l1 = torch.abs(pred_patches - target)
        pred_pix = pred_patches

    # Reconstruct 2D images in float32 for clean spatial gradient filtering
    B, N, _ = pred_patches.shape
    n_side = int(math.sqrt(N))
    pred_img = unpatchify(pred_pix, patch_size, 1, n_side).float()
    target_img = target_images.float()

    # 2D Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred_patches.device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=pred_patches.device, dtype=torch.float32).view(1, 1, 3, 3)

    pred_grad = torch.abs(torch.nn.functional.conv2d(pred_img, sobel_x, padding=1)) + \
                torch.abs(torch.nn.functional.conv2d(pred_img, sobel_y, padding=1))
    target_grad = torch.abs(torch.nn.functional.conv2d(target_img, sobel_x, padding=1)) + \
                  torch.abs(torch.nn.functional.conv2d(target_img, sobel_y, padding=1))

    grad_patches = patchify(torch.abs(pred_grad - target_grad), patch_size)

    # 3. Combine L1 with a gentle 0.1 edge weight
    total_per_patch = loss_l1.mean(dim=-1) + 0.1 * grad_patches.mean(dim=-1)
    return (total_per_patch * mask).sum() / mask.sum().clamp(min=1.0)


# ==========================================================================
# 3. VISUALIZATION
# ==========================================================================
@torch.no_grad()
def save_reconstruction(model, images, patch_size, mask_ratio, out_path, norm_pix=True):
    model.eval()
    pred, mask = model(images, mask_ratio)

    target = patchify(images, patch_size)
    
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred_pix = pred * torch.sqrt(var + 1e-6) + mean
    else:
        pred_pix = pred

    m = mask.unsqueeze(-1).bool()

    # Create masked input visualization (black out masked patches)
    masked_patches = torch.where(m, torch.zeros_like(target), target)
    
    # Overwrite masked regions with model predictions (Composite Blending)
    combo = torch.where(m, pred_pix, target)

    n_side = images.shape[-1] // patch_size
    masked_img = unpatchify(masked_patches, patch_size, images.shape[1], n_side)
    recon = unpatchify(combo, patch_size, images.shape[1], n_side)

    raw_np = images[0, 0].cpu().numpy()
    masked_np = masked_img[0, 0].cpu().numpy()
    recon_np = recon[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in zip(
        axes, [raw_np, masked_np, recon_np],
        ["1. Ground Truth", f"2. Masked Input ({int(mask_ratio*100)}%)", "3. MAE Reconstruction"],
    ):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    model.train()


# ==========================================================================
# 4. TRAINING
# ==========================================================================
def main():
    
    ap = argparse.ArgumentParser(description="Lunar MAE foundation model pre-training")
    
    ap.add_argument("--data", default="lunar_tiles.npz", help="Path to .npz dataset")
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--decoder-embed-dim", type=int, default=128)
    ap.add_argument("--decoder-depth", type=int, default=2)
    ap.add_argument("--decoder-heads", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32) # Lowered for local GPU safety
    ap.add_argument("--lr", type=float, default=1e-3)
    
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0) # Local Windows fix (must be 0)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="Smoke test run")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.batch_size, args.warmup_epochs = 1, 4, 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | torch threads: {torch.get_num_threads()}", flush=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}.")

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    print("1. Loading compressed lunar dataset into RAM...", flush=True)
    dataset = LunarNPZDataset(args.data)
    
    if args.quick:
        subset_indices = list(range(min(32, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    # CRITICAL LOCAL GPU FIX: pin_memory=False to prevent Windows memory lockup
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=False)

    print("2. Building MAE model...", flush=True)
    model = LunarMAE(args.tile_size, args.patch_size, 1,
                     args.embed_dim, args.depth, args.heads,
                     args.decoder_embed_dim, args.decoder_depth, args.decoder_heads,
                     args.mask_ratio).to(device)
    
    enc_params = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith(("patch_embed", "pos_embed", "encoder", "enc_norm")))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params/1e6:.2f}M total | {enc_params/1e6:.2f}M encoder | "
          f"enc(dim={args.embed_dim},depth={args.depth}) "
          f"dec(dim={args.decoder_embed_dim},depth={args.decoder_depth})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    start_epoch, best_loss = 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0)
        best_loss = ck.get("best_loss", float("inf"))
        print(f"   Resumed from {args.resume} @ epoch {start_epoch}", flush=True)

    steps_per_epoch = max(1, len(loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    print("3. Training (self-supervised masked reconstruction)...", flush=True)
    global_step = start_epoch * steps_per_epoch
    model.train()

    epoch_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = 0.0
        for i, images in enumerate(loader):
            images = images.to(device)

            lr = lr_at(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            use_amp = (device.type == 'cuda')
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred, mask = model(images, args.mask_ratio)
                loss = mae_loss(pred, images, mask, args.patch_size, norm_pix=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1
            if (i + 1) % 10 == 0 or (i + 1) == steps_per_epoch:
                print(f"   epoch {epoch+1}/{args.epochs} "
                      f"step {i+1}/{steps_per_epoch} "
                      f"loss {running/(i+1):.4f} lr {lr:.2e}", flush=True)

        epoch_loss = running / steps_per_epoch
        dt = time.time() - t0
        print(f"-- epoch {epoch+1} done | avg loss {epoch_loss:.4f} | {dt:.1f}s", flush=True)

        try:
            vis_batch = next(iter(loader)).to(device)
            save_reconstruction(model, vis_batch, args.patch_size, args.mask_ratio,
                                os.path.join(sample_dir, f"recon_epoch{epoch+1:02d}.png"),
                                norm_pix=True)
        except Exception as e:
            print(f"   (viz skipped: {e})", flush=True)

        # Checkpoints
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_loss": best_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "lunar_mae_last.pt"))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt["best_loss"] = best_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "lunar_mae_best.pt"))
            
            encoder_state = {
                k: v for k, v in model.state_dict().items()
                if k.startswith(("patch_embed", "pos_embed", "encoder", "enc_norm"))
            }
            torch.save({
                "encoder": encoder_state,
                "config": {
                    "img_size": args.tile_size, "patch_size": args.patch_size,
                    "in_chans": 1, "embed_dim": args.embed_dim,
                    "depth": args.depth, "heads": args.heads,
                },
            }, os.path.join(ckpt_dir, "lunar_encoder.pt"))
            print(f"   * new best {best_loss:.4f} -> saved lunar_encoder.pt", flush=True)

        # Record epoch loss and plot graph (updated every epoch)
        epoch_losses.append(epoch_loss)
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='#1f77b4', linewidth=2, label='Train Loss')
        plt.title('MAE Pre-training Loss vs. Epoch', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        
        plot_path = os.path.join(args.out_dir, 'loss_plot.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f" Graph updated: {plot_path}", flush=True)

    print("\n=== DONE ===", flush=True)
    print(f"Best loss: {best_loss:.4f}", flush=True)
    print(f"Foundation model encoder: {os.path.join(ckpt_dir, 'lunar_encoder.pt')}", flush=True)
    print(f"Reconstruction samples:   {sample_dir}", flush=True)


if __name__ == "__main__":
    main()