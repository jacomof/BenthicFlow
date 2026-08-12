"""U-Net Continuous Flow Matching model with Classifier-Free Guidance."""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """Residual block with Adaptive Group Normalization (AdaGN)."""
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.emb_proj = nn.Linear(emb_ch, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        
        # AdaGN injection
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class UNetCFM(nn.Module):
    """Fully convolutional vector field network for Flow Matching."""
    def __init__(self, 
                 latent_dim: int = 1024, 
                 cond_dim: int = 768, 
                 base_ch: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Normalization anchors
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std",  torch.ones(latent_dim))

        # Classifier-Free Guidance null token
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        emb_ch = base_ch * 4
        self.t_embed = TimestepEmbedder(emb_ch)
        self.c_embed = nn.Sequential(
            nn.Linear(cond_dim, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        # U-Net Architecture (16x16 -> 8x8 -> 16x16)
        self.in_conv = nn.Conv2d(latent_dim, base_ch, 3, padding=1)
        
        self.down1 = ResBlock(base_ch, base_ch * 2, emb_ch)
        self.down2 = ResBlock(base_ch * 2, base_ch * 2, emb_ch)
        
        self.mid1 = ResBlock(base_ch * 2, base_ch * 2, emb_ch)
        self.mid2 = ResBlock(base_ch * 2, base_ch * 2, emb_ch)
        
        self.up1 = ResBlock(base_ch * 4, base_ch * 2, emb_ch)
        self.up2 = ResBlock(base_ch * 3, base_ch, emb_ch)
        
        self.out_norm = nn.GroupNorm(32, base_ch)
        self.out_conv = nn.Conv2d(base_ch, latent_dim, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def normalize(self, z):   
        return (z - self.latent_mean.view(1, -1, 1, 1)) / self.latent_std.view(1, -1, 1, 1)
        
    def denormalize(self, z): 
        return z * self.latent_std.view(1, -1, 1, 1) + self.latent_mean.view(1, -1, 1, 1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z_t:  [B, 1024, H, W] (Noisy latent)
        t:    [B]             (Timesteps in [0, 1])
        cond: [B, 768]        (Global DINO embedding)
        """
        emb = self.t_embed(t) + self.c_embed(cond)
        
        h0 = self.in_conv(z_t)
        
        h1 = self.down1(h0, emb)
        h1_pool = F.avg_pool2d(h1, 2)
        
        h2 = self.down2(h1_pool, emb)
        
        h_mid = self.mid1(h2, emb)
        h_mid = self.mid2(h_mid, emb)
        
        # Use the skip connection's exact spatial size (not scale_factor): robust
        # under torch.compile, which can mis-resolve scale_factor on recompile.
        h_up = F.interpolate(h_mid, size=h1.shape[-2:], mode="nearest")
        h_up = torch.cat([h_up, h1], dim=1)
        h_up = self.up1(h_up, emb)
        
        h_up = torch.cat([h_up, h0], dim=1)
        h_up = self.up2(h_up, emb)
        
        out = self.out_conv(F.silu(self.out_norm(h_up)))
        return out


@torch.no_grad()
def cfm_sample_cfg(model: UNetCFM,
                   cond: torch.Tensor,
                   canvas_shape: tuple[int, int] = (16, 16),
                   n_steps: int = 50,
                   cfg_scale: float = 3.0,
                   z0: torch.Tensor | None = None) -> torch.Tensor:
    """Euler integration with Classifier-Free Guidance.

    z0: optional fixed initial noise [B, D, H, W]. Pass the same z0 with two
    different conds to compare them with the noise held constant.
    """
    B = cond.shape[0]
    H, W = canvas_shape
    device = cond.device
    D = model.latent_dim

    z = torch.randn(B, D, H, W, device=device) if z0 is None else z0.clone()
    null_cond = model.null_cond.unsqueeze(0).expand(B, -1)
    
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    
    for i in range(n_steps):
        t_cur = ts[i].expand(B)
        t_next = ts[i + 1]
        
        # Batched CFG forward pass
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([t_cur, t_cur], dim=0)
        c_in = torch.cat([cond, null_cond], dim=0)
        
        v_pred = model(z_in, t_in, c_in)
        v_cond, v_uncond = v_pred.chunk(2, dim=0)
        
        # Guidance formula
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # dt is a scalar (uniform step); subtracting the [B]-shaped t_cur here
        # would broadcast against the width dim instead of the batch dim.
        dt = ts[i + 1] - ts[i]
        z = z + dt * v

    return z
