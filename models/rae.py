"""Representation Autoencoder (RAE) for RGBD underwater imagery.

Architecture (concat-fusion, DINOv2-B, small depth encoder, conv decoder):

  RGB image (3 ch)            Depth map (1 ch)
       |                            |
       v                            v
  frozen DINOv2-B            trainable depth ViT
  [H, W, 768]                [H, W, 256]
       \\                          /
        +--- concat along channel --+
                  |
                  v
        fused latent [H, W, 1024]
                  |
                  v   (optional noise injection during training)
        z + n,  n ~ N(0, sigma^2),  sigma ~ |N(0, tau^2)|
                  |
                  v
        trainable conv decoder (SD-VAE style)
                  |
                  v
            RGBD output (4 ch, 14H x 14W)

Both the depth encoder and the decoder are now resolution-agnostic: the same
weights handle 224x224 (H=W=16), 448x448 (H=W=32), and non-square inputs, as
long as the spatial dimensions are multiples of the DINOv2 patch size (14).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class TokenLayerNorm(nn.Module):
    """LayerNorm applied per-token across channels with NO affine parameters.

    The RAE paper applies this to encoder outputs to give each token zero mean
    and unit variance across channels. Frozen DINOv2 already has an affine
    LayerNorm; we cancel that affinity here on the fused latent so the FM
    model later sees a stable, identity-normalized latent space.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (used by the depth encoder)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


def _sincos_2d_pos_embed(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """2D sin-cos positional embedding. Returns [grid_h*grid_w, dim].

    Generalised to non-square grids so we can recompute on the fly for
    arbitrary input resolutions.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos"
    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)

    pos_h = torch.arange(grid_h, dtype=torch.float32)
    pos_w = torch.arange(grid_w, dtype=torch.float32)
    gh, gw = torch.meshgrid(pos_h, pos_w, indexing="ij")
    gh = gh.flatten()
    gw = gw.flatten()

    out_h = torch.cat([torch.sin(gh[:, None] * omega),
                       torch.cos(gh[:, None] * omega)], dim=-1)
    out_w = torch.cat([torch.sin(gw[:, None] * omega),
                       torch.cos(gw[:, None] * omega)], dim=-1)
    return torch.cat([out_h, out_w], dim=-1)


# ---------------------------------------------------------------------------
# Depth encoder (resolution-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class DepthEncoderConfig:
    patch_size:  int = 14
    dim:         int = 256
    depth:       int = 8
    num_heads:   int = 8
    out_dim:     int = 256


class DepthEncoder(nn.Module):
    """Small ViT mapping a 1-channel depth map to an [H/14, W/14, out_dim] latent.

    Positional embeddings are computed on the fly for whatever grid size the
    input dictates, and cached so we don't pay the build cost every step.
    """

    def __init__(self, cfg: DepthEncoderConfig | None = None):
        super().__init__()
        cfg = cfg or DepthEncoderConfig()
        self.cfg = cfg

        self.patchify = nn.Conv2d(1, cfg.dim,
                                   kernel_size=cfg.patch_size,
                                   stride=cfg.patch_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.num_heads)
            for _ in range(cfg.depth)
        ])
        self.norm_out = TokenLayerNorm()

        self.proj = (nn.Linear(cfg.dim, cfg.out_dim)
                     if cfg.out_dim != cfg.dim else nn.Identity())

        # Pos-embed cache (not a buffer, so it doesn't get saved/loaded).
        self._cached_pos:   torch.Tensor | None = None
        self._cached_shape: tuple[int, int] | None = None

    def _get_pos_embed(self, gh: int, gw: int,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (self._cached_shape != (gh, gw)
                or self._cached_pos is None
                or self._cached_pos.device != device
                or self._cached_pos.dtype  != dtype):
            pos = _sincos_2d_pos_embed(gh, gw, self.cfg.dim)
            self._cached_pos = pos.unsqueeze(0).to(device=device, dtype=dtype)
            self._cached_shape = (gh, gw)
        return self._cached_pos

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth.shape
        p = self.cfg.patch_size
        assert H % p == 0 and W % p == 0, \
            f"depth spatial dims ({H}x{W}) must be multiples of patch_size ({p})"
        gh, gw = H // p, W // p

        x = self.patchify(depth)                         # [B, dim, gh, gw]
        x = x.flatten(2).transpose(1, 2)                  # [B, gh*gw, dim]
        x = x + self._get_pos_embed(gh, gw, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        x = self.proj(x)
        return x.view(B, gh, gw, -1)


# ---------------------------------------------------------------------------
# Conv decoder building blocks
# ---------------------------------------------------------------------------

def _gn(num_channels: int, groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a divisor-safe group count (drops to next divisor if needed)."""
    g = groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvResBlock(nn.Module):
    """Pre-norm GroupNorm/SiLU residual block (SD-VAE style)."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.norm1 = _gn(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch, groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    """Single-head self-attention over spatial tokens. Cheap at low resolution.

    Used in the decoder bottleneck so the network can do non-local reasoning
    once before purely-local conv processing takes over at higher resolutions.
    """

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.norm = _gn(channels, groups)
        self.qkv  = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, C, -1).transpose(1, 2)              # [B, HW, C]
        k = k.view(B, C, -1)                               # [B, C, HW]
        v = v.view(B, C, -1).transpose(1, 2)              # [B, HW, C]
        attn = torch.softmax((q @ k) * (C ** -0.5), dim=-1)
        out  = (attn @ v).transpose(1, 2).view(B, C, H, W)
        return x + self.proj(out)


class _UpsampleConv(nn.Module):
    """Nearest-neighbor upsample + 3x3 conv (no checkerboard artifacts)."""

    def __init__(self, channels: int, scale_factor: float):
        super().__init__()
        self.scale = scale_factor
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Conv RGBD decoder
# ---------------------------------------------------------------------------

@dataclass
class ConvDecoderConfig:
    """Fully-convolutional decoder.

    Input  : fused latent [B, H, W, in_dim] (channel-last, matches existing API).
    Output : RGBD image   [B, out_channels, sH, sW]  where  s = prod(upsample_factors).

    For DINOv2-B (patch 14), the default (2, 7) -> s=14, which inverts the
    encoder's patchification exactly.

    Invariant: len(hidden_dims) == len(upsample_factors) + 1
    (one channel width per resolution stage, from latent res to full res).
    """
    in_dim:            int = 768 + 256
    hidden_dims:       Tuple[int, ...] = (512, 256, 128)
    upsample_factors:  Tuple[int, ...] = (2, 7)
    num_res_blocks:    int = 2
    groups:            int = 32
    out_channels:      int = 4
    use_mid_attention: bool = True
    # Depth head activation. Base variant: relative depth in [0,1] -> sigmoid.
    # DPF variant: unbounded metric depth -> softplus (a sigmoid would cap it at
    # 1 and make deeper scenes unreconstructable; softplus keeps it strictly
    # positive, as required by the log-space SILog loss). The default follows
    # REEF_VARIANT so DPF checkpoints are never silently decoded with the wrong
    # head; it can still be overridden explicitly.
    depth_activation:  str = field(default_factory=lambda: (
        "softplus" if (os.environ.get("BENTHICFLOW_VARIANT") or os.environ.get("REEF_VARIANT", "")).lower() == "dpf" else "sigmoid"))


class ConvDecoder(nn.Module):
    """Fully-convolutional RGBD decoder. Resolution-agnostic by construction."""

    def __init__(self, cfg: ConvDecoderConfig | None = None):
        super().__init__()
        cfg = cfg or ConvDecoderConfig()
        self.cfg = cfg
        assert len(cfg.hidden_dims) == len(cfg.upsample_factors) + 1, (
            "hidden_dims must have one more entry than upsample_factors"
        )

        # 1x1 conv from fused-latent channels to first hidden width.
        self.input_proj = nn.Conv2d(cfg.in_dim, cfg.hidden_dims[0], 1)

        # Bottleneck at latent resolution.
        mid_ch = cfg.hidden_dims[0]
        mid_layers: list[nn.Module] = [ConvResBlock(mid_ch, mid_ch, cfg.groups)]
        if cfg.use_mid_attention:
            mid_layers.append(SpatialSelfAttention(mid_ch, cfg.groups))
        mid_layers.append(ConvResBlock(mid_ch, mid_ch, cfg.groups))
        self.mid = nn.Sequential(*mid_layers)

        # Up stages. Channel reduction happens inside the first ResBlock of
        # each new stage (after the spatial upsample, which preserves channels).
        self.up_stages = nn.ModuleList()
        for i, factor in enumerate(cfg.upsample_factors):
            in_ch  = cfg.hidden_dims[i]
            out_ch = cfg.hidden_dims[i + 1]
            stage = nn.ModuleList([_UpsampleConv(in_ch, factor)])
            for j in range(cfg.num_res_blocks):
                stage.append(
                    ConvResBlock(in_ch if j == 0 else out_ch, out_ch, cfg.groups)
                )
            self.up_stages.append(stage)

        # Output head.
        last_ch = cfg.hidden_dims[-1]
        self.out_norm = _gn(last_ch, cfg.groups)
        self.out_conv = nn.Conv2d(last_ch, cfg.out_channels, 3, padding=1)

    def get_last_layer(self) -> nn.Parameter:
        """Used by adaptive GAN weighting."""
        return self.out_conv.weight

    def forward(self, fused_latent: torch.Tensor) -> torch.Tensor:
        # [B, H, W, C] -> [B, C, H, W]
        x = fused_latent.permute(0, 3, 1, 2).contiguous()
        x = self.input_proj(x)
        x = self.mid(x)
        for stage in self.up_stages:
            for layer in stage:
                x = layer(x)
        x = self.out_conv(F.silu(self.out_norm(x)))
        rgb   = torch.sigmoid(x[:, :3])
        if self.cfg.depth_activation == "softplus":
            depth = F.softplus(x[:, 3:4])
        else:
            depth = torch.sigmoid(x[:, 3:4])
        return torch.cat([rgb, depth], dim=1)


# ---------------------------------------------------------------------------
# Full RAE wrapper with noise-augmented decoder training
# ---------------------------------------------------------------------------

@dataclass
class RAEConfig:
    """Top-level RAE config."""
    depth_cfg:   DepthEncoderConfig | None = None
    decoder_cfg: ConvDecoderConfig  | None = None

    # Noise-augmented decoding (RAE paper Section 3.3).
    # tau controls the scale of injected noise: sigma ~ |N(0, tau^2)|.
    # Set tau = 0.0 to disable. Paper uses tau = 0.8 as default.
    noise_tau: float = 0.8


class RAE(nn.Module):
    """Trainable parts of the RGBD RAE: depth encoder + conv decoder + fusion.

    Frozen DINOv2-B is loaded externally and applied with no_grad. RAE only
    holds the trainable depth encoder and conv decoder, plus the fusion +
    noise injection logic on the latent.
    """

    def __init__(self, cfg: RAEConfig | None = None):
        super().__init__()
        cfg = cfg or RAEConfig()
        self.cfg = cfg
        depth_cfg   = cfg.depth_cfg   or DepthEncoderConfig()
        decoder_cfg = cfg.decoder_cfg or ConvDecoderConfig()

        self.depth_encoder = DepthEncoder(depth_cfg)
        self.decoder       = ConvDecoder(decoder_cfg)
        self.fused_norm    = TokenLayerNorm()

    # ----- pieces (useful when training the FM later) -----

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return self.depth_encoder(depth)

    def fuse(self, rgb_latent: torch.Tensor,
             depth_latent: torch.Tensor) -> torch.Tensor:
        return self.fused_norm(torch.cat([rgb_latent, depth_latent], dim=-1))

    def inject_noise(self, fused: torch.Tensor) -> torch.Tensor:
        """Noise-augmented decoder training (RAE paper Section 3.3).

        sigma ~ |N(0, tau^2)| sampled per-batch-element, broadcast over tokens.
        Disabled in eval() and when tau == 0.
        """
        if not self.training or self.cfg.noise_tau <= 0:
            return fused
        B = fused.shape[0]
        sigma = torch.randn(B, 1, 1, 1, device=fused.device) * self.cfg.noise_tau
        sigma = sigma.abs()
        return fused + torch.randn_like(fused) * sigma

    # ----- main forward -----

    def forward(self,
                rgb_latent: torch.Tensor,
                depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """rgb_latent: [B, H, W, 768] from frozen DINOv2-B (already normalized).
           depth:      [B, 1, 14H, 14W] in [0, 1].

        Returns:
          rgbd_pred: [B, 4, 14H, 14W] in [0, 1]
          fused:     [B, H, W, 1024] CLEAN fused latent (without noise) -- used
                                     by FM training and downstream tools.
        """
        depth_latent = self.encode_depth(depth)
        fused = self.fuse(rgb_latent, depth_latent)
        # Noise applied only on the path going to the decoder; the fused tensor
        # we return for downstream use is the clean version.
        decoder_input = self.inject_noise(fused)
        rgbd = self.decoder(decoder_input)
        return rgbd, fused