"""Conditional Flow Matching over RAE fused latents.

Predicts rectified-flow velocity field v = z_1 - z_0 for generating 16x16 fused
RAE latents with mask conditioning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ===========================================================================
# Config
# ===========================================================================


@dataclass
class MaskedCFMConfig:
    grid_size: int = 16
    latent_dim: int = 1024
    model_dim: int = 1024
    depth: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0


# ===========================================================================
# Positional + time embeddings
# ===========================================================================


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """2D sin-cos position embedding -> [grid*grid, embed_dim]."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    pos = torch.arange(grid_size, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(pos, pos, indexing="xy")

    def axis(p: torch.Tensor) -> torch.Tensor:
        d_quarter = embed_dim // 4
        omega = torch.arange(d_quarter, dtype=torch.float32) / d_quarter
        omega = 1.0 / (10000**omega)
        out = p.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([out.sin(), out.cos()], dim=-1)

    return torch.cat([axis(grid_h), axis(grid_w)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding + MLP."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


# ===========================================================================
# DiT blocks
# ===========================================================================


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), sh1, sc1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = _modulate(self.norm2(x), sh2, sc2)
        h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        sh, sc = self.adaLN(c).chunk(2, dim=-1)
        x = _modulate(self.norm(x), sh, sc)
        return self.linear(x)


# ===========================================================================
# Masked CFM velocity model
# ===========================================================================


class MaskedCFM(nn.Module):
    """DiT velocity network. See module docstring."""

    def __init__(self, cfg: MaskedCFMConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 2 * cfg.latent_dim + 1  # z_t, M*z_known_t, M

        self.input_proj = nn.Linear(in_dim, cfg.model_dim)
        pos = get_2d_sincos_pos_embed(cfg.model_dim, cfg.grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.t_embed = TimestepEmbedder(cfg.model_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(cfg.model_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.final = FinalLayer(cfg.model_dim, cfg.latent_dim)

        self.register_buffer("latent_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("latent_std", torch.ones(cfg.latent_dim))

    def normalize(self, z):
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z):
        return z * self.latent_std + self.latent_mean

    def forward(self, z_t, t, mask, z_known):
        """Inputs:
        z_t      [B, N, D]   noisy latent
        t        [B]         in [0, 1]
        mask     [B, N, 1]   1 where token is conditioned
        z_known  [B, N, D]   already-noised conditioning at time t
                             (caller's responsibility; see cfm_loss
                             and cfm_sample for the noising)
        """
        cond = z_known * mask
        x = torch.cat([z_t, cond, mask], dim=-1)
        x = self.input_proj(x) + self.pos_embed
        c = self.t_embed(t)
        for block in self.blocks:
            x = block(x, c)
        return self.final(x, c)


# ===========================================================================
# Mask sampling
# ===========================================================================


def sample_rectangle_mask(
    B: int,
    grid: int = 16,
    p_uncond: float = 0.30,
    p_full: float = 0.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Random axis-aligned rectangle as the *known* region (M=1).

    Returns: [B, grid, grid, 1] float in {0, 1}.
    """
    h = torch.randint(1, grid + 1, (B,), device=device)
    w = torch.randint(1, grid + 1, (B,), device=device)
    y0 = (torch.rand(B, device=device) * (grid - h + 1).float()).long()
    x0 = (torch.rand(B, device=device) * (grid - w + 1).float()).long()

    ys = torch.arange(grid, device=device).view(1, grid, 1)
    xs = torch.arange(grid, device=device).view(1, 1, grid)
    mask = (
        (ys >= y0.view(B, 1, 1))
        & (ys < (y0 + h).view(B, 1, 1))
        & (xs >= x0.view(B, 1, 1))
        & (xs < (x0 + w).view(B, 1, 1))
    ).float()

    r = torch.rand(B, device=device).view(B, 1, 1)
    is_uncond = (r < p_uncond).float()
    is_full = ((r >= p_uncond) & (r < p_uncond + p_full)).float()
    mask = mask * (1 - is_uncond) * (1 - is_full) + is_full
    return mask.unsqueeze(-1)


# ===========================================================================
# Training loss
# ===========================================================================


def cfm_loss(
    model: MaskedCFM,
    z1_norm: torch.Tensor,
    p_uncond: float = 0.30,
    p_full: float = 0.0,
    loss_on: str = "all",
) -> dict:
    """Masked rectified-flow loss with noised conditioning.

    z1_norm: [B, N, D]  target latent, already in the model's normalized space.

    The conditioning channel sees z_known_t (z_1 noised to the same level
    as z_t), not the clean z_1. This forces the model to denoise on known
    tokens just like on unknown tokens — no algebraic shortcut to exploit.
    """
    B, N, D = z1_norm.shape
    grid = int(math.isqrt(N))
    device = z1_norm.device

    # Mask
    mask = sample_rectangle_mask(
        B, grid=grid, p_uncond=p_uncond, p_full=p_full, device=device
    )
    mask = mask.view(B, N, 1)

    # Forward process: independent noise for z_t and for z_known_t
    t = torch.rand(B, device=device)
    t_ = t.view(B, 1, 1)
    z0 = torch.randn_like(z1_norm)
    eps_cond = torch.randn_like(z1_norm)

    zt = (1.0 - t_) * z0 + t_ * z1_norm  # the target trajectory
    z_known_t = (1.0 - t_) * eps_cond + t_ * z1_norm  # the noised condition
    v_target = z1_norm - z0  # rectified-flow target

    v_pred = model(zt, t, mask, z_known_t)

    if loss_on == "unknown":
        w = 1.0 - mask  # [B, N, 1]
        denom = w.sum() * D + 1e-8
        loss = ((v_pred - v_target).pow(2) * w).sum() / denom
    elif loss_on == "all":
        loss = (v_pred - v_target).pow(2).mean()
    else:
        raise ValueError(f"loss_on must be 'unknown' or 'all', got {loss_on}")

    return {"loss": loss, "frac_known": float(mask.mean())}


# ===========================================================================
# Euler sampler with noised conditioning + anchored known tokens
# ===========================================================================


@torch.no_grad()
def cfm_sample(
    model: MaskedCFM,
    z_known_norm: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int = 50,
    anchor_known: bool = True,
) -> torch.Tensor:
    """Generate a latent by Euler integration of the velocity field.
    Inputs in model's normalized space; caller denormalizes the return.
    z_known_norm: [B, N, D]   clean tokens (used only where mask=1)
    mask:         [B, N, 1]   1 = condition this token to z_known_norm
    At every step we noise z_known_norm to the current t and pass that
    """
    device = z_known_norm.device
    B, N, D = z_known_norm.shape
    z = torch.randn(B, N, D, device=device)

    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i + 1]

        # Noise the conditioning to the current t (matches training).
        eps_cond = torch.randn_like(z)
        z_known_t = (1.0 - t_cur) * eps_cond + t_cur * z_known_norm

        if anchor_known:
            # Force known tokens onto the same noised trajectory.
            z = mask * z_known_t + (1.0 - mask) * z

        v = model(z, t_cur.expand(B), mask, z_known_t)
        z = z + (t_next - t_cur) * v

    # Pin known tokens exactly at t=1.
    z = mask * z_known_norm + (1.0 - mask) * z
    return z


# ===========================================================================
# MultiDiffusion-style tiled sampler for panorama growth
# ===========================================================================


@torch.no_grad()
def cfm_sample_tiled(
    model: MaskedCFM,
    canvas_known_norm: torch.Tensor,
    canvas_mask: torch.Tensor,
    tile: int = 16,
    stride: int = 8,
    n_steps: int = 50,
    anchor_known: bool = True,
) -> torch.Tensor:
    """MultiDiffusion sampler over an arbitrarily large canvas.
    canvas_known_norm: [1, H, W, D]   normalized known content
    canvas_mask:       [1, H, W, 1]   1 where condition is provided
    At every Euler step we (a) noise the canvas's known content to t,
    (b) optionally anchor z's known tokens to that noised target, and
    """
    assert canvas_known_norm.shape[0] == 1, "tiled sampler is single-canvas"
    device = canvas_known_norm.device
    _, H, W, D = canvas_known_norm.shape
    assert H >= tile and W >= tile
    G = model.cfg.grid_size
    assert tile == G, f"tile size {tile} must match model.grid_size {G}"

    def origins(L: int) -> list[int]:
        xs = list(range(0, L - tile + 1, stride))
        if xs[-1] != L - tile:
            xs.append(L - tile)
        return xs

    y_origins = origins(H)
    x_origins = origins(W)

    z = torch.randn(1, H, W, D, device=device)

    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i + 1]

        # Noise the canvas's known content to current t
        eps_cond = torch.randn_like(z)
        canvas_known_t = (1.0 - t_cur) * eps_cond + t_cur * canvas_known_norm

        if anchor_known:
            z = canvas_mask * canvas_known_t + (1.0 - canvas_mask) * z

        # Accumulate velocity per token (averaged over containing tiles)
        v_sum = torch.zeros_like(z)
        v_cnt = torch.zeros(1, H, W, 1, device=device)

        for y in y_origins:
            for x in x_origins:
                z_tile = z[:, y : y + tile, x : x + tile, :]
                k_tile = canvas_known_t[:, y : y + tile, x : x + tile, :]
                m_tile = canvas_mask[:, y : y + tile, x : x + tile, :]
                z_flat = z_tile.reshape(1, tile * tile, D)
                k_flat = k_tile.reshape(1, tile * tile, D)
                m_flat = m_tile.reshape(1, tile * tile, 1)
                v_flat = model(z_flat, t_cur.expand(1), m_flat, k_flat)
                v_tile = v_flat.reshape(1, tile, tile, D)
                v_sum[:, y : y + tile, x : x + tile, :] += v_tile
                v_cnt[:, y : y + tile, x : x + tile, :] += 1.0

        v = v_sum / v_cnt
        z = z + (t_next - t_cur) * v

    # Pin known tokens to clean values at t=1
    z = canvas_mask * canvas_known_norm + (1.0 - canvas_mask) * z
    return z
