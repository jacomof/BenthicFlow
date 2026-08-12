# scripts/test_panorama.py
"""Panorama generation with the (unmasked) global-conditioned UNetCFM.

MultiDiffusion-style flow matching: a wide latent canvas is integrated with
overlapping 16x16 windows. Each window is conditioned on a linear interpolation
of two seed conditions (a Hawaii image on the left, a Batemans image on the
right), so the panorama transitions from one seabed type to the other. Window
velocities are blended in the overlaps and the whole latent is decoded once
(the RAE ConvDecoder is fully convolutional -> size-agnostic, no pos-embed
patching needed).

Seeds are real images sampled from each campaign; over/under-exposed (near-white
or near-black) or flat frames are rejected and resampled, and both seeds are
plotted next to the panorama so you can see what conditioned it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt

from benthicflow import CKPT_ROOT, FIG_ROOT, SCRATCH_ROOT
from models.unet_cfm import UNetCFM
from scripts.train_cfm_cfg import load_rae_frozen

CFM_RUN_NAME = "unet_cfm_final"
RAE_CKPT = CKPT_ROOT / "rae" / "last.pt"
GRID = 16                       # CFM latent grid (per tile)

LEFT_CAMPAIGN = "Hawaii201801"
RIGHT_CAMPAIGN = "Batemans201011"

# Shared (not node-local) source arrays; this is a one-off viz, a few reads.
RGB_SRC = SCRATCH_ROOT / "rgb"
FEAT_SRC = SCRATCH_ROOT / "features"


def sample_conditioning(campaign, rng, device,
                        white_thr=0.9, black_thr=0.1, min_std=0.03, max_tries=30, rgb_src=RGB_SRC, feat_src=FEAT_SRC):
    """Pick a well-exposed image from `campaign`.

    Returns (rgb_chw in [0,1] for display, cond [1,768] = mean-pooled DINO).
    Rejects near-white / near-black / near-flat frames and resamples, matching
    the request to avoid degenerate (over/under-saturated) seeds.
    """
    deployments = sorted(p.stem for p in (rgb_src / campaign).glob("*.npy"))
    if not deployments:
        raise FileNotFoundError(f"No RGB arrays under {rgb_src / campaign}")

    for attempt in range(1, max_tries + 1):
        dep = str(rng.choice(deployments))
        rgb_arr = np.load(rgb_src / campaign / f"{dep}.npy", mmap_mode="r")
        feat_arr = np.load(feat_src / campaign / f"{dep}.npy", mmap_mode="r")
        idx = int(rng.integers(len(rgb_arr)))

        rgb = np.asarray(rgb_arr[idx], dtype=np.float32) / 255.0       # [S, S, 3]
        bright, contrast = float(rgb.mean()), float(rgb.std())
        if bright > white_thr or bright < black_thr or contrast < min_std:
            why = "white" if bright > white_thr else "black" if bright < black_thr else "flat"
            print(f"  [reject {attempt:02d}] {campaign}/{dep}[{idx}] "
                  f"bright={bright:.3f} std={contrast:.3f} ({why})")
            continue

        print(f"  [accept]    {campaign}/{dep}[{idx}] bright={bright:.3f} std={contrast:.3f}")
        dino = torch.from_numpy(np.asarray(feat_arr[idx], dtype=np.float32))   # [G, G, 768]
        cond = dino.reshape(-1, dino.shape[-1]).mean(0, keepdim=True).to(device)  # [1, 768]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)                         # [3, S, S]
        return rgb_t, cond

    raise RuntimeError(f"No well-exposed image found in {campaign} after {max_tries} tries "
                       f"(thresholds: black<{black_thr}, white>{white_thr}, std<{min_std}).")


@torch.no_grad()
def generate_panorama(model, rae, cond_left, cond_right, n_tiles_w=4,
                      n_steps=100, stride=8, cfg_scale=3.0, temperature=1.0, device="cuda"):
    """Integrate a [16, 16*n_tiles_w] latent canvas with overlapping windows,
    each conditioned on lerp(cond_left, cond_right). Returns (rgb, depth) panos."""
    grid, D = GRID, model.latent_dim
    H, W = grid, grid * n_tiles_w
    null = model.null_cond.unsqueeze(0)                                  # [1, 768]

    # Hann-like blend window along the width so overlapping tiles fuse smoothly.
    win = torch.sin(torch.linspace(0.05, 0.95, grid, device=device) * torch.pi)
    win = win.view(1, 1, 1, grid)                                        # broadcasts over W

    x = torch.randn(1, D, H, W, device=device) * temperature            # [1, D, 16, W]
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = ts[1] - ts[0]
    starts = list(range(0, W - grid + 1, stride))

    for i in range(n_steps):
        t = ts[i].expand(1)
        vel = torch.zeros_like(x)
        wacc = torch.zeros(1, 1, 1, W, device=device)
        for w0 in starts:
            w1 = w0 + grid
            alpha = (w0 + grid / 2) / W                                  # 0=left .. 1=right
            cond_w = (1.0 - alpha) * cond_left + alpha * cond_right       # [1, 768]
            xc = x[:, :, :, w0:w1]                                        # [1, D, 16, 16]

            v_cond = model(xc, t, cond_w)
            if cfg_scale != 1.0:
                v_uncond = model(xc, t, null)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            vel[:, :, :, w0:w1] += v * win
            wacc[:, :, :, w0:w1] += win
        x = x + (vel / (wacc + 1e-8)) * dt

    z = model.denormalize(x)                                            # [1, D, 16, W]
    fused = z.permute(0, 2, 3, 1).contiguous()                         # [1, 16, W, D]
    rgbd = rae.decoder(fused)                                           # [1, 4, 16*14, W*14]
    return rgbd[:, :3].clamp(0, 1), rgbd[:, 3:4].clamp(0, 1)


def plot_panorama(seed_l, seed_r, pano_rgb, pano_depth, out_path, n_tiles_w):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_l = seed_l.permute(1, 2, 0).numpy()
    seed_r = seed_r.permute(1, 2, 0).numpy()
    pano_rgb_np = pano_rgb[0].permute(1, 2, 0).cpu().numpy()
    pano_depth_np = pano_depth[0, 0].cpu().numpy()

    fig = plt.figure(figsize=(2 + 2 * n_tiles_w + 2, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, n_tiles_w, 1], height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(seed_l); ax.set_title(f"Seed L\n{LEFT_CAMPAIGN}", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(pano_rgb_np); ax.set_title("RGB panorama (cond interp L→R)", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(seed_r); ax.set_title(f"Seed R\n{RIGHT_CAMPAIGN}", fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[1, :]); ax.imshow(pano_depth_np, cmap="turbo", vmin=0, vmax=1); ax.set_title("Depth panorama", fontsize=9); ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- models ----
    cfm_ckpt = CKPT_ROOT / CFM_RUN_NAME / "best_model.pt"
    model = UNetCFM().to(DEVICE)
    model.load_state_dict(torch.load(cfm_ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    rae = load_rae_frozen(RAE_CKPT, DEVICE)

    # ---- seeds (with saturation rejection) ----
    rng = np.random.default_rng()                  # fresh samples each run
    print(f"Sampling LEFT seed from {LEFT_CAMPAIGN}:")
    seed_l_rgb, cond_l = sample_conditioning(LEFT_CAMPAIGN, rng, DEVICE)
    print(f"Sampling RIGHT seed from {RIGHT_CAMPAIGN}:")
    seed_r_rgb, cond_r = sample_conditioning(RIGHT_CAMPAIGN, rng, DEVICE)

    # ---- panorama ----
    print("Generating condition-interpolated panorama...")
    pano_rgb, pano_depth = generate_panorama(
        model, rae, cond_l, cond_r,
        n_tiles_w=4, n_steps=100, stride=8, cfg_scale=3.0, temperature=1.0, device=DEVICE,
    )

    out = FIG_ROOT / CFM_RUN_NAME / "panorama_test.png"
    plot_panorama(seed_l_rgb, seed_r_rgb, pano_rgb, pano_depth, out, n_tiles_w=4)
    print(f"Done! Saved to {out}")
