import argparse
import json
from pathlib import Path
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from tqdm import tqdm

from models.unet_cfm import UNetCFM, cfm_sample_cfg
from scripts.train_cfm_cfg import load_rae_frozen

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def load_models(rae_ckpt: str, cfm_ckpt: str, device: torch.device):
    """Loads frozen RAE and trained UNetCFM."""
    print("Loading RAE...")
    rae = load_rae_frozen(Path(rae_ckpt), device)

    print("Loading UNetCFM...")
    cfm = UNetCFM().to(device)
    ck_cfm = torch.load(cfm_ckpt, map_location=device, weights_only=True)
    state = ck_cfm.get("ema", ck_cfm.get("model", ck_cfm))
    cfm.load_state_dict(state, strict=False)

    if "model" in ck_cfm:
        cfm.latent_mean.copy_(ck_cfm["model"].get("latent_mean", cfm.latent_mean))
        cfm.latent_std.copy_(ck_cfm["model"].get("latent_std", cfm.latent_std))
    elif "latent_mean" in ck_cfm:
        cfm.latent_mean.copy_(ck_cfm["latent_mean"])
        cfm.latent_std.copy_(ck_cfm["latent_std"])

    cfm.eval().requires_grad_(False)
    return rae, cfm

def tensor_to_mesh(rgb_tensor, depth_tensor, out_prefix, min_d=3.0, max_d=4.5, focal_mm=14.0):
    """Converts generated RGB-D tensors to highly optimized PLY point clouds and meshes."""
    rgb = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    depth_norm = depth_tensor.squeeze().cpu().numpy()
    
    depth_z = depth_norm * (max_d - min_d) + min_d
    h, w = rgb.shape[:2]
    
    sensor_width_mm = 17.3
    fx = focal_mm * (w / sensor_width_mm)
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    x = (u - cx) * depth_z / fx
    y = (v - cy) * depth_z / fy

    points = np.stack((x, y, depth_z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.0003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(f"{out_prefix}_points.ply", pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    o3d.io.write_triangle_mesh(f"{out_prefix}_mesh.ply", mesh)
    print(f"Saved optimized 3D assets to {out_prefix}_mesh.ply")

@torch.no_grad()
def cfm_sample_multidiffusion_hybrid(model: UNetCFM, cond_left: torch.Tensor | None = None,
                                     cond_right: torch.Tensor | None = None,
                                     tiles_h: int = 2, tiles_w: int = 4,
                                     n_steps: int = 150, cfg_scale: float = 3.0,
                                     window_overlap: float = 0.1) -> torch.Tensor:
    """
    MultiDiffusion sampling with UNetCFM across overlapping 16x16 windows.
    Interpolates conditioning vectors (cond_left to cond_right) across the panorama width.
    """
    device = next(model.parameters()).device
    grid = 16
    D = model.latent_dim
    H, W = grid * tiles_h, grid * tiles_w

    if cond_left is None:
        cond_left = model.null_cond.unsqueeze(0)
    if cond_right is None:
        cond_right = model.null_cond.unsqueeze(0)

    # 2D Hann window for smooth spatial blending across overlapping tiles
    win_h = torch.sin(torch.linspace(0.05, 0.95, grid, device=device) * torch.pi).view(1, 1, grid, 1)
    win_w = torch.sin(torch.linspace(0.05, 0.95, grid, device=device) * torch.pi).view(1, 1, 1, grid)
    win = win_h * win_w

    z = torch.randn(1, D, H, W, device=device)
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = ts[1] - ts[0]

    stride = int(grid * (1.0 - window_overlap))
    y_origins = list(range(0, H - grid + 1, stride))
    if y_origins[-1] != H - grid:
        y_origins.append(H - grid)
    x_origins = list(range(0, W - grid + 1, stride))
    if x_origins[-1] != W - grid:
        x_origins.append(W - grid)

    null_cond = model.null_cond.unsqueeze(0)

    print(f"Using stride {stride} for UNetCFM MultiDiffusion sampling.")

    for i in range(n_steps):
        t_cur = ts[i].expand(1)
        vel = torch.zeros_like(z)
        wacc = torch.zeros(1, 1, H, W, device=device)

        for y0 in y_origins:
            y1 = y0 + grid
            for x0 in x_origins:
                x1 = x0 + grid
                # Interpolate conditioning across panorama width
                alpha = (x0 + grid / 2.0) / W
                cond_w = (1.0 - alpha) * cond_left + alpha * cond_right

                zc = z[:, :, y0:y1, x0:x1]

                v_cond = model(zc, t_cur, cond_w)
                if cfg_scale != 1.0:
                    v_uncond = model(zc, t_cur, null_cond)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond

                vel[:, :, y0:y1, x0:x1] += v * win
                wacc[:, :, y0:y1, x0:x1] += win

        z = z + (vel / (wacc + 1e-8)) * dt

    return z

@torch.no_grad()
def generate_fig1_panorama(rae, cfm, latents_mm, device, out_dir, window_overlap=0.1, avg_end_t=1.0, num_image=None):
    """Generates a panorama using MultiDiffusion with UNetCFM."""
    print(f"Generating Figure {num_image} (MultiDiffusion Panorama with UNetCFM)...")
    
    cond_left, cond_right = None, None
    if latents_mm is not None and len(latents_mm) > 0:
        num_records = len(latents_mm)
        idx_start, idx_end = np.random.choice(num_records, size=2, replace=False)
        feat_l = np.array(latents_mm[idx_start])
        feat_r = np.array(latents_mm[idx_end])
        cond_left = torch.from_numpy(feat_l).float().reshape(1, -1).to(device)[:, :768]
        cond_right = torch.from_numpy(feat_r).float().reshape(1, -1).to(device)[:, :768]
    else:
        print("Warning: No latent cache provided, generating unconditional panorama.")

    z_canvas = cfm_sample_multidiffusion_hybrid(
        cfm, cond_left, cond_right, tiles_h=2, tiles_w=4,
        n_steps=150, cfg_scale=3.0, window_overlap=window_overlap
    )

    z_denorm = cfm.denormalize(z_canvas)
    fused = z_denorm.permute(0, 2, 3, 1).contiguous()
    rgbd = rae.decoder(fused)
    rgb, depth = rgbd[0, :3].clamp(0, 1), rgbd[0, 3:4].clamp(0, 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(rgb.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("UNetCFM MultiDiffusion Panorama (RGB)")
    axes[0].axis("off")
    
    im = axes[1].imshow(depth.squeeze().cpu().numpy(), cmap="viridis")
    axes[1].set_title("UNetCFM MultiDiffusion Panorama (Depth)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.015, pad=0.02)
    
    out_file = out_dir / f"fig1_panorama_{num_image}.pdf"
    print(f"Saving Figure {num_image} to {out_file}")
    plt.savefig(out_file)
    plt.close()

@torch.no_grad()
def generate_fig2_conditional(rae, cfm, cond, device, out_dir):
    """Shows unconditional generation vs conditioned UNetCFM generation."""
    print("Generating Figure 2 (Unconditional vs Conditional UNetCFM)...")
    
    # 1. Unconditional (CFG=1.0)
    z_uncond = cfm_sample_cfg(cfm, cfm.null_cond.unsqueeze(0), canvas_shape=(16, 16), n_steps=150, cfg_scale=1.0)
    rgbd_uncond = rae.decoder(cfm.denormalize(z_uncond).permute(0, 2, 3, 1).contiguous()).clamp(0, 1)[0]
    
    # 2. Conditional (CFG=3.0)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    cond = cond.to(device)
    z_cond = cfm_sample_cfg(cfm, cond, canvas_shape=(16, 16), n_steps=150, cfg_scale=3.0)
    rgbd_cond = rae.decoder(cfm.denormalize(z_cond).permute(0, 2, 3, 1).contiguous()).clamp(0, 1)[0]
    
    decoded = [rgbd_uncond, rgbd_cond]
    titles = ["Unconditional Generation", "Conditioned Generation (CFG=3.0)"]
    
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.05)
    
    for i, (img, title) in enumerate(zip(decoded, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img[:3].permute(1, 2, 0).cpu().numpy())
        ax.set_title(title)
        ax.axis("off")
        
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img[3].cpu().numpy(), cmap="viridis")
        ax.axis("off")
        
    plt.savefig(out_dir / "fig2_conditional.pdf")
    plt.close()

@torch.no_grad()
def generate_fig3_trajectory(rae, cfm, device, out_dir):
    """Visualizes the predicted endpoint z_1 trajectories over integration steps."""
    print("Generating Figure 3 (ODE Trajectory Prediction)...")
    
    B, C, H, W = 1, cfm.latent_dim, 16, 16
    z_t = torch.randn(B, C, H, W, device=device)
    null_cond = cfm.null_cond.unsqueeze(0).to(device)
    
    steps = 150
    ts = torch.linspace(0, 1, steps + 1, device=device)
    dt = ts[1] - ts[0]
    
    save_idx = [0, steps // 4, steps // 2, 3 * steps // 4, steps - 1]
    saved_z = []

    for i in range(steps):
        t_cur = ts[i].expand(B)
        v_pred = cfm(z_t, t_cur, null_cond)
        
        z_1_pred = z_t + (1.0 - ts[i]) * v_pred
        
        if i in save_idx:
            saved_z.append(z_1_pred.clone())
            
        z_t = z_t + v_pred * dt

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, len(saved_z), wspace=0.02, hspace=0.02)
    
    for i, z in enumerate(saved_z):
        fused = cfm.denormalize(z).permute(0, 2, 3, 1).contiguous()
        rgbd = rae.decoder(fused).clamp(0, 1)[0]
        t_val = save_idx[i] / steps
        
        ax_rgb = fig.add_subplot(gs[0, i])
        ax_rgb.imshow(rgbd[:3].permute(1, 2, 0).cpu().numpy())
        ax_rgb.set_title(f"t = {t_val:.2f}")
        ax_rgb.axis("off")
        
        ax_depth = fig.add_subplot(gs[1, i])
        ax_depth.imshow(rgbd[3].cpu().numpy(), cmap="viridis")
        ax_depth.axis("off")
        
    plt.savefig(out_dir / "fig3_trajectory.pdf")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae-ckpt", type=str, required=True)
    parser.add_argument("--cfm-ckpt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--out-dir", type=str, default="figures_paper")
    parser.add_argument("--window-overlap", type=float, default=0.1, help="Overlap ratio for MultiDiffusion tiles")
    parser.add_argument("--avg-end-t", type=float, default=1.0, help="Time to switch in MultiDiffusion")
    parser.add_argument("--num-images", type=int, default=1, help="Number of diverse panorama images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rae, cfm = load_models(args.rae_ckpt, args.cfm_ckpt, device)

    latents_mm = None
    real_latent = None
    if args.cache_dir:
        try:
            latents_path = Path(args.cache_dir) / "latents.npy"
            latents_mm = np.load(latents_path, mmap_mode="r")
            real_latent = torch.from_numpy(np.array(latents_mm[0])).float().reshape(-1, 768).mean(0, keepdim=True)
        except Exception as e:
            print(f"Warning: Could not load real latents from {args.cache_dir}: {e}")
            latents_mm = None
    
    for num_image in range(args.num_images):
        generate_fig1_panorama(rae, cfm, latents_mm, device, out_dir, args.window_overlap, args.avg_end_t, num_image)

    print(f"Success! Optimized assets and figures compiled to {out_dir}")