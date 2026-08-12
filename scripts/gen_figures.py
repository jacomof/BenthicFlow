import argparse
import json
from pathlib import Path
 
import numpy as np
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from tqdm import tqdm
 
# Adjust these imports based on your exact project structure
from models.rae import RAE, RAEConfig, DepthEncoderConfig, ConvDecoderConfig
from models.cfm import MaskedCFM, MaskedCFMConfig, cfm_sample
 

 
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
    """Loads frozen RAE and trained CFM."""
    print("Loading RAE...")
    rae_cfg = RAEConfig(
        depth_cfg=DepthEncoderConfig(patch_size=14, dim=256, depth=8, num_heads=8, out_dim=256),
        decoder_cfg=ConvDecoderConfig(
            in_dim=768 + 256, hidden_dims=(512, 256, 128), upsample_factors=(2, 7),
            num_res_blocks=2, groups=32, out_channels=4, use_mid_attention=True
        ),
        noise_tau=0.8,
    )
    rae = RAE(rae_cfg).to(device)
    ck_rae = torch.load(rae_ckpt, map_location=device, weights_only=True)
    rae.load_state_dict(ck_rae.get("ema", ck_rae.get("rae", ck_rae)), strict=False)
    rae.eval().requires_grad_(False)
 
    print("Loading CFM...")
    ck_cfm = torch.load(cfm_ckpt, map_location=device, weights_only=True)
    args = ck_cfm["args"]
    cfm_cfg = MaskedCFMConfig(
        grid_size=16, latent_dim=args["model_dim"], model_dim=args["model_dim"],
        depth=args["model_depth"], num_heads=args["model_heads"], mlp_ratio=4.0, dropout=0.0
    )
    cfm = MaskedCFM(cfm_cfg).to(device)
    cfm.load_state_dict(ck_cfm["ema"])
    
    if "model" in ck_cfm:
        cfm.latent_mean.copy_(ck_cfm["model"].get("latent_mean", cfm.latent_mean))
        cfm.latent_std.copy_(ck_cfm["model"].get("latent_std", cfm.latent_std))
        
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
def cfm_sample_multidiffusion_hybrid(model, canvas_known, canvas_mask, n_steps=150, avg_end_t=1.0, window_overlap=0.1):
    """
    Hybrid MultiDiffusion: Averages overlapping tiles early to build global structure,
    but switches to independent hard-overwrites late in the ODE to prevent textural ghosting.
    """
    device = canvas_known.device
    _, H, W, D = canvas_known.shape
    grid = model.cfg.grid_size
 
    z = torch.randn(1, H, W, D, device=device)
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    stride = int(grid * (1 - window_overlap))
    print(f"Using stride {stride} for hybrid multi-diffusion sampling.")
    print(f"Averaging until t = {avg_end_t} and using overlap ratio of {window_overlap} for tile sampling.")

    
    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        is_averaging = t_cur <= avg_end_t
        y_origins = list(range(0, H - grid + 1, stride))
        if y_origins[-1] != H - grid:
            y_origins.append(H - grid)
        x_origins = list(range(0, W - grid + 1, stride))
        if x_origins[-1] != W - grid:
            x_origins.append(W - grid)
 
        v_sum = torch.zeros_like(z)
        v_cnt = torch.zeros(1, H, W, 1, device=device)
 
        for y in y_origins:
            for x in x_origins:
                z_tile = z[:, y:y + grid, x:x + grid, :]
                k_tile = torch.zeros_like(z_tile)  # No known latents in this setting
                m_tile = torch.zeros_like(z_tile[:, :, :, :1])  # No mask in this setting
                
                v_flat = model(
                    z_tile.reshape(1, grid * grid, D),
                    t_cur.expand(1),
                    m_tile.reshape(1, grid * grid, 1),
                    k_tile.reshape(1, grid * grid, D)
                )
                v_tile = v_flat.reshape(1, grid, grid, D)
                
                if is_averaging:
                    v_sum[:, y:y + grid, x:x + grid, :] += v_tile
                    v_cnt[:, y:y + grid, x:x + grid, :] += 1.0
                else:
                    v_sum[:, y:y + grid, x:x + grid, :] = v_tile
                    v_cnt[:, y:y + grid, x:x + grid, :] = 1.0
 
        v = v_sum / v_cnt
        z = z + (t_next - t_cur) * v
 
    # Final spatial constraint
    z = canvas_mask * canvas_known + (1.0 - canvas_mask) * z
    return z
 
@torch.no_grad()
def generate_fig1_panorama(rae, cfm, latents_mm, device, out_dir, window_overlap=0.1, avg_end_t=1.0, num_image=None):
    """Generates a highly diverse panorama using Sparse Seeding and Hybrid MultiDiffusion."""
    print(f"Generating Figure {num_image} (Hybrid MultiDiffusion Panorama)...")
    grid = cfm.cfg.grid_size
    print(f"Using grid size: {grid}")
    D = cfm.cfg.latent_dim
    tiles_h, tiles_w = 2, 4  
    H, W = grid * tiles_h, grid * tiles_w
    
    canvas = torch.ones(1, H, W, D, device=device)
    mask = torch.zeros(1, H, W, 1, device=device)

    print(f"latents_mm is: {latents_mm}")
    if latents_mm is not None:
        num_records = len(latents_mm)
        
        num_random_patches = 15
        for _ in range(num_random_patches):
            rand_idx = np.random.randint(0, num_records)
            z_rand = torch.from_numpy(np.array(latents_mm[rand_idx])).float().to(device)
            z_rand_norm = cfm.normalize(z_rand).view(1, grid, grid, D)
            
            r = torch.randint(0, H - 2, (1,)).item()
            c = torch.randint(0, W - 2, (1,)).item()
            r_z = torch.randint(0, grid - 2, (1,)).item()
            c_z = torch.randint(0, grid - 2, (1,)).item()
            
            canvas[:, r:r+2, c:c+2, :] = z_rand_norm[:, r_z:r_z+2, c_z:c_z+2, :]
            mask[:, r:r+2, c:c+2, :] = 1.0
 
        # Anchor extreme ends
        idx_start, idx_end = np.random.randint(0, num_records, size=2)
        z_start = cfm.normalize(torch.from_numpy(np.array(latents_mm[idx_start])).float().to(device)).view(1, grid, grid, D)
        z_end = cfm.normalize(torch.from_numpy(np.array(latents_mm[idx_end])).float().to(device)).view(1, grid, grid, D)
        
        canvas[:, 0:grid, 0:grid, :] = z_start
        mask[:, 0:grid, 0:grid, :] = 1.0
        canvas[:, -grid:, -grid:, :] = z_end
        mask[:, -grid:, -grid:, :] = 1.0
    else:
        print("Warning: No latent cache provided, generating pure random panorama.")
        canvas = torch.zeros(1, H, W, D, device=device)
        mask = torch.zeros(1, H, W, 1, device=device)
 
    z_canvas = cfm_sample_multidiffusion_hybrid(cfm, canvas, mask, n_steps=150, avg_end_t=avg_end_t, window_overlap=window_overlap)
 
    z_denorm = cfm.denormalize(z_canvas)
    rgbd = rae.decoder(z_denorm)
    rgb, depth = rgbd[0, :3].clamp(0, 1), rgbd[0, 3:4].clamp(0, 1)
 
    #tensor_to_mesh(rgb, depth, str(out_dir / "fig1_panorama"))
 
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(rgb.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Hybrid MultiDiffusion Panorama (RGB)")
    axes[0].axis("off")
    
    im = axes[1].imshow(depth.squeeze().cpu().numpy(), cmap="viridis")
    axes[1].set_title("Hybrid MultiDiffusion Panorama (Depth)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.015, pad=0.02)
    
    print(f"Saving Figure {num_image} to {out_dir / f'fig1_panorama_{num_image}.pdf'}")
    plt.savefig(out_dir / f"fig1_panorama_{num_image}.pdf")
    plt.close()
 
@torch.no_grad()
def generate_fig2_conditional(rae, cfm, real_latent, device, out_dir):
    """Shows original, masked context (with random scattered patches), and generation."""
    print("Generating Figure 2 (Conditional with Scattered Patches)...")
    grid = cfm.cfg.grid_size
    z_real = cfm.normalize(real_latent.to(device).unsqueeze(0))
    
    mask = torch.zeros(1, grid, grid, 1, device=device)
    
    mask[:, 0:2, :, :] = 1.0
    mask[:, -2:, :, :] = 1.0
    mask[:, :, 0:2, :] = 1.0
    mask[:, :, -2:, :] = 1.0
    
    num_random_patches = 10
    for _ in range(num_random_patches):
        r = torch.randint(2, grid-3, (1,)).item()
        c = torch.randint(2, grid-3, (1,)).item()
        mask[:, r:r+2, c:c+2, :] = 1.0
        
    mask_seq = mask.reshape(1, -1, 1)
    
    z_pred = cfm_sample(cfm, z_real, mask_seq, n_steps=150)
    
    rgbd_real = rae.decoder(cfm.denormalize(z_real.reshape(1, grid, grid, -1))).clamp(0, 1)[0]
    rgbd_pred = rae.decoder(cfm.denormalize(z_pred.reshape(1, grid, grid, -1))).clamp(0, 1)[0]
    
    pixel_mask = F.interpolate(mask.permute(0, 3, 1, 2), size=rgbd_real.shape[1:], mode='nearest')[0]
    rgbd_masked = rgbd_real * pixel_mask
    
    decoded = [rgbd_real, rgbd_masked, rgbd_pred]
    titles = ["Ground Truth", "Scattered Conditioning", "CFM Generation"]
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, wspace=0.05, hspace=0.05)
    
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
    """Visualizes the clean manifold target vector field predictions over time."""
    print("Generating Figure 3 (ODE Trajectory Prediction)...")
    
    b, L, D = 1, cfm.cfg.grid_size**2, cfm.cfg.latent_dim
    z_t = torch.randn(b, L, D, device=device)
    
    steps = 150
    t_vals = torch.linspace(0, 1, steps, device=device)
    
    save_idx = [0, steps//4, steps//2, 3*steps//4, steps-1]
    saved_z = []
 
    dummy_mask = torch.zeros(b, L, 1, device=device)
    dummy_z_known = torch.zeros_like(z_t)
 
    for i in range(steps):
        t = t_vals[i] * torch.ones(b, device=device)
        v_pred = cfm(z_t, t, dummy_mask, dummy_z_known)
        
        z_1_pred = z_t + (1.0 - t_vals[i]) * v_pred
        
        if i in save_idx:
            saved_z.append(z_1_pred.clone())
            
        z_t = z_t + v_pred * (1.0 / steps)
 
    if (steps - 1) not in save_idx:
        saved_z.append(z_1_pred.clone())
 
    grid = cfm.cfg.grid_size
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, len(saved_z), wspace=0.02, hspace=0.02)
    
    for i, z in enumerate(saved_z):
        rgbd = rae.decoder(cfm.denormalize(z.reshape(1, grid, grid, -1))).clamp(0, 1)[0]
        t_val = save_idx[i] / steps if i < len(save_idx) else 1.0
        
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
    parser.add_argument("--window-overlap", type=float, default=0.1, help="Overlap ratio for hybrid multi-diffusion tiles")
    parser.add_argument("--avg-end-t", type=float, default=1.0, help="Time to switch from averaging to hard-overwrite in hybrid multi-diffusion")
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
    try:
        latents_path = Path(args.cache_dir) / "latents.npy"
        latents_mm = np.load(latents_path, mmap_mode="r")
        real_latent = torch.from_numpy(np.array(latents_mm[0])).float()
    except Exception as e:
        print(f"Warning: Could not load real latents from {args.cache_dir}: {e}")
        latents_mm = None
    
    for num_image in range(args.num_images):
        generate_fig1_panorama(rae, cfm, latents_mm, device, out_dir, args.window_overlap, args.avg_end_t, num_image)
    # generate_fig3_trajectory(rae, cfm, device, out_dir)
 
    # if real_latent is not None:
    #     generate_fig2_conditional(rae, cfm, real_latent, device, out_dir)
 
    print(f"Success! Optimized assets and figures compiled to {out_dir}")
 