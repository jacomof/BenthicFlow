#!/usr/bin/env python3
"""
Lift a single RGB image into a 2.5D point cloud / Gaussian surfel splat.

Pipeline:
  RGB image
    -> Depth Anything relative disparity
    -> robust normalize disparity to [0,1]
    -> invert to depth01 = 1 - disparity01
    -> unproject RGBD into a point cloud
    -> build surface-aligned flat Gaussian surfels
    -> save pointcloud.ply, surfels.ply, depth maps, front/tilted renders, panel

This is intended for method-overview figures, not metric reconstruction.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


SH_C0 = 0.28209479177387814


# =============================================================================
# I/O and Depth Anything
# =============================================================================

def load_rgb(path: Path, max_side: int | None, device: str):
    img = Image.open(path).convert("RGB")

    if max_side is not None and max_side > 0:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    rgb = torch.from_numpy(arr).to(device)
    return img, rgb


def robust_normalize(x: torch.Tensor, q_low=0.02, q_high=0.98, eps=1e-8):
    """Robust min-max normalization to [0,1]."""
    lo = torch.quantile(x.flatten(), q_low)
    hi = torch.quantile(x.flatten(), q_high)
    x = (x - lo) / (hi - lo + eps)
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def estimate_depth_anything(
    image_pil: Image.Image,
    model_id: str,
    device: str,
    invert: bool = True,
):
    """
    Depth Anything output is treated as relative disparity / inverse depth.

    We first normalize the raw prediction to disparity01 in [0,1].
    Then, by default, invert it:
        depth01 = 1 - disparity01

    So larger depth01 means farther away in the later unprojection.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)

    pred = outputs.predicted_depth  # [B, h, w]
    H, W = image_pil.height, image_pil.width

    pred = F.interpolate(
        pred.unsqueeze(1),
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )[0, 0].float()

    disparity01 = robust_normalize(pred)

    if invert:
        depth01 = 1.0 - disparity01
    else:
        depth01 = disparity01

    depth01 = depth01.clamp(0.0, 1.0)
    return depth01, disparity01


# =============================================================================
# Geometry
# =============================================================================

def make_K(fx, fy, w, h, device):
    return torch.tensor(
        [
            [fx, 0.0, (w - 1) / 2.0],
            [0.0, fy, (h - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def unproject(depth01, focal, z_near, z_span):
    """
    Convert depth01 into camera-space XYZ.

    Camera convention follows the original script:
      X right, Y down, Z forward.
    """
    H, W = depth01.shape
    dev = depth01.device

    vv, uu = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    Z = z_near + depth01 * z_span
    X = (uu - cx) * Z / focal
    Y = (vv - cy) * Z / focal

    return torch.stack([X, Y, Z], dim=-1)


def look_at_viewmat(eye, target, up, device):
    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)

    fwd = F.normalize(target - eye, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, up), dim=0)
    true_up = torch.linalg.cross(right, fwd)

    R = torch.stack([right, -true_up, fwd], dim=0)

    vm = torch.eye(4, device=device)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ eye
    return vm


def _rot_x(a, device):
    c, s = np.cos(a), np.sin(a)
    return torch.tensor(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=torch.float32,
        device=device,
    )


def _rot_y(a, device):
    c, s = np.cos(a), np.sin(a)
    return torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=torch.float32,
        device=device,
    )


def tilt_viewmat(points, tilt_deg, azim_deg, device):
    """
    Orbit the source camera around the scene centroid.
    This reveals relief while keeping disocclusions moderate.
    """
    c0 = points.reshape(-1, 3).mean(dim=0)
    eye0 = torch.zeros(3, device=device)

    Rr = _rot_y(np.deg2rad(azim_deg), device) @ _rot_x(np.deg2rad(tilt_deg), device)
    eye = c0 + Rr @ (eye0 - c0)

    return look_at_viewmat(eye, c0, (0, -1, 0), device)


# =============================================================================
# Surface-aligned surfel construction
# =============================================================================

def _sqrt_pos(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def matrix_to_quaternion(matrix):
    """
    Convert rotation matrix [...,3,3] to quaternion [...,4] in wxyz order.
    """
    batch = matrix.shape[:-2]
    m = matrix.reshape(*batch, 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m, dim=-1)

    q_abs = _sqrt_pos(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2], -1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    cand = quat_by_rijk / (2.0 * torch.maximum(q_abs[..., None], flr))

    sel = F.one_hot(q_abs.argmax(dim=-1), 4) > 0.5
    return cand[sel, :].reshape(*batch, 4)


def _axis_spacing(P, axis):
    """
    One-sided minimum neighbour spacing.

    This avoids stretching a surfel across a depth discontinuity.
    """
    H, W = P.shape[:2]
    d = torch.linalg.norm(torch.diff(P, dim=axis), dim=-1)

    sp = torch.full((H, W), float("inf"), device=P.device)

    if axis == 1:
        sp[:, :-1] = torch.minimum(sp[:, :-1], d)
        sp[:, 1:] = torch.minimum(sp[:, 1:], d)
    else:
        sp[:-1, :] = torch.minimum(sp[:-1, :], d)
        sp[1:, :] = torch.minimum(sp[1:, :], d)

    return sp


def build_surfels(
    points,
    colors,
    k_overlap=0.75,
    thin_ratio=0.10,
    max_scale_mult=3.0,
    normal_smooth=1,
    opacity=0.999,
):
    H, W, _ = points.shape
    dev = points.device

    # Tangents by finite differences.
    du = torch.zeros_like(points)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    du[:, 0] = points[:, 1] - points[:, 0]
    du[:, -1] = points[:, -1] - points[:, -2]

    dv = torch.zeros_like(points)
    dv[1:-1, :] = points[2:, :] - points[:-2, :]
    dv[0, :] = points[1, :] - points[0, :]
    dv[-1, :] = points[-1, :] - points[-2, :]

    # Surface normal.
    n = F.normalize(torch.linalg.cross(du, dv, dim=-1), dim=-1, eps=1e-8)

    # Flip normals to face the source camera.
    n = torch.where(n[..., 2:3] > 0, -n, n)

    # Smooth only normals, not depth.
    if normal_smooth > 0:
        k = 2 * normal_smooth + 1
        n = F.avg_pool2d(
            n.permute(2, 0, 1)[None],
            kernel_size=k,
            stride=1,
            padding=normal_smooth,
        )[0].permute(1, 2, 0)
        n = F.normalize(n, dim=-1, eps=1e-8)

    # Tangent frame.
    t1 = F.normalize(du - (du * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    t2 = torch.linalg.cross(n, t1, dim=-1)

    R = torch.stack(
        [t1.reshape(-1, 3), t2.reshape(-1, 3), n.reshape(-1, 3)],
        dim=-1,
    )
    quats = matrix_to_quaternion(R)

    su = _axis_spacing(points, axis=1)
    sv = _axis_spacing(points, axis=0)

    finite_su = su[torch.isfinite(su)]
    med = torch.median(finite_su)

    su = su.clamp(max=max_scale_mult * med)
    sv = sv.clamp(max=max_scale_mult * med)

    s_u = (k_overlap * su).reshape(-1)
    s_v = (k_overlap * sv).reshape(-1)
    s_n = thin_ratio * 0.5 * (s_u + s_v)

    scales = torch.stack([s_u, s_v, s_n], dim=-1).clamp_min(1e-6)

    means = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3).clamp(0, 1)
    opac = torch.full((means.shape[0],), float(opacity), device=dev)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opac,
        "colors": cols,
    }


# =============================================================================
# Rendering with gsplat
# =============================================================================

def render_gaussians(
    g,
    viewmat,
    K,
    width,
    height,
    rasterize_mode="antialiased",
    render_mode="RGB+ED",
    near=0.01,
    far=1e8,
    clamp_rgb=True,
):
    from gsplat import rasterization

    out, alpha, _ = rasterization(
        means=g["means"],
        quats=g["quats"],
        scales=g["scales"],
        opacities=g["opacities"],
        colors=g["colors"],
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
    )

    rgb = out[0, ..., :3]
    alpha = alpha[0, ..., 0]

    if clamp_rgb:
        rgb = rgb.clamp(0, 1)

    return rgb, alpha


def render_view(
    g,
    viewmat,
    focal,
    ref_w,
    ref_h,
    out_w,
    out_h,
    ss,
    bg,
    rasterize_mode,
):
    """
    Render with same FOV as focal@reference resolution.
    Supersampling is used only for display quality.
    """
    res_w = out_w * ss
    res_h = out_h * ss

    fx = focal * res_w / ref_w
    fy = focal * res_h / ref_h

    K = make_K(fx, fy, res_w, res_h, viewmat.device)

    rgb, alpha = render_gaussians(
        g,
        viewmat,
        K,
        width=res_w,
        height=res_h,
        rasterize_mode=rasterize_mode,
    )

    comp = rgb * alpha[..., None] + bg.view(1, 1, 3) * (1 - alpha[..., None])

    comp = F.interpolate(
        comp.permute(2, 0, 1)[None],
        size=(out_h, out_w),
        mode="area",
    )[0].permute(1, 2, 0)

    alpha_ds = F.interpolate(
        alpha[None, None],
        size=(out_h, out_w),
        mode="area",
    )[0, 0]

    return comp.clamp(0, 1), alpha_ds.clamp(0, 1)


# =============================================================================
# PLY writers
# =============================================================================

def write_gaussian_ply(path, g):
    means = g["means"].detach().cpu().numpy().astype(np.float32)
    cols = g["colors"].detach().cpu().numpy().astype(np.float32)
    sc = g["scales"].detach().cpu().numpy().astype(np.float32)
    op = g["opacities"].detach().cpu().numpy().astype(np.float32)
    qt = g["quats"].detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    f_dc = (cols - 0.5) / SH_C0

    log_scale = np.log(np.clip(sc, 1e-8, None))
    opc = np.clip(op, 1e-6, 1 - 1e-6)
    op_logit = np.log(opc / (1 - opc))

    props = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    ).encode()

    rows = np.concatenate(
        [
            means,
            np.zeros((n, 3), np.float32),
            f_dc,
            op_logit[:, None],
            log_scale,
            qt,
        ],
        axis=1,
    ).astype(np.float32)

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rows.tobytes())


def write_pointcloud_ply(path, points, colors):
    pts = points.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
    cols = colors.detach().cpu().numpy().reshape(-1, 3)
    cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)

    n = pts.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    arr = np.empty(n, dtype=dtype)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    arr["red"] = cols[:, 0]
    arr["green"] = cols[:, 1]
    arr["blue"] = cols[:, 2]

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(arr.tobytes())


# =============================================================================
# Figures and metrics
# =============================================================================

def psnr(a, b):
    mse = F.mse_loss(a, b)
    return 99.0 if mse < 1e-12 else float(-10.0 * torch.log10(mse))


def ssim(a_hw3, b_hw3, ws=11, sigma=1.5):
    a = a_hw3.permute(2, 0, 1).unsqueeze(0)
    b = b_hw3.permute(2, 0, 1).unsqueeze(0)

    coords = torch.arange(ws, device=a.device) - ws // 2
    gk = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gk = gk / gk.sum()

    win = (gk[:, None] * gk[None, :]).view(1, 1, ws, ws)
    win = win.repeat(a.shape[1], 1, 1, 1)

    pad = ws // 2
    C = a.shape[1]

    mu_a = F.conv2d(a, win, padding=pad, groups=C)
    mu_b = F.conv2d(b, win, padding=pad, groups=C)

    va = F.conv2d(a * a, win, padding=pad, groups=C) - mu_a * mu_a
    vb = F.conv2d(b * b, win, padding=pad, groups=C) - mu_b * mu_b
    cov = F.conv2d(a * b, win, padding=pad, groups=C) - mu_a * mu_b

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    score = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)
    )

    return float(score.mean())


def save_panel(rgb, disparity01, depth01, front, tilt, out_path):
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    items = [
        (rgb, "input RGB"),
        (disparity01, "Depth Anything disparity"),
        (depth01, "inverted depth"),
        (front, "front-view surfel render"),
        (tilt, "tilted 3D render"),
    ]

    for a, (im, title) in zip(ax, items):
        arr = im.detach().cpu().numpy()
        if arr.ndim == 2:
            a.imshow(arr, cmap="gray")
        else:
            a.imshow(arr)
        a.set_title(title, fontsize=10)
        a.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tensor_image(path, x, cmap=None):
    arr = x.detach().cpu().numpy()
    if arr.ndim == 2:
        plt.imsave(path, arr, cmap=cmap or "gray", vmin=0, vmax=1)
    else:
        plt.imsave(path, np.clip(arr, 0, 1))


# =============================================================================
# Driver
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Lift one image with Depth Anything into a point cloud / surfel splat."
    )

    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)

    p.add_argument(
        "--model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace Depth Anything model ID.",
    )

    p.add_argument(
        "--max-side",
        type=int,
        default=672,
        help="Resize longest image side before depth/lifting. Use 0 to keep original.",
    )

    p.add_argument("--no-invert", action="store_true",
                   help="Disable disparity-to-depth inversion.")

    # Geometry / relief.
    p.add_argument("--focal", type=float, default=None,
                   help="Focal length in pixels. Default: image width.")
    p.add_argument("--z-near", type=float, default=2.0)
    p.add_argument("--z-span", type=float, default=0.45,
                   help="Relief amplitude. Larger means more visible 3D shape.")

    # Surfel construction.
    p.add_argument("--k-overlap", type=float, default=0.75)
    p.add_argument("--thin-ratio", type=float, default=0.10)
    p.add_argument("--normal-smooth", type=int, default=1)
    p.add_argument("--opacity", type=float, default=0.999)
    p.add_argument("--max-scale-mult", type=float, default=3.0)

    # Rendering.
    p.add_argument("--render-max-side", type=int, default=1200)
    p.add_argument("--ss", type=int, default=2)
    p.add_argument("--rasterize-mode", choices=["classic", "antialiased"],
                   default="antialiased")
    p.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--tilt-deg", type=float, default=20.0)
    p.add_argument("--azimuth-deg", type=float, default=15.0)

    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_side = None if args.max_side == 0 else args.max_side

    out_dir = args.out
    if out_dir is None:
        out_dir = Path("figures") / "single_image_surfels" / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = torch.tensor(args.bg, dtype=torch.float32, device=device)

    print(f"Loading image: {args.image}")
    image_pil, rgb = load_rgb(args.image, max_side=max_side, device=device)
    H, W = rgb.shape[:2]

    print(f"Image size after optional resize: {W}x{H}")
    print(f"Estimating Depth Anything disparity with {args.model_id}")

    depth01, disparity01 = estimate_depth_anything(
        image_pil=image_pil,
        model_id=args.model_id,
        device=device,
        invert=not args.no_invert,
    )

    focal = args.focal or float(W)
    print(f"Using focal={focal:.2f}, z_near={args.z_near}, z_span={args.z_span}")

    points = unproject(
        depth01=depth01,
        focal=focal,
        z_near=args.z_near,
        z_span=args.z_span,
    )

    g = build_surfels(
        points=points,
        colors=rgb,
        k_overlap=args.k_overlap,
        thin_ratio=args.thin_ratio,
        max_scale_mult=args.max_scale_mult,
        normal_smooth=args.normal_smooth,
        opacity=args.opacity,
    )

    print(f"Built {g['means'].shape[0]} surface-aligned surfels")

    # Save raw assets.
    save_tensor_image(out_dir / "input_rgb.png", rgb)
    save_tensor_image(out_dir / "depth_anything_disparity01.png", disparity01, cmap="gray")
    save_tensor_image(out_dir / "depth01_inverted.png", depth01, cmap="gray")

    write_pointcloud_ply(out_dir / "pointcloud.ply", points, rgb)
    write_gaussian_ply(out_dir / "surfels.ply", g)

    print("Wrote pointcloud.ply and surfels.ply")

    # Render views.
    aspect = W / H
    if W >= H:
        out_w = args.render_max_side
        out_h = int(round(args.render_max_side / aspect))
    else:
        out_h = args.render_max_side
        out_w = int(round(args.render_max_side * aspect))

    vm_front = torch.eye(4, device=device)
    vm_tilt = tilt_viewmat(points, args.tilt_deg, args.azimuth_deg, device=device)

    print("Rendering front and tilted views with gsplat")

    front_native, alpha_native = render_view(
        g=g,
        viewmat=vm_front,
        focal=focal,
        ref_w=W,
        ref_h=H,
        out_w=W,
        out_h=H,
        ss=args.ss,
        bg=bg,
        rasterize_mode=args.rasterize_mode,
    )

    front_disp, _ = render_view(
        g=g,
        viewmat=vm_front,
        focal=focal,
        ref_w=W,
        ref_h=H,
        out_w=out_w,
        out_h=out_h,
        ss=args.ss,
        bg=bg,
        rasterize_mode=args.rasterize_mode,
    )

    tilt_disp, tilt_alpha = render_view(
        g=g,
        viewmat=vm_tilt,
        focal=focal,
        ref_w=W,
        ref_h=H,
        out_w=out_w,
        out_h=out_h,
        ss=args.ss,
        bg=bg,
        rasterize_mode=args.rasterize_mode,
    )

    save_tensor_image(out_dir / "front_render.png", front_disp)
    save_tensor_image(out_dir / "tilted_render.png", tilt_disp)

    rgba = torch.cat([tilt_disp, tilt_alpha[..., None]], dim=-1)
    plt.imsave(out_dir / "beauty_rgba.png", np.clip(rgba.detach().cpu().numpy(), 0, 1))

    save_panel(
        rgb=rgb,
        disparity01=disparity01,
        depth01=depth01,
        front=front_disp,
        tilt=tilt_disp,
        out_path=out_dir / "panel.png",
    )

    cov_mask = alpha_native > 0.5
    coverage = float(cov_mask.float().mean())

    if cov_mask.any():
        masked_mse = F.mse_loss(front_native[cov_mask], rgb[cov_mask])
        front_psnr_covered = (
            99.0 if masked_mse < 1e-12 else float(-10.0 * torch.log10(masked_mse))
        )
    else:
        front_psnr_covered = 0.0

    metrics = {
        "image": str(args.image),
        "model_id": args.model_id,
        "width": int(W),
        "height": int(H),
        "n_points": int(points.numel() // 3),
        "n_surfels": int(g["means"].shape[0]),
        "focal": float(focal),
        "z_near": float(args.z_near),
        "z_span": float(args.z_span),
        "depth_inverted": bool(not args.no_invert),
        "pointcloud_ply_mb": round((out_dir / "pointcloud.ply").stat().st_size / 1e6, 3),
        "surfels_ply_mb": round((out_dir / "surfels.ply").stat().st_size / 1e6, 3),
        "front_psnr": round(psnr(front_native, rgb), 3),
        "front_ssim": round(ssim(front_native, rgb), 4),
        "front_l1": round(float((front_native - rgb).abs().mean()), 4),
        "coverage": round(coverage, 4),
        "front_psnr_covered": round(front_psnr_covered, 3),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nDone. Outputs written to: {out_dir}")
    print("Open surfels.ply in SuperSplat, or use panel.png / beauty_rgba.png for the method overview.")


if __name__ == "__main__":
    main()