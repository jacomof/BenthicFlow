"""Lift 2D panoramas into 3D surfel point clouds and meshes.

Unprojects generated RGB-D panoramas into 3D point clouds and exports PLY models.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from benthicflow import CKPT_ROOT, FIG_ROOT
from models.unet_cfm import UNetCFM
from scripts.test_panorama import sample_conditioning
from scripts.train_cfm_cfg import load_rae_frozen

# ----------------------------------------------------------------------------- #
CFM_RUN_NAME = "unet_cfm_final"
RAE_CKPT = CKPT_ROOT / "rae" / "last.pt"
GRID = 16
DECODER_UPSAMPLE = 14
NATIVE_PX = GRID * DECODER_UPSAMPLE
# 0th-degree Spherical Harmonics constant: 1 / (2 * sqrt(pi)) for 3D Gaussian Splatting color encoding
SH_C0 = 0.28209479177387814
LEFT_CAMPAIGN = "Hawaii201801"
RIGHT_CAMPAIGN = "Batemans201011"


# ============================================================================= #
#  Generation (verbatim from lift_panorama_3d.py)                               #
# ============================================================================= #
def ease(a, gamma=1.0):
    """Symmetric rational ease on [0,1]: gamma=1 is the identity; gamma>1
    plateaus near 0/1 and steepens mid-transition, so corner windows keep a
    near-pure seed conditioning and the appearance swap is compressed into the
    canvas centre (addresses the abrupt corner-to-corner transitions)."""
    if gamma == 1.0:
        return a
    return a**gamma / (a**gamma + (1.0 - a) ** gamma)


def fractal_cond_jitter(ny, nx, dim, levels=3, seed=None):
    """Value-noise fBm on the window lattice: per level, a small Gaussian grid
    (all `dim` channels at once) is bilinearly upsampled to (ny, nx) and summed
    with halving amplitude -- the cheap equivalent of diamond-square midpoint
    jitter (Infinite-Leagues-style), without its lattice artifacts.
    Levels start at the FIRST midpoint grid (3x3 -> 5x5 -> 9x9 with amplitudes
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    field = torch.zeros(ny, nx, dim)
    for k in range(levels):
        g = 2 ** (k + 1) + 1  # 3, 5, 9, ...
        grid = torch.randn(1, dim, g, g, generator=gen)
        up = F.interpolate(grid, size=(ny, nx), mode="bilinear", align_corners=True)
        field += (0.5**k) * up[0].permute(1, 2, 0)
    return field


@torch.no_grad()
def generate_canvas(
    model,
    rae,
    corners,
    n=2,
    n_steps=100,
    stride=8,
    cfg_scale=3.0,
    temperature=1.0,
    device="cuda",
    verbose=False,
    cond_gamma=1.0,
    pin_corners=False,
    cond_jitter=0.0,
    jitter_seed=None,
):
    grid, D = GRID, model.latent_dim
    H = W = grid * n
    null = model.null_cond.unsqueeze(0)
    c_tl, c_tr, c_bl, c_br = corners
    w1d = torch.sin(torch.linspace(0.05, 0.95, grid, device=device) * torch.pi)
    win = (w1d[:, None] * w1d[None, :]).view(1, 1, grid, grid)
    x = torch.randn(1, D, H, W, device=device) * temperature
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = ts[1] - ts[0]
    ys = list(range(0, H - grid + 1, stride))
    xs = list(range(0, W - grid + 1, stride))

    # Per-window conditioning, fixed across time. Bilinear over the four corner
    # conds, with two knobs on the axis weights:
    #   pin_corners: normalise window CENTRES over their own range instead of
    #     the canvas, so the outermost windows get the PURE corner seeds
    #     (legacy centre/H sampling leaves them a grid/2H mixture, e.g. 12.5%
    #     at n=4 -- part of why seed patterns died near the corners);
    #   cond_gamma: ease() the weights so the transition starts slow at the
    #     corners and accelerates toward the centre.
    # Optional correlated conditioning jitter (fractal mosaic transitions):
    # scaled so the mean per-window offset norm = cond_jitter * (mean pairwise
    # distance between the four corner conds), and gated to zero AT the corners
    # so pinned seeds stay exact.
    jit = None
    if cond_jitter > 0:
        jit = fractal_cond_jitter(
            len(ys), len(xs), c_tl.shape[-1], seed=jitter_seed
        ).to(device)
        corner_sep = torch.pdist(torch.cat([c_tl, c_tr, c_bl, c_br], 0)).mean()
        jit *= cond_jitter * corner_sep / (jit.norm(dim=-1).mean() + 1e-8)
        print(
            f"  cond jitter: mean offset {float(jit.norm(dim=-1).mean()):.2f} = "
            f"{cond_jitter:.2f} x corner sep {float(corner_sep):.2f} "
            f"(seed {jitter_seed})"
        )

    conds = {}
    for iy, y0 in enumerate(ys):
        ay = y0 / max(H - grid, 1) if pin_corners else (y0 + grid / 2) / H
        ay = ease(ay, cond_gamma)
        left = (1 - ay) * c_tl + ay * c_bl
        right = (1 - ay) * c_tr + ay * c_br
        for ix, x0 in enumerate(xs):
            ax = x0 / max(W - grid, 1) if pin_corners else (x0 + grid / 2) / W
            ax = ease(ax, cond_gamma)
            cond_w = (1 - ax) * left + ax * right
            if jit is not None:
                max_w = max((1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay)
                gate = (1.0 - max_w) / 0.75  # 0 at corners, 1 at centre
                cond_w = cond_w + gate * jit[iy, ix]
            conds[(y0, x0)] = cond_w

    for i in tqdm(range(n_steps), desc="  generating canvas", disable=not verbose):
        t = ts[i].expand(1)
        vel = torch.zeros_like(x)
        wacc = torch.zeros(1, 1, H, W, device=device)
        for y0 in ys:
            for x0 in xs:
                xc = x[:, :, y0 : y0 + grid, x0 : x0 + grid]
                v_cond = model(xc, t, conds[(y0, x0)])
                if cfg_scale != 1.0:
                    v_uncond = model(xc, t, null)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
                vel[:, :, y0 : y0 + grid, x0 : x0 + grid] += v * win
                wacc[:, :, y0 : y0 + grid, x0 : x0 + grid] += win
        x = x + (vel / (wacc + 1e-8)) * dt
    z = model.denormalize(x)
    fused = z.permute(0, 2, 3, 1).contiguous()
    rgbd = rae.decoder(fused)
    rgb = rgbd[:, :3].clamp(0, 1)[0].permute(1, 2, 0).contiguous()
    depth = rgbd[:, 3:4].clamp(0, 1)[0, 0].contiguous()
    return rgb, depth


# ============================================================================= #
#  Geometry + cameras                                                            #
# ============================================================================= #
def make_K(f, w, h, device):
    return torch.tensor(
        [[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]], device=device
    )


def unproject(depth01, f, z_near, z_span):
    H, W = depth01.shape
    dev = depth01.device
    vv, uu = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    Z = z_near + depth01 * z_span
    X = (uu - cx) * Z / f
    Y = (vv - cy) * Z / f
    return torch.stack([X, Y, Z], -1)


def look_at_viewmat(eye, target, up, device):
    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)
    fwd = F.normalize(target - eye, dim=0)
    right = F.normalize(torch.linalg.cross(fwd, up), dim=0)
    true_up = torch.linalg.cross(right, fwd)
    R = torch.stack([right, -true_up, fwd], 0)
    vm = torch.eye(4, device=device)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ eye
    return vm


def _rot_x(a, device):
    c, s = np.cos(a), np.sin(a)
    return torch.tensor(
        [[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32, device=device
    )


def _rot_y(a, device):
    c, s = np.cos(a), np.sin(a)
    return torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=device
    )


def tilt_viewmat(points, tilt_deg, azim_deg, device):
    """Orbit the source camera (at origin) around the scene centroid by a gentle
    tilt+azimuth, looking back at the centroid. Reveals relief without exposing
    large disocclusions, since the surfels are watertight."""
    c0 = points.reshape(-1, 3).mean(0)
    eye0 = torch.zeros(3, device=device)
    Rr = _rot_y(np.deg2rad(azim_deg), device) @ _rot_x(np.deg2rad(tilt_deg), device)
    eye = c0 + Rr @ (eye0 - c0)
    return look_at_viewmat(eye, c0, (0, -1, 0), device)


# ============================================================================= #
#  Surface-aligned surfel construction  (math validated in NumPy)               #
# ============================================================================= #
def _sqrt_pos(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def matrix_to_quaternion(matrix):
    """[...,3,3] rotation -> [...,4] quaternion (w,x,y,z). pytorch3d formulation."""
    batch = matrix.shape[:-2]
    m = matrix.reshape(*batch, 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m, -1)
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
    sel = F.one_hot(q_abs.argmax(-1), 4) > 0.5
    return cand[sel, :].reshape(*batch, 4)


def _axis_spacing(P, axis):
    """Per-pixel inter-neighbour distance along `axis`, ONE-SIDED MINIMUM so a
    surfel at a depth edge is sized by its own surface, never across the gap."""
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
    k_overlap=0.65,
    thin_ratio=0.1,
    max_scale_mult=3.0,
    normal_smooth=1,
    opacity=0.999,
):
    H, W, _ = points.shape
    dev = points.device

    # tangents (central diff, replicated edges)
    du = torch.zeros_like(points)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    du[:, 0] = points[:, 1] - points[:, 0]
    du[:, -1] = points[:, -1] - points[:, -2]
    dv = torch.zeros_like(points)
    dv[1:-1, :] = points[2:, :] - points[:-2, :]
    dv[0, :] = points[1, :] - points[0, :]
    dv[-1, :] = points[-1, :] - points[-2, :]

    n = F.normalize(torch.linalg.cross(du, dv, dim=-1), dim=-1, eps=1e-8)
    n = torch.where(n[..., 2:3] > 0, -n, n)  # face camera (+Z forward)
    if normal_smooth > 0:  # denoise normals (not depth)
        k = 2 * normal_smooth + 1
        n = F.avg_pool2d(n.permute(2, 0, 1)[None], k, 1, normal_smooth)[0].permute(
            1, 2, 0
        )
        n = F.normalize(n, dim=-1, eps=1e-8)

    t1 = F.normalize(du - (du * n).sum(-1, keepdim=True) * n, dim=-1, eps=1e-8)
    t2 = torch.linalg.cross(n, t1, dim=-1)
    R = torch.stack([t1.reshape(-1, 3), t2.reshape(-1, 3), n.reshape(-1, 3)], dim=-1)
    quats = matrix_to_quaternion(R)  # [HW,4] wxyz

    su = _axis_spacing(points, 1)
    sv = _axis_spacing(points, 0)
    med = torch.median(su[torch.isfinite(su)])
    su = su.clamp(max=max_scale_mult * med)
    sv = sv.clamp(max=max_scale_mult * med)
    s_u = (k_overlap * su).reshape(-1)
    s_v = (k_overlap * sv).reshape(-1)
    s_n = thin_ratio * 0.5 * (s_u + s_v)
    scales = torch.stack([s_u, s_v, s_n], -1).clamp_min(1e-6)

    means = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    opac = torch.full((means.shape[0],), float(opacity), device=dev)
    return dict(means=means, quats=quats, scales=scales, opacities=opac, colors=cols)


# ============================================================================= #
#  gsplat rendering (+ supersample + composite)                                  #
# ============================================================================= #
def render_gaussians(
    g,
    viewmat,
    K,
    width,
    height,
    rasterize_mode="classic",
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
    if clamp_rgb:
        rgb = rgb.clamp(0, 1)
    return rgb, alpha[0, ..., 0]


def render_view(g, viewmat, focal, ref_size, out_size, ss, bg, mode):
    """Render at out_size*ss with the SAME FOV as `focal`@ref_size, composite over
    `bg`, area-downsample to out_size. Output resolution is DECOUPLED from the
    panorama grid: raise out_size for a crisp far field (the receding part of a
    tilted view is render-undersampled, not surfel-soft). Returns (rgb, alpha)."""
    res = out_size * ss
    f_res = focal * res / ref_size  # keep FOV, scale focal to res
    K = make_K(f_res, res, res, viewmat.device)
    rgb, alpha = render_gaussians(g, viewmat, K, res, res, mode)
    comp = rgb * alpha[..., None] + bg.view(1, 1, 3) * (1 - alpha[..., None])
    comp = F.interpolate(
        comp.permute(2, 0, 1)[None], size=(out_size, out_size), mode="area"
    )[0].permute(1, 2, 0)
    a = F.interpolate(alpha[None, None], size=(out_size, out_size), mode="area")[0, 0]
    return comp.clamp(0, 1), a.clamp(0, 1)


# ============================================================================= #
#  Anchored surfel appearance refinement                                         #
# ============================================================================= #
def _logit(x, eps=1e-6):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _grad_l1(a, b):
    """Small differentiable sharpness/fidelity term; avoids LPIPS dependency."""
    ax = a[:, 1:, :] - a[:, :-1, :]
    bx = b[:, 1:, :] - b[:, :-1, :]
    ay = a[1:, :, :] - a[:-1, :, :]
    by = b[1:, :, :] - b[:-1, :, :]
    return F.l1_loss(ax, bx) + F.l1_loss(ay, by)


def refine_surfels_source_view(
    g, target_rgb, focal, bg, rasterize_mode, steps=500, log_every=100
):
    """Constrained front-view fit.
    Fixed: positions, rotations, normal thickness.
    Optimised: DC colour, opacity, bounded in-plane scale residuals.
    The only public hyperparameter is `steps`; the rest are deliberately fixed so
    this behaves like a renderer calibration stage rather than a new method with a
    """
    if steps <= 0:
        return g

    dev = target_rgb.device
    H, W = target_rgb.shape[:2]
    K = make_K(focal, W, H, dev)
    viewmat = torch.eye(4, device=dev)

    means = g["means"].detach()
    quats = g["quats"].detach()
    scale0 = g["scales"].detach()
    color0 = g["colors"].detach().clamp(1e-4, 1 - 1e-4)
    opac0 = g["opacities"].detach().clamp(1e-4, 1 - 1e-4)

    color_logits = torch.nn.Parameter(_logit(color0))
    opacity_delta = torch.nn.Parameter(torch.zeros_like(opac0))
    scale_delta = torch.nn.Parameter(torch.zeros_like(scale0[:, :2]))

    # One optimizer, fixed internal rates. Colour needs to move most; opacity and
    # scale are conservative residual corrections.
    opt = torch.optim.Adam(
        [
            {"params": [color_logits], "lr": 3e-2},
            {"params": [opacity_delta], "lr": 5e-3},
            {"params": [scale_delta], "lr": 5e-3},
        ]
    )

    # Hard-coded, intentionally small design constants.
    scale_bound = 0.20  # exp(±0.20) ~= ±22% in-plane scale freedom
    opacity_bound = 1.00  # logit residual; 0.999 -> roughly [0.997, 0.9996]
    best = None
    best_mse = float("inf")

    for it in range(steps):
        opt.zero_grad(set_to_none=True)

        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(
            _logit(opac0) + opacity_bound * torch.tanh(opacity_delta)
        )
        scales = scale0.clone()
        scales[:, :2] = scale0[:, :2] * torch.exp(scale_bound * torch.tanh(scale_delta))
        # Keep the normal axis fixed: this is the anti-blur geometry prior.
        scales[:, 2] = scale0[:, 2]

        gr = dict(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors
        )
        pred, alpha = render_gaussians(
            gr, viewmat, K, W, H, rasterize_mode=rasterize_mode, clamp_rgb=False
        )
        pred = pred * alpha[..., None] + bg.view(1, 1, 3) * (1 - alpha[..., None])
        pred = pred.clamp(0, 1)

        mse = F.mse_loss(pred, target_rgb)
        l1 = F.l1_loss(pred, target_rgb)
        grad = _grad_l1(pred, target_rgb)
        coverage = ((1.0 - alpha).clamp_min(0.0) ** 2).mean()

        # Anchors keep the refinement honest. They are weak for colour, stronger
        # for opacity/scale because those are the blur-prone degrees of freedom.
        color_anchor = F.mse_loss(colors, color0)
        opacity_anchor = F.mse_loss(opacities, opac0)
        scale_anchor = (torch.tanh(scale_delta) ** 2).mean()

        loss = (
            mse
            + 0.20 * l1
            + 0.05 * grad
            + 0.02 * coverage
            + 0.002 * color_anchor
            + 0.05 * opacity_anchor
            + 0.10 * scale_anchor
        )
        loss.backward()
        opt.step()

        mse_val = float(mse.detach())
        if mse_val < best_mse:
            best_mse = mse_val
            best = dict(
                colors=colors.detach(),
                opacities=opacities.detach(),
                scales=scales.detach(),
            )

        if log_every and ((it + 1) % log_every == 0 or it == 0 or it + 1 == steps):
            cur_psnr = 99.0 if mse_val < 1e-12 else -10.0 * np.log10(mse_val)
            print(
                f"  refine {it + 1:4d}/{steps}: loss={float(loss):.5f} "
                f"mse={mse_val:.6f} psnr={cur_psnr:.2f} "
                f"alpha={float(alpha.mean()):.3f}"
            )

    out = {k: v.detach() for k, v in g.items()}
    if best is not None:
        out["colors"] = best["colors"]
        out["opacities"] = best["opacities"]
        out["scales"] = best["scales"]
    return out


# ============================================================================= #
#  Optional colour-only perceptual polish                                        #
# ============================================================================= #
class _VGGFeatureLoss(torch.nn.Module):
    """Small VGG feature loss used only when pretrained weights are available.

    This is intentionally local and dependency-light: LPIPS is preferred when the
    `lpips` package is installed; otherwise we try torchvision's pretrained VGG16.
    If neither is available, the caller falls back to a gradient/L1 polish.
    """

    def __init__(self, vgg_features):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                vgg_features[:4].eval(),  # relu1_2-ish
                vgg_features[4:9].eval(),  # relu2_2-ish
                vgg_features[9:16].eval(),  # relu3_3-ish
            ]
        )
        for p in self.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred_bchw, target_bchw):
        pred = (pred_bchw.clamp(0, 1) - self.mean) / self.std
        target = (target_bchw.clamp(0, 1) - self.mean) / self.std
        loss = pred.new_tensor(0.0)
        for block in self.blocks:
            pred = block(pred)
            with torch.no_grad():
                target = block(target)
            loss = loss + F.l1_loss(pred, target)
        return loss / len(self.blocks)


def _make_perceptual_loss(device):
    """Return (backend_name, callable_or_None).

    The function never raises: perceptual losses are useful but not required for
    the renderer to run on clusters without LPIPS/VGG weights.
    """
    try:
        import lpips  # type: ignore

        net = lpips.LPIPS(net="vgg").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)

        def lpips_loss(pred_bchw, target_bchw):
            # LPIPS expects [-1, 1]
            return net(pred_bchw * 2.0 - 1.0, target_bchw * 2.0 - 1.0).mean()

        return "lpips-vgg", lpips_loss
    except Exception as exc:
        print(f"  perceptual: LPIPS unavailable ({exc.__class__.__name__}: {exc})")

    try:
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        loss_fn = _VGGFeatureLoss(vgg).to(device).eval()
        return "vgg16", loss_fn
    except Exception as exc:
        print(
            f"  perceptual: pretrained VGG16 unavailable ({exc.__class__.__name__}: {exc})"
        )
        return "gradient-fallback", None


def refine_surfels_color_perceptual(
    g, target_rgb, focal, bg, rasterize_mode, steps=200, log_every=50
):
    """Final colour-only source-view polish.
    Fixed: positions, rotations, opacity, all scales including normal thickness.
    Optimised: DC colours only.
    This is the safest place to use LPIPS/VGG: it can improve texture and
    perceptual sharpness, but it cannot change coverage or bloat splats because
    """
    if steps <= 0:
        return g, {"perceptual_backend": "disabled"}

    dev = target_rgb.device
    H, W = target_rgb.shape[:2]
    K = make_K(focal, W, H, dev)
    viewmat = torch.eye(4, device=dev)

    means = g["means"].detach()
    quats = g["quats"].detach()
    scales = g["scales"].detach()
    opacities = g["opacities"].detach()
    color0 = g["colors"].detach().clamp(1e-4, 1 - 1e-4)
    color_logits = torch.nn.Parameter(_logit(color0))

    backend, perceptual_loss = _make_perceptual_loss(dev)
    print(f"  colour-only polish backend: {backend}")

    opt = torch.optim.Adam([color_logits], lr=1e-2)
    target_bchw = target_rgb.permute(2, 0, 1)[None]
    best = None
    best_score = float("inf")

    for it in range(steps):
        opt.zero_grad(set_to_none=True)

        colors = torch.sigmoid(color_logits)
        gr = dict(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors
        )
        pred, alpha = render_gaussians(
            gr, viewmat, K, W, H, rasterize_mode=rasterize_mode, clamp_rgb=False
        )
        pred = pred * alpha[..., None] + bg.view(1, 1, 3) * (1 - alpha[..., None])
        pred = pred.clamp(0, 1)
        pred_bchw = pred.permute(2, 0, 1)[None]

        mse = F.mse_loss(pred, target_rgb)
        l1 = F.l1_loss(pred, target_rgb)
        grad = _grad_l1(pred, target_rgb)
        color_anchor = F.mse_loss(colors, color0)

        if perceptual_loss is not None:
            perc = perceptual_loss(pred_bchw, target_bchw)
            loss = mse + 0.15 * l1 + 0.05 * grad + 0.05 * perc + 0.001 * color_anchor
            score = float((mse + 0.02 * perc).detach())
        else:
            perc = pred.new_tensor(0.0)
            # Still useful when VGG/LPIPS are unavailable: colour-only high-frequency polish.
            loss = mse + 0.15 * l1 + 0.10 * grad + 0.001 * color_anchor
            score = float(mse.detach())

        loss.backward()
        opt.step()

        if score < best_score:
            best_score = score
            best = colors.detach()

        if log_every and ((it + 1) % log_every == 0 or it == 0 or it + 1 == steps):
            mse_val = float(mse.detach())
            cur_psnr = 99.0 if mse_val < 1e-12 else -10.0 * np.log10(mse_val)
            print(
                f"  polish {it + 1:4d}/{steps}: loss={float(loss):.5f} "
                f"mse={mse_val:.6f} psnr={cur_psnr:.2f} "
                f"perc={float(perc.detach()):.4f}"
            )

    out = {k: v.detach() for k, v in g.items()}
    if best is not None:
        out["colors"] = best
    return out, {"perceptual_backend": backend}


# ============================================================================= #
#  Metrics + PLY (PLY writer verbatim)                                           #
# ============================================================================= #
def psnr(a, b):
    mse = F.mse_loss(a, b)
    return 99.0 if mse < 1e-12 else float(-10 * torch.log10(mse))


def ssim(a_hw3, b_hw3, ws=11, sigma=1.5):
    a = a_hw3.permute(2, 0, 1).unsqueeze(0)
    b = b_hw3.permute(2, 0, 1).unsqueeze(0)
    coords = torch.arange(ws, device=a.device) - ws // 2
    gk = torch.exp(-(coords**2) / (2 * sigma**2))
    gk /= gk.sum()
    win = (gk[:, None] * gk[None, :]).view(1, 1, ws, ws).repeat(a.shape[1], 1, 1, 1)
    pad, C = ws // 2, a.shape[1]
    mu_a = F.conv2d(a, win, padding=pad, groups=C)
    mu_b = F.conv2d(b, win, padding=pad, groups=C)
    va = F.conv2d(a * a, win, padding=pad, groups=C) - mu_a * mu_a
    vb = F.conv2d(b * b, win, padding=pad, groups=C) - mu_b * mu_b
    cov = F.conv2d(a * b, win, padding=pad, groups=C) - mu_a * mu_b
    C1, C2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / (
        (mu_a**2 + mu_b**2 + C1) * (va + vb + C2)
    )
    return float(s.mean())


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
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    ).encode()
    rows = np.concatenate(
        [means, np.zeros((n, 3), np.float32), f_dc, op_logit[:, None], log_scale, qt], 1
    ).astype(np.float32)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rows.tobytes())


# ============================================================================= #
#  Figures                                                                       #
# ============================================================================= #
def save_panel(ref_rgb, front, tilt, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(
        ax,
        [ref_rgb, front, tilt],
        [
            "reference RGB-D (input)",
            "front view (fidelity check)",
            "tilted view (3D relief)",
        ],
    ):
        a.imshow(im.detach().cpu().numpy())
        a.set_title(t, fontsize=11)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_angle_sheet(
    g, points, focal, ref_size, out_size, ss, bg, mode, out_path, device
):
    tilts, azims = [12, 24], [-25, 0, 25]
    fig, ax = plt.subplots(
        len(tilts), len(azims), figsize=(4 * len(azims), 4 * len(tilts))
    )
    for i, tl in enumerate(tilts):
        for j, az in enumerate(azims):
            vm = tilt_viewmat(points, tl, az, device)
            img, _ = render_view(g, vm, focal, ref_size, out_size, ss, bg, mode)
            ax[i, j].imshow(img.detach().cpu().numpy())
            ax[i, j].set_title(f"tilt {tl}, azim {az}", fontsize=9)
            ax[i, j].axis("off")
    fig.suptitle("Pick a hero angle (or open surfels.ply in SuperSplat)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================================= #
#  Driver                                                                        #
# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--bilinear", action="store_true")
    p.add_argument(
        "--cond-gamma",
        type=float,
        default=1.0,
        help="ease exponent for the corner-conditioning weights: 1 = "
        "linear (legacy); >1 keeps windows near the corners "
        "close to the pure seed and compresses the transition "
        "into the canvas centre (try 2.5)",
    )
    p.add_argument(
        "--pin-corners",
        action="store_true",
        help="normalise window centres over their own range so the "
        "outermost windows receive the PURE corner seeds "
        "(legacy sampling leaves them a grid/2H mixture)",
    )
    p.add_argument(
        "--cond-jitter",
        type=float,
        default=0.0,
        help="correlated (fractal) conditioning jitter: mean offset "
        "norm as a fraction of the corner-cond separation "
        "(try 0.15); 0 disables. Gated to zero at the corners.",
    )
    p.add_argument(
        "--jitter-seed",
        type=int,
        default=None,
        help="dedicated RNG seed for the jitter field (default: "
        "follow the global torch RNG)",
    )
    # geometry / relief
    p.add_argument("--focal", type=float, default=None, help="px; default=image width")
    p.add_argument("--z-near", type=float, default=2.0)
    p.add_argument(
        "--z-span",
        type=float,
        default=0.35,
        help="relief: Z in [z_near, z_near+z_span]; raise for more 3D",
    )
    # surfel look
    p.add_argument(
        "--k-overlap",
        type=float,
        default=0.75,
        help="surfel std as fraction of pixel spacing (lower=sharper, higher=safer vs holes)",
    )
    p.add_argument("--thin-ratio", type=float, default=0.1, help="normal-axis flatness")
    p.add_argument(
        "--normal-smooth", type=int, default=1, help="normal denoise radius (0=off)"
    )
    p.add_argument("--opacity", type=float, default=0.999)
    p.add_argument("--max-scale-mult", type=float, default=3.0)
    p.add_argument(
        "--refine-steps",
        type=int,
        default=500,
        help="anchored appearance-refinement steps; 0 disables",
    )
    p.add_argument(
        "--perceptual-steps",
        type=int,
        default=200,
        help="optional colour-only LPIPS/VGG polish steps after refinement; 0 disables",
    )
    # rendering
    p.add_argument(
        "--ss", type=int, default=2, help="supersample factor for crisp output"
    )
    p.add_argument(
        "--render-size",
        type=int,
        default=1200,
        help="output px of DISPLAY renders, decoupled from the panorama grid; "
        "raise this (not --ss alone) to sharpen the far field",
    )
    p.add_argument(
        "--rasterize-mode", choices=["classic", "antialiased"], default="antialiased"
    )
    p.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--tilt-deg", type=float, default=20.0)
    p.add_argument("--azimuth-deg", type=float, default=15.0)
    # caching to iterate on the look without regenerating
    p.add_argument("--save-rgbd", type=Path, default=None)
    p.add_argument("--load-rgbd", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    out_dir = args.out or (FIG_ROOT / CFM_RUN_NAME / "surfels")
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.tensor(args.bg, dtype=torch.float32, device=device)

    # ---- get the RGB-D panorama (load cache or generate) ----
    if args.load_rgbd and args.load_rgbd.exists():
        print(f"Loading cached RGB-D from {args.load_rgbd}")
        blob = torch.load(args.load_rgbd, map_location=device)
        rgb, depth = blob["rgb"].to(device), blob["depth"].to(device)
        ref_panel = blob.get("seed_l")
    else:
        print("Loading CFM + RAE ...")
        model = UNetCFM().to(device)
        model.load_state_dict(
            torch.load(
                CKPT_ROOT / CFM_RUN_NAME / "best_model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        rae = load_rae_frozen(RAE_CKPT, device)
        rae.eval()
        print(f"Sampling seeds: L={LEFT_CAMPAIGN}, R={RIGHT_CAMPAIGN}")
        seed_l, c_l = sample_conditioning(LEFT_CAMPAIGN, rng, device)
        seed_r, c_r = sample_conditioning(RIGHT_CAMPAIGN, rng, device)
        if args.bilinear:
            _, c_l2 = sample_conditioning(LEFT_CAMPAIGN, rng, device)
            _, c_r2 = sample_conditioning(RIGHT_CAMPAIGN, rng, device)
            corners = (c_l, c_r, c_l2, c_r2)
        else:
            corners = (c_l, c_r, c_l, c_r)
        print(f"Generating {args.n}x{args.n} panorama ...")
        rgb, depth = generate_canvas(
            model,
            rae,
            corners,
            n=args.n,
            n_steps=args.steps,
            stride=args.stride,
            cfg_scale=args.cfg,
            temperature=args.temperature,
            device=device,
            cond_gamma=args.cond_gamma,
            pin_corners=args.pin_corners,
            cond_jitter=args.cond_jitter,
            jitter_seed=args.jitter_seed,
        )
        if args.save_rgbd:
            torch.save({"rgb": rgb.cpu(), "depth": depth.cpu()}, args.save_rgbd)
            print(f"  cached RGB-D -> {args.save_rgbd}")

    Himg, Wimg = depth.shape
    focal = args.focal or float(Wimg)
    print(f"  panorama {Himg}x{Wimg}px, focal={focal:.0f}")

    # ---- build surfels from the reference RGB-D ----
    points = unproject(depth, focal, args.z_near, args.z_span)
    g = build_surfels(
        points,
        rgb,
        k_overlap=args.k_overlap,
        thin_ratio=args.thin_ratio,
        max_scale_mult=args.max_scale_mult,
        normal_smooth=args.normal_smooth,
        opacity=args.opacity,
    )
    print(f"  built {g['means'].shape[0]} surface-aligned surfels")

    # ---- anchored appearance refinement: no position/rotation/thickness updates ----
    perceptual_info = {"perceptual_backend": "disabled"}
    if args.refine_steps > 0:
        print(f"  refining DC appearance for {args.refine_steps} steps ...")
        g = refine_surfels_source_view(
            g, rgb, focal, bg, args.rasterize_mode, steps=args.refine_steps
        )

    # ---- optional final colour-only LPIPS/VGG polish: geometry/opacity/scale frozen ----
    if args.perceptual_steps > 0:
        print(f"  colour-only perceptual polish for {args.perceptual_steps} steps ...")
        g, perceptual_info = refine_surfels_color_perceptual(
            g, rgb, focal, bg, args.rasterize_mode, steps=args.perceptual_steps
        )

    ply = out_dir / "surfels.ply"
    write_gaussian_ply(ply, g)
    print(f"  wrote {ply.name} ({ply.stat().st_size / 1e6:.2f} MB)")

    # ---- render views ----
    rs = args.render_size
    vm_front = torch.eye(4, device=device)
    vm_tilt = tilt_viewmat(points, args.tilt_deg, args.azimuth_deg, device)

    # Fidelity is measured at NATIVE panorama resolution vs the reference (a fair
    # comparison); the DISPLAY renders use --render-size for a crisp far field.
    front_metric, alpha_metric = render_view(
        g, vm_front, focal, Wimg, Wimg, args.ss, bg, args.rasterize_mode
    )
    front_disp, _ = render_view(
        g, vm_front, focal, Wimg, rs, args.ss, bg, args.rasterize_mode
    )
    tilt, tilt_a = render_view(
        g, vm_tilt, focal, Wimg, rs, args.ss, bg, args.rasterize_mode
    )

    save_panel(rgb, front_disp, tilt, out_dir / "panel.png")
    save_angle_sheet(
        g,
        points,
        focal,
        Wimg,
        rs,
        args.ss,
        bg,
        args.rasterize_mode,
        out_dir / "angles.png",
        device,
    )
    # standalone beauty render with transparent background (RGBA)
    rgba = torch.cat([tilt, tilt_a[..., None]], -1).detach().cpu().numpy()
    plt.imsave(out_dir / "beauty.png", np.clip(rgba, 0, 1))

    # ---- fidelity metrics (front view vs reference, at native resolution) ----
    # coverage + front_psnr_covered decompose the score: if front_psnr falls but
    # front_psnr_covered stays high while coverage < 1, the loss is background
    # bleeding through micro-seams (raise --k-overlap), NOT a real fidelity drop.
    cov_mask = alpha_metric > 0.5
    coverage = float(cov_mask.float().mean())
    if cov_mask.any():
        masked_mse = F.mse_loss(front_metric[cov_mask], rgb[cov_mask])
        masked_psnr = (
            99.0 if masked_mse < 1e-12 else float(-10 * torch.log10(masked_mse))
        )
    else:
        masked_psnr = 0.0
    metrics = dict(
        n_surfels=int(g["means"].shape[0]),
        ply_mb=round(ply.stat().st_size / 1e6, 3),
        front_psnr=round(psnr(front_metric, rgb), 3),
        front_ssim=round(ssim(front_metric, rgb), 4),
        front_l1=round(float((front_metric - rgb).abs().mean()), 4),
        coverage=round(coverage, 4),
        front_psnr_covered=round(masked_psnr, 3),
        refine_steps=int(args.refine_steps),
        perceptual_steps=int(args.perceptual_steps),
        perceptual_backend=str(perceptual_info.get("perceptual_backend", "unknown")),
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  fidelity (front vs reference): {metrics}")
    print(f"\nDone. Outputs in {out_dir}")
    print(
        "Tip: open surfels.ply in SuperSplat (https://playcanvas.com/supersplat) "
        "to frame the hero shot interactively. Use --refine-steps 0 for the "
        "deterministic baseline; add --perceptual-steps 200 for the final "
        "colour-only polish. Tune --k-overlap / --z-span / --tilt-deg "
        "with --load-rgbd to iterate without regenerating."
    )


if __name__ == "__main__":
    main()
