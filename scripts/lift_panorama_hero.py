# scripts/lift_panorama_hero.py
"""HEADLINE-FIGURE driver for the surfel-lifted seabed panorama.

Builds on lift_panorama_surfels.py (all working pieces imported verbatim) and
adds exactly what the paper figure needs:

  1. FOUR-CORNER CONDITIONING. One seed per corner (TL, TR, BL, BR), each from
     a DIFFERENT campaign, bilinearly blended across the canvas by
     generate_canvas. Pass e.g.:
       --campaigns Hawaii201801 Batemans201211 Batemans201011 ScottReef201108
     (order = TL TR BL BR).

  2. PROVENANCE. Every corner seed image is saved individually
     (corner_TL_<campaign>.png, ...), plus provenance.png (the panorama with
     the four labelled seeds inset at their corners) and provenance.json
     (campaigns, corner assignment, seed shapes, RNG seed). Seeds are also
     stored inside the RGB-D cache blob so --load-rgbd keeps provenance.

  3. MAX FIDELITY DEFAULTS. Denser sliding-window overlap (stride 4), full ODE
     budget (1000 steps), antialiased rasterisation, 3x supersampling,
     1600 px display renders, 800-step anchored appearance refinement +
     300-step colour-only perceptual polish, fixed RNG seed. Fidelity metrics
     are computed BEFORE and AFTER refinement so the paper can report the gain.

  4. EXHAUSTIVE REPORTING under --out:
       surfels.ply                    interactive hero framing (SuperSplat)
       rgb.png / depth.png / normals.png   the raw 2.5D inputs to the lift
       panel.png                     reference | front render | tilted render
       provenance.png                corners-in-context headline candidate
       angles.png                    3 tilts x 4 azimuths contact sheet
       angles/beauty_tXX_aYY.png     every contact-sheet angle as RGBA
                                     (with --save-all-angles)
       beauty.png                    chosen hero angle, RGBA transparent bg
       error_map.png                 |front render - reference| heatmap
       coverage_map.png              front-view alpha (where seams leak)
       metrics.json                  everything numeric, pre/post refinement
       config.json                   the exact invocation, reproducible
"""

import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from REEF import CKPT_ROOT, FIG_ROOT
from models.unet_cfm import UNetCFM
from scripts.train_cfm_cfg import load_rae_frozen
from scripts.test_panorama import sample_conditioning, RGB_SRC, FEAT_SRC
from scripts.lift_panorama_3d import (
    CFM_RUN_NAME, RAE_CKPT,
    generate_canvas, unproject, make_K, tilt_viewmat,
    build_surfels, refine_surfels_source_view, refine_surfels_color_perceptual,
    render_view, render_gaussians, write_gaussian_ply,
    psnr, ssim,
)

CORNERS = ["TL", "TR", "BL", "BR"]


def resolve_seed_path(path_str):
    """Image path (.../CAMPAIGN/DEPLOYMENT/IMAGE.jpg) -> (campaign, deployment,
    idx), resolving the frame index from the {deployment}_keys.npy filename
    index (written at feature extraction; its row order is shared by the RGB,
    DINO and depth arrays -- the same mapping CSVSplitDataset relies on)."""
    p = Path(path_str)
    if len(p.parts) < 3:
        sys.exit(f"Seed path '{path_str}' too short: need .../CAMPAIGN/DEPLOYMENT/IMAGE")
    campaign, deployment, name = p.parts[-3], p.parts[-2], p.name
    keys_npy = FEAT_SRC / campaign / f"{deployment}_keys.npy"
    if not keys_npy.exists():
        sys.exit(f"No keys index for {campaign}/{deployment}: {keys_npy}")
    keys = np.load(keys_npy)
    hits = np.nonzero(keys == name)[0]
    if hits.size == 0:
        sys.exit(f"'{name}' not found in {keys_npy.name} ({keys.size} frames); "
                 f"was the image rejected at extraction?")
    idx = int(hits[0])
    print(f"  resolved {name} -> {campaign}/{deployment}[{idx}]")
    return campaign, deployment, idx


def parse_seed_spec(spec):
    """Corner-seed spec -> (campaign, deployment, idx). Three forms:
      * an image path ending in .jpg/.jpeg/.png (frame index looked up in the
        deployment's keys file -- no need to know the row index);
      * 'CAMPAIGN/DEPLOYMENT[IDX]' (explicit frame);
      * 'CAMPAIGN/DEPLOYMENT' (idx None -> well-exposed random frame)."""
    spec = spec.strip()
    if spec.lower().endswith((".jpg", ".jpeg", ".png")):
        return resolve_seed_path(spec)
    m = re.match(r"^([^/]+)/([^\[\]]+?)(?:\[(\d+)\])?$", spec)
    if not m:
        sys.exit(f"Bad --seeds spec '{spec}': expected an image path, "
                 f"CAMPAIGN/DEPLOYMENT, or CAMPAIGN/DEPLOYMENT[IDX]")
    return m.group(1), m.group(2), (int(m.group(3)) if m.group(3) else None)


def load_seed(campaign, deployment, idx, rng, device,
              white_thr=0.9, black_thr=0.1, min_std=0.03, max_tries=30):
    """Pinned-corner variant of sample_conditioning: a fixed deployment, and
    optionally a fixed frame index. With an explicit idx the exposure gates are
    only reported (the frame was hand-picked); without one, a well-exposed
    frame is drawn from that deployment with the usual rejection loop.
    Returns (rgb [3,S,S] in [0,1], cond [1,768], 'campaign/deployment[idx]')."""
    rgb_npy = RGB_SRC / campaign / f"{deployment}.npy"
    if not rgb_npy.exists():
        avail = sorted(p.stem for p in (RGB_SRC / campaign).glob("*.npy"))
        sys.exit(f"Seed {campaign}/{deployment}: no such deployment. Available:\n  "
                 + "\n  ".join(avail))
    rgb_arr = np.load(rgb_npy, mmap_mode="r")
    feat_arr = np.load(FEAT_SRC / campaign / f"{deployment}.npy", mmap_mode="r")

    for attempt in range(1, max_tries + 1):
        i = idx if idx is not None else int(rng.integers(len(rgb_arr)))
        if not 0 <= i < len(rgb_arr):
            sys.exit(f"Seed {campaign}/{deployment}[{i}]: index out of range "
                     f"(deployment has {len(rgb_arr)} frames).")
        rgb = np.asarray(rgb_arr[i], dtype=np.float32) / 255.0
        bright, contrast = float(rgb.mean()), float(rgb.std())
        bad = bright > white_thr or bright < black_thr or contrast < min_std
        if bad and idx is None:
            print(f"  [reject {attempt:02d}] {campaign}/{deployment}[{i}] "
                  f"bright={bright:.3f} std={contrast:.3f}")
            continue
        note = "  (WARNING: fails exposure gates)" if bad else ""
        print(f"  [pinned]    {campaign}/{deployment}[{i}] "
              f"bright={bright:.3f} std={contrast:.3f}{note}")
        dino = torch.from_numpy(np.asarray(feat_arr[i], dtype=np.float32))
        cond = dino.reshape(-1, dino.shape[-1]).mean(0, keepdim=True).to(device)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        return rgb_t, cond, f"{campaign}/{deployment}[{i}]"

    raise RuntimeError(f"No well-exposed frame found in {campaign}/{deployment} "
                       f"after {max_tries} tries; pin an index explicitly.")


# ============================================================================= #
#  Seed-image handling (robust to CHW/HWC, RGB/RGBD, batched or not)            #
# ============================================================================= #
def seed_to_hwc_rgb(seed):
    """Best-effort conversion of whatever sample_conditioning returns for the
    seed into an HxWx3 float image in [0,1] for provenance display."""
    t = seed.detach().float().cpu()
    while t.dim() > 3:                      # strip batch dims
        t = t[0]
    if t.dim() == 2:                        # grayscale -> RGB
        t = t.unsqueeze(0).repeat(3, 1, 1)
    if t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[0] < t.shape[-1]:
        t = t.permute(1, 2, 0)              # CHW -> HWC
    if t.shape[-1] == 4:                    # RGBD -> RGB
        t = t[..., :3]
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 3)
    if float(t.max()) > 1.5:                # 0-255 -> 0-1
        t = t / 255.0
    return t.clamp(0, 1).numpy()


def save_img(path, arr_or_tensor, cmap=None):
    a = arr_or_tensor
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    plt.imsave(path, np.clip(a, 0, 1) if cmap is None else a, cmap=cmap)


# ============================================================================= #
#  Provenance figure: panorama with the 4 labelled seeds inset at its corners   #
# ============================================================================= #
def save_provenance_figure(rgb, seeds_hwc, campaigns, out_path, inset_frac=0.22):
    img = rgb.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img)
    ax.axis("off")
    # inset positions in figure coords: (x0, y0) per corner
    m, s = 0.015, inset_frac
    pos = {"TL": (m, 1 - m - s), "TR": (1 - m - s, 1 - m - s),
           "BL": (m, m), "BR": (1 - m - s, m)}
    for corner, camp in zip(CORNERS, campaigns):
        x0, y0 = pos[corner]
        ia = fig.add_axes([x0, y0, s, s])
        ia.imshow(seeds_hwc[corner])
        ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_edgecolor("white"); sp.set_linewidth(2.5)
        label_y = y0 - 0.005 if corner in ("TL", "TR") else y0 + s + 0.005
        va = "top" if corner in ("TL", "TR") else "bottom"
        fig.text(x0 + s / 2, label_y, f"{corner}: {camp}", color="white",
                 fontsize=10, ha="center", va=va, weight="bold",
                 bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_seed_grid(seeds_hwc, campaigns, out_path):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    order = [("TL", ax[0, 0]), ("TR", ax[0, 1]), ("BL", ax[1, 0]), ("BR", ax[1, 1])]
    for (corner, a), camp in zip(order, campaigns):
        a.imshow(seeds_hwc[corner])
        a.set_title(f"{corner}: {camp}", fontsize=11)
        a.axis("off")
    fig.suptitle("Conditioning seeds (provenance)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================================= #
#  Extra diagnostics                                                            #
# ============================================================================= #
def quats_to_normals(quats):
    """Third column of the rotation matrix encoded by (w,x,y,z) quaternions."""
    w, x, y, z = quats.unbind(-1)
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], -1)


def save_normals_png(g, H, W, out_path):
    n = quats_to_normals(g["quats"]).reshape(H, W, 3)
    save_img(out_path, (n * 0.5 + 0.5).clamp(0, 1))


def save_heatmap(arr2d, out_path, title, cmap="inferno", vmax=None):
    a = arr2d.detach().cpu().numpy() if torch.is_tensor(arr2d) else arr2d
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(a, cmap=cmap, vmax=vmax)
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(); fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def front_fidelity(front, alpha, ref):
    cov_mask = alpha > 0.5
    coverage = float(cov_mask.float().mean())
    if cov_mask.any():
        m = F.mse_loss(front[cov_mask], ref[cov_mask])
        masked_psnr = 99.0 if m < 1e-12 else float(-10 * torch.log10(m))
    else:
        masked_psnr = 0.0
    return dict(
        psnr=round(psnr(front, ref), 3),
        ssim=round(ssim(front, ref), 4),
        l1=round(float((front - ref).abs().mean()), 4),
        coverage=round(coverage, 4),
        psnr_covered=round(masked_psnr, 3),
    )


def save_panel(ref_rgb, front, tilt, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, [ref_rgb, front, tilt],
                        ["reference RGB-D (input)", "front view (fidelity check)",
                         "tilted view (3D relief)"]):
        a.imshow(im.detach().cpu().numpy()); a.set_title(t, fontsize=11); a.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def save_angle_sheet(g, points, focal, ref_size, out_size, ss, bg, mode, out_dir,
                     device, save_individual=False):
    tilts, azims = [10, 18, 26], [-30, -10, 10, 30]
    angle_dir = out_dir / "angles"
    if save_individual:
        angle_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(len(tilts), len(azims),
                           figsize=(4 * len(azims), 4 * len(tilts)))
    for i, tl in enumerate(tilts):
        for j, az in enumerate(azims):
            vm = tilt_viewmat(points, tl, az, device)
            img, a = render_view(g, vm, focal, ref_size, out_size, ss, bg, mode)
            ax[i, j].imshow(img.detach().cpu().numpy())
            ax[i, j].set_title(f"tilt {tl}, azim {az}", fontsize=9)
            ax[i, j].axis("off")
            if save_individual:
                rgba = torch.cat([img, a[..., None]], -1).detach().cpu().numpy()
                plt.imsave(angle_dir / f"beauty_t{tl:02d}_a{az:+03d}.png",
                           np.clip(rgba, 0, 1))
    fig.suptitle("Hero-angle contact sheet (or frame surfels.ply in SuperSplat)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "angles.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================================= #
#  Driver                                                                       #
# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaigns", nargs=4, metavar=("TL", "TR", "BL", "BR"),
                   default=None,
                   help="four DIFFERENT campaigns, one per corner (TL TR BL BR), "
                        "e.g. --campaigns Hawaii201801 Batemans201011 "
                        "ScottReef2009 ScottReef2015")
    p.add_argument("--seeds", nargs=4, metavar=("TL", "TR", "BL", "BR"),
                   default=None,
                   help="pin the corner seeds explicitly. Each spec is either "
                        "a path to the reference image (data_normalized/"
                        "CAMPAIGN/DEPLOYMENT/IMG.jpg; the frame index is "
                        "resolved automatically via the deployment's keys "
                        "file), or CAMPAIGN/DEPLOYMENT[IDX] (IDX optional -> "
                        "well-exposed random frame from that deployment). "
                        "Overrides --campaigns.")
    # generation (hero defaults: dense overlap, full ODE budget)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--stride", type=int, default=4,
                   help="sliding-window stride; 4 = maximum blend smoothness")
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cond-gamma", type=float, default=1.0,
                   help="ease exponent for the corner-conditioning weights: 1 = "
                        "linear (legacy); >1 keeps windows near the corners "
                        "close to the pure seed and compresses the transition "
                        "into the canvas centre. " \
                        "Not used by default."
                        "Try 1.5 or 2.0 for a more varied corner appearance.")
    p.add_argument("--pin-corners", action="store_true",
                   help="normalise window centres over their own range so the "
                        "outermost windows receive the PURE corner seeds ")
    p.add_argument("--cond-jitter", type=float, default=0.0,
                   help="correlated (fractal) conditioning jitter: mean offset "
                        "norm as a fraction of the corner-cond separation"
                        "(try 0.15); 0 disables. Gated to zero at the corners." \
                        "Not used by default.")
    p.add_argument("--jitter-seed", type=int, default=None,
                   help="dedicated RNG seed for the jitter field (default: "
                        "follow the global torch RNG)")
    # geometry / relief
    p.add_argument("--focal", type=float, default=None)
    p.add_argument("--z-near", type=float, default=2.0)
    p.add_argument("--z-span", type=float, default=0.35)
    # surfel look
    p.add_argument("--k-overlap", type=float, default=0.75)
    p.add_argument("--thin-ratio", type=float, default=0.1)
    p.add_argument("--normal-smooth", type=int, default=1)
    p.add_argument("--opacity", type=float, default=0.999)
    p.add_argument("--max-scale-mult", type=float, default=3.0)
    p.add_argument("--refine-steps", type=int, default=800)
    p.add_argument("--perceptual-steps", type=int, default=300)
    # rendering (hero defaults: heavy supersampling, large display renders)
    p.add_argument("--ss", type=int, default=3)
    p.add_argument("--render-size", type=int, default=1600)
    p.add_argument("--rasterize-mode", choices=["classic", "antialiased"],
                   default="antialiased")
    p.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--tilt-deg", type=float, default=18.0)
    p.add_argument("--azimuth-deg", type=float, default=10.0)
    p.add_argument("--save-all-angles", action="store_true",
                   help="save every contact-sheet angle as an RGBA beauty render")
    # caching / reproducibility
    p.add_argument("--save-rgbd", type=Path, default=None)
    p.add_argument("--load-rgbd", type=Path, default=None)
    p.add_argument("--seed", type=int, default=44,
                   help="fixed by default: headline figures must be reproducible")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--experiment-label", type=str, default=None,
                   help="optional string to append to the output directory name")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.out or (FIG_ROOT / CFM_RUN_NAME / f"hero_4_{args.cond_jitter:.2f}_{args.experiment_label}")
    if args.experiment_label:
        out_dir = out_dir / args.experiment_label
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.tensor(args.bg, dtype=torch.float32, device=device)

    # ---- reproducibility: dump the exact invocation ----
    (out_dir / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        indent=2))

    # ------------------------------------------------------------------ #
    #  1+2: four diverse corner seeds, saved for provenance               #
    # ------------------------------------------------------------------ #
    if args.load_rgbd and args.load_rgbd.exists():
        print(f"Loading cached RGB-D from {args.load_rgbd}")
        blob = torch.load(args.load_rgbd, map_location="cpu")
        rgb = blob["rgb"].to(device)
        depth = blob["depth"].to(device)
        campaigns = blob["campaigns"]
        seed_ids = blob.get("seed_ids", list(campaigns))
        seeds_hwc = {c: blob["seeds"][c] for c in CORNERS}
    else:
        specs = None
        if args.seeds is not None:
            specs = [parse_seed_spec(s) for s in args.seeds]
            campaigns = [c for c, _, _ in specs]
            if args.campaigns is not None:
                print("NOTE: --seeds given; --campaigns is ignored.")
        elif args.campaigns is not None:
            campaigns = args.campaigns
        else:
            sys.exit("ERROR: pass --campaigns TL TR BL BR (random seed per "
                     "campaign) or --seeds TL TR BL BR (pinned "
                     "CAMPAIGN/DEPLOYMENT[IDX]) when generating.")
        if len(set(campaigns)) < 4:
            print("WARNING: corners are not all distinct campaigns:", campaigns)

        print("Loading CFM + RAE ...")
        model = UNetCFM().to(device)
        model.load_state_dict(torch.load(CKPT_ROOT / CFM_RUN_NAME / "best_model.pt",
                                         map_location=device, weights_only=True))
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        rae = load_rae_frozen(RAE_CKPT, device)
        rae.eval()

        seeds_hwc, conds, seed_ids = {}, [], []
        for k, (corner, camp) in enumerate(zip(CORNERS, campaigns)):
            if specs is not None:
                print(f"  loading pinned {corner} seed:")
                seed, cond, sid = load_seed(*specs[k], rng, device)
            else:
                print(f"  sampling {corner} seed from {camp}")
                seed, cond = sample_conditioning(camp, rng, device)
                sid = camp
            seeds_hwc[corner] = seed_to_hwc_rgb(seed)
            conds.append(cond)
            seed_ids.append(sid)
        corners_cond = tuple(conds)  # (TL, TR, BL, BR) = generate_canvas order

        print(f"Generating {args.n}x{args.n} panorama "
              f"(stride={args.stride}, steps={args.steps}) ...")
        rgb, depth = generate_canvas(model, rae, corners_cond, n=args.n,
                                     n_steps=args.steps, stride=args.stride,
                                     cfg_scale=args.cfg,
                                     temperature=args.temperature, device=device, verbose=True,
                                     cond_gamma=args.cond_gamma,
                                     pin_corners=args.pin_corners,
                                     cond_jitter=args.cond_jitter,
                                     jitter_seed=args.jitter_seed)
        if args.save_rgbd:
            torch.save({"rgb": rgb.cpu(), "depth": depth.cpu(),
                        "campaigns": campaigns, "seeds": seeds_hwc,
                        "seed_ids": seed_ids,
                        "rng_seed": args.seed}, args.save_rgbd)
            print(f"  cached RGB-D + provenance -> {args.save_rgbd}")

    # provenance outputs
    for corner, camp in zip(CORNERS, campaigns):
        save_img(out_dir / f"corner_{corner}_{camp}.png", seeds_hwc[corner])
    save_seed_grid(seeds_hwc, campaigns, out_dir / "seeds_grid.png")
    save_provenance_figure(rgb, seeds_hwc, campaigns, out_dir / "provenance.png")
    (out_dir / "provenance.json").write_text(json.dumps(dict(
        corner_campaigns=dict(zip(CORNERS, campaigns)),
        corner_seeds=dict(zip(CORNERS, seed_ids)),
        rng_seed=args.seed,
        seed_shapes={c: list(np.asarray(seeds_hwc[c]).shape) for c in CORNERS},
        canvas=dict(n=args.n, steps=args.steps, stride=args.stride,
                    cfg=args.cfg, temperature=args.temperature,
                    cond_gamma=args.cond_gamma, pin_corners=args.pin_corners,
                    cond_jitter=args.cond_jitter, jitter_seed=args.jitter_seed),
    ), indent=2))
    print(f"  provenance saved: {', '.join(f'{c}={m}' for c, m in zip(CORNERS, seed_ids))}")

    # raw 2.5D inputs to the lift
    save_img(out_dir / "rgb.png", rgb)
    save_heatmap(depth, out_dir / "depth.png", "depth (normalised)", cmap="viridis")

    # ------------------------------------------------------------------ #
    #  3: max-fidelity lift + refinement, with before/after metrics       #
    # ------------------------------------------------------------------ #
    Himg, Wimg = depth.shape
    focal = args.focal or float(Wimg)
    points = unproject(depth, focal, args.z_near, args.z_span)
    g = build_surfels(points, rgb, k_overlap=args.k_overlap,
                      thin_ratio=args.thin_ratio,
                      max_scale_mult=args.max_scale_mult,
                      normal_smooth=args.normal_smooth, opacity=args.opacity)
    print(f"  built {g['means'].shape[0]} surfels")
    save_normals_png(g, Himg, Wimg, out_dir / "normals.png")

    vm_front = torch.eye(4, device=device)

    # deterministic baseline metrics (pre-refinement) at native resolution
    front0, alpha0 = render_view(g, vm_front, focal, Wimg, Wimg, args.ss, bg,
                                 args.rasterize_mode)
    fid_pre = front_fidelity(front0, alpha0, rgb)
    print(f"  pre-refinement fidelity: {fid_pre}")

    perceptual_info = {"perceptual_backend": "disabled"}
    if args.refine_steps > 0:
        print(f"  anchored appearance refinement ({args.refine_steps} steps) ...")
        g = refine_surfels_source_view(g, rgb, focal, bg, args.rasterize_mode,
                                       steps=args.refine_steps)
    if args.perceptual_steps > 0:
        print(f"  colour-only perceptual polish ({args.perceptual_steps} steps) ...")
        g, perceptual_info = refine_surfels_color_perceptual(
            g, rgb, focal, bg, args.rasterize_mode, steps=args.perceptual_steps)

    ply = out_dir / "surfels.ply"
    write_gaussian_ply(ply, g)
    print(f"  wrote {ply.name} ({ply.stat().st_size / 1e6:.2f} MB)")

    # ------------------------------------------------------------------ #
    #  4: renders + exhaustive reporting                                  #
    # ------------------------------------------------------------------ #
    rs = args.render_size
    vm_tilt = tilt_viewmat(points, args.tilt_deg, args.azimuth_deg, device)

    front_metric, alpha_metric = render_view(g, vm_front, focal, Wimg, Wimg,
                                             args.ss, bg, args.rasterize_mode)
    front_disp, _ = render_view(g, vm_front, focal, Wimg, rs, args.ss, bg,
                                args.rasterize_mode)
    tilt, tilt_a = render_view(g, vm_tilt, focal, Wimg, rs, args.ss, bg,
                               args.rasterize_mode)

    fid_post = front_fidelity(front_metric, alpha_metric, rgb)
    print(f"  post-refinement fidelity: {fid_post}")

    save_panel(rgb, front_disp, tilt, out_dir / "panel.png")
    save_angle_sheet(g, points, focal, Wimg, rs, args.ss, bg, args.rasterize_mode,
                     out_dir, device, save_individual=args.save_all_angles)
    rgba = torch.cat([tilt, tilt_a[..., None]], -1).detach().cpu().numpy()
    plt.imsave(out_dir / "beauty.png", np.clip(rgba, 0, 1))

    err = (front_metric - rgb).abs().mean(-1)
    save_heatmap(err, out_dir / "error_map.png",
                 f"|front render - reference|  (mean {float(err.mean()):.4f})",
                 vmax=max(0.1, float(err.max())))
    save_heatmap(alpha_metric, out_dir / "coverage_map.png",
                 "front-view alpha (coverage)", cmap="gray", vmax=1.0)

    metrics = dict(
        corner_campaigns=dict(zip(CORNERS, campaigns)),
        rng_seed=args.seed,
        panorama_px=[int(Himg), int(Wimg)],
        n_surfels=int(g["means"].shape[0]),
        ply_mb=round(ply.stat().st_size / 1e6, 3),
        depth_stats=dict(min=round(float(depth.min()), 4),
                         max=round(float(depth.max()), 4),
                         mean=round(float(depth.mean()), 4),
                         std=round(float(depth.std()), 4)),
        front_pre_refinement=fid_pre,
        front_post_refinement=fid_post,
        tilt_view=dict(tilt_deg=args.tilt_deg, azimuth_deg=args.azimuth_deg,
                       alpha_mean=round(float(tilt_a.mean()), 4)),
        refine_steps=int(args.refine_steps),
        perceptual_steps=int(args.perceptual_steps),
        perceptual_backend=str(perceptual_info.get("perceptual_backend", "unknown")),
        render=dict(ss=args.ss, render_size=rs, mode=args.rasterize_mode,
                    focal=focal, z_near=args.z_near, z_span=args.z_span),
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nDone. Everything in {out_dir}")
    print("Headline candidates: provenance.png (corners-in-context), "
          "beauty.png / angles/ (hero renders), panel.png (fidelity story). "
          "For the very best framing open surfels.ply in SuperSplat.")


if __name__ == "__main__":
    main()