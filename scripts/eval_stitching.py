# scripts/eval_stitching.py
"""Quantify MultiDiffusion stitching artifacts in generated images.

Tiled / MultiDiffusion generation (see test_panorama.py) integrates a large
latent canvas with overlapping windows and blends them; this can leave visible
seams where adjacent tiles meet -- an artifact reported by other tiled works.
Here we expand a *square* canvas in both x and y (a `scale x scale` grid of
tiles) and measure how abrupt the resulting seams are, and how the amount of
window overlap affects them.

Metric (seam gradient ratio)
----------------------------
Seams are axis-aligned lines, so we look at the gradient magnitude across them:
vertical seams show up in the horizontal gradient |dI/dx|, horizontal seams in
the vertical gradient |dI/dy|. Along each axis we form the 1D profile
    g(u) = mean_{other axis, channels} |I(u+1) - I(u)|
and report

    seam_ratio = mean(g over seam bands) / mean(g over non-seam columns/rows).

This is a *ratio* of per-pixel gradient densities, so it does not grow with the
"stitching area" (number of seams, or overlap width) -- it only reflects how
much sharper the image is at the seams than elsewhere. Informally:
  ~1.0  -> seams indistinguishable from natural texture (good stitching)
  >>1.0 -> abrupt, visible seams (bad stitching).
We combine the x- and y-axis ratios into one score per image.

We sweep several `stride` values (=> several overlap fractions) and, to isolate
the effect of overlap, reuse the same seeds and the same initial noise across
all strides for a given image index.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from benthicflow import CKPT_ROOT, FIG_ROOT, SCRATCH_ROOT, SCRATCH_NODE_ROOT, RGB_ROOT, FEAT_ROOT
from models.unet_cfm import UNetCFM
from scripts.train_cfm_cfg import load_rae_frozen
from scripts.test_panorama import (
    sample_conditioning, GRID, CFM_RUN_NAME, RAE_CKPT,
)

TOP_LEFT_CAMPAIGN = "ScottReef201503"
TOP_RIGHT_CAMPAIGN = "Batemans201011"
BOTTOM_LEFT_CAMPAIGN = "Hawaii201801"
BOTTOM_RIGHT_CAMPAIGN = "ScottReef201108" 


from tqdm import tqdm

# Decoder upsamples each latent cell to PATCH pixels (RAE ConvDecoder, 14px/cell);
# see the `W*14` decode in test_panorama.generate_panorama.
PATCH = 14

# ECCV 2026 (Springer LNCS) style for the summary figure. Built at HALF the
# 122 mm text width: include with \includegraphics[width=0.5\textwidth] (or as
# a subfigure) and the 8 pt fonts keep their true printed size. fonttype 42
# embeds TrueType, as the submission checker requires. Applied via rc_context
# so the diagnostic figures keep their default styling.
ECCV_TEXTWIDTH_IN = 122 / 25.4
ECCV_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _bilerp_cond(x0, y0, grid, W, H, cond_tl, cond_tr, cond_bl, cond_br):
    """Bilinearly interpolated [B, 768] conditioning for the window at (x0, y0).

    cond_tl/tr/bl/br: [B, 768] (one corner condition per image in the batch).
    """
    ax = (x0 + grid / 2) / W                                              # 0=left .. 1=right
    ay = (y0 + grid / 2) / H                                              # 0=top .. 1=bottom
    return ((1.0 - ax) * (1.0 - ay) * cond_tl
           + ax * (1.0 - ay) * cond_tr
           + (1.0 - ax) * ay * cond_bl
           + ax * ay * cond_br)


@torch.no_grad()
def generate_grid(model, rae, cond_tl, cond_tr, cond_bl, cond_br, scale=3,
                  n_steps=100, stride=8, cfg_scale=3.0, temperature=1.0,
                  batch_size=64, decode_batch_size=None, use_hann=True, device="cuda"):
    """2D MultiDiffusion: integrate a square [grid*scale, grid*scale] latent
    canvas with overlapping windows sliding in *both* x and y, blended with a
    separable Hann window. Conditioning is bilinearly interpolated between the
    four corner seeds (top-left, top-right, bottom-left, bottom-right), so it
    varies along both x and y instead of being row-independent.

    `use_hann=False` swaps the tapered window for a uniform (box) one -- vanilla
    MultiDiffusion averaging -- so the seams the taper normally hides become
    visible (an ablation of the blend window itself).

    cond_tl/tr/bl/br: [B, 768], one corner-condition set per image. All B
    images (independent noise, shared tiling) are generated together.

    Each image's windows and, on top of that, all B images are flattened into
    one (image, window) task pool and batched up to `batch_size` tasks per
    model forward pass (cond+uncond fused into one call like cfm_sample_cfg).
    This matters most when a single image has few windows (large stride --
    a lone image badly underfills the batch) and least when window count
    alone already saturates batch_size (small stride).

    The final decode is batched separately by `decode_batch_size` whole images
    (defaults to `batch_size`). Keep this small independently of `batch_size`:
    decoding runs the RAE mid-attention over the *whole* stitched canvas, whose
    memory grows as decode_batch_size * (grid*scale)**4, so it blows up far
    faster with scale than the fixed 16x16 windows of the integration loop.

    Returns (rgb, depth), each [B, C, grid*scale*14, ...].
    """
    grid, D = GRID, model.latent_dim
    B = cond_tl.shape[0]
    H = W = grid * scale
    null = model.null_cond.unsqueeze(0)                                  # [1, 768]

    # Separable Hann-like 2D blend window so overlapping tiles fuse smoothly.
    # use_hann=False -> uniform (box) window: equal-weight averaging over the
    # overlaps (vanilla MultiDiffusion), used to expose the otherwise-hidden seams.
    if use_hann:
        win1d = torch.sin(torch.linspace(0.05, 0.95, grid, device=device) * torch.pi)
        win2d = (win1d.view(grid, 1) * win1d.view(1, grid)).view(1, 1, grid, grid)
    else:
        win2d = torch.ones(1, 1, grid, grid, device=device)

    x = torch.randn(B, D, H, W, device=device) * temperature
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = ts[1] - ts[0]
    starts_y = window_starts(H, grid, stride)
    starts_x = window_starts(W, grid, stride)
    windows = [(y0, x0) for y0 in starts_y for x0 in starts_x]

    # Flatten (window, image) into one static task pool: (b, y0, x0, cond_vec).
    # Static across steps (only the `x` crop depends on the step), so built once.
    tasks = []
    for (y0, x0) in windows:
        cond_w = _bilerp_cond(x0, y0, grid, W, H, cond_tl, cond_tr, cond_bl, cond_br)  # [B, 768]
        for b in range(B):
            tasks.append((b, y0, x0, cond_w[b:b + 1]))

    step_desc = f"generate_grid stride={stride} scale={scale}"
    for i in tqdm(range(n_steps), desc=step_desc, total=n_steps):
        t = ts[i]
        vel = torch.zeros_like(x)
        wacc = torch.zeros(B, 1, H, W, device=device)

        for c0 in range(0, len(tasks), batch_size):
            chunk = tasks[c0:c0 + batch_size]
            n = len(chunk)

            xc_batch = torch.cat(
                [x[b:b + 1, :, y0:y0 + grid, x0:x0 + grid] for (b, y0, x0, _) in chunk], dim=0)
            cond_batch = torch.cat([cv for (_, _, _, cv) in chunk], dim=0)   # [n, 768]

            if cfg_scale != 1.0:
                xc_in = torch.cat([xc_batch, xc_batch], dim=0)          # [2n, D, grid, grid]
                cond_in = torch.cat([cond_batch, null.expand(n, -1)], dim=0)
                t_in = t.expand(2 * n)
                v_pred = model(xc_in, t_in, cond_in)
                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                v_batch = v_uncond + cfg_scale * (v_cond - v_uncond)     # [n, D, grid, grid]
            else:
                v_batch = model(xc_batch, t.expand(n), cond_batch)

            for k, (b, y0, x0, _) in enumerate(chunk):
                vel[b:b + 1, :, y0:y0 + grid, x0:x0 + grid] += v_batch[k:k + 1] * win2d
                wacc[b:b + 1, :, y0:y0 + grid, x0:x0 + grid] += win2d

        x = x + (vel / (wacc + 1e-8)) * dt

    # Decode at the stitched-image level in batches: denormalize, reshape, and
    # decode one chunk of whole images at a time so no full-B intermediate (the
    # [B, D, H, W] copy from .contiguous() or the upsampled [B, 4, H*14, W*14])
    # is ever materialized. Batching here is over whole images (not windows like
    # the integration loop above), and uses its own decode_batch_size so a small,
    # memory-safe decode chunk can coexist with a large generation batch_size.
    dbs = decode_batch_size or batch_size
    outs = []
    for i in range(0, B, dbs):
        z = model.denormalize(x[i:i + dbs])                             # [b, D, H, W]
        fused = z.permute(0, 2, 3, 1).contiguous()                      # [b, H, W, D]
        outs.append(rae.decoder(fused))                                 # [b, 4, H*14, W*14]
    rgbd = torch.cat(outs, dim=0)                                       # [B, 4, H*14, W*14]
    return rgbd[:, :3].clamp(0, 1), rgbd[:, 3:4].clamp(0, 1)


def window_starts(length, grid, stride):
    """Window start offsets covering [0, length): the regular stride sequence,
    plus a final window pinned at length-grid whenever the stride does not
    divide (length - grid) evenly. Without the pin, the trailing rows/columns
    are covered by NO window, their latents never integrate (they stay initial
    noise), and the decoder quietly turns that into plausible-looking texture
    -- silently corrupting both the image border and the seam-ratio baseline."""
    starts = list(range(0, length - grid + 1, stride))
    if starts[-1] != length - grid:
        starts.append(length - grid)
    return starts


def seam_positions_px(n_tiles, grid, stride):
    """Pixel locations of the seams along one axis for the given tiling.

    Mirrors generate_grid's layout via window_starts. A seam sits at the centre
    of the overlap between consecutive windows a < b: the overlap is
    [b, a + grid], so the seam is at (a + b + grid) / 2 -- which collapses to
    the shared tile edge when stride == grid (no overlap), and handles the
    pinned (shorter-gap) final window. Same for x and y (square, one stride).
    """
    N = grid * n_tiles
    starts = window_starts(N, grid, stride)
    seams_latent = [(a + b + grid) / 2.0 for a, b in zip(starts[:-1], starts[1:])]
    return [round(sl * PATCH) for sl in seams_latent]


def gradient_profile(img_chw, axis):
    """1D gradient-magnitude profile across seams on `axis`.

    axis=2 (x): horizontal gradient |dI/dx|, profile over columns  -> vertical seams.
    axis=1 (y): vertical gradient   |dI/dy|, profile over rows      -> horizontal seams.
    Averaged over channels and the other spatial axis.
    """
    if axis == 2:
        g = (img_chw[:, :, 1:] - img_chw[:, :, :-1]).abs()   # [C, H, Wpx-1]
        return g.mean(dim=(0, 1)).cpu().numpy()              # [Wpx-1]
    g = (img_chw[:, 1:, :] - img_chw[:, :-1, :]).abs()       # [C, Hpx-1, W]
    return g.mean(dim=(0, 2)).cpu().numpy()                  # [Hpx-1]


def seam_ratio_1d(prof, seams_px, band, edge_margin):
    """Return (ratio, seam_mask, base_mask) for one axis' profile.

    ratio = mean(g within `band` px of any seam) / mean(g over remaining
    interior columns/rows). `band`/`edge_margin` are fixed in pixels
    (independent of overlap), so the ratio is not inflated by wider overlaps.
    """
    L = prof.shape[0]
    locs = np.arange(L) + 0.5                                 # location of each diff
    seam_mask = np.zeros(L, dtype=bool)
    for s in seams_px:
        seam_mask |= np.abs(locs - s) <= band

    interior = (locs >= edge_margin) & (locs <= (L + 1) - edge_margin)
    base_mask = interior & ~seam_mask
    seam_mask = interior & seam_mask                          # keep seams off borders too

    if seam_mask.sum() == 0 or base_mask.sum() == 0:
        return float("nan"), seam_mask, base_mask
    ratio = float(prof[seam_mask].mean() / prof[base_mask].mean())
    return ratio, seam_mask, base_mask


def seam_gradient_ratio(img_chw, seams_px, band, edge_margin):
    """Combined (x+y) seam ratio for a square tiled image, plus per-axis parts."""
    prof_x = gradient_profile(img_chw, axis=2)
    prof_y = gradient_profile(img_chw, axis=1)
    rx, smx, bmx = seam_ratio_1d(prof_x, seams_px, band, edge_margin)
    ry, smy, bmy = seam_ratio_1d(prof_y, seams_px, band, edge_margin)
    combined = float(np.nanmean([rx, ry]))
    diag = dict(prof_x=prof_x, prof_y=prof_y, smx=smx, bmx=bmx, smy=smy, bmy=bmy,
                rx=rx, ry=ry)
    return combined, diag


def plot_diagnostic(pano_rgb, seams_px, diag, ratio, stride, overlap_pct, out_path):
    """Square image with seam grid lines + per-axis gradient profiles.

    ECCV-ready like plot_summary, built at HALF the 122 mm text width so two
    diagnostics sit side by side: include each with
    \\includegraphics[width=0.5\\textwidth] and the fonts keep their true
    printed size. The two profile panels share the y-axis (labels on the left
    one only) under a single shared legend. Saves PDF + PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_np = pano_rgb[0].permute(1, 2, 0).cpu().numpy()
    Hpx, Wpx = rgb_np.shape[:2]

    with plt.rc_context(ECCV_RC):
        half_w = 0.5 * ECCV_TEXTWIDTH_IN
        fig = plt.figure(figsize=(half_w, 1.46 * half_w))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax_img = fig.add_subplot(gs[0, :])
        ax_x = fig.add_subplot(gs[1, 0])
        ax_y = fig.add_subplot(gs[1, 1], sharey=ax_x)

        ax_img.imshow(rgb_np, extent=[0, Wpx, Hpx, 0])
        for s in seams_px:
            ax_img.axvline(s, color="cyan", ls="--", lw=0.6, alpha=0.8)
            ax_img.axhline(s, color="cyan", ls="--", lw=0.6, alpha=0.8)
        ax_img.set_title(f"stride {stride} ({overlap_pct:.1f}% overlap)   "
                         f"seam ratio {ratio:.3f}", fontsize=7)
        ax_img.axis("off")

        for ax, prof, sm, bm, lbl in [
            (ax_x, diag["prof_x"], diag["smx"], diag["bmx"],
             r"$|\partial I/\partial x|$ (vert. seams)"),
            (ax_y, diag["prof_y"], diag["smy"], diag["bmy"],
             r"$|\partial I/\partial y|$ (horiz. seams)"),
        ]:
            u = np.arange(prof.shape[0]) + 0.5
            ax.plot(u, prof, color="0.4", lw=0.3)
            ax.plot(u[bm], prof[bm], ".", ms=1.0, color="tab:green", label="baseline")
            ax.plot(u[sm], prof[sm], ".", ms=1.0, color="tab:red", label="seam")
            for s in seams_px:
                ax.axvline(s, color="cyan", ls="--", lw=0.5, alpha=0.8)
            ax.set_title(lbl, fontsize=7)
            ax.set_xlabel("px", fontsize=7)
            ax.margins(x=0)
            ax.tick_params(labelsize=6, length=2)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_linewidth(0.6)
        ax_x.yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.setp(ax_y.get_yticklabels(), visible=False)
        lo, hi = ax_x.get_ylim()                  # shared headroom so the
        ax_x.set_ylim(lo, hi + 0.24 * (hi - lo))  # legend sits clear of data

        fig.tight_layout(h_pad=0.5)
        # One legend for both profile panels, centred in their headroom band
        # (per-panel legends don't fit a ~1 in panel at print size).
        bb_l, bb_r = ax_x.get_position(), ax_y.get_position()
        fig.legend(*ax_x.get_legend_handles_labels(), ncol=2, frameon=False,
                   loc="upper center", fontsize=6,
                   bbox_to_anchor=((bb_l.x0 + bb_r.x1) / 2, bb_l.y1),
                   handletextpad=0.1, columnspacing=0.8, markerscale=2.2)
        for ext in ("pdf", "png"):
            fig.savefig(out_path.with_suffix(f".{ext}"))
        plt.close(fig)


def ci95_halfwidth(vals):
    """Half-width of the 95% t-confidence interval of the mean of `vals`.

    Far tighter than the raw std for large n (CI ~ std/sqrt(n)): the plot's
    error bars then show uncertainty about the MEAN ratio, which is what the
    overlap sweep argues about, not the per-image spread."""
    from scipy.stats import t
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return 0.0
    return float(t.ppf(0.975, vals.size - 1) * vals.std(ddof=1) / np.sqrt(vals.size))


# Legend label + fixed identity styling per blend-window variant (Okabe-Ito
# blue/vermillion, distinct markers so identity is never colour-alone).
# "Sine window": the taper is a truncated sine (sin, not the sin^2 of a true
# Hann window, and never zero at the window edges), so this is the accurate
# name. Internal keys stay "hann"/"uniform" (dir names, JSON configs).
WINDOW_LABELS = {"hann": "Sine window", "uniform": "Uniform window"}
SERIES_STYLE = {"hann": dict(color="#0072B2", marker="o"),
                "uniform": dict(color="#D55E00", marker="s")}


def plot_summary(series, out_path):
    """ECCV-ready summary: Times, half-column size, 95% CI-of-mean error bars.

    `series` is a list of (variant_key, overlaps, means, ci95) tuples with
    variant_key in WINDOW_LABELS: one entry for a single-variant run, two for
    the combined Hann-vs-uniform figure (--window both)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(ECCV_RC):
        fig, ax = plt.subplots(figsize=(0.5 * ECCV_TEXTWIDTH_IN, 1.9))
        ax.axhline(1.0, color="0.45", ls=(0, (4, 3)), lw=0.8, zorder=1,
                   label="seamless (ratio = 1)")
        all_x = set()
        for key, overlaps, means, cis in series:
            order = np.argsort(np.asarray(overlaps, dtype=float))
            x = np.asarray(overlaps, dtype=float)[order]
            y = np.asarray(means, dtype=float)[order]
            e = np.asarray(cis, dtype=float)[order]
            st = SERIES_STYLE[key]
            ax.errorbar(x, y, yerr=e, color=st["color"], lw=1.2,
                        marker=st["marker"], ms=3.2, mfc=st["color"],
                        mec="white", mew=0.5, elinewidth=0.8, capsize=1.8,
                        capthick=0.8, zorder=3, label=WINDOW_LABELS[key])
            all_x.update(x.tolist())
        xt = sorted(all_x)
        ax.set_xlabel("Window overlap (%)")
        ax.set_ylabel("Seam gradient ratio")
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{v:g}" for v in xt])
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.6)
        ax.legend(frameon=False, handlelength=1.8)
        for ext in ("pdf", "png"):
            fig.savefig(out_path.with_suffix(f".{ext}"))
        plt.close(fig)


def save_seed_previews(seed_rgbs, out_dir):
    """Save the four corner-seed RGB images used per panorama, laid out in their
    spatial 2x2 arrangement (TL/TR top row, BL/BR bottom), each corner drawn
    from its own campaign (TOP_LEFT/TOP_RIGHT/BOTTOM_LEFT/BOTTOM_RIGHT), so you
    can eyeball which seeds drove each result. `seed_rgbs`: list of (rgb_tl, rgb_tr, rgb_bl, rgb_br),
    each [3, S, S] in [0, 1] -- one entry per previewed image."""
    if not seed_rgbs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, (rgb_tl, rgb_tr, rgb_bl, rgb_br) in enumerate(seed_rgbs):
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        for ax, rgb, title in [
            (axes[0, 0], rgb_tl, f"TL ({TOP_LEFT_CAMPAIGN})"),
            (axes[0, 1], rgb_tr, f"TR ({TOP_RIGHT_CAMPAIGN})"),
            (axes[1, 0], rgb_bl, f"BL ({BOTTOM_LEFT_CAMPAIGN})"),
            (axes[1, 1], rgb_br, f"BR ({BOTTOM_RIGHT_CAMPAIGN})"),
        ]:
            ax.imshow(rgb.permute(1, 2, 0).cpu().numpy())
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Corner seeds -- image {p}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"seeds_image{p:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(seed_rgbs)} corner-seed preview(s) to {out_dir}")


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays (and non-finite floats) so the
    metrics dict can be json.dump'd. NaN/inf -> null (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def _restore(v):
    """JSON -> what the plotters expect: lists back to numpy arrays, null->nan."""
    if isinstance(v, list):
        return np.asarray(v)
    return float("nan") if v is None else v


def summary_from_metrics(metrics):
    """(overlaps, means, ci95) recomputed from the per-image ratios stored in a
    metrics JSON, so old runs whose summary lacks "ci95" replot correctly
    (fallback to the stored summary + normal approximation for very old files)."""
    overlaps, means, cis = [], [], []
    for rec in metrics["strides"].values():
        vals = [im["ratio"] for im in rec.get("images", []) if im["ratio"] is not None]
        if not vals:
            continue
        overlaps.append(rec["overlap_pct"])
        means.append(float(np.mean(vals)))
        cis.append(ci95_halfwidth(vals))
    if not overlaps:                       # very old JSON without per-image data
        summ = metrics["summary"]
        n = metrics["config"]["n_images"]
        overlaps, means = summ["overlaps"], summ["means"]
        cis = [1.96 * s / np.sqrt(n) for s in summ["stds"]]
    return overlaps, means, cis


def replot_both(json_hann, json_uniform):
    """Combined two-line Hann-vs-uniform summary from two saved metrics JSONs
    (CPU-only). Written next to the two variant dirs; per-variant diagnostics
    are not re-rendered here (use --from_json on each JSON for those)."""
    series = []
    for key, path in (("hann", json_hann), ("uniform", json_uniform)):
        with open(path) as f:
            metrics = json.load(f)
        if metrics["config"].get("use_hann", True) != (key == "hann"):
            print(f"[warn] {path}: config says use_hann="
                  f"{metrics['config'].get('use_hann')} but it was passed as "
                  f"the {WINDOW_LABELS[key]!r} series -- check the order of "
                  f"--from_json / --from_json_uniform.")
        series.append((key, *summary_from_metrics(metrics)))
    out = Path(json_hann).resolve().parent.parent / "stitching_seam_ratio_both.png"
    plot_summary(series, out)
    print(f"Saved combined summary to {out.with_suffix('.pdf')} / .png")


def replot_from_json(json_path):
    """Re-render the figures from a saved metrics JSON, no models or generation
    needed -- for iterating on plot styling. The summary plot comes straight from
    the stored arrays; each diagnostic plot is rebuilt from the stored profiles
    plus its saved background panorama (rgb_stride*.png)."""
    json_path = Path(json_path)
    with open(json_path) as f:
        metrics = json.load(f)
    out_dir = json_path.parent

    key = "hann" if metrics["config"].get("use_hann", True) else "uniform"
    plot_summary([(key, *summary_from_metrics(metrics))],
                 out_dir / "stitching_seam_ratio.png")

    for stride_str, rec in metrics["strides"].items():
        stride = int(stride_str)
        for diag in rec.get("diags", []):
            if not diag.get("rgb_png"):
                continue
            rgb_path = out_dir / diag["rgb_png"]
            p = diag.get("image", 0)
            if not rgb_path.exists():
                print(f"  skip diag stride {stride_str} img {p}: missing {rgb_path.name}")
                continue
            rgb = plt.imread(rgb_path)[..., :3]                      # [H, W, 3], 0..1
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
            diag_np = {k: _restore(v) for k, v in diag.items()
                       if k not in ("ratio", "rgb_png", "image")}
            plot_diagnostic(rgb_t, rec["seams_px"], diag_np, _restore(diag["ratio"]),
                            stride, rec["overlap_pct"], out_dir / f"diag_stride{stride:02d}_img{p:02d}.png")

    print(f"Re-rendered figures under {out_dir} (from {json_path.name})")


def run_variant(window, model, rae, conds, seed_rgbs, args, device):
    """One full stride sweep with the given blend window ('hann' | 'uniform').

    Writes its metrics JSON, per-variant summary plot and diagnostics into its
    own out dir (stitching / stitching_no_hann) and returns (overlaps, means,
    ci95) for the combined figure. Conds/seeds come from the caller, so
    --window both compares the two variants on IDENTICAL conditioning; the
    per-stride torch.manual_seed further makes the initial noise identical
    across variants, so the comparison isolates the blend window alone.
    """
    use_hann = window == "hann"
    cond_tl_all, cond_tr_all, cond_bl_all, cond_br_all = conds
    out_dir = FIG_ROOT / CFM_RUN_NAME / ("stitching" if use_hann else "stitching_no_hann")
    save_seed_previews(seed_rgbs, out_dir)
    results = {}   # stride -> list of per-image combined ratios

    # Everything needed to rebuild the figures without regenerating images.
    metrics = {
        "config": {
            "scale": args.scale, "strides": args.strides, "n_images": args.n_images,
            "batch_size": args.batch_size, "decode_batch_size": args.decode_batch_size,
            "n_steps": args.n_steps,
            "cfg_scale": args.cfg_scale, "temperature": args.temperature,
            "seam_band": args.seam_band, "edge_margin": args.edge_margin,
            "seed": args.seed, "grid": GRID, "patch": PATCH,
            "n_diag_images": args.n_diag_images,
            "use_hann": use_hann,
            "cfm_run_name": CFM_RUN_NAME,
        },
        "strides": {},   # str(stride) -> per-stride results
        "summary": {},   # filled after the sweep
    }

    for stride in args.strides:
        if stride > GRID:
            print(f"Skipping stride={stride} (> window size {GRID}, would leave gaps).")
            continue
        overlap_pct = 100.0 * (GRID - stride) / GRID
        seams_px = seam_positions_px(args.scale, GRID, stride)
        results[stride] = []
        stride_rec = {
            "overlap_pct": overlap_pct,
            "seams_px": seams_px,
            "images": [],      # per-image {ratio, rx, ry}
            "diags": [],       # per-diagnostic-image profiles/masks (first n_diag_images)
        }
        metrics["strides"][str(stride)] = stride_rec

        # Fixed seed each stride -> identical initial noise across the sweep
        # (shape only depends on scale, not stride), so differences in the
        # metric are due to overlap, not the random draw.
        torch.manual_seed(args.seed)
        pano_rgb, _ = generate_grid(
            model, rae, cond_tl_all, cond_tr_all, cond_bl_all, cond_br_all,
            scale=args.scale, n_steps=args.n_steps,
            stride=stride, cfg_scale=args.cfg_scale,
            temperature=args.temperature, batch_size=args.batch_size,
            decode_batch_size=args.decode_batch_size,
            use_hann=use_hann, device=device,
        )

        for p in tqdm(range(args.n_images), desc=f"Evaluating stride {stride}"):
            ratio, diag = seam_gradient_ratio(
                pano_rgb[p], seams_px, band=args.seam_band, edge_margin=args.edge_margin)
            results[stride].append(ratio)
            stride_rec["images"].append(
                {"ratio": ratio, "rx": diag["rx"], "ry": diag["ry"]})
            print(f"stride={stride:2d} overlap={overlap_pct:5.1f}%  image {p}: "
                  f"seam_ratio={ratio:.4f} (x={diag['rx']:.4f}, y={diag['ry']:.4f})")

            if p < args.n_diag_images:
                # Save the raw background panorama separately (one per image/overlap)
                # so the diagnostic figure can be re-rendered later without generating.
                out_dir.mkdir(parents=True, exist_ok=True)
                rgb_png = out_dir / f"rgb_stride{stride:02d}_img{p:02d}.png"
                plt.imsave(rgb_png, pano_rgb[p].permute(1, 2, 0).cpu().numpy())
                stride_rec["diags"].append(
                    {"image": p, "ratio": ratio, "rgb_png": rgb_png.name, **diag})
                plot_diagnostic(pano_rgb[p:p + 1], seams_px, diag, ratio, stride, overlap_pct,
                                out_dir / f"diag_stride{stride:02d}_img{p:02d}.png")

    # ---- summary ----
    strides_sorted = sorted(results.keys(), reverse=True)   # low -> high overlap
    overlaps, means, stds, cis = [], [], [], []
    print("\n=== Stitching seam gradient ratio (mean +/- 95% CI over images) ===")
    print(f"{'stride':>6} {'overlap%':>9} {'seam_ratio':>12}")
    for stride in strides_sorted:
        vals = np.array(results[stride], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        overlap_pct = 100.0 * (GRID - stride) / GRID
        overlaps.append(overlap_pct)
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))
        cis.append(ci95_halfwidth(vals))
        print(f"{stride:>6} {overlap_pct:>9.1f} {vals.mean():>8.4f} +/- {cis[-1]:.4f} "
              f"(std {vals.std():.4f}, n={vals.size})")

    metrics["summary"] = {
        "strides": strides_sorted, "overlaps": overlaps,
        "means": means, "stds": stds, "ci95": cis,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "stitching_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    plot_summary([(window, overlaps, means, cis)],
                 out_dir / "stitching_seam_ratio.png")
    print(f"Done ({WINDOW_LABELS[window]})! Figures saved under {out_dir}")
    return overlaps, means, cis


def main(args):
    if args.from_json and args.from_json_uniform:
        replot_both(args.from_json, args.from_json_uniform)
        return
    if args.from_json:
        replot_from_json(args.from_json)
        return

    window = args.window or ("uniform" if args.no_hann else "hann")
    variants = ("hann", "uniform") if window == "both" else (window,)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- models ----
    cfm_ckpt = CKPT_ROOT / CFM_RUN_NAME / "best_model.pt"
    model = UNetCFM().to(DEVICE)
    model.load_state_dict(torch.load(cfm_ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    rae = load_rae_frozen(RAE_CKPT, DEVICE)

    print("Loaded models and starting generation")

    # ---- seeds: sample once, reuse across all strides AND variants ----
    # Each corner drawn (independently) from its own campaign: TL/TR/BL/BR from
    # TOP_LEFT/TOP_RIGHT/BOTTOM_LEFT/BOTTOM_RIGHT, so the panorama transitions
    # between all four campaigns along both x and y.
    rng = np.random.default_rng(args.seed)
    tl_list, tr_list, bl_list, br_list = [], [], [], []
    seed_rgbs = []   # (tl,tr,bl,br) display images for the first n_seed_previews images
    for p in tqdm(range(args.n_images), desc="Sampling corner seeds"):
        print(f"Sampling corner seeds for image {p}:")
        rgb_tl, cond_tl = sample_conditioning(TOP_LEFT_CAMPAIGN, rng, DEVICE, rgb_src=RGB_ROOT, feat_src=FEAT_ROOT)
        rgb_tr, cond_tr = sample_conditioning(TOP_RIGHT_CAMPAIGN, rng, DEVICE, rgb_src=RGB_ROOT, feat_src=FEAT_ROOT)
        rgb_bl, cond_bl = sample_conditioning(BOTTOM_LEFT_CAMPAIGN, rng, DEVICE, rgb_src=RGB_ROOT, feat_src=FEAT_ROOT)
        rgb_br, cond_br = sample_conditioning(BOTTOM_RIGHT_CAMPAIGN, rng, DEVICE, rgb_src=RGB_ROOT, feat_src=FEAT_ROOT)
        tl_list.append(cond_tl); tr_list.append(cond_tr)
        bl_list.append(cond_bl); br_list.append(cond_br)
        if p < args.n_seed_previews:
            seed_rgbs.append((rgb_tl, rgb_tr, rgb_bl, rgb_br))

    # Stack into [n_images, 768] so all images are generated together in one
    # batched call per stride (see generate_grid's (image, window) task pool).
    conds = (torch.cat(tl_list, dim=0), torch.cat(tr_list, dim=0),
             torch.cat(bl_list, dim=0), torch.cat(br_list, dim=0))

    curves = {}
    for wv in variants:
        print(f"\n===== blend-window variant: {WINDOW_LABELS[wv]} =====")
        curves[wv] = run_variant(wv, model, rae, conds, seed_rgbs, args, DEVICE)

    if len(curves) == 2:
        out = FIG_ROOT / CFM_RUN_NAME / "stitching_seam_ratio_both.png"
        plot_summary([(k, *curves[k]) for k in ("hann", "uniform")], out)
        print(f"Saved combined summary to {out.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate 2D tiled-generation stitching via seam gradients.")
    ap.add_argument("--scale", type=int, default=3,
                    help="Scaling factor: canvas is a scale x scale grid of tiles "
                         f"(output ~ {GRID}*scale*{PATCH} px per side).")
    ap.add_argument("--strides", type=int, nargs="+", default=[16, 12, 8, 4, 2],
                    help=f"Window strides to sweep (<= window size {GRID}). "
                         "overlap%% = (grid - stride)/grid.")
    ap.add_argument("--n_images", type=int, default=3, help="Images averaged per stride.")
    ap.add_argument("--n_seed_previews", type=int, default=3,
                    help="Save corner-seed montages (seeds_image*.png) for the first N "
                         "images so you can see which seeds were sampled (0 to disable).")
    ap.add_argument("--n_diag_images", type=int, default=3,
                    help="Per stride, save this many per-image diagnostic figures "
                         "(diag_stride*_img*.png) + their background panoramas, for the "
                         "first N images. Raise it to inspect stitching over more samples "
                         "for the paper (0 to disable).")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Max windows batched together per model forward pass "
                         "(cond+uncond fused into one call). Lower this if you hit OOM "
                         "at small strides / large scale, where window counts can reach the hundreds.")
    ap.add_argument("--decode_batch_size", type=int, default=None,
                    help="Whole images decoded together in the final RAE decode "
                         "(defaults to --batch_size). Keep this small at large --scale: "
                         "decode memory grows as decode_batch_size * (grid*scale)**4 via the "
                         "decoder's full-canvas attention, so decouple it from the (large) "
                         "--batch_size used to keep generation fast.")
    ap.add_argument("--n_steps", type=int, default=100, help="Flow-matching integration steps.")
    ap.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Initial-noise scale.")
    ap.add_argument("--seam_band", type=int, default=PATCH,
                    help="Half-width (px) of the band around each seam (fixed across strides).")
    ap.add_argument("--edge_margin", type=int, default=PATCH,
                    help="Pixels excluded at each border from the baseline.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for seed sampling and noise.")
    ap.add_argument("--window", choices=["hann", "uniform", "both"], default=None,
                    help="Blend window over overlapping tiles: 'hann' (tapered, "
                         "default), 'uniform' (box averaging, vanilla "
                         "MultiDiffusion), or 'both' to run the two sweeps back "
                         "to back on identical seeds/noise -- each into its own "
                         "dir (stitching / stitching_no_hann) -- plus a combined "
                         "two-line summary figure.")
    ap.add_argument("--no_hann", action="store_true",
                    help="Legacy alias for --window uniform (ignored when "
                         "--window is given).")
    ap.add_argument("--from_json", type=str, default=None,
                    help="Skip generation entirely; reload a saved stitching_metrics.json "
                         "and re-render the figures (summary + diagnostics from the saved "
                         "rgb_stride*.png backgrounds) to refine plot styling.")
    ap.add_argument("--from_json_uniform", type=str, default=None,
                    help="With --from_json (the Hann JSON): also load this "
                         "uniform-window JSON and render only the combined "
                         "two-line summary figure.")
    main(ap.parse_args())
