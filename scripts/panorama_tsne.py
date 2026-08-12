# scripts/panorama_tsne.py
"""Embedding-space trajectory of condition-interpolated panoramas (paper figure).
Generates N panoramas, each interpolating between a random Hawaii seed (left)
and a random Batemans seed (right) exactly as in test_panorama.py, but with a
canvas sized so the generation windows *are* the analysis windows: 4 abutting
windows of 16 latent columns (stride 16, no overlap) -> latent canvas W = 64.
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # headless; set before any pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# DINOv2 layers warn once each that xFormers is available; informational only.
warnings.filterwarnings("ignore", message="xFormers is available")

from benthicflow import CKPT_ROOT, FIG_ROOT
from models.unet_cfm import UNetCFM
from scripts.test_panorama import CFM_RUN_NAME, FEAT_SRC, GRID, RAE_CKPT, RGB_SRC
from scripts.train_cfm_cfg import load_rae_frozen

LEFT_CAMPAIGN = "Hawaii201801"
RIGHT_CAMPAIGN = "ScottReef200907"  # the other end of the transect
# Pin each side to one deployment (None = random deployment per panorama).
# Deployments of one campaign can sit in different DINO clusters, which would
# smear the endpoints; see --list-deployments for the available names.
LEFT_DEPLOYMENT = None
RIGHT_DEPLOYMENT = None
DINO_NATIVE = 518  # stored seed features were extracted at 518 (37x37)
OUT_DIR = FIG_ROOT / CFM_RUN_NAME
CACHE_NPZ = OUT_DIR / "panorama_tsne_embeddings.npz"

# ----------------------------------------------------------------------------- #
# ECCV 2026 (Springer LNCS) figure style.
# LNCS text block is 122 mm wide -> 4.803 in. The figure is built at exact half of
# that width so in a a half column minipage \includegraphics[width=\textwidth]
# applies no scaling: an 8 pt label here is an 8 pt label on the printed page
# (LNCS body is 10 pt, captions 9 pt). fonttype 42 embeds TrueType (no Type-3),
# as required by the submission checker.
# ----------------------------------------------------------------------------- #
ECCV_TEXTWIDTH_IN = 122 / 25.4  # 2.4015 in
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 8 * 1.5,
        "axes.labelsize": 8 * 1.5,
        "legend.fontsize": 7.5 * 1.1,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sample_conditioning(
    campaign,
    rng,
    device,
    deployment=None,
    anchor=None,
    sim_thr=0.85,
    white_thr=0.9,
    black_thr=0.1,
    min_std=0.03,
    max_tries=200,
):
    """test_panorama.sample_conditioning, plus a `deployment` pin and an
    `anchor` similarity gate.
    If `anchor` (a pooled [1,768] cond from a previous call) is given, a
    candidate is only accepted when the cosine similarity of its pooled DINO
    embedding to the anchor is >= sim_thr. Anchoring every panorama's seed to
    """
    deployments = sorted(p.stem for p in (RGB_SRC / campaign).glob("*.npy"))
    if not deployments:
        raise FileNotFoundError(f"No RGB arrays under {RGB_SRC / campaign}")
    if deployment is not None and deployment not in deployments:
        raise FileNotFoundError(
            f"Deployment '{deployment}' not found in {campaign}. Available:\n  "
            + "\n  ".join(deployments)
        )

    for attempt in range(1, max_tries + 1):
        dep = deployment if deployment is not None else str(rng.choice(deployments))
        rgb_arr = np.load(RGB_SRC / campaign / f"{dep}.npy", mmap_mode="r")
        feat_arr = np.load(FEAT_SRC / campaign / f"{dep}.npy", mmap_mode="r")
        idx = int(rng.integers(len(rgb_arr)))

        rgb = np.asarray(rgb_arr[idx], dtype=np.float32) / 255.0  # [S, S, 3]
        bright, contrast = float(rgb.mean()), float(rgb.std())
        if bright > white_thr or bright < black_thr or contrast < min_std:
            why = (
                "white"
                if bright > white_thr
                else "black" if bright < black_thr else "flat"
            )
            print(
                f"  [reject {attempt:02d}] {campaign}/{dep}[{idx}] "
                f"bright={bright:.3f} std={contrast:.3f} ({why})"
            )
            continue

        dino = torch.from_numpy(
            np.asarray(feat_arr[idx], dtype=np.float32)
        )  # [G, G, 768]
        cond = (
            dino.reshape(-1, dino.shape[-1]).mean(0, keepdim=True).to(device)
        )  # [1, 768]

        sim_note = ""
        if anchor is not None and sim_thr > 0:
            sim = F.cosine_similarity(cond, anchor).item()
            if sim < sim_thr:
                print(
                    f"  [reject {attempt:02d}] {campaign}/{dep}[{idx}] "
                    f"cos={sim:.3f} < {sim_thr} (dissimilar to anchor)"
                )
                continue
            sim_note = f" cos={sim:.3f}"

        print(
            f"  [accept]    {campaign}/{dep}[{idx}] "
            f"bright={bright:.3f} std={contrast:.3f}{sim_note}"
        )
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, S, S]
        return rgb_t, cond, f"{dep}[{idx}]"

    raise RuntimeError(
        f"No acceptable image found in {campaign} after {max_tries} tries "
        f"(exposure: black<{black_thr}, white>{white_thr}, std<{min_std}"
        + (
            f"; cosine>={sim_thr} to anchor"
            if anchor is not None and sim_thr > 0
            else ""
        )
        + "). Lower --sim-thr or pin an easier deployment."
    )


@torch.no_grad()
def generate_interp_panorama(
    model,
    rae,
    cond_left,
    cond_right,
    n_windows=4,
    n_steps=100,
    cfg_scale=3.0,
    temperature=1.0,
    device="cuda",
):
    """test_panorama.generate_panorama, but parameterised by *window count*.

    stride = GRID (abutting tiles, zero overlap), latent W = n_windows * GRID,
    so the flow-matching windows coincide with the crops we later embed. With no
    overlap the Hann weights cancel (vel/wacc = v) and each tile integrates
    independently. Returns (pano_rgb [1,3,224,W*14], window px starts, alphas).
    """
    grid, D = GRID, model.latent_dim
    stride = grid
    H, W = grid, grid + (n_windows - 1) * stride
    null = model.null_cond.unsqueeze(0)

    win = torch.sin(torch.linspace(0.05, 0.95, grid, device=device) * torch.pi)
    win = win.view(1, 1, 1, grid)

    x = torch.randn(1, D, H, W, device=device) * temperature
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = ts[1] - ts[0]
    starts = list(range(0, W - grid + 1, stride))
    assert len(starts) == n_windows
    alphas = [(w0 + grid / 2) / W for w0 in starts]  # 0.125, 0.375, 0.625, 0.875

    for i in range(n_steps):
        t = ts[i].expand(1)
        vel = torch.zeros_like(x)
        wacc = torch.zeros(1, 1, 1, W, device=device)
        for w0, alpha in zip(starts, alphas):
            cond_w = (1.0 - alpha) * cond_left + alpha * cond_right
            xc = x[:, :, :, w0 : w0 + grid]
            v_cond = model(xc, t, cond_w)
            if cfg_scale != 1.0:
                v_uncond = model(xc, t, null)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            vel[:, :, :, w0 : w0 + grid] += v * win
            wacc[:, :, :, w0 : w0 + grid] += win
        x = x + (vel / (wacc + 1e-8)) * dt

    z = model.denormalize(x)
    fused = z.permute(0, 2, 3, 1).contiguous()
    rgbd = rae.decoder(fused)  # [1, 4, 224, W*14]
    px_starts = [w0 * 14 for w0 in starts]
    return rgbd[:, :3].clamp(0, 1), px_starts, alphas


@torch.no_grad()
def embed_windows(dino_model, crops):
    """Pooled DINO embedding of [B,3,224,224] crops in [0,1], at 518 native res
    to match the resolution the stored seed features were extracted at."""
    from scripts.test_generation import dino_patch_embed

    crops = F.interpolate(
        crops, size=(DINO_NATIVE, DINO_NATIVE), mode="bilinear", align_corners=False
    )
    return dino_patch_embed(dino_model, crops).mean(dim=1)  # [B, 768]


def compute_embeddings(args, device):
    """Generate args.n_panoramas panoramas and return everything cached to npz."""
    from scripts.extract_features import load_dinov2

    model = UNetCFM().to(device)
    model.load_state_dict(
        torch.load(
            CKPT_ROOT / CFM_RUN_NAME / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    rae = load_rae_frozen(RAE_CKPT, device)
    dino_model = load_dinov2()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # flow-matching init noise (torch.randn)
    n_items = args.n_windows + 2
    emb = np.zeros((args.n_panoramas, n_items, 768), dtype=np.float32)
    panos, seeds_l, seeds_r, keys_l, keys_r = [], [], [], [], []
    alphas = None
    anchor_l = anchor_r = None  # first accepted seed per side

    for k in range(args.n_panoramas):
        print(f"\n=== panorama {k + 1}/{args.n_panoramas} ===")
        seed_l_rgb, cond_l, key_l = sample_conditioning(
            LEFT_CAMPAIGN,
            rng,
            device,
            deployment=args.left_deployment,
            anchor=anchor_l,
            sim_thr=args.sim_thr,
        )
        seed_r_rgb, cond_r, key_r = sample_conditioning(
            RIGHT_CAMPAIGN,
            rng,
            device,
            deployment=args.right_deployment,
            anchor=anchor_r,
            sim_thr=args.sim_thr,
        )
        if anchor_l is None:
            anchor_l, anchor_r = cond_l, cond_r
        keys_l.append(key_l)
        keys_r.append(key_r)

        pano_rgb, px_starts, win_alphas = generate_interp_panorama(
            model,
            rae,
            cond_l,
            cond_r,
            n_windows=args.n_windows,
            n_steps=args.n_steps,
            cfg_scale=args.cfg_scale,
            device=device,
        )

        crops = torch.cat(
            [pano_rgb[:, :, :, p : p + 224] for p in px_starts]
        )  # [n_w,3,224,224]
        win_emb = embed_windows(dino_model, crops)  # [n_w, 768]

        # Seeds pooled from their stored grids = the conditioning vectors.
        emb[k, 0] = cond_l.squeeze(0).cpu().numpy()
        emb[k, 1:-1] = win_emb.cpu().numpy()
        emb[k, -1] = cond_r.squeeze(0).cpu().numpy()
        alphas = np.array([0.0] + list(win_alphas) + [1.0], dtype=np.float32)

        # Keep uint8 copies (downsized seeds) for a possible qualitative figure.
        panos.append(
            (pano_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        for lst, s in ((seeds_l, seed_l_rgb), (seeds_r, seed_r_rgb)):
            s224 = F.interpolate(s.unsqueeze(0), size=(224, 224), mode="area")[0]
            lst.append((s224.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_NPZ,
        embeddings=emb,
        alphas=alphas,
        panos=np.stack(panos),
        seeds_l=np.stack(seeds_l),
        seeds_r=np.stack(seeds_r),
        seed_keys_l=np.array(keys_l),
        seed_keys_r=np.array(keys_r),
        left_campaign=LEFT_CAMPAIGN,
        right_campaign=RIGHT_CAMPAIGN,
        seed=args.seed,
    )
    print(f"\nCached embeddings to {CACHE_NPZ}")
    return emb


def demo_embeddings(args):
    """Synthetic two-cluster + noisy-interpolation embeddings to preview layout."""
    rng = np.random.default_rng(args.seed)
    n, n_items = args.n_panoramas, args.n_windows + 2
    a, b = rng.normal(size=768), rng.normal(size=768)
    curve = rng.normal(size=768)  # bow the path off the chord
    alphas = np.linspace(0, 1, n_items)
    emb = np.zeros((n, n_items, 768), dtype=np.float32)
    for k in range(n):
        for i, al in enumerate(alphas):
            noise = 0.25 if i in (0, n_items - 1) else 0.45
            emb[k, i] = (
                (1 - al) * a
                + al * b
                + 0.8 * al * (1 - al) * curve
                + noise * rng.normal(size=768)
            )
    return emb


def reduce_2d(emb, seed, perplexity=None):
    """L2-normalise (cosine geometry) -> PCA(50) -> t-SNE(2)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n, n_items, d = emb.shape
    flat = emb.reshape(-1, d).astype(np.float64)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12

    n_pts = flat.shape[0]
    flat = PCA(n_components=min(50, n_pts, d), random_state=seed).fit_transform(flat)
    if perplexity is None:
        perplexity = min(30, max(5, n_pts // 5))
    print(f"t-SNE on {n_pts} points (perplexity={perplexity})")
    xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(flat)
    return xy.reshape(n, n_items, 2)


def plot_contact_sheet(seeds_l, panos, seeds_r, out_path):
    """Diagnostic sheet (not a paper figure): one row per panorama showing
    left seed | generated panorama | right seed, so you can eyeball exactly
    which images each embedding trajectory compares."""
    n = len(panos)
    ratios = [1.0, panos.shape[2] / panos.shape[1], 1.0]  # pano aspect (W/H)
    row_h = 7.0 / sum(ratios)  # seed cell is square
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(7.0, row_h * n * 1.06),
        squeeze=False,
        gridspec_kw={"width_ratios": ratios, "wspace": 0.02, "hspace": 0.08},
    )
    for k in range(n):
        for ax, im in zip(axes[k], (seeds_l[k], panos[k], seeds_r[k])):
            ax.imshow(im)
            ax.set_axis_off()
        axes[k, 1].text(
            0.01,
            0.95,
            f"#{k}",
            transform=axes[k, 1].transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
        )
    for ax, title in zip(
        axes[0],
        (
            f"Seed L\n{LEFT_CAMPAIGN}",
            "Panorama (cond. interp. L→R)",
            f"Seed R\n{RIGHT_CAMPAIGN}",
        ),
    ):
        ax.set_title(title, fontsize=7, pad=2)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_tsne(xy, out_stem, images=None, example=-1):
    """ECCV-column figure: t-SNE trajectory scatter; if `images` is given
    (seeds_l, panos, seeds_r), one example's seed L / panorama / seed R strip is
    placed below with arrows from each image to its point in the embedding."""

    name_mappings = {
        "Hawaii": "Hawaii",
        "ScottReef": "Scott Reef",
        "Batemans": "Batemans Bay",
    }

    n, n_items, _ = xy.shape
    # t-SNE is reflection-invariant, so orient the map with the left seed's
    # cluster on the left: the embedding then reads in the same left-to-right
    # direction as the panorama strip, and the arrows barely cross.
    if xy[:, 0, 0].mean() > xy[:, -1, 0].mean():
        xy = xy * np.array([-1.0, 1.0])
    left_site = name_mappings.get(
        LEFT_CAMPAIGN.rstrip("0123456789"), LEFT_CAMPAIGN.rstrip("0123456789")
    )
    right_site = name_mappings.get(
        RIGHT_CAMPAIGN.rstrip("0123456789"), RIGHT_CAMPAIGN.rstrip("0123456789")
    )
    labels = (
        [f"{left_site} seed"]
        + [f"Window {i}" for i in range(1, n_items - 1)]
        + [f"{right_site} seed"]
    )
    # Ordinal ramp for an ordered variable (window index); CVD-safe, monotone
    # lightness. Seeds additionally get a diamond marker so the endpoints are
    # identified by shape as well, not colour alone.
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, n_items))

    if images is None:
        fig, ax = plt.subplots(figsize=(ECCV_TEXTWIDTH_IN, 3.1))
    else:
        n_w = n_items - 2
        strip_h = ECCV_TEXTWIDTH_IN / (n_w + 2)  # square seed cells
        fig = plt.figure(figsize=(ECCV_TEXTWIDTH_IN, 2.75 + strip_h + 0.45))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[2.75, strip_h],
            width_ratios=[1, n_w, 1],
            hspace=0.16,
            wspace=0.05,
        )
        ax = fig.add_subplot(gs[0, :])
        ax_strip = [fig.add_subplot(gs[1, j]) for j in range(3)]

    for k in range(n):  # trajectories underneath
        ax.plot(
            xy[k, :, 0],
            xy[k, :, 1],
            color="0.55",
            lw=0.5,
            alpha=0.35,
            zorder=1,
            solid_capstyle="round",
        )

    for i in range(n_items):
        is_seed = i in (0, n_items - 1)
        ax.scatter(
            xy[:, i, 0],
            xy[:, i, 1],
            s=26 if is_seed else 17,
            marker="D" if is_seed else "o",
            facecolor=colors[i],
            edgecolor="0.25",
            linewidth=0.35,
            label=labels[i],
            zorder=3 if is_seed else 2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("0.35")
    ax.set_ylabel("t-SNE dim. 2", labelpad=2)
    if images is None:  # arrows would cross it otherwise
        ax.set_xlabel("t-SNE dim. 1", labelpad=2)

    # Legend keeps the window sequence but oriented to match the plot: the
    # endpoint whose cluster sits further left leads (after the flip above this
    # is the forward order, but keep it adaptive rather than hardcoded).
    order = np.arange(n_items)
    if xy[:, -1, 0].mean() < xy[:, 0, 0].mean():
        order = order[::-1]
    handles, lbls = ax.get_legend_handles_labels()
    legend_kw = dict(
        ncol=n_items,
        frameon=False,
        handletextpad=0.15,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    if images is None:
        ax.legend(
            [handles[j] for j in order],
            [lbls[j] for j in order],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
            **legend_kw,
        )
    else:
        ax.legend(
            [handles[j] for j in order],
            [lbls[j] for j in order],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            **legend_kw,
        )

    if images is not None:
        seeds_l, panos, seeds_r = images
        # Example: the most prototypical trajectory (closest to the per-window
        # centroids), unless a specific panorama index was requested.
        cent = xy.mean(axis=0)  # [n_items, 2]
        k = (
            example
            if 0 <= example < n
            else int(np.argmin(((xy - cent) ** 2).sum(-1).sum(-1)))
        )
        print(f"Example panorama in the strip: #{k}")

        # Highlight the example's trajectory on top of the cloud.
        ax.plot(xy[k, :, 0], xy[k, :, 1], color="0.1", lw=0.9, zorder=4)
        for i in range(n_items):
            is_seed = i in (0, n_items - 1)
            ax.scatter(
                *xy[k, i],
                s=34 if is_seed else 24,
                marker="D" if is_seed else "o",
                facecolor=colors[i],
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

        for a, im, lab in zip(
            ax_strip,
            (seeds_l[k], panos[k], seeds_r[k]),
            (f"{left_site} seed", "Generated panorama", f"{right_site} seed"),
        ):
            a.imshow(im)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.5)
                s.set_color("0.5")
            a.set_xlabel(lab, fontsize=7.5 * 1.5, color="0.2", labelpad=2)

        # One arrow per item: seeds from their cell centres, windows from the
        # centre of their (abutting) tile in the panorama. Anchors in image
        # data coords, so letterboxing inside the axes cannot misplace them.
        w_px = panos.shape[2] / n_w
        anchors = (
            [(ax_strip[0], (seeds_l.shape[2] / 2, 0), 0)]
            + [(ax_strip[1], ((i + 0.5) * w_px, 0), i + 1) for i in range(n_w)]
            + [(ax_strip[2], (seeds_r.shape[2] / 2, 0), n_items - 1)]
        )
        for axA, xyA, i in anchors:
            fig.add_artist(
                ConnectionPatch(
                    xyA=xyA,
                    coordsA="data",
                    axesA=axA,
                    xyB=tuple(xy[k, i]),
                    coordsB="data",
                    axesB=ax,
                    arrowstyle="-|>",
                    mutation_scale=5,
                    lw=0.55,
                    color="0.3",
                    alpha=0.85,
                    shrinkB=2.5,
                )
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"Saved {out_stem}.pdf / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-panoramas", type=int, default=24)
    ap.add_argument("--n-windows", type=int, default=4, help="intermediate windows")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--left-deployment",
        default=LEFT_DEPLOYMENT,
        help="pin the left seed to one deployment (None = random per panorama)",
    )
    ap.add_argument(
        "--right-deployment",
        default=RIGHT_DEPLOYMENT,
        help="pin the right seed to one deployment",
    )
    ap.add_argument(
        "--list-deployments",
        action="store_true",
        help="print available deployments for both campaigns and exit",
    )
    ap.add_argument(
        "--sim-thr",
        type=float,
        default=0.85,
        help="accept a seed only if its pooled-DINO cosine similarity "
        "to the first seed of its side is >= this (0 disables)",
    )
    ap.add_argument("--perplexity", type=float, default=None)
    ap.add_argument(
        "--example",
        type=int,
        default=-1,
        help="panorama index shown in the image strip "
        "(-1 = most prototypical trajectory)",
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="skip generation, re-plot from the cached npz (CPU)",
    )
    ap.add_argument(
        "--demo",
        action="store_true",
        help="synthetic embeddings; preview figure layout (CPU)",
    )
    args = ap.parse_args()

    if args.list_deployments:
        for c in (LEFT_CAMPAIGN, RIGHT_CAMPAIGN):
            deps = sorted(p.stem for p in (RGB_SRC / c).glob("*.npy"))
            n_imgs = {
                d: len(np.load(RGB_SRC / c / f"{d}.npy", mmap_mode="r")) for d in deps
            }
            print(f"\n{c} ({len(deps)} deployments):")
            for d in deps:
                print(f"  {d}  ({n_imgs[d]} images)")
        sys.exit(0)

    if args.demo:
        emb, images = demo_embeddings(args), None
        out_stem = OUT_DIR / "panorama_tsne_demo"
    else:
        if args.plot_only:
            if not CACHE_NPZ.exists():
                sys.exit(
                    f"--plot-only but no cache at {CACHE_NPZ}; run without it first."
                )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_embeddings(args, device)
        data = np.load(CACHE_NPZ)
        emb = data["embeddings"]
        images = (data["seeds_l"], data["panos"], data["seeds_r"])
        out_stem = OUT_DIR / "panorama_tsne"

    xy = reduce_2d(emb, seed=args.seed, perplexity=args.perplexity)
    plot_tsne(xy, out_stem, images=images, example=args.example)

    if images is not None:
        plot_contact_sheet(*images, OUT_DIR / "panorama_tsne_seeds.png")
