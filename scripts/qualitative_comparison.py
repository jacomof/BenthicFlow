# scripts/qualitative_comparison.py
"""Qualitative comparison: Original / FLUX.2-dev / BenthicFlow / BenthicFlow-DPF.

5 rows x 6 columns. Three hand-picked conditioning images (1 per site:
Batemans, Hawaii, Scott Reef), each shown as an [image, depth] column pair --
there is no second *photo* per site anymore; the second column is always that
row's own depth estimate, turbo-colored. A horizontal rule separates the
reef-domain rows from the DPF (enhanced-domain) section:

  row 1  "Reference": the raw uncropped photo, plus Depth-Anything-V2-Large's
         estimate on that same photo (extract_depth.py's non-metric variant,
         per-image min-max normalized to [0, 1] -- "reverse" depth, closer =
         higher value -- computed full-aspect/no center-crop, unlike the
         cached training-target pipeline, to match this row's own uncropped
         image);
  row 2  FLUX.2-dev (4-bit NF4), reference-conditioned on the full uncropped
         frame with the shared REEF_PROMPT -- FLUX has no depth output, so its
         depth cell is left empty;
  row 3  BenthicFlow (reef CFM, unet_cfm_final), conditioned on the pooled
         DINO of the reference (518 native pipeline, as in training), sampled
         at a NEAR-NATIVE 16x21 latent (224x294 px: training height, aspect-
         matched width) and only upsampled at assembly -- like the DPF row.
         Sampling at the 518-class 37x49 canvas (~7x the 16x16 training area)
         made the fully-convolutional UNet/decoder tile repetitive texture;
         depth is the RAE decoder's own 4th channel (sigmoid head,
         models/rae.py) -- normalized *reverse* depth in [0, 1], the same
         convention as row 1;
  ------ separator: everything below lives in the DPF-enhanced domain ------
  row 4  "Reference (DPF-Net Enhanced)": the DPF-enhanced counterpart of the
         same image (DATA_NORM_DPF, 256 px) -- the conditioning input of row 5
         -- plus DPF-Net's own stored depth *target* (DPEM's x_d, from
         clean_and_extract.py): unnormalized **metric** depth (softplus head,
         not sigmoid -- see models/rae.py's variant-driven depth head), with
         no shared scale
         across images, so it's colored per-image (that image's own min/max);
  row 5  BenthicFlow-DPF (CFM trained on DPF-enhanced imagery), generated
         in-process by --stage dpf (the DPF pipeline lives in this repo since
         the unification -- no more separate-repo subprocess), conditioned on
         the pooled DINO of row 4 (266 native pipeline, as in training) and
         sampled at its NATIVE 19x19 latent (266 px) -- rows 4-5
         are only upsampled at assembly, since generating at the 518-class
         canvas made the content look zoomed-out vs the training domain. Its
         depth is that same model's own 4th channel: the same unnormalized
         metric convention as row 4, also colored per-image.

The two depth conventions are NOT interchangeable: reef's CFM (rows 1, 3)
outputs normalized *reverse* depth in [0, 1] (closer = higher; fixed vmin=0,
vmax=1, turbo -> red = close), while DPF-Net (rows 4, 5) outputs unnormalized
metric *distance* (closer = LOWER) with an arbitrary per-image scale (vmin/
vmax taken from that image alone). To keep the color language consistent
across the figure -- red = close everywhere -- the DPF rows are rendered with
the REVERSED turbo colormap. The DPF-domain maps also look intrinsically much
smoother than row 1's: x_d comes from DAv2-vits at 256 px inside DPEM (vs
DAv2-Large at full resolution for row 1), and row 5 is decoded from a 19x19
latent -- that blobbiness is in the data, not a plotting artifact. Finally,
DPEM's metric depth is known to be sign-unstable per-image (independent min/
max heads with nothing enforcing max > min), so a single DPF-domain panel
that still reads inverted relative to its neighbors is a known model quirk,
not a plotting bug.

All generations keep the original's aspect ratio, no cropping. Both CFM rows
sample at (or just beyond) their native training scale and are upsampled only
for display: sampling far beyond the training canvas is where conv models
degrade into repetitive texture, so the figure shows each model at the scale
it was trained for. FLUX generates at 512 x (aspect-matched multiple of 16),
its own native scale.

Stages (separate processes; each caches its row as PNGs/NPYs so any stage can
be re-run alone, and assembly is CPU-only):

  --stage cfm       BenthicFlow row + Reference depth (GPU)
                     -> rows/cfm_{i}.png, rows/cfm_{i}_depth.npy, rows/ref_{i}_depth.npy
  --stage dpf       BenthicFlow-DPF row (GPU)
                     -> rows/dpf_{i}.png, rows/dpf_{i}_depth.npy
  --stage flux      FLUX.2-dev row (GPU)          -> rows/flux_{i}.png
  --stage assemble  figure from cached rows (CPU) -> qualitative_comparison.pdf/.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from benthicflow import CKPT_ROOT, FIG_ROOT, SCRATCH_ROOT

# ---- the 3 hand-picked conditioning images (1 per site, column order) ------ #
DATA_NORM = Path("./data_normalized/")
# DPF-enhanced counterparts (same campaign/deployment/filename, 256 px); the
# BenthicFlow-DPF row is conditioned on these, since that model was trained on
# the enhanced imagery.
DATA_NORM_DPF = Path("./data_normalized-dpf")
# DPF-Net's stored DPEM depth *targets* (x_d): metric, unnormalized, keyed by
# bare stem (no ".jpg") -- see clean_and_extract.py / the dpfnet-depth-target
# note. Read directly here (plain .npy on shared scratch); no DPF-Net import
# needed for this row's ground-truth depth.
DEPTH_ROOT_DPF = Path("./depth-dpf")
# IMAGES = [
#     DATA_NORM / "Batemans201011/r20101119_204443_site6guz_10_densegrids/PR_20101119_204942_996_LC16.jpg",
#     DATA_NORM / "Batemans201011/r20101120_222627_site2guz_14_densegrids/PR_20101120_223131_607_LC16.jpg",
#     DATA_NORM / "Hawaii201801/r20180206_184219_SS16_waikoloa_legs_mid_south/PR_20180206_184722_173_LC16.jpg",
#     DATA_NORM / "Hawaii201801/r20180203_182731_SS12_waikoloa_broad_cross_200_150_c_shallow/PR_20180203_183824_386_LC16.jpg",
#     DATA_NORM / "ScottReef201108/r20110811_222428_08_scott_grids_deep_auv2/PR_20110811_223113_978_LC16.jpg",
#     DATA_NORM / "ScottReef201503/r20150331_050931_10_scott_long_leg_auv2/PR_20150331_053958_722_LC16.jpg",
# ]
IMAGES = [
    DATA_NORM / "Batemans201411/r20141112_004500_02_Tollgates_site4sz_dense/PR_20141112_005105_839_LC16.jpg",
    DATA_NORM / "Hawaii201801/r20180203_182731_SS12_waikoloa_broad_cross_200_150_c_shallow/PR_20180203_183824_386_LC16.jpg",
    DATA_NORM / "ScottReef201503/r20150330_225013_09_scott_grids_deep_auv2/PR_20150330_225510_723_LC16.jpg",
]
SITES = ["Batemans", "Hawaii", "Scott Reef"]        # one header per column pair
# Rotated ylabels must be shorter than the ~45 pt row pitch at 8.45 pt, so
# the BenthicFlow labels are hyphenated onto extra lines (first line renders
# outermost).
ROW_LABELS = ["Reference", "FLUX.2-dev", "Benthic-\nFlow\n(Ours)",
              "Reference\n(DPF-Net)", "Benthic-\nFlow-DPF\n(Ours)"]

CFM_CKPT = CKPT_ROOT / "unet_cfm_final" / "best_model.pt"
RAE_CKPT = CKPT_ROOT / "rae" / "last.pt"
# BenthicFlow-DPF checkpoints live under the dpf-variant root (what REEF would
# resolve as CKPT_ROOT with REEF_VARIANT=dpf; this process stays base-variant,
# so spell it out).
CKPT_ROOT_DPF = SCRATCH_ROOT / "checkpoints-dpf"
CFM_CKPT_DPF = CKPT_ROOT_DPF / "unet_cfm_final_dpf" / "best_model.pt"
RAE_CKPT_DPF = CKPT_ROOT_DPF / "rae" / "last.pt"

OUT_DIR = FIG_ROOT / "qualitative_comparison"
ROWS_DIR = OUT_DIR / "rows"

PATCH = 14
DINO_NATIVE = 518                       # reef feature-extraction resolution
DINO_NATIVE_DPF = 266                   # features-dpf extraction resolution

# ECCV 2026 (LNCS): build at the exact 122 mm text width (see panorama_tsne
# .py). Fonts are 30% larger than the at-\textwidth sizing because this figure
# is placed side by side with another one, so LaTeX scales it down.
ECCV_TEXTWIDTH_IN = 122 / 25.4
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@torch.no_grad()
def pooled_dino(dino_model, pil_img, native):
    """Mean-pooled DINOv2 patch embedding, replicating extract_features.py's
    Resize(native) + CenterCrop(native) preprocessing -> [1, 768]."""
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(native, antialias=True),
        transforms.CenterCrop(native),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(pil_img).unsqueeze(0).to(next(dino_model.parameters()).device)
    tokens = dino_model.forward_features(x)["x_norm_patchtokens"]  # [1, N, 768]
    return tokens.mean(dim=1)


def save_rgb(rgb_chw, path):
    arr = (rgb_chw.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)
    print(f"  saved {path}")


@torch.no_grad()
def dav2_depth_of(dav2_run, pil_img):
    """Depth-Anything-V2's raw estimate for the "Reference" row: resized to
    the source image's own (h, w) and per-image min-max normalized to [0, 1]
    (reverse depth: closer = higher value -- the same convention extract_depth
    .py uses for the training targets), but full-aspect / no center-crop, to
    match this row's own uncropped photo."""
    w, h = pil_img.size
    d = np.asarray(dav2_run([pil_img])[0], dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(np.squeeze(d)))[None, None]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    d = t[0, 0].numpy()
    lo, hi = float(d.min()), float(d.max())
    return (d - lo) / max(hi - lo, 1e-6)


def load_dpf_target_depth(p: Path):
    """DPF-Net's stored DPEM depth target (x_d) for one enhanced image --
    (256, 256) float32, unnormalized metric -- or None if not cached for that
    deployment."""
    campaign, deployment = p.parts[-3], p.parts[-2]
    depth_npy = DEPTH_ROOT_DPF / campaign / f"{deployment}.npy"
    keys_npy = DEPTH_ROOT_DPF / campaign / f"{deployment}_keys.npy"
    if not depth_npy.exists():
        return None
    keys = np.load(keys_npy)
    idx = np.where(keys == p.stem)[0]                  # DPF-Net keys are bare stems
    if len(idx) == 0:
        return None
    depths = np.load(depth_npy, mmap_mode="r")
    return np.asarray(depths[idx[0], 0], dtype=np.float32)


def resize_depth_array(arr, size_wh):
    """Bilinear-resize a raw (metric or normalized) depth array to (w, h),
    e.g. to match the squashed display size used for the DPF-domain rows."""
    w, h = size_wh
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def stage_cfm(args, device):
    """Reef CFM row: sampled at the model's near-native canvas (training
    height 16, aspect-matched width -> 16x21, 224x294 px) and upsampled only
    at assembly; the 518-class 37x49 canvas produced repetitive tiled texture
    (see module docstring). Also caches each image's depth: the CFM's own
    decoded depth channel (RAE sigmoid head -- normalized reverse depth,
    [0, 1]) and, for the Reference row, Depth-Anything-V2's estimate on the
    raw photo itself."""
    from models.unet_cfm import UNetCFM, cfm_sample_cfg
    from scripts.train_cfm_cfg import load_rae_frozen, TRAIN_GRID
    from scripts.extract_features import load_dinov2
    from scripts.extract_depth import load_model as load_dav2

    w, h = Image.open(IMAGES[0]).size
    lat_h = TRAIN_GRID                                     # 16 -> 224 px
    lat_w = round(TRAIN_GRID * w / h)                      # aspect-matched, 21
    print(f"CFM canvas: {lat_h}x{lat_w} latent -> {lat_h * PATCH}x{lat_w * PATCH} px "
          f"(near-native; upsampled at assembly)")

    model = UNetCFM().to(device)
    model.load_state_dict(torch.load(CFM_CKPT, map_location=device, weights_only=True))
    model.eval()
    rae = load_rae_frozen(RAE_CKPT, device)
    dino_model = load_dinov2()
    dav2 = load_dav2("Large", metric=False, processed=False)

    torch.manual_seed(args.seed)
    for i, p in enumerate(IMAGES):
        pil = Image.open(p).convert("RGB")
        cond = pooled_dino(dino_model, pil, DINO_NATIVE)
        z = cfm_sample_cfg(model, cond, canvas_shape=(lat_h, lat_w),
                           n_steps=args.n_steps, cfg_scale=args.cfg_scale)
        rgbd = rae.decoder(model.denormalize(z).permute(0, 2, 3, 1).contiguous())
        save_rgb(rgbd[0, :3], ROWS_DIR / f"cfm_{i}.png")
        np.save(ROWS_DIR / f"cfm_{i}_depth.npy", rgbd[0, 3].cpu().numpy().astype(np.float32))

        ref_depth = dav2_depth_of(dav2, pil)
        np.save(ROWS_DIR / f"ref_{i}_depth.npy", ref_depth)
        print(f"  saved cfm_{i}_depth.npy / ref_{i}_depth.npy")


def stage_dpf(args, device):
    """BenthicFlow-DPF row (in-process since the DPF-Net unification; formerly
    a separate-repo subprocess): conditioned on the pooled DINO of the
    DPF-enhanced counterpart (Resize(266) + CenterCrop(266), exactly the
    features-dpf extraction pipeline) and sampled at the DPF-NATIVE 19x19
    latent (266 px decoded, the training-domain scale of the 256 px enhanced
    imagery); assembly upsamples the row for display. Sampling at the reef
    518-class canvas instead made the content look zoomed-out relative to the
    training crops. Also caches the decoder's own 4th channel (unnormalized
    metric depth) for this row's depth panel."""
    from models.unet_cfm import UNetCFM, cfm_sample_cfg
    from scripts.train_cfm_cfg import load_rae_frozen
    from scripts.extract_features import load_dinov2

    lat = DINO_NATIVE_DPF // PATCH                         # 19 (square canvas)
    print(f"DPF-CFM canvas: {lat}x{lat} latent -> {lat * PATCH}x{lat * PATCH} px "
          f"(native; upsampled at assembly)")

    enhanced = []
    for p in IMAGES:
        q = DATA_NORM_DPF / p.parts[-3] / p.parts[-2] / p.name
        if not q.exists():
            sys.exit(f"Enhanced counterpart not found: {q}")
        enhanced.append(q)

    model = UNetCFM().to(device)
    model.load_state_dict(torch.load(CFM_CKPT_DPF, map_location=device, weights_only=True))
    model.eval()
    # Explicit softplus head: this process runs base-variant, so ConvDecoder
    # Config's REEF_VARIANT-driven default would silently decode this DPF
    # checkpoint's unbounded metric depth through a sigmoid (see models/rae.py).
    rae = load_rae_frozen(RAE_CKPT_DPF, device, depth_activation="softplus")
    dino_model = load_dinov2()

    torch.manual_seed(args.seed)
    for i, q in enumerate(enhanced):
        cond = pooled_dino(dino_model, Image.open(q).convert("RGB"), DINO_NATIVE_DPF)
        z = cfm_sample_cfg(model, cond, canvas_shape=(lat, lat),
                           n_steps=args.n_steps, cfg_scale=args.cfg_scale)
        rgbd = rae.decoder(model.denormalize(z).permute(0, 2, 3, 1).contiguous())
        save_rgb(rgbd[0, :3], ROWS_DIR / f"dpf_{i}.png")
        np.save(ROWS_DIR / f"dpf_{i}_depth.npy", rgbd[0, 3].cpu().numpy().astype(np.float32))
        print(f"  saved dpf_{i}_depth.npy")


def stage_flux(args, device):
    """FLUX.2-dev (4-bit) row: full uncropped frame as reference + REEF_PROMPT."""
    from scripts.flux_generation import build_pipeline, REEF_PROMPT, resize_to_multiple_of_16

    w, h = Image.open(IMAGES[0]).size
    flux_h = args.flux_size
    flux_w = round(flux_h * w / h / 16) * 16
    print(f"FLUX canvas: {flux_h}x{flux_w} px")

    pipe = build_pipeline("dev")
    for i, p in enumerate(IMAGES):
        ref = resize_to_multiple_of_16(Image.open(p).convert("RGB"), flux_w, flux_h)
        out = pipe(
            prompt=REEF_PROMPT,
            image=[ref],
            height=flux_h, width=flux_w,
            num_inference_steps=args.flux_steps,
            guidance_scale=args.flux_guidance,
            generator=torch.Generator(device).manual_seed(args.seed + i),
        ).images[0]
        out.save(ROWS_DIR / f"flux_{i}.png")
        print(f"  saved flux_{i}.png")


def stage_assemble(args):
    """5x6 grid at ECCV column width: 3 sites, each an [image, depth] column
    pair -- Reference / FLUX / BenthicFlow, a separator rule, then the DPF
    section (enhanced reference / BenthicFlow-DPF). Depth panels use turbo,
    but the two model families' depth conventions differ (see module
    docstring): reef rows (1, 3) get a fixed vmin=0, vmax=1; DPF rows (4, 5)
    are unnormalized metric *distance* with no shared scale, so each is
    colored against its own min/max with REVERSED turbo (red = close, same
    orientation as the reef rows). FLUX has no depth output -- empty cell."""
    n = len(IMAGES)
    n_cols = 2 * n
    w, h = Image.open(IMAGES[0]).size
    aspect = w / h
    lat_h = round(DINO_NATIVE / PATCH)
    # DPF-section rows are produced/kept at their native 256-class resolution
    # (the DPF training domain) and only upsampled here for display; stretching
    # the square enhanced frames to the cell aspect undoes their aspect squash.
    up_size = (round(lat_h * aspect) * PATCH, lat_h * PATCH)

    def cached_rgb(tag, i, resize=None):
        f = ROWS_DIR / f"{tag}_{i}.png"
        if not f.exists():
            print(f"[warn] missing {f} -- gray placeholder (run its stage first)")
            return np.full((up_size[1], up_size[0], 3), 200, dtype=np.uint8)
        img = Image.open(f).convert("RGB")
        if resize is not None and img.size != resize:
            img = img.resize(resize, Image.LANCZOS)
        return np.asarray(img)

    def cached_depth_arr(tag, i):
        f = ROWS_DIR / f"{tag}_{i}_depth.npy"
        if not f.exists():
            print(f"[warn] missing {f} -- blank depth cell (run its stage first)")
            return None
        return np.load(f)

    def depth01_cell(arr):
        """Reef-domain depth: normalized reverse depth (closer = higher),
        fixed [0, 1] scale, turbo -> red = close."""
        return ("empty",) if arr is None else ("depth", arr, 0.0, 1.0, "turbo")

    def depth_metric_cell(arr):
        """DPF-domain depth: unnormalized *metric distance* (closer = LOWER
        value, opposite of the reef rows' reverse depth), colored against its
        own min/max (no shared scale across images). Reversed turbo so red =
        close, matching the reef rows' orientation."""
        if arr is None:
            return ("empty",)
        return ("depth", arr, float(arr.min()), float(arr.max()), "turbo_r")

    def interleave(rgb_arrays, depth_cells):
        cells = []
        for img, d in zip(rgb_arrays, depth_cells):
            cells.append(("rgb", img))
            cells.append(d)
        return cells

    enh, enh_depth = [], []
    for p in IMAGES:
        q = DATA_NORM_DPF / p.parts[-3] / p.parts[-2] / p.name
        if q.exists():
            enh.append(np.asarray(Image.open(q).convert("RGB").resize(up_size, Image.LANCZOS)))
        else:
            print(f"[warn] missing enhanced counterpart {q} -- gray placeholder")
            enh.append(np.full((up_size[1], up_size[0], 3), 200, dtype=np.uint8))
        d = load_dpf_target_depth(p)
        enh_depth.append(d if d is None else resize_depth_array(d, up_size))

    # CFM and DPF rows are generated at their near-native canvases (224/266 px)
    # and only upsampled here for display.
    cfm_depth, dpf_depth = [], []
    for i in range(n):
        d = cached_depth_arr("cfm", i)
        cfm_depth.append(d if d is None else resize_depth_array(d, up_size))
        d = cached_depth_arr("dpf", i)
        dpf_depth.append(d if d is None else resize_depth_array(d, up_size))

    rows = [
        interleave([np.asarray(Image.open(p).convert("RGB")) for p in IMAGES],
                   [depth01_cell(cached_depth_arr("ref", i)) for i in range(n)]),
        interleave([cached_rgb("flux", i) for i in range(n)],
                   [("empty",) for _ in range(n)]),
        interleave([cached_rgb("cfm", i, resize=up_size) for i in range(n)],
                   [depth01_cell(d) for d in cfm_depth]),
        interleave(enh,
                   [depth_metric_cell(d) for d in enh_depth]),
        interleave([cached_rgb("dpf", i, resize=up_size) for i in range(n)],
                   [depth_metric_cell(d) for d in dpf_depth]),
    ]

    sep = 0.18                                    # separator gap, in cell heights
    cell_w = ECCV_TEXTWIDTH_IN / n_cols
    cell_h = cell_w / aspect
    fig_h = (len(rows) + sep) * cell_h * 1.02 + 0.18
    fig = plt.figure(figsize=(ECCV_TEXTWIDTH_IN, fig_h))
    gs = fig.add_gridspec(6, n_cols, height_ratios=[1, 1, 1, sep, 1, 1],
                          wspace=0.03, hspace=0.05)
    grid_rows = [0, 1, 2, 4, 5]                   # gridspec row per image row

    axes = np.empty((len(rows), n_cols), dtype=object)
    for r, gr in enumerate(grid_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[gr, c])
            cell = rows[r][c]
            if cell[0] == "rgb":
                ax.imshow(cell[1])
            elif cell[0] == "depth":
                _, arr, vmin, vmax, cmap = cell
                ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            # "empty": leave the cell blank (FLUX has no depth output).
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            axes[r, c] = ax
        axes[r, 0].set_ylabel(ROW_LABELS[r], fontsize=8.45)

    # Horizontal rule through the spacer row: reef domain above, DPF below.
    y = (axes[2, 0].get_position().y0 + axes[3, 0].get_position().y1) / 2
    fig.add_artist(plt.Line2D([axes[2, 0].get_position().x0,
                               axes[2, -1].get_position().x1], [y, y],
                              transform=fig.transFigure, color="0.45", lw=0.6))

    # One site header centred over each column pair (positions are final once
    # the gridspec is built; no tight_layout that would move them afterwards).
    for s, site in enumerate(SITES):
        left = axes[0, 2 * s].get_position()
        right = axes[0, 2 * s + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.012, site,
                 ha="center", va="bottom", fontsize=9.75)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"qualitative_comparison.{ext}")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'qualitative_comparison.pdf'} / .png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["cfm", "dpf", "flux", "assemble"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="CFM Euler steps (default: 1000 for --stage cfm, 50 for --stage dpf)")
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--flux-steps", type=int, default=28)
    ap.add_argument("--flux-guidance", type=float, default=4.0)
    ap.add_argument("--flux-size", type=int, default=512, help="FLUX output height")
    args = ap.parse_args()
    if args.n_steps is None:
        args.n_steps = {"cfm": 1000, "dpf": 50}.get(args.stage)

    for p in IMAGES:
        if not p.exists():
            sys.exit(f"Conditioning image not found: {p}")

    if args.stage == "assemble":
        stage_assemble(args)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ROWS_DIR.mkdir(parents=True, exist_ok=True)
        {"cfm": stage_cfm, "dpf": stage_dpf, "flux": stage_flux}[args.stage](args, device)
