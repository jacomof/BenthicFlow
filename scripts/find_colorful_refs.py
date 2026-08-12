# scripts/find_colorful_refs.py
"""Rank reference images by colourfulness and report the top-K.
Scans the precomputed 518x518 RGB arrays (SCRATCH_ROOT/rgb) for the requested
campaigns and scores every frame with the Hasler-Suesstrunk colourfulness
metric (Hasler & Suesstrunk, "Measuring colourfulness in natural images",
SPIE 2003): with opponent channels rg = R-G and yb = (R+G)/2 - B,
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benthicflow import FIG_ROOT, PROJECT_ROOT
from scripts.test_panorama import RGB_SRC, FEAT_SRC

DATA_NORM = PROJECT_ROOT / "data_normalized"
OUT_DIR = FIG_ROOT / "colorful_refs"


def colorfulness(img, mu_weight=0.1):
    """Hasler-Suesstrunk metric for one [H, W, 3] float image in [0, 1].

    mu_weight scales the colour-CAST term (original paper: 0.3). A uniform
    saturated cast (open blue water) maxes that term with zero reef content,
    so for "colourful reef" ranking the chroma-SPREAD term should dominate;
    default 0.1 (0 = pure spread)."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + mu_weight * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def scan_campaign(campaign, frame_stride, pixel_stride, mu_weight,
                  white_thr=0.9, black_thr=0.1, min_std=0.03):
    """Yield (score, brightness, campaign, deployment, idx) per scanned frame.

    Frames failing the exposure/flatness gates (same thresholds as
    sample_conditioning) are skipped: near-white/black frames and structureless
    water columns should never rank, however saturated their cast."""
    deps = sorted(p.stem for p in (RGB_SRC / campaign).glob("*.npy"))
    if not deps:
        print(f"[warn] no RGB arrays under {RGB_SRC / campaign}")
    for dep in deps:
        arr = np.load(RGB_SRC / campaign / f"{dep}.npy", mmap_mode="r")
        n = len(arr)
        t0 = time.time()
        kept = 0
        for i in range(0, n, frame_stride):
            img = np.asarray(arr[i, ::pixel_stride, ::pixel_stride],
                             dtype=np.float32) / 255.0
            bright, lum_std = float(img.mean()), float(img.mean(-1).std())
            if bright > white_thr or bright < black_thr or lum_std < min_std:
                continue
            kept += 1
            yield colorfulness(img, mu_weight), bright, campaign, dep, i
        print(f"  {campaign}/{dep}: kept {kept}/{len(range(0, n, frame_stride))} "
              f"scanned frames ({n} total) in {time.time() - t0:.0f}s")


def main(args):
    entries = []          # (score, brightness, campaign, dep, idx)
    campaigns = args.campaigns or sorted(
        p.name for p in RGB_SRC.iterdir() if p.name.startswith(args.prefix))
    print(f"Scanning campaigns: {campaigns} "
          f"(frame stride {args.frame_stride}, pixel stride {args.pixel_stride})")
    for campaign in campaigns:
        entries.extend(scan_campaign(campaign, args.frame_stride,
                                     args.pixel_stride, args.mu_weight))

    entries.sort(key=lambda e: -e[0])
    # Greedy selection with a minimum frame separation per deployment:
    # consecutive AUV frames overlap heavily, so without this the top-K is
    # bursts of near-duplicates of a few colourful spots.
    top, picked = [], {}
    for e in entries:
        _, _, campaign, dep, idx = e
        near = picked.setdefault((campaign, dep), [])
        if any(abs(idx - j) < args.min_sep for j in near):
            continue
        near.append(idx)
        top.append(e)
        if len(top) == args.top:
            break
    print(f"\nScanned {len(entries)} frames; "
          f"score range [{entries[-1][0]:.3f}, {entries[0][0]:.3f}]; "
          f"picked {len(top)} with min separation {args.min_sep}")

    # resolve filenames via the keys index (row order shared with RGB arrays)
    keys_cache = {}
    rows = []
    for rank, (score, bright, campaign, dep, idx) in enumerate(top, 1):
        kf = (campaign, dep)
        if kf not in keys_cache:
            keys_cache[kf] = np.load(FEAT_SRC / campaign / f"{dep}_keys.npy")
        name = str(keys_cache[kf][idx])
        rows.append(dict(rank=rank, score=f"{score:.4f}", brightness=f"{bright:.3f}",
                         campaign=campaign, deployment=dep, index=idx, filename=name,
                         path=str(DATA_NORM / campaign / dep / name)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.prefix.lower()
    csv_path = OUT_DIR / f"{tag}_top{args.top}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved {csv_path}")

    # contact sheet: rank + score on each thumbnail
    ncol = 10
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.6))
    axes = np.atleast_2d(axes)
    for a in axes.flat:
        a.set_axis_off()
    for k, (score, bright, campaign, dep, idx) in enumerate(top):
        arr = np.load(RGB_SRC / campaign / f"{dep}.npy", mmap_mode="r")
        ax = axes.flat[k]
        ax.imshow(np.asarray(arr[idx, ::4, ::4]))
        ax.set_title(f"#{k + 1}  M={score:.2f}\n{campaign.replace('ScottReef', 'SR')}[{idx}]",
                     fontsize=6)
    fig.tight_layout()
    png_path = OUT_DIR / f"{tag}_top{args.top}.png"
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", default="ScottReef",
                    help="scan all campaigns whose name starts with this")
    ap.add_argument("--campaigns", nargs="*", default=None,
                    help="explicit campaign list (overrides --prefix)")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="scan every k-th frame (10 = fast preview pass)")
    ap.add_argument("--pixel-stride", type=int, default=4,
                    help="subsample pixels for the metric (4 -> 130x130 samples)")
    ap.add_argument("--mu-weight", type=float, default=0.1,
                    help="weight of the colour-cast term (Hasler default 0.3; "
                         "lower favours multi-coloured reef over uniform casts)")
    ap.add_argument("--min-sep", type=int, default=50,
                    help="minimum frame-index separation between picks from "
                         "the same deployment (AUV frames overlap heavily)")
    ap.add_argument("--tag", default=None, help="output filename tag (default: prefix)")
    main(ap.parse_args())
