"""Precompute per-deployment RGB arrays for fast node-local mmap loading.
"""
from __future__ import annotations
# DPF-Net (BenthicFlow-DPF) variant of scripts/precompute_rgb.py.
# Differences vs the base pipeline: DPF-Net-processed imagery at a 266px /
# 19-patch grid, unbounded metric depth (softplus head + SILog loss), and
# '-dpf'-suffixed data roots. Forces REEF_VARIANT=dpf so REEF resolves
# features-dpf/, depth-dpf/, rgb-dpf/, checkpoints-dpf/, data_normalized-dpf/.
import os as _os
_os.environ["BENTHICFLOW_VARIANT"] = "dpf"
_os.environ["REEF_VARIANT"] = "dpf"


import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from benthicflow import DATA_NORM_ROOT, SCRATCH_ROOT

# Source keys live next to the depth arrays. DEPTH_ROOT may point at node-local;
# the keys are also on the shared source, so resolve from SCRATCH_ROOT here.
# (REEF_VARIANT=dpf is forced above, so all roots carry the -dpf suffix.)
DEPTH_SRC = SCRATCH_ROOT / "depth-dpf"
RGB_DST = SCRATCH_ROOT / "rgb-dpf"   # precomputed per-deployment uint8 RGB arrays

NATIVE_SIZE = 266   # must match the training NATIVE_SIZE

# DPF-Net-processed images (output of dpf_net/clean_and_extract.py)
RGB_ROOT = DATA_NORM_ROOT

def _load_resize(args: tuple[str, int]) -> np.ndarray:
    """Open one JPEG and apply the deterministic Resize+CenterCrop -> uint8 HWC."""
    path, size = args
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, size, antialias=True)     # smaller edge -> size
    img = TF.center_crop(img, [size, size])
    return np.asarray(img, dtype=np.uint8)         # [S, S, 3]


def build_deployment(campaign: str, deployment: str, size: int, workers: int,
                     overwrite: bool = False) -> None:
    keys_path = DEPTH_SRC / campaign / f"{deployment}_keys.npy"
    if not keys_path.exists():
        print(f"  [skip] no keys file: {keys_path}")
        return

    out_path = RGB_DST / campaign / f"{deployment}.npy"
    keys = np.load(keys_path, allow_pickle=True)
    n = len(keys)

    # Resume-safe: skip if a complete array already exists.
    if out_path.exists() and not overwrite:
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape == (n, size, size, 3):
            print(f"  [done] {campaign}/{deployment} ({n})")
            return

    img_dir = RGB_ROOT / campaign / deployment
    # Keys may be stored as bare stems (DPF-Net) or with the .jpg suffix (reef);
    # the image files on disk always carry an extension.
    def _fname(k):
        k = str(k)
        return k if k.lower().endswith((".jpg", ".jpeg", ".png")) else f"{k}.jpg"
    paths = [(str(img_dir / _fname(k)), size) for k in keys]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".npy.tmp")
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8,
                                    shape=(n, size, size, 3))
    with Pool(workers) as pool:
        for i, im in enumerate(tqdm(pool.imap(_load_resize, paths, chunksize=16),
                                    total=n, desc=f"{campaign}/{deployment}", leave=False)):
            arr[i] = im
    arr.flush()
    del arr
    tmp_path.rename(out_path)   # atomic: only a complete file ends up at out_path
    print(f"  [built] {campaign}/{deployment} ({n})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--native-size", type=int, default=NATIVE_SIZE)
    ap.add_argument("--campaigns", nargs="*", default=None,
                    help="Optional subset of campaign names.")
    ap.add_argument("--deployments", nargs="*", default=None,
                    help="Optional subset of deployment names.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Discover deployments from the depth keys files (the canonical ordering).
    deployments: list[tuple[str, str]] = []

    if args.deployments is not None:
        # User-specified deployments: must be in the form campaign/deployment.
        for d in args.deployments:
            try:
                campaign, deployment = d.split("/", 1)
            except ValueError:
                raise ValueError(f"Deployment must be specified as campaign/deployment: {d}")
            deployments.append((campaign, deployment))
    else:
        for keys_file in sorted(DEPTH_SRC.glob("*/*_keys.npy")):
            campaign = keys_file.parent.name
            deployment = keys_file.name[: -len("_keys.npy")]
            if args.campaigns is None or campaign in args.campaigns:
                deployments.append((campaign, deployment))

    print(f"Building RGB arrays for {len(deployments)} deployments -> {RGB_DST}")
    print(f"  native size {args.native_size}, {args.workers} workers")
    for campaign, deployment in deployments:
        build_deployment(campaign, deployment, args.native_size, args.workers,
                         overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
