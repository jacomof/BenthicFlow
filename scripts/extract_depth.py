"""Extract Depth-Anything-V2 depth maps natively at 518x518 from raw directories.

Automatically crawls REEF_SCRATCH_ROOT/data_normalized and extracts depth maps 
for all discovered campaigns and deployments.

Outputs raw, uncompressed .npy files to a flat hierarchy to prevent SLURM 
CPU-RAM OOM kills during multiprocessing DataLoader reads.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(os.environ.get(
    "REEF_SCRATCH_ROOT",
    PROJECT_ROOT / "scratch",
))

DATA_ROOT = Path("PROJECT_ROOT") / "data_normalized"
DEPTH_ROOT = SCRATCH_ROOT / "depth"

NATIVE_SIZE = 518


# ============================================================================
# Model loading
# ============================================================================

VARIANT_REPOS = {
    "Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Large-hf",
}

VARIANT_REPOS_METRIC = {
    "Small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Base":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def load_model(variant: str = "Large", metric: bool = False, processed: bool = False):
    """Load Depth-Anything-V2 via HuggingFace pipeline."""
    from transformers import pipeline

    repo = VARIANT_REPOS_METRIC[variant] if metric else VARIANT_REPOS[variant]
    print(f"Loading {repo} on {DEVICE}...")

    # More robust for transformers.pipeline than passing "cuda" directly.
    device_id = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=repo, device=device_id)

    if processed:
        print("Using post-processed 'depth' output from the pipeline.")
    else:
        print("Using raw 'predicted_depth' output from the pipeline.")

    def run(pil_images):
        results = pipe(pil_images)

        if isinstance(results, dict):
            results = [results]

        if processed:
            return [np.asarray(r["depth"], dtype=np.float32) for r in results]

        return [np.asarray(r["predicted_depth"], dtype=np.float32) for r in results]

    return run


# ============================================================================
# Path handling
# ============================================================================

def flat_depth_paths(campaign: str, deployment: str, label: str | None = None) -> tuple[Path, Path]:
    """Returns paths for the raw depth array and the corresponding keys."""
    suffix = f"_{label}" if label else ""
    depth_npy = DEPTH_ROOT / campaign / f"{deployment}{suffix}.npy"
    keys_npy = DEPTH_ROOT / campaign / f"{deployment}{suffix}_keys.npy"
    return depth_npy, keys_npy


def crawl_data_directory() -> dict[tuple[str, str], list[Path]]:
    """Crawls DATA_ROOT and returns a dictionary mapping (campaign, deployment) to image paths."""
    deployments = {}
    if not DATA_ROOT.exists():
        sys.exit(f"Data root not found: {DATA_ROOT}")

    for campaign_dir in DATA_ROOT.iterdir():
        if not campaign_dir.is_dir():
            continue
        campaign = campaign_dir.name
        
        for dep_dir in campaign_dir.iterdir():
            if not dep_dir.is_dir():
                continue
            deployment = dep_dir.name
            
            images = list(dep_dir.glob("*.jpg")) + list(dep_dir.glob("*.png"))
            if images:
                deployments[(campaign, deployment)] = images
                
    return deployments


# ============================================================================
# Preprocessing
# ============================================================================

def resize_shorter_then_center_crop(img: Image.Image, size: int = NATIVE_SIZE) -> Image.Image:
    """Resize shorter side to `size`, then center-crop to size x size.
    Must match RGB preprocessing used elsewhere to prevent drift.
    """
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.CenterCrop(size),
    ])(img)


# ============================================================================
# Dataset
# ============================================================================

class DirectoryRGBDataset(Dataset):
    """Yields preprocessed 518x518 PIL images directly from a list of Paths."""

    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        key = path.name

        try:
            img = Image.open(path).convert("RGB")
            img = resize_shorter_then_center_crop(img, NATIVE_SIZE)
            return img, key, True
        except Exception as e:
            print(f"Failed to load image: {path} ({e})")
            return Image.new("RGB", (NATIVE_SIZE, NATIVE_SIZE)), key, False


def collate(batch):
    imgs, keys, oks = zip(*batch)
    return list(imgs), list(keys), list(oks)


# ============================================================================
# Depth post-processing
# ============================================================================

def resize_depth_to_native(depth: np.ndarray) -> np.ndarray:
    """Resize a HxW depth array to NATIVE_SIZE x NATIVE_SIZE without cv2."""
    d = np.asarray(depth, dtype=np.float32)

    if d.shape == (NATIVE_SIZE, NATIVE_SIZE):
        return d.astype(np.float32)

    if d.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {d.shape}")

    t = torch.from_numpy(np.ascontiguousarray(d))[None, None]  # 1 x 1 x H x W

    t = F.interpolate(
        t,
        size=(NATIVE_SIZE, NATIVE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    return t[0, 0].numpy().astype(np.float32)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization for non-metric depth."""
    d = np.asarray(depth, dtype=np.float32)

    lo, hi = float(d.min()), float(d.max())

    if hi - lo < 1e-6:
        d = np.full_like(d, 0.5, dtype=np.float32)
    else:
        d = (d - lo) / (hi - lo)

    d = resize_depth_to_native(d)
    return d.astype(np.float32)


# ============================================================================
# Processing
# ============================================================================

def process_deployment(
    run,
    image_paths: list[Path],
    depth_path: Path,
    keys_path: Path,
    batch_size: int,
    num_workers: int,
    metric: bool = False,
) -> None:
    if not image_paths:
        print(f"empty image list for {depth_path}")
        return

    ds = DirectoryRGBDataset(image_paths)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate,
    )

    all_depths = []
    all_keys = []

    pbar = tqdm(dl, desc=depth_path.stem, leave=False, dynamic_ncols=True)

    for imgs, keys, oks in pbar:
        good = [(im, k) for im, k, ok in zip(imgs, keys, oks) if ok]

        if not good:
            continue

        good_imgs, good_keys = zip(*good)

        try:
            depths = run(list(good_imgs))
        except Exception as e:
            print(f"\nBatch failed for {depth_path.stem}: {e}")
            continue

        for d, k in zip(depths, good_keys):
            if metric:
                if d.shape != (NATIVE_SIZE, NATIVE_SIZE):
                    d = resize_depth_to_native(d)
                all_depths.append(d.astype(np.float32))
            else:
                all_depths.append(normalize_depth(d))

            all_keys.append(str(k))

    if not all_depths:
        print(f"all images failed for {depth_path}")
        return

    depth_path.parent.mkdir(parents=True, exist_ok=True)

    depths_arr = np.stack(all_depths, axis=0)
    keys_arr = np.array(all_keys, dtype=str)

    # Save as raw, uncompressed binary arrays
    np.save(depth_path, depths_arr)
    np.save(keys_path, keys_arr)

    print(f"saved {depth_path} shape={depths_arr.shape}")


# ============================================================================
# Driver
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    ap.add_argument(
        "--variant",
        default="Large",
        choices=list(VARIANT_REPOS.keys()),
    )
    ap.add_argument(
        "--metric",
        action="store_true",
        help="Use the metric outdoor Depth-Anything-V2 variant.",
    )
    ap.add_argument(
        "--processed",
        action="store_true",
        help="Use the post-processed 'depth' output instead of raw 'predicted_depth'.",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if args.metric and args.processed:
        sys.exit(
            "The metric variant does not expose the same processed output. "
            "Use either --metric or --processed, not both."
        )

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"SCRATCH_ROOT: {SCRATCH_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"DEPTH_ROOT: {DEPTH_ROOT}")

    DEPTH_ROOT.mkdir(parents=True, exist_ok=True)

    deployments = crawl_data_directory()
    if not deployments:
        sys.exit(f"No images found in {DATA_ROOT}")

    print(f"\nDiscovered {len(deployments)} deployments.")

    run = load_model(
        args.variant,
        metric=args.metric,
        processed=args.processed,
    )

    if args.metric:
        label = "metric"
    elif args.processed:
        label = "processed"
    else:
        label = None

    print(
        f"\nProcessing device={DEVICE}, variant={args.variant}, "
        f"metric={args.metric}, processed={args.processed}, size={NATIVE_SIZE}x{NATIVE_SIZE}\n"
    )

    for (campaign, deployment), image_paths in deployments.items():
        depth_path, keys_path = flat_depth_paths(
            campaign,
            deployment,
            label=label,
        )

        if depth_path.exists() and keys_path.exists() and not args.overwrite:
            print(f"skip cached {depth_path.name}")
            continue

        print(
            f"\nExtracting depth: {campaign} / {deployment} / N={len(image_paths)}"
        )

        process_deployment(
            run,
            image_paths,
            depth_path,
            keys_path,
            batch_size=args.batch,
            num_workers=args.workers,
            metric=args.metric,
        )


if __name__ == "__main__":
    main()