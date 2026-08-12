"""Extract DINOv2 patch-token grids natively at 518x518 from raw directories.

Automatically crawls REEF_SCRATCH_ROOT/data_normalized and extracts features
for all discovered campaigns and deployments.

Outputs raw, uncompressed .npy files to a flat hierarchy to prevent SLURM
CPU-RAM OOM kills during multiprocessing DataLoader reads.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(
    os.environ.get(
        "REEF_SCRATCH_ROOT",
        PROJECT_ROOT / "scratch",
    )
)

DATA_ROOT = PROJECT_ROOT / "data_normalized"
FEAT_ROOT = SCRATCH_ROOT / "features"
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"FEAT_ROOT: {FEAT_ROOT}")

NATIVE_SIZE = 518
PATCH_SIZE = 14
GRID = NATIVE_SIZE // PATCH_SIZE  # 37


def flat_feature_paths(campaign: str, deployment: str) -> tuple[Path, Path]:
    """Returns paths for the raw feature array and the corresponding keys."""
    feat_npy = FEAT_ROOT / campaign / f"{deployment}.npy"
    keys_npy = FEAT_ROOT / campaign / f"{deployment}_keys.npy"
    return feat_npy, keys_npy


def load_dinov2(variant: str = "dinov2_vitb14"):
    print(f"Loading {variant} on {DEVICE}...")
    model = torch.hub.load("facebookresearch/dinov2", variant)
    return model.eval().to(DEVICE)


class DirectoryImagesNative(Dataset):
    """Yields preprocessed images directly from a list of Path objects."""

    def __init__(self, image_paths: list[Path]):
        self.image_paths = image_paths
        self.tfm = transforms.Compose(
            [
                transforms.Resize(NATIVE_SIZE, antialias=True),
                transforms.CenterCrop(NATIVE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        try:
            img = Image.open(path).convert("RGB")
            return self.tfm(img), i, True
        except Exception as e:
            print(f"Failed to load image: {path} ({e})")
            return torch.zeros(3, NATIVE_SIZE, NATIVE_SIZE), i, False


@torch.no_grad()
def extract_deployment(
    model,
    image_paths: list[Path],
    feat_path: Path,
    keys_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    if not image_paths:
        return

    ds = DirectoryImagesNative(image_paths)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    feats: np.ndarray | None = None
    valid = np.zeros(len(ds), dtype=bool)

    for x, idx, ok in tqdm(dl, desc=feat_path.stem, leave=False):
        x = x.to(DEVICE, non_blocking=True)

        out = model.forward_features(x)
        patch = out["x_norm_patchtokens"]

        B, N, D = patch.shape
        assert N == GRID * GRID, f"expected {GRID * GRID} patches, got {N}"

        patch = patch.view(B, GRID, GRID, D).float().cpu().numpy()

        if feats is None:
            feats = np.zeros((len(ds), GRID, GRID, D), dtype=np.float32)

        idx_np = idx.numpy()
        feats[idx_np] = patch
        valid[idx_np] = ok.numpy()

    valid_feats = feats[valid]
    # We use the filename (e.g., 'PR_001.jpg') as the definitive key for alignment later
    valid_keys = np.array(
        [p.name for i, p in enumerate(image_paths) if valid[i]], dtype=str
    )

    feat_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as raw, uncompressed binary arrays
    np.save(feat_path, valid_feats)
    np.save(keys_path, valid_keys)

    print(f"saved {feat_path}  shape={valid_feats.shape}")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dinov2_vitb14")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"FEAT_ROOT: {FEAT_ROOT}")

    deployments = crawl_data_directory()
    if not deployments:
        sys.exit(f"No images found in {DATA_ROOT}")

    print(f"\nDiscovered {len(deployments)} deployments.")

    model = load_dinov2(args.variant)

    for (campaign, deployment), image_paths in deployments.items():
        feat_path, keys_path = flat_feature_paths(campaign, deployment)

        skip = False
        if feat_path.exists() and keys_path.exists() and not args.overwrite:
            try:
                _ = np.load(feat_path, mmap_mode="r")
                _ = np.load(keys_path, allow_pickle=True)
                print(f"skipping existing {feat_path.name}")
                skip = True
            except Exception as e:
                print(f"found corrupted {feat_path.name} ({e}), re-extracting")

        if skip:
            continue

        print(f"\nExtracting: {campaign} / {deployment} / N={len(image_paths)}")
        extract_deployment(
            model,
            image_paths,
            feat_path,
            keys_path,
            batch_size=args.batch,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
