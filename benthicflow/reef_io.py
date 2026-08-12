"""Filesystem layout, manifest loading, feature caching."""

import glob
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from . import DATA_ROOT, FEAT_ROOT, DATA_NORM_ROOT, DEPTH_ROOT


def iter_deployments(campaigns: Optional[Sequence[str]] = None) -> Iterator[tuple[str, str, Path, Path]]:
    """Yield (campaign, deployment, manifest_path, image_dir) for each deployment."""
    print(f"iter_deployments: campaigns={campaigns}")
    if campaigns is None:
        pattern = str(DATA_ROOT / "*" / "*" / "manifest.csv")
    else:
        pattern = None  

    print(f"iter_deployments: pattern={pattern}")
    if pattern is not None:
        manifests = sorted(glob.glob(pattern))
        print(f"Found {len(manifests)} deployments matching pattern: {pattern}")
    else:
        manifests = []
        for c in campaigns:
            print("Collecting manifests for campaign:", c)
            manifests += sorted(glob.glob(str(DATA_ROOT / c / "*" / "manifest.csv")))

    for m in manifests:
        mp = Path(m)
        yield mp.parts[-3], mp.parts[-2], mp, mp.parent / "images"


def load_manifest(manifest_path: Path, image_dir: Path) -> pd.DataFrame:
    """Load a manifest"""
    df = pd.read_csv(manifest_path)
    df["img_path"] = df["key"].apply(lambda k: image_dir / f"{k}.jpg")
    df = df[df["img_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    return df


def feature_path(campaign: str, deployment: str) -> Path:
    return FEAT_ROOT / campaign / deployment / "feats.npz"


def normalized_image_path(campaign: str, deployment: str, key: str) -> Path:
    """Path to one exposure-normalized JPEG."""
    return DATA_NORM_ROOT / campaign / deployment / f"{key}.jpg"


def save_features(path: Path, *, features: np.ndarray, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.astype(np.float16),    
        keys=df["key"].astype(str).to_numpy(),
    )


def iter_feature_files() -> Iterator[tuple[str, str, Path]]:
    """Yield (campaign, deployment, npz_path) for every cached feature file."""
    for f in sorted(glob.glob(str(FEAT_ROOT / "*" / "*" / "feats.npz"))):
        fp = Path(f)
        yield fp.parts[-3], fp.parts[-2], fp


def load_all_features() -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    """Concatenate every cached feature matrix; return (F, sources)."""
    blocks, sources = [], []
    for camp, depl, fp in iter_feature_files():
        z = np.load(fp)
        blocks.append(z["features"])
        sources.append((camp, depl, z["features"].shape[0]))
    if not blocks:
        raise SystemExit("No feature files found — run extract_features.py first.")
    return np.concatenate(blocks, axis=0), sources


def depth_paths(campaign: str, deployment: str, label: str=None) -> tuple[Path, Path]:
    """Paths to a deployment's uncompressed Depth-Anything-V2 maps and keys."""
    
    if label:
        base = Path(str(DEPTH_ROOT)+label) / campaign 
    else:
        base = DEPTH_ROOT / campaign
        
    depths = base / f"{deployment}.npy"
    keys = base / f"{deployment}_keys.npy"

    return depths, keys

def depth_path(campaign: str, deployment: str, label: str=None) -> Path:
    """Base directory for one deployment's depth cache."""
    if label:
        return DEPTH_ROOT / campaign / f"{deployment}_{label}.npy"
    else:
        return DEPTH_ROOT / campaign / f"{deployment}.npy"

def flat_depth_paths(campaign: str, deployment: str, label: str | None = None) -> tuple[Path, Path]:
    """Returns paths for the raw depth array and the corresponding keys."""
    suffix = f"_{label}" if label else ""
    depth_npy = DEPTH_ROOT / campaign / f"{deployment}{suffix}.npy"
    keys_npy = DEPTH_ROOT / campaign / f"{deployment}{suffix}_keys.npy"
    return depth_npy, keys_npy

def save_depths(path: Path, *, depths: np.ndarray, keys: np.ndarray) -> None:
    """Cache normalized depth maps into memory-mappable uncompressed formats.

    Accepts either:
      - a deployment directory: .../<campaign>/<deployment>
      - or an explicit file path: .../<campaign>/<deployment>/depths.npy
    """
    path = Path(path)

    if path.suffix == ".npy":
        base = path.parent
    else:
        base = path

    base.mkdir(parents=True, exist_ok=True)

    np.save(base / "depths.npy", depths.astype(np.float16))
    np.save(base / "depths_keys.npy", keys.astype(str))