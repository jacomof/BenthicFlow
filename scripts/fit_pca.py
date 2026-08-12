"""Fit global PCA bases on all cached DINOv2 patch features."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from benthicflow import PCA_ROOT
from benthicflow.reef_io import iter_feature_files


def collect_patch_pool(max_samples: int | None) -> np.ndarray:
    """Concatenate features from every deployment, flattening patch grids.

    Returns [M, D] where M is total number of patches across the corpus.
    Subsamples to max_samples if given, with even allocation per deployment.
    """
    files = list(iter_feature_files())
    if not files:
        raise SystemExit("No feature files found — run extract_features.py first.")

    if max_samples is None:
        # Load everything (could be very big — ~256× more than image count)
        blocks = []
        for camp, depl, fp in files:
            z = np.load(fp)
            f = z["features"].astype(np.float32)  # [N, 16, 16, D]
            blocks.append(f.reshape(-1, f.shape[-1]))
        F = np.concatenate(blocks, axis=0)
        print(
            f"Loaded {F.shape[0]:,} patch vectors of dim {F.shape[1]} "
            f"from {len(files)} deployments."
        )
        return F

    # Subsample evenly across deployments
    per_deployment = max_samples // len(files)
    blocks = []
    rng = np.random.default_rng(0)
    for camp, depl, fp in files:
        z = np.load(fp)
        f = z["features"].astype(np.float32).reshape(-1, z["features"].shape[-1])
        if len(f) > per_deployment:
            idx = rng.choice(len(f), per_deployment, replace=False)
            f = f[idx]
        blocks.append(f)
    F = np.concatenate(blocks, axis=0)
    print(
        f"Sampled {F.shape[0]:,} patch vectors of dim {F.shape[1]} "
        f"from {len(files)} deployments (max_samples={max_samples})."
    )
    return F


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--max-samples",
        type=int,
        default=5_000_000,
        help="Subsample patches to keep PCA fit tractable. "
        "5M patches is plenty (default).",
    )
    args = ap.parse_args()

    PCA_ROOT.mkdir(exist_ok=True)
    F = collect_patch_pool(args.max_samples)

    # 16-D basis — the model uses this
    pca16 = PCA(n_components=16, random_state=0).fit(F)
    joblib.dump(pca16, PCA_ROOT / "pca16.joblib")
    cum = pca16.explained_variance_ratio_.cumsum()
    print(f"\nPCA-16  cumulative explained variance: {cum[-1]:.3f}")
    print(f"        per-component: {np.round(pca16.explained_variance_ratio_, 3)}\n")

    # 3-D basis — for visualization
    pca3 = PCA(n_components=3, random_state=0).fit(F)
    joblib.dump(pca3, PCA_ROOT / "pca3.joblib")
    print(
        f"PCA-3   cumulative explained variance: "
        f"{pca3.explained_variance_ratio_.sum():.3f}"
    )

    # Global normalization range — fit on a fresh subsample to capture tail behavior
    Z3 = pca3.transform(F)
    # Use 1st/99th percentiles instead of min/max to avoid outlier-dominated stretching
    vmin = np.percentile(Z3, 1, axis=0)
    vmax = np.percentile(Z3, 99, axis=0)
    np.savez(PCA_ROOT / "pca3_norm.npz", vmin=vmin, vmax=vmax)

    print(f"\nSaved → {PCA_ROOT}/pca16.joblib")
    print(f"Saved → {PCA_ROOT}/pca3.joblib")
    print(f"Saved → {PCA_ROOT}/pca3_norm.npz")


if __name__ == "__main__":
    main()
