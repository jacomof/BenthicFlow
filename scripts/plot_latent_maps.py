"""Patch-level latent visualization for REEF."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from benthicflow import FIG_ROOT
from benthicflow.reef_io import iter_feature_files
from benthicflow.viz import features_to_rgb, load_pca_rgb

# ----------------------------- helpers ------------------------------------


def grid_to_rgb_mosaic(grid_features: np.ndarray, pca3, vmin, vmax) -> np.ndarray:
    """Convert a [16, 16, D] patch grid to a [16, 16, 3] RGB mosaic."""
    H, W, D = grid_features.shape
    flat = grid_features.reshape(-1, D).astype(np.float32)
    rgb = features_to_rgb(flat, pca3, vmin, vmax)  # [256, 3] in [0, 1]
    return rgb.reshape(H, W, 3)


def image_to_mean_rgb(grid_features: np.ndarray, pca3, vmin, vmax) -> np.ndarray:
    """Average patch features → one RGB color per image (for timeline view)."""
    mean_feat = grid_features.reshape(-1, grid_features.shape[-1]).mean(0)
    rgb = features_to_rgb(mean_feat[None, :].astype(np.float32), pca3, vmin, vmax)
    return rgb[0]


def collect_mean_features(npz_path: Path) -> np.ndarray:
    """Load deployment features and return [N, D] of per-image mean patches."""
    z = np.load(npz_path)
    feats = z["features"].astype(np.float32)  # [N, 16, 16, D]
    return feats.reshape(feats.shape[0], -1, feats.shape[-1]).mean(1)


def fit_global_kmeans(n_clusters: int) -> KMeans:
    """Fit K-Means on per-image MEAN patch features across all deployments."""
    print(f"Fitting global K-Means (K={n_clusters}) on per-image mean features...")
    blocks = []
    for _, _, fp in iter_feature_files():
        blocks.append(collect_mean_features(fp))
    F = np.concatenate(blocks, axis=0)
    print(f"  {F.shape[0]:,} images")
    return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(F)


# ----------------------------- per-image patch mosaics --------------------


def plot_patch_mosaics(
    npz_path: Path, pca3, vmin, vmax, out: Path, n_samples: int = 24
) -> None:
    """Show n_samples random images as patch mosaics."""
    z = np.load(npz_path)
    feats = z["features"].astype(np.float32)  # [N, 16, 16, D]
    n = len(feats)
    if n == 0:
        return

    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(n_samples, n), replace=False)
    cols = 8
    rows = (len(idx) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.6, rows * 1.6), squeeze=False
    )
    for ax, i in zip(axes.ravel(), idx):
        mosaic = grid_to_rgb_mosaic(feats[i], pca3, vmin, vmax)
        ax.imshow(mosaic, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(z["keys"][i])[:14], fontsize=6)
    for ax in axes.ravel()[len(idx) :]:
        ax.axis("off")

    fig.suptitle(f"{npz_path.parent.name} — within-image patch structure", fontsize=11)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


# ----------------------------- timeline (sequence) ------------------------


def plot_deployment_timeline(
    npz_path: Path, pca3, vmin, vmax, kmeans: KMeans, out: Path
) -> None:
    z = np.load(npz_path)
    feats = z["features"].astype(np.float32)
    n = len(feats)
    if n == 0:
        return

    means = feats.reshape(n, -1, feats.shape[-1]).mean(1)  # [N, D]

    # Continuous PCA-RGB strip
    cont_rgb = features_to_rgb(means, pca3, vmin, vmax)  # [N, 3]

    # Categorical K-Means strip
    labels = kmeans.predict(means)
    cmap = plt.get_cmap("tab10")
    cat_rgb = cmap(labels)[:, :3]  # [N, 3]

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 4), gridspec_kw={"height_ratios": [1, 1]}
    )

    axes[0].imshow(
        cont_rgb[None, :, :],
        aspect="auto",
        extent=[0, n, 0, 1],
        interpolation="nearest",
    )
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_title(
        f"Continuous PCA-RGB timeline — {npz_path.parent.name} " f"({n} images)"
    )

    axes[1].imshow(
        cat_rgb[None, :, :], aspect="auto", extent=[0, n, 0, 1], interpolation="nearest"
    )
    axes[1].set_yticks([])
    axes[1].set_xlabel("Image sequence index")
    axes[1].set_title(f"K-Means biome labels (K={kmeans.n_clusters})")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


# ----------------------------- campaign overview --------------------------


def plot_campaign_grid(
    campaign: str, deployment_files: list[Path], kmeans: KMeans, out: Path
) -> None:
    """One row per deployment, showing the K-Means biome timeline."""
    n = len(deployment_files)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(2, n * 1.2)), squeeze=False)

    cmap = plt.get_cmap("tab10")
    for ax, fp in zip(axes.ravel(), deployment_files):
        means = collect_mean_features(fp)
        if len(means) == 0:
            ax.axis("off")
            continue
        labels = kmeans.predict(means)
        cat_rgb = cmap(labels)[:, :3]
        ax.imshow(
            cat_rgb[None, :, :],
            aspect="auto",
            extent=[0, len(means), 0, 1],
            interpolation="nearest",
        )
        ax.set_yticks([])
        ax.set_title(fp.parent.name, fontsize=9, loc="left")
        ax.set_xlabel("seq idx", fontsize=7)

    fig.suptitle(f"{campaign} — biome timelines (K={kmeans.n_clusters})", fontsize=12)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


# ----------------------------- driver -------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--per-deployment",
        action="store_true",
        help="Also save per-deployment timeline + patch-mosaic figures.",
    )
    ap.add_argument("--clusters", type=int, default=4)
    args = ap.parse_args()

    pca3, vmin, vmax = load_pca_rgb()
    kmeans = fit_global_kmeans(args.clusters)

    by_campaign: dict[str, list[Path]] = {}
    for campaign, _deployment, fp in iter_feature_files():
        by_campaign.setdefault(campaign, []).append(fp)

    if not by_campaign:
        sys.exit("No feature files found — run extract_features.py first.")

    print(f"Visualizing {len(by_campaign)} campaign(s): {sorted(by_campaign)}")
    for campaign, files in sorted(by_campaign.items()):
        out = FIG_ROOT / f"latent_map_{campaign}.png"
        plot_campaign_grid(campaign, files, kmeans, out)
        print(f"  saved {out}")

    if args.per_deployment:
        for campaign, _deployment, fp in iter_feature_files():
            base = FIG_ROOT / "per_deployment" / campaign / fp.parent.name
            plot_deployment_timeline(
                fp,
                pca3,
                vmin,
                vmax,
                kmeans,
                base.with_name(f"{base.name}_timeline.png"),
            )
            plot_patch_mosaics(
                fp, pca3, vmin, vmax, base.with_name(f"{base.name}_mosaics.png")
            )
        print(f"\nPer-deployment figures in {FIG_ROOT}/per_deployment/")


if __name__ == "__main__":
    main()
