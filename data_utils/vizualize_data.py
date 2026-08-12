"""
Sanity-check the pulled Squidle+ data:
  1. Tabulate images per campaign / deployment
  2. Plot AUV trajectories (lat/lon, colored by depth)
  3. Show altitude distribution (should be ~2m for Sirius)
  4. Spot-check a few thumbnails per campaign

Run after pull_dreamsea_data.py.
"""

import glob, random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data")


def load_all_manifests():
    """Concatenate every manifest.csv into one DataFrame with campaign/deployment cols."""
    rows = []
    for m in glob.glob(str(ROOT / "*" / "*" / "manifest.csv")):
        d = pd.read_csv(m)
        parts = Path(m).parts
        d["campaign"]   = parts[-3]
        d["deployment"] = parts[-2]
        d["img_path"]   = str(Path(m).parent / "images") + "/" + d["key"] + ".jpg"
        rows.append(d)
    if not rows:
        raise SystemExit("No manifests found under data/. Run pull_dreamsea_data.py first.")
    return pd.concat(rows, ignore_index=True)


def summary_table(df):
    print("\n=== Per-campaign summary ===")
    g = df.groupby("campaign").agg(
        n_deployments=("deployment", "nunique"),
        n_images=("media_id", "count"),
        mean_alt=("alt", "mean"),
        mean_depth=("depth", "mean"),
        lat_range=("lat", lambda s: f"{s.min():.3f}…{s.max():.3f}"),
        lon_range=("lon", lambda s: f"{s.min():.3f}…{s.max():.3f}"),
    )
    print(g.to_string())

    print("\n=== Per-deployment image counts (top 20) ===")
    g2 = df.groupby(["campaign", "deployment"]).size().sort_values(ascending=False).head(20)
    print(g2.to_string())


def plot_trajectories(df, out="trajectories.png"):
    """One subplot per campaign: lat/lon scatter, colored by depth."""
    camps = sorted(df["campaign"].unique())
    n = len(camps)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax, camp in zip(axes.ravel(), camps):
        g = df[df["campaign"] == camp].dropna(subset=["lat", "lon"])
        if g.empty:
            ax.set_title(f"{camp} (no pose)"); ax.axis("off"); continue
        sc = ax.scatter(g["lon"], g["lat"], c=g["depth"], s=0.5,
                        cmap="viridis", rasterized=True)
        ax.set_title(f"{camp}\n{len(g):,} images, {g['deployment'].nunique()} deployments")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_aspect("equal", adjustable="datalim")
        plt.colorbar(sc, ax=ax, label="depth (m)", shrink=0.8)

    # Hide unused axes
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close()


def plot_altitude_hist(df, out="altitudes.png"):
    """Sirius should be at ~2 m altitude; outliers indicate bad poses."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for camp, g in df.groupby("campaign"):
        ax.hist(g["alt"].dropna(), bins=80, alpha=0.4, label=camp, range=(0, 6))
    ax.set_xlabel("altitude above seafloor (m)")
    ax.set_ylabel("count")
    ax.set_title("Altitude distribution (Sirius nominally ~2 m)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def thumbnail_grid(df, per_campaign=8, out="thumbnails.png"):
    """Random sample of images per campaign for visual eyeballing."""
    camps = sorted(df["campaign"].unique())
    fig, axes = plt.subplots(len(camps), per_campaign,
                             figsize=(2 * per_campaign, 2 * len(camps)),
                             squeeze=False)
    for r, camp in enumerate(camps):
        g = df[df["campaign"] == camp]
        sample = g.sample(min(per_campaign, len(g)), random_state=0)
        for c, (_, row) in enumerate(sample.iterrows()):
            ax = axes[r, c]
            try:
                img = Image.open(row["img_path"])
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(camp, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def main():
    df = load_all_manifests()
    summary_table(df)
    plot_trajectories(df)
    plot_altitude_hist(df)
    thumbnail_grid(df)


if __name__ == "__main__":
    main()