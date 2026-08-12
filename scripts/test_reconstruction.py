# scripts/test_reconstruction.py
"""Evaluate RAE reconstruction fidelity with FID and KID."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benthicflow import FIG_ROOT
from scripts.test_generation import (
    DEVICE,
    CSVSplitDataset,
    build_metrics,
    format_metrics,
    print_campaign_summary,
    select_campaigns,
    update_results_json,
)
from scripts.train_cfm_cfg import load_rae_frozen

RESULTS_JSON = FIG_ROOT / "rae" / "test_reconstruction" / "eval_results.json"


def reconstruct(rae, dino, depth):
    """RAE encode->decode reconstruction. Returns RGB in [0, 1], [B, 3, H, W].

    dino:  [B, gh, gw, 768] frozen DINOv2 RGB features (the RAE's rgb_latent).
    depth: [B, 1, 14*gh, 14*gw] in [0, 1].
    Matches the RAE forward used at train time (train_cfm_cfg.py).
    """
    rgbd, _ = rae(
        dino.to(DEVICE, non_blocking=True), depth.to(DEVICE, non_blocking=True)
    )
    return rgbd[:, :3].clamp(0, 1)


def save_reconstruction_debug(tag, orig_rgb, recon_rgb, n_samples):
    """Save [original | reconstruction] RGB pairs for eyeballing."""
    n = min(n_samples, orig_rgb.shape[0], recon_rgb.shape[0])
    out_dir = FIG_ROOT / "rae" / "test_reconstruction" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5), squeeze=False)
    for j in range(n):
        axes[0, j].imshow(orig_rgb[j].permute(1, 2, 0).cpu().numpy())
        axes[1, j].imshow(recon_rgb[j].permute(1, 2, 0).cpu().numpy())
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("RAE recon", fontsize=10)
    fig.suptitle(f"RAE reconstruction — {tag}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "reconstruction_samples.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] saved debug samples to {out_path}")


def evaluate_reconstruction(
    rae, metrics, args, campaign=None, tag="global", debug_samples=0
):
    """FID/KID between RAE reconstructions and originals over one (sub)set.

    Returns a JSON-friendly score dict, or None if the set is too small.
    """
    ds = CSVSplitDataset(split="test", load_rgb=True, campaign=campaign)
    if len(ds) < 2:
        print(f"[skip] {tag}: only {len(ds)} images (need >= 2)")
        return None
    # shuffle=False: FID/KID are order-independent and the whole (sub)set is
    # scanned, so sequential reads keep the big RGB mmaps cheap.
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    metrics.reset()
    n = 0
    data_wait = gpu_time = 0.0
    debug_saved = False
    t_prev = time.perf_counter()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"{tag} recon")
        for rgb, depth, dino in pbar:
            t0 = time.perf_counter()
            data_wait += t0 - t_prev

            orig_rgb = rgb.to(DEVICE, non_blocking=True)
            recon_rgb = reconstruct(rae, dino, depth)
            metrics.update(orig_rgb, real=True)
            metrics.update(recon_rgb, real=False)

            if debug_samples > 0 and not debug_saved:
                save_reconstruction_debug(tag, orig_rgb, recon_rgb, debug_samples)
                debug_saved = True

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_prev = time.perf_counter()
            gpu_time += t_prev - t0
            n += rgb.shape[0]
            pbar.set_postfix(data=f"{data_wait:.1f}s", gpu=f"{gpu_time:.1f}s")

    # One reconstruction per real image, so n_real == n_fake == n.
    score = metrics.compute(n, n)
    print(
        f"Reconstruction [{tag}]: {format_metrics(score)}  (n={n})  "
        f"-- dataloader wait {data_wait:.1f}s, gpu {gpu_time:.1f}s"
    )
    return score


def main(args):
    metrics = build_metrics(args)
    kid_note = "off" if args.skip_kid else f"on (subset_size={args.kid_subset_size})"
    print(f"Feature extractor: {metrics.extractor_label}, KID {kid_note}")

    rae = load_rae_frozen(args.rae_ckpt, DEVICE)

    sections = {}
    if args.mode in ("global", "both"):
        score = evaluate_reconstruction(
            rae,
            metrics,
            args,
            campaign=None,
            tag="global",
            debug_samples=args.debug_samples,
        )
        if score is not None:
            sections["global"] = score

    if args.mode in ("per_campaign", "both"):
        campaigns = select_campaigns(args)
        print(
            f"\nPer-campaign reconstruction over {len(campaigns)} campaigns: {campaigns}"
        )
        per_campaign = {}
        for campaign in campaigns:
            score = evaluate_reconstruction(
                rae,
                metrics,
                args,
                campaign=campaign,
                tag=campaign,
                debug_samples=args.debug_samples,
            )
            if score is not None:
                per_campaign[campaign] = score
        print_campaign_summary("Per-campaign RAE reconstruction FID/KID", per_campaign)
        sections["per_campaign"] = per_campaign

    config = {
        "feature_extractor": metrics.extractor_label,
        "kid_subset_size": args.kid_subset_size,
        "skip_kid": args.skip_kid,
        "batch_size": args.batch_size,
        "rae_ckpt": args.rae_ckpt,
    }
    run_key = f"{Path(args.rae_ckpt).stem}_{metrics.extractor_label}"
    update_results_json(RESULTS_JSON, run_key, config, sections)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Evaluate RAE encode->decode reconstruction fidelity (FID/KID) "
        "vs the original test images, globally and per campaign."
    )

    argparser.add_argument(
        "--rae_ckpt", type=str, required=True, help="Path to the RAE checkpoint."
    )
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    argparser.add_argument(
        "--n_workers", type=int, default=4, help="Number of DataLoader workers."
    )
    argparser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["global", "per_campaign", "both"],
        help="global: FID/KID over the whole test split; per_campaign: "
        "one FID/KID per campaign; both.",
    )
    argparser.add_argument(
        "--feature_extractor",
        type=str,
        default="dino",
        choices=["dino", "inception"],
        help="Feature extractor for FID/KID: DINOv2 CLS token (dino) or InceptionV3.",
    )
    argparser.add_argument(
        "--skip_kid",
        action="store_true",
        help="Only compute FID (skip the complementary KID metric).",
    )
    argparser.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        help="KID MMD subset size per resample; clamped down to the available "
        "sample count per comparison.",
    )
    argparser.add_argument(
        "--campaigns",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated campaigns to evaluate (per_campaign mode). "
        "Default: all campaigns in the test split.",
    )
    argparser.add_argument(
        "--debug_samples",
        type=int,
        default=4,
        help="Per tag, save this many [original | reconstruction] pairs to "
        "figs/rae/test_reconstruction/<tag>/. 0 disables.",
    )

    args = argparser.parse_args()
    main(args)
