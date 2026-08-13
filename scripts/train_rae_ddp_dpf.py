"""Train the RAE for RGBD underwater imagery with Grid-Snapped Cropping.
Pipeline per step:
1. Load (rgb, depth, dino_features) perfectly aligned batch from disk via CSV.
2. Apply per-token LayerNorm to offline DINO features.
3. RAE forward: depth encoder + concat fusion + noise injection + conv decoder.
"""

from __future__ import annotations

# DPF-Net (BenthicFlow-DPF) variant of scripts/train_rae_ddp.py.
# Differences vs the base pipeline: DPF-Net-processed imagery at a 266px /
# 19-patch grid, unbounded metric depth (softplus head + SILog loss), and
# '-dpf'-suffixed data roots. Forces REEF_VARIANT=dpf so REEF resolves
# features-dpf/, depth-dpf/, rgb-dpf/, checkpoints-dpf/, data_normalized-dpf/.
import os as _os

_os.environ["BENTHICFLOW_VARIANT"] = "dpf"
_os.environ["REEF_VARIANT"] = "dpf"


import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from benthicflow import (
    CKPT_ROOT,
    DATA_NORM_ROOT,
    DEPTH_ROOT,
    FEAT_ROOT,
    FIG_ROOT,
    PROJECT_ROOT,
    SCRATCH_ROOT,
)
from models.losses import (
    DinoDiscriminator,
    RAELoss,
    multiscale_gradient_loss,
    silog_loss,
)
from models.rae import (
    RAE,
    ConvDecoderConfig,
    DepthEncoderConfig,
    RAEConfig,
    TokenLayerNorm,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing them for compatibility
# SCRATCH_ROOT = Path(os.environ.get(
#     "REEF_SCRATCH_ROOT",
#     PROJECT_ROOT / "scratch",
# ))
# DEPTH_ROOT = SCRATCH_ROOT / "depth"
# FEAT_ROOT = SCRATCH_ROOT / "features"

# DPF-Net geometry: DINOv2 features are extracted at 266px (patch 14 -> 19x19
# grid). Depth is DPF-Net's 256px output, resized to NATIVE_SIZE in the loader
# so RGB, depth, and features all share the same 266px / 19-patch grid.
NATIVE_SIZE = 266  # 19 patches * 14
TRAIN_SIZE = 224  # 16 patches * 14
PATCH_SIZE = 14
NATIVE_GRID = NATIVE_SIZE // PATCH_SIZE  # 19
TRAIN_GRID = TRAIN_SIZE // PATCH_SIZE  # 16

print(
    f"Training RAE on device {DEVICE} with native size {NATIVE_SIZE}, train size {TRAIN_SIZE}, patch size {PATCH_SIZE}, native grid {NATIVE_GRID}, train grid {TRAIN_GRID}"
)
print(f"Project root: {PROJECT_ROOT}")
print(f"Scratch root: {SCRATCH_ROOT}")
print(f"Depth root: {DEPTH_ROOT}")
print(f"Feature root: {FEAT_ROOT}")
print(f"Checkpoint root: {CKPT_ROOT}")


def get_model(model: nn.Module) -> nn.Module:
    """Helper to unwrap DDP models safely."""
    return model.module if isinstance(model, DDP) else model


# ===========================================================================
# CSV-Driven Grid-Snapped Dataset
# ===========================================================================


class CSVSplitDataset(Dataset):
    """Loads RGB, Depth, and pre-extracted DINO features from a flat hierarchy."""

    def __init__(
        self,
        split: str,
        campaigns: list[str] | None = None,
        horizontal_flip: bool = True,
    ):
        self.split = split
        self.train = split == "train"
        self.horizontal_flip = horizontal_flip and self.train

        self.records: list[tuple[str, str, str, int]] = []
        self._depth_mmap_paths: dict[tuple[str, str], Path] = {}
        self._dino_mmap_paths: dict[tuple[str, str], Path] = {}
        self._worker_mmaps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

        csv_path = PROJECT_ROOT / "data_split" / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        depth_key_cache: dict[tuple[str, str], dict[str, int]] = {}

        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                campaign = row.get("campaign")
                deployment = row.get("deployment")
                rel_path = row.get("path")

                if not campaign or not deployment or not rel_path:
                    print(f"Skipping malformed row in {csv_path}: {row}")
                    continue
                if campaigns is not None and campaign not in campaigns:
                    print(
                        f"Skipping row with campaign '{campaign}' not in filter list."
                    )
                    continue

                dep_key = (campaign, deployment)

                # Fetch target filenames
                depth_npy = DEPTH_ROOT / campaign / f"{deployment}.npy"
                keys_npy = DEPTH_ROOT / campaign / f"{deployment}_keys.npy"
                dino_npy = FEAT_ROOT / campaign / f"{deployment}.npy"

                if (
                    not depth_npy.exists()
                    or not keys_npy.exists()
                    or not dino_npy.exists()
                ):
                    raise FileNotFoundError(
                        f"Missing required files for {dep_key}: {depth_npy}, {keys_npy}, {dino_npy}"
                    )

                if dep_key not in depth_key_cache:
                    keys = np.load(keys_npy)
                    # Create lookup: filename (e.g., PR_...jpg) -> integer index
                    depth_key_cache[dep_key] = {str(k): i for i, k in enumerate(keys)}
                    self._depth_mmap_paths[dep_key] = depth_npy
                    self._dino_mmap_paths[dep_key] = dino_npy

                # Match against the keys array, which stores stems WITHOUT the
                # .jpg suffix (unlike the reef repo). Use .stem, not .name.
                filename = Path(rel_path).stem
                depth_keys = depth_key_cache[dep_key]

                if filename in depth_keys:
                    self.records.append(
                        (campaign, deployment, rel_path, depth_keys[filename])
                    )

        if not self.records:
            raise RuntimeError(
                f"CSVSplitDataset is empty for {csv_path}. Check paths and extractions."
            )

        self.rgb_align = transforms.Compose(
            [
                transforms.Resize(NATIVE_SIZE, antialias=True),
                transforms.CenterCrop(NATIVE_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _sample_grid_snapped_crop(self) -> tuple[int, int]:
        max_patch_idx = NATIVE_GRID - TRAIN_GRID
        y_patch = int(torch.randint(0, max_patch_idx + 1, (1,)).item())
        x_patch = int(torch.randint(0, max_patch_idx + 1, (1,)).item())
        return y_patch, x_patch

    def __getitem__(self, i: int):
        campaign, deployment, rel_path, idx = self.records[i]
        dep_key = (campaign, deployment)
        # Lazy MMap Loader
        if dep_key not in self._worker_mmaps:
            depth_mmap = np.load(self._depth_mmap_paths[dep_key], mmap_mode="r")
            try:
                dino_mmap = np.load(self._dino_mmap_paths[dep_key], mmap_mode="r")
            except Exception as e:
                print(f"Error loading DINO MMap for {dep_key}: {e}")
                raise e
            self._worker_mmaps[dep_key] = (depth_mmap, dino_mmap)

        depth_arr, dino_arr = self._worker_mmaps[dep_key]

        # Load RGB directly from CSV path
        full_path = DATA_NORM_ROOT / campaign / deployment / Path(rel_path).name
        rgb = Image.open(full_path).convert("RGB")
        rgb = self.rgb_align(rgb)

        # Load targets from MMap. depth_arr[idx] is already (1, 256, 256), so do
        # NOT unsqueeze. Resize to the 266px native grid so depth aligns with the
        # RGB/feature patch grid for grid-snapped cropping.
        depth = torch.from_numpy(
            np.array(depth_arr[idx], dtype=np.float32)
        )  # (1, 256, 256)
        depth = TF.resize(
            depth, [NATIVE_SIZE, NATIVE_SIZE], antialias=True
        )  # (1, 266, 266)
        dino = torch.from_numpy(np.asarray(dino_arr[idx], dtype=np.float32))

        # Grid-Snapped Cropping
        if self.train:
            y_patch, x_patch = self._sample_grid_snapped_crop()
            y0, x0 = y_patch * PATCH_SIZE, x_patch * PATCH_SIZE

            rgb = TF.crop(rgb, y0, x0, TRAIN_SIZE, TRAIN_SIZE)
            depth = TF.crop(depth, y0, x0, TRAIN_SIZE, TRAIN_SIZE)
            dino = dino[
                y_patch : y_patch + TRAIN_GRID, x_patch : x_patch + TRAIN_GRID, :
            ]

            if self.horizontal_flip and torch.rand(1).item() < 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
                dino = torch.flip(dino, dims=[1])
        else:
            y_patch, x_patch = (NATIVE_GRID - TRAIN_GRID) // 2, (
                NATIVE_GRID - TRAIN_GRID
            ) // 2
            y0, x0 = y_patch * PATCH_SIZE, x_patch * PATCH_SIZE

            rgb = TF.crop(rgb, y0, x0, TRAIN_SIZE, TRAIN_SIZE)
            depth = TF.crop(depth, y0, x0, TRAIN_SIZE, TRAIN_SIZE)
            dino = dino[
                y_patch : y_patch + TRAIN_GRID, x_patch : x_patch + TRAIN_GRID, :
            ]

        return rgb, depth, dino


# ===========================================================================
# EMA helper
# ===========================================================================


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module) -> dict:
        backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }
        ema_state = {k: model.state_dict()[k].clone() for k in model.state_dict()}
        for k in self.shadow:
            ema_state[k] = self.shadow[k]
        model.load_state_dict(ema_state, strict=False)
        return backup

    def restore(self, model: nn.Module, backup: dict):
        state = model.state_dict()
        for k, v in backup.items():
            state[k] = v
        model.load_state_dict(state, strict=False)


# ===========================================================================
# Validation & Test logic
# ===========================================================================


@torch.no_grad()
def save_recon_grid(rae: RAE, batch, out_path: Path, n: int = 8):
    rae.eval()
    rgb, depth, rgb_latent = batch
    rgb, depth = rgb[:n].to(DEVICE), depth[:n].to(DEVICE)
    rgb_latent = rgb_latent[:n].to(DEVICE)

    token_norm = TokenLayerNorm().to(DEVICE)
    rgb_latent = token_norm(rgb_latent)

    rgbd_pred, _ = rae(rgb_latent, depth)
    rgb_pred, depth_pred = rgbd_pred[:, :3], rgbd_pred[:, 3:4]

    n = rgb.shape[0]
    rgb_np = rgb.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_pred_np = rgb_pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    depth_np = depth[:, 0].cpu().numpy()
    depth_pred_np = depth_pred[:, 0].cpu().numpy()

    # Depths are metric (Depth-Anything-V2 Metric-Outdoor): values are physical
    # distances where LARGER = FARTHER from the camera. Each image sits at its
    # own absolute scale, so colour-map each column against the *real* depth's
    # min/max (in metres) and reuse that exact range for the prediction, so the
    # two depth rows are directly comparable and the colour encodes true scale.
    # 'turbo': low (near) -> blue, high (far) -> red.
    rows = ["RGB", "RGB recon", "Depth [m]\n(near=blue, far=red)", "Depth recon [m]"]
    fig, axes = plt.subplots(
        len(rows), n, figsize=(2.2 * n, 2.4 * len(rows)), squeeze=False
    )

    for j in range(n):
        vmin = float(depth_np[j].min())
        vmax = float(depth_np[j].max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        axes[0][j].imshow(rgb_np[j])
        axes[1][j].imshow(rgb_pred_np[j])
        im_r = axes[2][j].imshow(depth_np[j], cmap="turbo", vmin=vmin, vmax=vmax)
        im_p = axes[3][j].imshow(depth_pred_np[j], cmap="turbo", vmin=vmin, vmax=vmax)
        # Compact colourbars so the metric range (in metres) is readable.
        fig.colorbar(im_r, ax=axes[2][j], fraction=0.046, pad=0.04)
        fig.colorbar(im_p, ax=axes[3][j], fraction=0.046, pad=0.04)
        axes[2][j].set_title(f"{vmin:.2f}–{vmax:.2f} m", fontsize=7)

        for r in range(len(rows)):
            axes[r][j].set_xticks([])
            axes[r][j].set_yticks([])

    for r, label in enumerate(rows):
        axes[r][0].set_ylabel(label, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    rae.train()


@torch.no_grad()
def run_evaluation(
    rae: RAE, loss_fn: RAELoss, dataloader, max_batches: int | None = None
) -> dict:
    rae.eval()
    token_norm = TokenLayerNorm().to(DEVICE)

    sums = {
        "l1_rgb": 0.0,
        "l1_depth": 0.0,
        "silog_depth": 0.0,
        "depth_grad": 0.0,
        "lpips": 0.0,
        "fused_mean": 0.0,
        "fused_std": 0.0,
        "n": 0,
    }
    lpips_fn = loss_fn.lpips

    for i, (rgb, depth, rgb_latent) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        rgb = rgb.to(DEVICE, non_blocking=True)
        depth = depth.to(DEVICE, non_blocking=True)
        rgb_latent = token_norm(rgb_latent.to(DEVICE, non_blocking=True))

        rgbd_pred, fused = rae(rgb_latent, depth)

        rgb_p, depth_p = rgbd_pred[:, :3], rgbd_pred[:, 3:4]
        b = rgb.shape[0]
        sums["l1_rgb"] += F.l1_loss(rgb_p, rgb, reduction="mean").item() * b
        sums["l1_depth"] += F.l1_loss(depth_p, depth, reduction="mean").item() * b
        sums["silog_depth"] += (
            silog_loss(
                depth_p, depth, lam=loss_fn.silog_lambda, eps=loss_fn.silog_eps
            ).item()
            * b
        )
        sums["depth_grad"] += (
            multiscale_gradient_loss(
                depth_p,
                depth,
                num_scales=loss_fn.depth_grad_scales,
                eps=loss_fn.silog_eps,
            ).item()
            * b
        )
        sums["lpips"] += lpips_fn(rgb_p, rgb).item() * b
        sums["fused_mean"] += float(fused.mean()) * b
        sums["fused_std"] += float(fused.std()) * b
        sums["n"] += b

    rae.train()
    return {
        "l1_rgb": sums["l1_rgb"] / sums["n"],
        "l1_depth": sums["l1_depth"] / sums["n"],
        "silog_depth": sums["silog_depth"] / sums["n"],
        "depth_grad": sums["depth_grad"] / sums["n"],
        "lpips": sums["lpips"] / sums["n"],
        "fused_mean": sums["fused_mean"] / sums["n"],
        "fused_std": sums["fused_std"] / sums["n"],
    }


def generate_test_report(metrics: dict, out_path: Path):
    report = f"""# Representation Autoencoder (RAE) - Test Split Report
*Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}*
### Core Reconstruction Metrics
| Metric | Value |
| :--- | :--- |
"""
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION REPORT")
    print("=" * 50)
    print(report)
    out_path.write_text(report)


def build_models() -> tuple[RAE, DinoDiscriminator, RAELoss, TokenLayerNorm]:
    rae = RAE(RAEConfig()).to(DEVICE)
    disc = DinoDiscriminator(head_channels=256).to(DEVICE)
    loss_fn = RAELoss(
        lpips_start_epoch=6,
        gan_start_epoch=8,
        w_l1_rgb=1.0,
        w_depth=5.0,
        w_lpips=1.0,
        w_gan=0.75,
        depth_loss="silog",
        silog_lambda=0.85,
        w_depth_grad=0.5,
        depth_grad_scales=4,
    ).to(DEVICE)
    token_norm = TokenLayerNorm().to(DEVICE)
    return rae, disc, loss_fn, token_norm


def build_optimizers(
    rae: nn.Module, disc: nn.Module, lr: float, total_epochs: int, steps_per_epoch: int
):
    g_opt = AdamW(rae.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=0.0)
    d_params = [p for p in get_model(disc).head.parameters() if p.requires_grad]
    d_opt = AdamW(d_params, lr=lr, betas=(0.5, 0.9), weight_decay=0.0)

    warmup_steps = steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def make_sched(opt):
        warm = LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cos = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=lr * 0.1)
        return SequentialLR(opt, schedulers=[warm, cos], milestones=[warmup_steps])

    return g_opt, d_opt, make_sched(g_opt), make_sched(d_opt)


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    global_step: int,
    rae: nn.Module,
    disc: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    g_sched,
    d_sched,
    ema: EMA,
    args: argparse.Namespace,
):
    checkpoint = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "rae": get_model(rae).state_dict(),
        "disc": get_model(disc).state_dict(),
        "g_opt": g_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "g_sched": g_sched.state_dict(),
        "d_sched": d_sched.state_dict(),
        "ema": ema.shadow,
        "args": vars(args),
    }

    epoch_path = ckpt_dir / f"checkpoint_e{epoch:03d}.pt"
    last_path = ckpt_dir / "last.pt"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, last_path)
    print(f"  [ckpt e{epoch}] saved {epoch_path} and {last_path}")


def load_checkpoint(
    resume_path: str | Path,
    rae: nn.Module,
    disc: nn.Module,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    g_sched,
    d_sched,
    ema: EMA,
) -> tuple[int, int]:
    checkpoint = torch.load(resume_path, map_location=DEVICE)

    get_model(rae).load_state_dict(checkpoint["rae"])
    get_model(disc).load_state_dict(checkpoint["disc"])
    g_opt.load_state_dict(checkpoint["g_opt"])
    d_opt.load_state_dict(checkpoint["d_opt"])
    g_sched.load_state_dict(checkpoint["g_sched"])
    d_sched.load_state_dict(checkpoint["d_sched"])
    ema.shadow = {k: v.to(DEVICE) for k, v in checkpoint["ema"].items()}

    return int(checkpoint["epoch"]), int(checkpoint["global_step"])


# ===========================================================================
# Main Training Driver
# ===========================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=1)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--run-name", type=str, default="rae")
    ap.add_argument("--amp", action="store_true", help="Use bfloat16 mixed precision.")
    args = ap.parse_args()

    # ---- DDP INIT ----
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        global DEVICE
        DEVICE = torch.device("cuda", local_rank)
    else:
        local_rank, global_rank = 0, 0
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = CKPT_ROOT / args.run_name
    fig_dir = FIG_ROOT / args.run_name

    if global_rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    if is_ddp:
        dist.barrier()

    # ---- Data ----
    train_ds = CSVSplitDataset(split="train")
    if is_ddp:
        train_sampler = DistributedSampler(train_ds)
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_sampler = None
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

    if global_rank == 0:
        val_ds = CSVSplitDataset(split="val")
        val_dl = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
        )
        val_batch_for_figs = next(iter(val_dl))

        test_ds = CSVSplitDataset(split="test")
        test_dl = DataLoader(
            test_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print(
            f"Dataset Setup:\n  Train: {len(train_ds)} imgs\n  Val: {len(val_ds)} imgs\n  Test: {len(test_ds)} imgs"
        )
    else:
        val_dl, val_batch_for_figs, test_dl = None, None, None

    # ---- Models ----
    rae, disc, loss_fn, token_norm = build_models()

    if is_ddp:
        rae = DDP(rae, device_ids=[local_rank], output_device=local_rank)
        disc = DDP(disc, device_ids=[local_rank], output_device=local_rank)

    ema = EMA(get_model(rae), decay=args.ema_decay)

    steps_per_epoch = len(train_dl)
    g_opt, d_opt, g_sched, d_sched = build_optimizers(
        rae, disc, args.lr, args.epochs, steps_per_epoch
    )

    start_epoch, global_step = 0, 0
    amp_dtype = torch.bfloat16 if args.amp else None

    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            args.resume, rae, disc, g_opt, d_opt, g_sched, d_sched, ema
        )
        if global_rank == 0:
            print(
                f"Resumed from {args.resume} at epoch {start_epoch}, step {global_step}"
            )

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        rae.train()
        disc.train()
        loss_fn.set_epoch(epoch)
        epoch_t0 = time.time()

        pbar = (
            tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)
            if global_rank == 0
            else train_dl
        )
        running = {}

        for rgb, depth, rgb_latent in pbar:
            rgb = rgb.to(DEVICE, non_blocking=True)
            depth = depth.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                rgb_latent = token_norm(rgb_latent.to(DEVICE, non_blocking=True))

            # =============== Generator step ===============
            g_opt.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    rgbd_pred, _ = rae(rgb_latent, depth)
                    rgbd_target = torch.cat([rgb, depth], dim=1)
                    g_out = loss_fn.generator_loss(
                        rgbd_pred=rgbd_pred,
                        rgbd_target=rgbd_target,
                        last_layer=get_model(rae).decoder.get_last_layer(),
                        disc=get_model(disc) if loss_fn.gan_active else None,
                    )
            else:
                rgbd_pred, _ = rae(rgb_latent, depth)
                rgbd_target = torch.cat([rgb, depth], dim=1)
                g_out = loss_fn.generator_loss(
                    rgbd_pred=rgbd_pred,
                    rgbd_target=rgbd_target,
                    last_layer=get_model(rae).decoder.get_last_layer(),
                    disc=get_model(disc) if loss_fn.gan_active else None,
                )

            g_loss = g_out["total"]
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(rae.parameters(), max_norm=1.0)
            g_opt.step()
            g_sched.step()
            ema.update(get_model(rae))

            # =============== Discriminator step ===============
            if loss_fn.gan_active:
                d_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if amp_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            rgbd_fake, _ = rae(rgb_latent, depth)
                    else:
                        rgbd_fake, _ = rae(rgb_latent, depth)
                    rgb_fake = rgbd_fake[:, :3]

                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        d_out = loss_fn.discriminator_loss(
                            disc=disc, rgb_real=rgb, rgb_fake=rgb_fake
                        )
                else:
                    d_out = loss_fn.discriminator_loss(
                        disc=disc, rgb_real=rgb, rgb_fake=rgb_fake
                    )

                d_loss = d_out["d_total"]
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                d_opt.step()
                d_sched.step()
            else:
                d_out = {"d_total": torch.tensor(0.0)}

            if global_rank == 0:
                for k, v in g_out.items():
                    if k != "total" and torch.is_tensor(v):
                        running.setdefault(k, []).append(float(v))
                for k, v in d_out.items():
                    if torch.is_tensor(v):
                        running.setdefault(k, []).append(float(v))

                if global_step % 50 == 0:
                    msg = " ".join(
                        f"{k}={sum(vs[-50:])/min(50,len(vs)):.3f}"
                        for k, vs in running.items()
                    )
                    pbar.set_postfix_str(msg)

            global_step += 1

        # ---- Validation ----
        if global_rank == 0:
            if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
                backup = ema.apply_to(get_model(rae))
                val_metrics = run_evaluation(
                    get_model(rae), loss_fn, val_dl, max_batches=16
                )
                save_recon_grid(
                    get_model(rae),
                    val_batch_for_figs,
                    fig_dir / f"recon_e{epoch:03d}.png",
                )
                ema.restore(get_model(rae), backup)

                msg = " | ".join(f"{k}={v:.5f}" for k, v in val_metrics.items())
                print(f"  [val e{epoch}] {msg}")
                with open(ckpt_dir / "metrics.jsonl", "a") as f:
                    f.write(
                        json.dumps({"epoch": epoch, "step": global_step, **val_metrics})
                        + "\n"
                    )

            if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
                print(f"  [ckpt e{epoch}] Saving checkpoint to {ckpt_dir}...")
                save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    epoch=epoch,
                    global_step=global_step,
                    rae=rae,
                    disc=disc,
                    g_opt=g_opt,
                    d_opt=d_opt,
                    g_sched=g_sched,
                    d_sched=d_sched,
                    ema=ema,
                    args=args,
                )

    # ---- Final Test Report ----
    if global_rank == 0:
        print(
            "\nTraining complete. Running final evaluation on test set using EMA weights..."
        )
        backup = ema.apply_to(get_model(rae))
        test_metrics = run_evaluation(
            get_model(rae), loss_fn, test_dl, max_batches=None
        )
        generate_test_report(test_metrics, ckpt_dir / "test_report.md")
        ema.restore(get_model(rae), backup)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
