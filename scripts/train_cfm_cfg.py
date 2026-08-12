"""Train the U-Net Continuous Flow Matching model via CFG.

Ingests split CSVs directly. Computes the global semantic condition
on the fly by pooling the 16x16 DINO feature grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from benthicflow import (
    CKPT_ROOT,
    DEPTH_ROOT,
    FEAT_ROOT,
    FIG_ROOT,
    PROJECT_ROOT,
    RGB_ROOT,
)
from models.rae import RAE, ConvDecoderConfig, DepthEncoderConfig, RAEConfig
from models.unet_cfm import UNetCFM, cfm_sample_cfg
from scripts.train_rae_ddp import EMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fast fp32 matmuls (TF32) + cuDNN autotuner. Input sizes are fixed (224 RGB /
# 16x16 latent), so benchmark autotune is a clean win. Safe on H100/Ampere+.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def amp_autocast(enabled: bool):
    """bf16 autocast context (no GradScaler needed for bf16) or a no-op."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


NATIVE_SIZE = 518
TRAIN_SIZE = 224
PATCH_SIZE = 14
NATIVE_GRID = NATIVE_SIZE // PATCH_SIZE
TRAIN_GRID = TRAIN_SIZE // PATCH_SIZE


def load_rae_frozen(
    ckpt_path: Path, device: torch.device, depth_activation: str | None = None
) -> RAE:
    # depth_activation overrides ConvDecoderConfig's REEF_VARIANT-driven default
    # ("sigmoid" base / "softplus" dpf) -- required when one process decodes
    # both variants' checkpoints (e.g. qualitative_comparison.py --stage dpf).
    dec_kwargs = (
        {} if depth_activation is None else {"depth_activation": depth_activation}
    )
    rae_cfg = RAEConfig(
        depth_cfg=DepthEncoderConfig(
            patch_size=14, dim=256, depth=8, num_heads=8, out_dim=256
        ),
        decoder_cfg=ConvDecoderConfig(
            in_dim=1024,
            hidden_dims=(512, 256, 128),
            upsample_factors=(2, 7),
            num_res_blocks=2,
            groups=32,
            out_channels=4,
            use_mid_attention=True,
            **dec_kwargs,
        ),
        noise_tau=0.0,
    )
    rae = RAE(rae_cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ck["ema"] if "ema" in ck else ck["rae"]
    rae.load_state_dict(state, strict=False)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad_(False)
    return rae


class CSVSplitDataset(Dataset):
    def __init__(self, split: str, load_rgb: bool = False):
        self.train = split == "train"
        # RGB is only needed for the generation reference grid, not for CFM
        # training/eval. Default off so we don't pay the (large) RGB read per step.
        self.load_rgb = load_rgb
        self.records = []
        self._depth_mmap_paths = {}
        self._dino_mmap_paths = {}
        self._rgb_mmap_paths = {}
        self._worker_mmaps = {}

        csv_path = PROJECT_ROOT / "data_split" / f"{split}.csv"
        depth_key_cache = {}

        with csv_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                campaign, deployment, rel_path = (
                    row.get("campaign"),
                    row.get("deployment"),
                    row.get("path"),
                )
                if not all([campaign, deployment, rel_path]):
                    continue

                dep_key = (campaign, deployment)
                depth_npy = DEPTH_ROOT / campaign / f"{deployment}.npy"
                keys_npy = DEPTH_ROOT / campaign / f"{deployment}_keys.npy"
                dino_npy = FEAT_ROOT / campaign / f"{deployment}.npy"
                rgb_npy = RGB_ROOT / campaign / f"{deployment}.npy"

                if not all(
                    [p.exists() for p in [depth_npy, keys_npy, dino_npy, rgb_npy]]
                ):
                    raise FileNotFoundError(
                        f"Missing depth/DINO/RGB files for {dep_key}: {depth_npy}, {keys_npy}, {dino_npy}, {rgb_npy}"
                    )

                if dep_key not in depth_key_cache:
                    keys = np.load(keys_npy)
                    depth_key_cache[dep_key] = {str(k): i for i, k in enumerate(keys)}
                    self._depth_mmap_paths[dep_key] = depth_npy
                    self._dino_mmap_paths[dep_key] = dino_npy
                    self._rgb_mmap_paths[dep_key] = rgb_npy

                filename = Path(rel_path).name
                if filename in depth_key_cache[dep_key]:
                    self.records.append(
                        (
                            campaign,
                            deployment,
                            rel_path,
                            depth_key_cache[dep_key][filename],
                        )
                    )

        # RGB is preprocessed (Resize+CenterCrop baked in) and stored as uint8
        # arrays, so no PIL transform is needed at load time.

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        campaign, deployment, rel_path, idx = self.records[i]
        dep_key = (campaign, deployment)

        if dep_key not in self._worker_mmaps:
            self._worker_mmaps[dep_key] = (
                np.load(self._depth_mmap_paths[dep_key], mmap_mode="r"),
                np.load(self._dino_mmap_paths[dep_key], mmap_mode="r"),
                np.load(self._rgb_mmap_paths[dep_key], mmap_mode="r"),
            )

        depth_arr, dino_arr, rgb_arr = self._worker_mmaps[dep_key]
        depth_slice = np.array(depth_arr[idx], dtype=np.float32, copy=True)
        dino_slice = np.array(dino_arr[idx], dtype=np.float32, copy=True)

        depth = torch.from_numpy(depth_slice).unsqueeze(0)
        dino = torch.from_numpy(dino_slice)

        # Skip the large RGB read unless this dataset is the generation source.
        rgb = torch.empty(0)
        if self.load_rgb:
            rgb_slice = np.array(rgb_arr[idx], copy=True)  # [S, S, 3] uint8
            rgb = torch.from_numpy(rgb_slice).permute(2, 0, 1).float().div_(255.0)

        # Grid-Snapped Crop logic
        if self.train:
            y_patch = int(torch.randint(0, NATIVE_GRID - TRAIN_GRID + 1, (1,)).item())
            x_patch = int(torch.randint(0, NATIVE_GRID - TRAIN_GRID + 1, (1,)).item())
            if torch.rand(1).item() < 0.5:
                depth = TF.hflip(depth)
                dino = torch.flip(dino, dims=[1])
                if self.load_rgb:
                    rgb = TF.hflip(rgb)
        else:
            y_patch = x_patch = (NATIVE_GRID - TRAIN_GRID) // 2

        y0, x0 = y_patch * PATCH_SIZE, x_patch * PATCH_SIZE
        depth = TF.crop(depth, y0, x0, TRAIN_SIZE, TRAIN_SIZE)
        dino = dino[y_patch : y_patch + TRAIN_GRID, x_patch : x_patch + TRAIN_GRID, :]
        if self.load_rgb:
            rgb = TF.crop(rgb, y0, x0, TRAIN_SIZE, TRAIN_SIZE)

        return rgb, depth, dino


@torch.no_grad()
def fit_normalizer(
    model: UNetCFM, rae: RAE, dataloader: DataLoader, max_batches: int = 100
):
    s = torch.zeros(1024, dtype=torch.float64, device=DEVICE)
    sq = torch.zeros(1024, dtype=torch.float64, device=DEVICE)
    n = 0
    for i, (rgb, depth, dino) in enumerate(tqdm(dataloader, desc="Fitting Normalizer")):
        if i >= max_batches:
            break
        rgb, depth = rgb.to(DEVICE), depth.to(DEVICE)

        # DINO formatting: [B, 16, 16, 768] -> [B, 768, 16, 16]
        dino_spatial = dino.to(DEVICE).permute(0, 3, 1, 2)

        _, fused = rae(
            dino_spatial.permute(0, 2, 3, 1), depth
        )  # RAE outputs channel-last
        fused = fused.permute(0, 3, 1, 2).double()  # [B, 1024, 16, 16]

        s += fused.sum(dim=(0, 2, 3))
        sq += (fused * fused).sum(dim=(0, 2, 3))
        n += fused.shape[0] * fused.shape[2] * fused.shape[3]

    mean = (s / n).float()
    std = ((sq / n).float() - mean.pow(2)).clamp(min=1e-6).sqrt()
    model.latent_mean.copy_(mean)
    model.latent_std.copy_(std)
    print(
        f"Normalizer fitted. Mean range: [{mean.min():.3f}, {mean.max():.3f}], Std range: [{std.min():.3f}, {std.max():.3f}]"
    )


def cfm_loss(
    model: UNetCFM, z1_norm: torch.Tensor, cond: torch.Tensor, p_uncond: float = 0.1
) -> torch.Tensor:
    """Rectified Flow loss with CFG dropout."""
    B = z1_norm.shape[0]

    # CFG Dropout
    if model.training and p_uncond > 0:
        drop_mask = torch.rand(B, device=DEVICE) < p_uncond

        # UNWRAP the model here to access the parameter safely
        base_model = get_model(model)
        cond = torch.where(
            drop_mask.unsqueeze(1), base_model.null_cond.unsqueeze(0), cond
        )

    t = torch.rand(B, device=DEVICE)
    t_view = t.view(B, 1, 1, 1)

    z0 = torch.randn_like(z1_norm)
    zt = (1.0 - t_view) * z0 + t_view * z1_norm
    v_target = z1_norm - z0

    v_pred = model(zt, t, cond)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def evaluate(
    model: UNetCFM,
    rae: RAE,
    dataloader: DataLoader,
    is_ddp: bool = False,
    amp: bool = False,
) -> float:
    """Distributed validation: each rank evaluates its DistributedSampler shard,
    then we all-reduce the summed loss and count so every rank returns the same
    global mean. Because all ranks call the all-reduce together, this also acts
    as the inter-rank sync point (no single-rank stall -> no barrier timeout)."""
    model.eval()
    base_model = model.module if isinstance(model, DDP) else model

    total_loss = torch.zeros((), device=DEVICE)
    total_n = torch.zeros((), device=DEVICE)

    for rgb, depth, dino in dataloader:
        dino_spatial = dino.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)
        cond = dino_spatial.mean(dim=(2, 3))  # Global Semantic Vector

        with amp_autocast(amp):
            _, fused = rae(
                dino_spatial.permute(0, 2, 3, 1), depth.to(DEVICE, non_blocking=True)
            )
        z1_norm = base_model.normalize(fused.float().permute(0, 3, 1, 2))

        with amp_autocast(amp):
            loss = cfm_loss(
                base_model, z1_norm, cond, p_uncond=0.0
            )  # No dropout on eval
        bs = rgb.shape[0]
        total_loss += loss.detach() * bs
        total_n += bs

    if is_ddp:
        dist.all_reduce(total_loss)
        dist.all_reduce(total_n)

    model.train()
    return (total_loss / total_n).item()


@torch.no_grad()
def save_generation_grid(
    model: UNetCFM,
    rae: RAE,
    batch,
    out_path: Path,
    n: int = 8,
    n_steps: int = 50,
    cfg_scale: float = 3.0,
):
    """Sample latents from the CFM and decode them through the frozen RAE to
    eyeball generation quality AND verify conditioning. Conditional and
    unconditional samples share the same initial noise, so any difference between
    those two rows is due to the DINO condition. Rows (labeled in the figure):
      Real RGB | Conditional gen | Unconditional gen | Conditional depth
    Column j of every row corresponds to the same source image / condition.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = get_model(model)
    was_training = base.training
    base.eval()

    rgb, depth, dino = batch
    rgb = rgb[:n].to(DEVICE)
    dino = dino[:n].to(DEVICE)

    # Per-image global pooled DINO vector (the train-time condition) vs the null
    # token (unconditional). Same z0 for both -> isolates the effect of cond.
    cond = dino.permute(0, 3, 1, 2).mean(dim=(2, 3))  # [n, 768]
    null = base.null_cond.unsqueeze(0).expand(n, -1)  # [n, 768]
    z0 = torch.randn(n, base.latent_dim, TRAIN_GRID, TRAIN_GRID, device=DEVICE)

    def decode(z_norm):
        z = base.denormalize(z_norm)
        rgbd = rae.decoder(z.permute(0, 2, 3, 1).contiguous())  # [n, 4, 224, 224]
        return rgbd[:, :3].clamp(0, 1), rgbd[:, 3:4].clamp(0, 1)

    z_cond = cfm_sample_cfg(base, cond, n_steps=n_steps, cfg_scale=cfg_scale, z0=z0)
    z_uncond = cfm_sample_cfg(base, null, n_steps=n_steps, cfg_scale=1.0, z0=z0)
    rgb_cond, depth_cond = decode(z_cond)
    rgb_uncond, depth_uncond = decode(z_uncond)

    # reef depth is sigmoid -> [0, 1], so grayscale repeat is the right viz
    # (matches train_rae_ddp.py's recon grid; no turbo/metric handling needed).
    rows = [
        ("Real RGB", rgb),
        ("Conditional gen", rgb_cond),
        ("Unconditional gen", rgb_uncond),
        ("Conditional depth", depth_cond.repeat(1, 3, 1, 1)),
        ("Unconditional depth", depth_uncond.repeat(1, 3, 1, 1)),
    ]

    fig, axes = plt.subplots(len(rows), 1, figsize=(2 * n, 2 * len(rows)))
    for ax, (label, imgs) in zip(axes, rows):
        row_img = make_grid(imgs, nrow=n, padding=2).permute(1, 2, 0).cpu().numpy()
        ax.imshow(row_img)
        ax.set_ylabel(label, fontsize=13, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(out_path.stem, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if was_training:
        base.train()


def get_model(model: nn.Module) -> nn.Module:
    """Helper to unwrap DDP models safely."""
    return model.module if isinstance(model, DDP) else model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rae-ckpt", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument(
        "--gen-every",
        type=int,
        default=20,
        help="Save a CFM generation grid every N epochs.",
    )
    ap.add_argument("--amp", action="store_true", help="bf16 mixed precision.")
    ap.add_argument("--compile", action="store_true", help="torch.compile the UNet.")
    ap.add_argument("--run-name", type=str, default="unet_cfm")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        global_rank = int(os.environ["RANK"])
    else:
        global_rank = 0

    ckpt_dir = CKPT_ROOT / args.run_name
    fig_dir = FIG_ROOT / "cfm" / args.run_name
    if global_rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CSVSplitDataset("train")
    val_ds = CSVSplitDataset("val")
    test_ds = CSVSplitDataset("test")

    train_sampler = DistributedSampler(train_ds) if is_ddp else None
    # Shard val/test across ranks too, so validation is ~world_size faster and
    # every rank participates (avoids the single-rank stall that timed out the
    # post-validation barrier).
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_ddp else None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch,
        sampler=test_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Fixed batch (rank 0 only) for periodic generation snapshots. This is the
    # ONLY place real RGB is needed, so load it from a tiny dedicated loader
    # (load_rgb=True) instead of paying the RGB read on every training step.
    gen_batch = None
    if global_rank == 0:
        gen_ds = CSVSplitDataset("val", load_rgb=True)
        gen_batch = next(
            iter(DataLoader(gen_ds, batch_size=8, shuffle=False, num_workers=2))
        )

    rae = load_rae_frozen(Path(args.rae_ckpt), DEVICE)
    model = UNetCFM().to(DEVICE)

    if global_rank == 0:
        fit_normalizer(model, rae, train_dl, max_batches=50)
    if is_ddp:
        dist.barrier()
        dist.broadcast(model.latent_mean, src=0)
        dist.broadcast(model.latent_std, src=0)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Compile in place (Module.compile) so the module identity is preserved -- EMA,
    # get_model(), and state_dict save/load all keep working unchanged.
    if args.compile:
        get_model(model).compile()

    ema = EMA(get_model(model), decay=0.9999)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))

    best_val = float("inf")

    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", disable=(global_rank != 0))
        for rgb, depth, dino in pbar:
            dino_spatial = dino.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2)
            cond = dino_spatial.mean(dim=(2, 3))  # Average pool for global CLS proxy

            with torch.no_grad():
                with amp_autocast(args.amp):
                    _, fused = rae(
                        dino_spatial.permute(0, 2, 3, 1),
                        depth.to(DEVICE, non_blocking=True),
                    )
                # Normalize the target in fp32 for numerical stability.
                z1_norm = get_model(model).normalize(fused.float().permute(0, 3, 1, 2))

            opt.zero_grad()
            with amp_autocast(args.amp):
                loss = cfm_loss(model, z1_norm, cond, p_uncond=args.p_uncond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(get_model(model))

            if global_rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ---- Validation (distributed: all ranks participate) ----
        # Evaluate (and snapshot/save) under EMA weights, so best_model.pt stores
        # exactly the weights the val_loss was measured on.
        backup = ema.apply_to(get_model(model))
        val_loss = evaluate(
            model, rae, val_dl, is_ddp=is_ddp, amp=args.amp
        )  # all-reduce inside == sync point

        if global_rank == 0:
            print(f"Validation Loss: {val_loss:.5f}")
            if (
                (epoch + 1) % args.gen_every == 0
                or epoch == args.epochs - 1
                or epoch == 0
            ):
                save_generation_grid(
                    model, rae, gen_batch, fig_dir / f"gen_e{epoch:03d}.png"
                )

        # val_loss is identical on every rank (all-reduced), so best_val stays in
        # sync; only rank 0 writes the file (the EMA weights, before restore).
        if val_loss < best_val:
            best_val = val_loss
            if global_rank == 0:
                torch.save(get_model(model).state_dict(), ckpt_dir / "best_model.pt")

        ema.restore(get_model(model), backup)
        # Non-zero ranks wait here while rank 0 finishes the (fast) generation grid.
        if is_ddp:
            dist.barrier()

    # ---- Final test (distributed) on the best EMA checkpoint ----
    if is_ddp:
        dist.barrier()  # ensure rank 0 has finished writing best_model.pt
    if global_rank == 0:
        print("\nRunning Final Test Set Evaluation...")
    # best_model.pt already holds the EMA weights, so load directly (no apply_to).
    # map_location=DEVICE keeps each rank's tensors on its own GPU.
    get_model(model).load_state_dict(
        torch.load(ckpt_dir / "best_model.pt", weights_only=True, map_location=DEVICE)
    )
    test_loss = evaluate(model, rae, test_dl, is_ddp=is_ddp, amp=args.amp)
    if global_rank == 0:
        print(f"Test Set CFG Flow Loss: {test_loss:.5f}")

    # Barrier so ranks 1..N-1 wait for rank 0 to finish the test eval before
    # tearing down the process group.
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
