"""Evaluate FLUX text-to-image and img2img generative quality.

Computes FID, KID, and depth consistency for FLUX baseline models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import datetime
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)
from transformers import (
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from tqdm import tqdm
import csv
import torchvision.transforms.functional as TF


from benthicflow import FIG_ROOT, PROJECT_ROOT, DEPTH_ROOT, FEAT_ROOT, RGB_ROOT
from scripts.test_generation import TRAIN_SIZE, _DINOv2CLS, DEVICE
from scripts.extract_features import load_dinov2
from scripts.flux_generation import REEF_PROMPT, resize_to_multiple_of_16, DEV_ID

RESULTS_JSON = FIG_ROOT / "flux" / "flux_eval_results.json"

NATIVE_SIZE = 518
TRAIN_SIZE  = 224     
PATCH_SIZE  = 14
NATIVE_GRID = NATIVE_SIZE // PATCH_SIZE
TRAIN_GRID  = TRAIN_SIZE // PATCH_SIZE

# Set by setup_distributed(); default to single-process values.
RANK, WORLD_SIZE, LOCAL_RANK = 0, 1, 0


class CSVSplitDataset(Dataset):
    def __init__(self, split: str, load_rgb: bool = False, campaign: str = None):
        self.train = split == "train"
        # RGB is only needed for the generation reference grid, not for CFM
        # training/eval. Default off so we don't pay the (large) RGB read per step.
        self.load_rgb = load_rgb
        self.records = []
        self._dino_mmap_paths = {}
        self._rgb_mmap_paths = {}
        self._worker_mmaps = {}

        csv_path = PROJECT_ROOT / "data_split" / f"{split}.csv"
        key_cache = {}

        available_campaigns = set()

        with csv_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                # NB: use row-local names here. Reusing `campaign` would shadow
                # the constructor's `campaign` filter argument and leave it set
                # to the last row's campaign, silently breaking per-campaign
                # selection (every campaign would resolve to the same subset).
                row_campaign, deployment, rel_path = row.get("campaign"), row.get("deployment"), row.get("path")
                if not all([row_campaign, deployment, rel_path]):
                    raise ValueError(f"Missing required fields in CSV row: {row}")

                available_campaigns.add(row_campaign)
                dep_key = (row_campaign, deployment)
                # The *_keys.npy index lives under DEPTH_ROOT but maps filenames
                # to the shared row order used by the DINO and RGB arrays too, so
                # it is still needed even though depth data is no longer loaded.
                keys_npy = DEPTH_ROOT / row_campaign / f"{deployment}_keys.npy"
                dino_npy = FEAT_ROOT / row_campaign / f"{deployment}.npy"
                rgb_npy = RGB_ROOT / row_campaign / f"{deployment}.npy"

                if not all([p.exists() for p in [keys_npy, dino_npy, rgb_npy]]):
                    raise FileNotFoundError(f"Missing keys/DINO/RGB files for {dep_key}: {keys_npy}, {dino_npy}, {rgb_npy}")

                if dep_key not in key_cache:
                    keys = np.load(keys_npy)
                    key_cache[dep_key] = {str(k): i for i, k in enumerate(keys)}
                    self._dino_mmap_paths[dep_key] = dino_npy
                    self._rgb_mmap_paths[dep_key] = rgb_npy

                filename = Path(rel_path).name
                if filename in key_cache[dep_key]:
                    self.records.append((row_campaign, deployment, rel_path, key_cache[dep_key][filename]))

        # Optionally restrict to a single campaign (per-campaign conditional FID).
        if campaign is not None:
            self.records = [r for r in self.records if r[0] == campaign]

        print(f"Available campaigns are {available_campaigns}, split '{split}' has {len(self.records)} records, load_rgb={self.load_rgb}")

        self.available_campaigns = available_campaigns
        # RGB is preprocessed (Resize+CenterCrop baked in) and stored as uint8
        # arrays, so no PIL transform is needed at load time.

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, i: int):
        campaign, deployment, rel_path, idx = self.records[i]
        dep_key = (campaign, deployment)

        if dep_key not in self._worker_mmaps:
            self._worker_mmaps[dep_key] = (
                np.load(self._dino_mmap_paths[dep_key], mmap_mode="r"),
                np.load(self._rgb_mmap_paths[dep_key], mmap_mode="r"),
            )

        dino_arr, rgb_arr = self._worker_mmaps[dep_key]
        dino = torch.from_numpy(np.array(dino_arr[idx], dtype=np.float32, copy=True))

        # Full-frame RGB, NO crop. FLUX conditions on (and is scored against) the
        # whole native image: an arbitrary TRAIN_SIZE crop upsampled to the gen
        # resolution gave FLUX a blurry partial scene, whereas the full frame is
        # sharp and keeps the entire benthic context, improving generation quality.
        # NB: this makes the FLUX metrics full-frame, so they are no longer directly
        # comparable to test_generation.py's TRAIN_SIZE-crop numbers.
        rgb = torch.empty(0)
        if self.load_rgb:
            rgb_slice = np.array(rgb_arr[idx], copy=True)          # [S, S, 3] uint8, S = NATIVE_SIZE
            rgb = torch.from_numpy(rgb_slice).permute(2, 0, 1).float().div_(255.0)

        return rgb, dino

def setup_distributed():
    """Init torch.distributed from torchrun env vars (data-parallel, one rank
    per GPU). Falls back to single-process when not launched via torchrun.

    Also repoints the module-level DEVICE at this rank's GPU so the pipeline,
    metrics and tensors all land on the right device.
    """
    global RANK, WORLD_SIZE, LOCAL_RANK, DEVICE
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        RANK = int(os.environ["RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(LOCAL_RANK)
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
    return RANK, WORLD_SIZE, LOCAL_RANK


def is_main():
    return RANK == 0


def rprint(*a, **k):
    """print only on rank 0, to avoid WORLD_SIZE-fold duplicated logs."""
    if is_main():
        print(*a, **k)


def tensor_to_pil(rgb_chw: torch.Tensor) -> Image.Image:
    """[3, H, W] float tensor in [0,1] -> PIL RGB image."""
    arr = (rgb_chw.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def resize_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear-resize a [B, 3, H, W] batch to (size, size) if not already that size.

    Brings full-frame real crops (and the generations) to the common --eval_size
    before the FID/KID/DINO judges, so both distributions are scored at the same
    resolution regardless of the (larger) generation resolution.
    """
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def pil_list_to_tensor(pil_images, size: int, device) -> torch.Tensor:
    """List of PIL images -> [N, 3, size, size] float tensor in [0,1] on device."""
    tensors = [torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
              .permute(2, 0, 1) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    if batch.shape[-2:] != (size, size):
        batch = F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)
    return batch


def flux_generate(pipe, rgb_seed, args):
    """Reference-condition FLUX.2 on each full-frame seed; return [B, 3, eval_size, eval_size].
FLUX.2 treats a list of images as multiple references for ONE generation and
replicates them across the batch, so distinct seeds can't be batched -- we
generate one image per seed, each using its own full-frame image as the single
reference. Generation runs at args.gen_size (FLUX's comfortable resolution),
"""
    gen_images = []
    for img in rgb_seed:
        ref = resize_to_multiple_of_16(tensor_to_pil(img), args.gen_size, args.gen_size)
        out = pipe(
            prompt=REEF_PROMPT,
            image=[ref],
            height=args.gen_size,
            width=args.gen_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]
        gen_images.append(out)
    return pil_list_to_tensor(gen_images, args.eval_size, DEVICE)


def flux_generate_unconditional(pipe, n, args):
    """Text-to-image: generate n images from REEF_PROMPT alone (no reference image).

    Batched in a single pipeline call (num_images_per_prompt=n) -- with no
    reference image FLUX.2 batches independent samples normally. Generation runs at
    args.gen_size, then outputs are resized down to args.eval_size so the generated
    and real distributions are scored at the same (lower) resolution.
    """
    gen_images = pipe(
        prompt=REEF_PROMPT,
        num_images_per_prompt=n,
        height=args.gen_size,
        width=args.gen_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    ).images
    return pil_list_to_tensor(gen_images, args.eval_size, DEVICE)


def build_metrics(feature_extractor, kid_subset_size):
    """Build (fid, kid, label). DINOv2 extractor is loaded once and shared."""
    if feature_extractor == "dino":
        dino_judge = load_dinov2()
        fid = FrechetInceptionDistance(feature=_DINOv2CLS(dino_judge)).to(DEVICE)
        kid = KernelInceptionDistance(feature=_DINOv2CLS(dino_judge),
                                      subset_size=kid_subset_size).to(DEVICE)
        return fid, kid, "DINOv2"
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    kid = KernelInceptionDistance(feature=2048, subset_size=kid_subset_size,
                                  normalize=True).to(DEVICE)
    return fid, kid, "InceptionV3"


@torch.no_grad()
def dino_patch_embed(model, x):
    """DINOv2 patch tokens for a [B, 3, H, W] float batch in [0,1].

    Mirrors _DINOv2CLS's preprocessing (resize to a multiple of the patch size,
    ImageNet normalise) but returns the per-patch tokens x_norm_patchtokens
    ([B, num_patches, embed_dim]) instead of the CLS token, for the per-sample
    content-fidelity MSE.
    """
    _PATCH = 14
    x = x.float().to(next(model.parameters()).device)
    h, w = x.shape[-2:]
    h14, w14 = (h // _PATCH) * _PATCH, (w // _PATCH) * _PATCH
    if h14 != h or w14 != w:
        x = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return model.forward_features((x - mean) / std)["x_norm_patchtokens"]


def save_flux_debug(tag, seed_rgb, gen_rgb, n_samples):
    """Save [img2img seed | flux-generated] pairs for eyeballing.

    Top row: the real crop fed to Flux as the img2img seed. Bottom row: the Flux
    generation produced from that seed. Mirrors test_generation.py's conditional
    debug grid so the two generators can be compared side by side.
    """
    n = min(n_samples, seed_rgb.shape[0], gen_rgb.shape[0])
    out_dir = FIG_ROOT / "flux" / "test_generation" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5), squeeze=False)
    for j in range(n):
        axes[0, j].imshow(seed_rgb[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[1, j].imshow(gen_rgb[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Img2img seed\n(real crop)", fontsize=10)
    axes[1, 0].set_ylabel("Flux generated", fontsize=10)
    fig.suptitle(f"Flux img2img — {tag}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "flux_samples.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] saved debug samples to {out_path}")


def save_flux_uncond_debug(tag, gen_rgb, n_samples):
    """Save a single row of prompt-only FLUX generations for eyeballing.

    Unconditional counterpart to save_flux_debug: there is no img2img seed, so
    only the generated row is shown.
    """
    n = min(n_samples, gen_rgb.shape[0])
    out_dir = FIG_ROOT / "flux" / "test_generation" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5), squeeze=False)
    for j in range(n):
        axes[0, j].imshow(gen_rgb[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    axes[0, 0].set_ylabel("Flux generated\n(prompt only)", fontsize=10)
    fig.suptitle(f"Flux text-to-image — {tag}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "flux_samples.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] saved debug samples to {out_path}")


def select_campaigns(args):
    """Campaigns to evaluate: those passed via --campaigns, else all in the test split.
Returned as a *sorted* list so every rank iterates campaigns in the identical
order. This is critical under data parallelism: evaluate_pair runs collective
ops (all_reduce, fid/kid.compute) once per campaign, so if two ranks walked
the campaigns in different orders they would desync and NCCL would time out.
"""
    available = CSVSplitDataset(split="test", load_rgb=False).available_campaigns
    rprint(f"\nFound {len(available)} campaigns in the test split.")
    rprint(f"Available campaigns: {sorted(available)}")
    if args.campaigns:
        unknown = [c for c in args.campaigns if c not in available]
        if unknown:
            raise ValueError(f"Unknown campaign(s) {unknown}. Available: {sorted(available)}")
        return sorted(c for c in available if c in args.campaigns)
    return sorted(available)


def evaluate_pair(pipe, fid, kid, dino_patch_model, real_loader, seed_loader, args, tag,
                  debug_samples=0, max_per_rank=None):
    """FID/KID for one (real reference, generation seed) loader pair.
real_loader supplies the real reference images; seed_loader supplies the
generation seeds/references. Metrics are reset first so this is reusable
across campaigns. When debug_samples > 0, the first generated batch is saved
as a [seed | generated] grid under figs/flux/test_generation/<tag>/.
"""
    fid.reset()
    kid.reset()

    # Instrumentation: split time spent *blocked waiting on the DataLoader*
    # (data) from time in the H2D copy + FID/KID feature extraction (gpu). If
    # `data` dominates, the phase is I/O-bound (disk/GPFS reads or too few
    # workers); if `gpu` dominates, it's the feature extractor.
    # Under data parallelism each rank processes its own shard of the loaders;
    # tqdm/timing are shown for rank 0's shard only. FID/KID states are synced
    # across ranks in .compute() below, so the reported scores are global.
    n_real = 0
    data_wait = gpu_time = 0.0
    t_prev = time.perf_counter()
    pbar = tqdm(real_loader, desc=f"{tag} real", disable=not is_main())
    for rgb_real, _ in pbar:
        t0 = time.perf_counter()
        data_wait += t0 - t_prev
        r = rgb_real.to(DEVICE, non_blocking=True)
        if max_per_rank is not None:               # cap the real reference at the budget
            r = r[: max_per_rank - n_real]
        r = resize_batch(r, args.eval_size)        # score reals at eval_size (gen is larger)
        fid.update(r, real=True)
        kid.update(r, real=True)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_prev = time.perf_counter()
        gpu_time += t_prev - t0
        n_real += r.shape[0]
        pbar.set_postfix(data=f"{data_wait:.1f}s", gpu=f"{gpu_time:.1f}s")
        if max_per_rank is not None and n_real >= max_per_rank:
            break
    rprint(f"[{tag}] real (rank0 shard): {n_real} imgs -- dataloader wait {data_wait:.1f}s, "
           f"gpu update {gpu_time:.1f}s ({data_wait / max(n_real, 1) * 1e3:.1f} ms/img waiting)")

    n_fake = 0
    debug_saved = False
    gen_data_wait = gen_time = 0.0
    # Content-fidelity metric: cosine similarity between the average-pooled DINOv2
    # tokens of each img2img seed (the reference/source image) and the image Flux
    # generated from it (range [-1, 1], higher = more faithful to the seed).
    # Paired per sample; accumulated as sum + count for a global mean.
    pooled_cossim_sum, pooled_cossim_n = 0.0, 0
    t_prev = time.perf_counter()
    pbar = tqdm(seed_loader, desc=f"{tag} flux gen", disable=not is_main())
    for rgb_seed, _ in pbar:
        t0 = time.perf_counter()
        gen_data_wait += t0 - t_prev
        if max_per_rank is not None:               # trim before generating: bounds FLUX calls
            rgb_seed = rgb_seed[: max_per_rank - n_fake]
        gen_rgb = flux_generate(pipe, rgb_seed, args)
        fid.update(gen_rgb, real=False)
        kid.update(gen_rgb, real=False)

        # Mean-pool each image's DINO patch tokens to one [D] vector, then cosine
        # similarity between the seed's and the generation's pooled embeddings.
        # Resize the full-frame seed to eval_size so it matches gen_rgb's resolution.
        seed_ev = resize_batch(rgb_seed.to(DEVICE, non_blocking=True), args.eval_size)
        seed_pooled = dino_patch_embed(dino_patch_model, seed_ev).mean(dim=1)     # [B, D]
        gen_pooled = dino_patch_embed(dino_patch_model, gen_rgb).mean(dim=1)      # [B, D]
        cos_per_img = F.cosine_similarity(seed_pooled, gen_pooled, dim=1)     # [B] in [-1, 1]
        pooled_cossim_sum += float(cos_per_img.sum().item())
        pooled_cossim_n += cos_per_img.shape[0]

        # Debug grid only from rank 0 (avoids WORLD_SIZE writers racing the file).
        if is_main() and debug_saved is False and debug_samples > 0:
            save_flux_debug(tag, rgb_seed, gen_rgb, debug_samples)
            debug_saved = True

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t_prev = time.perf_counter()
        gen_time += t_prev - t0
        n_fake += gen_rgb.shape[0]
        pbar.set_postfix(data=f"{gen_data_wait:.1f}s", genupd=f"{gen_time:.1f}s", n_fake=n_fake)
        if max_per_rank is not None and n_fake >= max_per_rank:
            break
    rprint(f"[{tag}] gen (rank0 shard): {n_fake} imgs -- dataloader wait {gen_data_wait:.1f}s, "
           f"generate+update {gen_time:.1f}s")

    # Global sample counts (across ranks) drive the KID subset clamp and the
    # reported n_real/n_fake. fid/kid .compute() are collective: every rank must
    # call them, and each gets the synced global result.
    g_real, g_fake = n_real, n_fake
    if WORLD_SIZE > 1:
        counts = torch.tensor([n_real, n_fake], device=DEVICE)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        g_real, g_fake = int(counts[0].item()), int(counts[1].item())
        # Reduce the pooled-cossim sum/count too, so the mean is over all ranks.
        pm = torch.tensor([pooled_cossim_sum, pooled_cossim_n], device=DEVICE, dtype=torch.float64)
        dist.all_reduce(pm, op=dist.ReduceOp.SUM)
        pooled_cossim_sum, pooled_cossim_n = pm[0].item(), pm[1].item()

    out = {"n_real": g_real, "n_fake": g_fake}
    out["dino_pooled_cossim"] = pooled_cossim_sum / pooled_cossim_n if pooled_cossim_n > 0 else None
    out["fid"] = float(fid.compute().item())

    # KID needs subset_size <= #samples in each (global) distribution; clamp so
    # small campaigns don't crash. Decision uses global counts so every rank
    # takes the same branch (else the collective compute() would deadlock).
    subset = min(args.kid_subset_size, g_real, g_fake)
    out["kid_subset_size"] = subset
    if subset >= 2:
        kid.subset_size = subset
        m, s = kid.compute()
        out["kid_mean"], out["kid_std"] = float(m.item()), float(s.item())
    else:
        rprint(f"[{tag}] KID skipped (only {subset} usable samples)")
        out["kid_mean"] = out["kid_std"] = None

    kid_str = "n/a" if out["kid_mean"] is None else f"{out['kid_mean']:.5f} +/- {out['kid_std']:.5f}"
    cos_str = "n/a" if out["dino_pooled_cossim"] is None else f"{out['dino_pooled_cossim']:.5f}"
    rprint(f"[{tag}] FID={out['fid']}  KID={kid_str}  DINO_pooled_cossim={cos_str}  "
           f"(real={g_real}, fake={g_fake})")
    return out


def evaluate_unconditional(pipe, fid, kid, real_loader, args, tag, debug_samples=0):
    """FID/KID for prompt-only (text-to-image) FLUX generation vs the real test set.
There is no img2img seed and no per-sample pairing, so only FID/KID are
reported (the DINO pooled-MSE, which needs a seed<->generation pair, is
omitted). Both the real reference set and the generated set are capped at
args.n_samples total, split evenly across ranks under data parallelism.
"""
    fid.reset()
    kid.reset()

    # Per-rank budget: split n_samples across ranks (the last rank may do a few
    # extra when it doesn't divide evenly). Global counts are recombined below.
    n_per_rank = (args.n_samples + WORLD_SIZE - 1) // WORLD_SIZE

    # ---- real reference images (capped, from a shuffled all-campaign loader) ----
    n_real = 0
    pbar = tqdm(real_loader, desc=f"{tag} real", disable=not is_main())
    for rgb_real, _ in pbar:
        if n_real >= n_per_rank:
            break
        take = min(rgb_real.shape[0], n_per_rank - n_real)
        r = resize_batch(rgb_real[:take].to(DEVICE, non_blocking=True), args.eval_size)
        fid.update(r, real=True)
        kid.update(r, real=True)
        n_real += r.shape[0]
        pbar.set_postfix(n_real=n_real)

    # ---- generated images (prompt only), same per-rank budget ----
    n_fake = 0
    debug_saved = False
    gen_time = 0.0
    pbar = tqdm(total=n_per_rank, desc=f"{tag} flux gen", disable=not is_main())
    while n_fake < n_per_rank:
        bs = min(args.batch_size, n_per_rank - n_fake)
        t0 = time.perf_counter()
        gen_rgb = flux_generate_unconditional(pipe, bs, args)
        fid.update(gen_rgb, real=False)
        kid.update(gen_rgb, real=False)

        # Debug grid only from rank 0 (avoids WORLD_SIZE writers racing the file).
        if is_main() and debug_saved is False and debug_samples > 0:
            save_flux_uncond_debug(tag, gen_rgb, debug_samples)
            debug_saved = True

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        gen_time += time.perf_counter() - t0
        n_fake += gen_rgb.shape[0]
        pbar.update(gen_rgb.shape[0])
        pbar.set_postfix(genupd=f"{gen_time:.1f}s")
    pbar.close()
    rprint(f"[{tag}] gen (rank0 shard): {n_fake} imgs -- generate+update {gen_time:.1f}s")

    # Global sample counts (across ranks) drive the KID subset clamp and the
    # reported counts. fid/kid .compute() are collective: every rank must call
    # them, and each gets the synced global result.
    g_real, g_fake = n_real, n_fake
    if WORLD_SIZE > 1:
        counts = torch.tensor([n_real, n_fake], device=DEVICE)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        g_real, g_fake = int(counts[0].item()), int(counts[1].item())

    out = {"n_real": g_real, "n_fake": g_fake}
    out["fid"] = float(fid.compute().item())

    # KID needs subset_size <= #samples in each (global) distribution; clamp so a
    # small n_samples doesn't crash. Global counts keep every rank on the same
    # branch (else the collective compute() would deadlock).
    subset = min(args.kid_subset_size, g_real, g_fake)
    out["kid_subset_size"] = subset
    if subset >= 2:
        kid.subset_size = subset
        m, s = kid.compute()
        out["kid_mean"], out["kid_std"] = float(m.item()), float(s.item())
    else:
        rprint(f"[{tag}] KID skipped (only {subset} usable samples)")
        out["kid_mean"] = out["kid_std"] = None

    kid_str = "n/a" if out["kid_mean"] is None else f"{out['kid_mean']:.5f} +/- {out['kid_std']:.5f}"
    rprint(f"[{tag}] FID={out['fid']}  KID={kid_str}  (real={g_real}, fake={g_fake})")
    return out


def loader_for(split, campaign, args, shuffle=False):
    """Build a dataset + loader, sharded across ranks under data parallelism.
Each rank sees a disjoint ~1/WORLD_SIZE slice (DistributedSampler); the union
is the full split, and FID/KID recombine the shards in .compute(). shuffle
defaults off so per-rank reads stay sequential (cheap once data is on
node-local disk) and order is irrelevant to FID/KID; the unconditional mode
"""
    ds = CSVSplitDataset(split=split, load_rgb=True, campaign=campaign)
    if WORLD_SIZE > 1:
        sampler = DistributedSampler(ds, num_replicas=WORLD_SIZE, rank=RANK,
                                     shuffle=shuffle, drop_last=False)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.n_workers, pin_memory=True)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                            num_workers=args.n_workers, pin_memory=True)
    return ds, loader


def update_results_json(path, run_key, config, sections):
    """Merge this run's results into the JSON file, preserving other runs.

    `sections` holds the freshly computed pieces ({"unconditional": {...}} or
    {"per_campaign": {...}}); only those keys are overwritten within run_key, so
    re-running a mode updates just its section and leaves other runs intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {path} was not valid JSON; starting fresh.")
    entry = data.get(run_key, {})
    entry["config"] = config
    entry["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    entry.update(sections)
    data[run_key] = entry
    path.write_text(json.dumps(data, indent=2))
    print(f"\nResults written to {path} under run key '{run_key}'")


def build_flux2_pipeline(local_rank):
    """Load the 4-bit NF4 FLUX.2-dev pipeline, resident on this rank's GPU.
The transformer and the ~24B Mistral3 text encoder are quantized to 4-bit
(bitsandbytes) so the pair fits with room to spare on an 80 GB H100; the VAE
stays bf16. 4-bit params can't be .to()-moved, so each quantized component is
placed on the rank's GPU at load via device_map, and only the (movable) bf16
"""
    bnb_diffusers = DiffusersBitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    bnb_transformers = TransformersBitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    transformer = Flux2Transformer2DModel.from_pretrained(
        DEV_ID, subfolder="transformer",
        quantization_config=bnb_diffusers, torch_dtype=torch.bfloat16,
        device_map=local_rank)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        DEV_ID, subfolder="text_encoder",
        quantization_config=bnb_transformers, torch_dtype=torch.bfloat16,
        device_map=local_rank)
    pipe = Flux2Pipeline.from_pretrained(
        DEV_ID, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16)
    # transformer + text_encoder are already on the GPU (device_map above); the
    # bf16 VAE loads on CPU, so move just it (a full pipe.to() would choke on the
    # 4-bit modules). Leaves the whole pipeline resident on this rank's GPU.
    pipe.vae.to(f"cuda:{local_rank}")
    # Silence the pipeline's own per-call denoising tqdm so it doesn't interleave
    # with our real/gen progress bars.
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main(args):
    setup_distributed()
    rprint(f"Distributed: WORLD_SIZE={WORLD_SIZE} (one FLUX.2 pipeline per GPU)")

    run_uncond = args.mode in ("unconditional", "both")
    run_cond   = args.mode in ("conditional", "both")

    # One resident 4-bit FLUX.2-dev pipeline per GPU (data parallel). The same
    # Flux2Pipeline serves both modes: prompt-only for unconditional, and
    # reference-conditioned (image=[seed]) for conditional -- there is no separate
    # img2img class, and no CPU offload (see build_flux2_pipeline).
    rprint("Loading FLUX.2-dev (4-bit) pipeline, resident on each GPU...")
    pipe = build_flux2_pipeline(LOCAL_RANK)

    fid, kid, label = build_metrics(args.feature_extractor, args.kid_subset_size)
    rprint(f"Feature extractor: {label}")

    # ---- unconditional: prompt-only text-to-image vs the whole test split ----
    if run_uncond:
        # No seed<->generation pairing here, so the DINO patch-MSE doesn't apply
        # (its model isn't loaded). Reals are a shuffled sample of the whole test
        # split (campaign=None) so every campaign is represented; both the real
        # and generated sides are capped at --n_samples.
        config = {
            "model": "FLUX.2-dev-4bit",
            "feature_extractor": label,
            "mode": "unconditional",
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "gen_size": args.gen_size,
            "eval_size": args.eval_size,
            "n_samples": args.n_samples,
            "kid_subset_size": args.kid_subset_size,
            "world_size": WORLD_SIZE,
        }
        run_key = f"flux2dev_uncond_{label}_sz{args.gen_size}_ev{args.eval_size}_steps{args.steps}_cfg{args.guidance_scale}_n{args.n_samples}"
        _, test_loader = loader_for("test", None, args, shuffle=True)
        section = evaluate_unconditional(
            pipe, fid, kid, test_loader, args, tag="unconditional",
            debug_samples=args.debug_samples)
        if is_main():
            update_results_json(RESULTS_JSON, run_key, config, {"unconditional": section})

    # ---- conditional: per-campaign img2img, test seeds vs the same test reals --
    if run_cond:
        # Dedicated DINOv2 for the per-sample patch-embedding MSE, independent of
        # the FID/KID feature extractor above. Re-homed onto this rank's DEVICE.
        dino_patch_model = load_dinov2().to(DEVICE)

        config = {
            "model": "FLUX.2-dev-4bit",
            "feature_extractor": label,
            "mode": "conditional",
            "gen_size": args.gen_size,
            "eval_size": args.eval_size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "per_campaign_size": args.per_campaign_size,
            "kid_subset_size": args.kid_subset_size,
            "world_size": WORLD_SIZE,
        }
        run_key = f"flux2dev_{label}_sz{args.gen_size}_ev{args.eval_size}_steps{args.steps}_cfg{args.guidance_scale}_pcs{args.per_campaign_size}"

        # Per-campaign budget, split across ranks. Capping the (expensive) FLUX
        # generations keeps each campaign bounded; shuffle the loader so the capped
        # subset is drawn from across the campaign, not the first deployments in
        # CSV order. DistributedSampler(shuffle=True) uses a fixed epoch-0 seed, so
        # the real and generation passes over the same loader see the same subset.
        cap_per_rank = (args.per_campaign_size + WORLD_SIZE - 1) // WORLD_SIZE
        campaigns = select_campaigns(args)
        rprint(f"\nConditional per-campaign eval over {len(campaigns)} campaigns "
               f"(<= {args.per_campaign_size} imgs each): {campaigns}")
        per_campaign = {}
        for campaign in campaigns:
            # Seeds and reals are both this campaign's test crops (references
            # scored against the same test images), so one loader serves as both.
            ds, loader = loader_for("test", campaign, args, shuffle=True)
            # Skip decision is on the full dataset size (identical across ranks),
            # so all ranks stay in lockstep for the collective compute() calls.
            if len(ds) < 2:
                rprint(f"[skip] {campaign}: only {len(ds)} test images")
                continue
            per_campaign[campaign] = evaluate_pair(
                pipe, fid, kid, dino_patch_model, loader, loader, args, tag=campaign,
                debug_samples=args.debug_samples, max_per_rank=cap_per_rank)
        if is_main():
            update_results_json(RESULTS_JSON, run_key, config, {"per_campaign": per_campaign})

    if WORLD_SIZE > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Evaluate FLUX.2-dev (4-bit) reef variations (FID/KID) on the test split: "
                    "per-campaign reference-conditioned (conditional) and/or prompt-only (unconditional).")

    argparser.add_argument("--mode", type=str, default="both",
                           choices=["conditional", "both", "unconditional"],
                           help="conditional: per-campaign reference-conditioned, that campaign's test "
                                "crops as references scored against the same test images (FID/KID + DINO "
                                "cossim). unconditional: prompt-only vs a random --n_samples sample of "
                                "the whole test split (FID/KID only, no reference). both: run both.")
    argparser.add_argument("--batch_size", type=int, default=8,
                           help="Unconditional: images generated per pipeline call (num_images_per_prompt). "
                                "Conditional: DataLoader/metric batch only -- FLUX.2 generates one image "
                                "per reference regardless.")
    argparser.add_argument("--n_workers", type=int, default=4, help="Number of DataLoader workers.")
    argparser.add_argument("--strength", type=float, default=0.60,
                           help="Ignored: FLUX.2 has no img2img strength knob. Kept only so existing "
                                "launch scripts that pass --strength still parse.")
    argparser.add_argument("--steps", type=int, default=28, help="Number of FLUX.2 denoising steps.")
    argparser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale (FLUX.2 default 4.0).")
    argparser.add_argument("--n_samples", type=int, default=1000,
                           help="Unconditional mode only: number of images to draw from both the "
                                "model and the test set for FID/KID (split across ranks).")
    argparser.add_argument("--per_campaign_size", type=int, default=500,
                           help="Conditional mode only: max real + generated images per campaign "
                                "(split across ranks). Caps the number of FLUX generations so the "
                                "per-campaign eval stays bounded.")
    argparser.add_argument("--gen_size", type=int, default=512,
                           help="Square resolution FLUX.2 generates at (multiple of 16), for both modes. "
                                "512 is a speed/quality sweet spot -- FLUX.2 is trained ~1MP and degrades "
                                "well below ~512, so don't set this to 224.")
    argparser.add_argument("--eval_size", type=int, default=224,
                           help="Resolution at which FID/KID/DINO are scored: reals and generations are "
                                "both resized to this before the metrics (decoupled from --gen_size, which "
                                "stays high for generation quality).")
    argparser.add_argument("--feature_extractor", type=str, default="dino", choices=["dino", "inception"],
                           help="Feature extractor for FID/KID: DINOv2 CLS token (dino) or standard InceptionV3 (inception).")
    argparser.add_argument("--kid_subset_size", type=int, default=1000,
                           help="KID subset size per resample; clamped down to the available sample count per tag.")
    argparser.add_argument("--campaigns", type=str, nargs="+", default=None,
                           help="Space-separated campaigns to evaluate (conditional mode). "
                                "Default: all campaigns in the test split.")
    argparser.add_argument("--debug_samples", type=int, default=4,
                           help="Per campaign (conditional mode), save this many [reference | "
                                "generated] pairs to figs/flux/test_generation/<campaign>/. 0 disables.")

    args = argparser.parse_args()
    main(args)
