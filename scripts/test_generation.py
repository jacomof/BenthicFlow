# scripts/test_panorama.py
"""Evaluate generated images with the Fréchet Inception Distance (FID) and
Kernel Inception Distance (KID) metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import datetime
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance
from tqdm import tqdm

from benthicflow import (
    CKPT_ROOT,
    DEPTH_ROOT,
    FEAT_ROOT,
    FIG_ROOT,
    PROJECT_ROOT,
    RGB_ROOT,
    SCRATCH_ROOT,
)
from models.unet_cfm import UNetCFM, cfm_sample_cfg
from scripts.extract_depth import load_model as load_dav2
from scripts.extract_depth import normalize_depth
from scripts.extract_features import DirectoryImagesNative, load_dinov2
from scripts.train_cfm_cfg import load_rae_frozen

# Merged results file, mirroring flux_test_generation.py's flux_eval_results.json.
RESULTS_JSON = FIG_ROOT / "cfm" / "test_generation" / "eval_results.json"

NATIVE_SIZE = 518
TRAIN_SIZE = 224
PATCH_SIZE = 14
NATIVE_GRID = NATIVE_SIZE // PATCH_SIZE
TRAIN_GRID = TRAIN_SIZE // PATCH_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(z_norm, model, rae):
    z = model.denormalize(z_norm)
    rgbd = rae.decoder(z.permute(0, 2, 3, 1).contiguous())  # [n, 4, 224, 224]
    return rgbd[:, :3].clamp(0, 1), rgbd[:, 3:4].clamp(0, 1)


class CSVSplitDataset(Dataset):
    def __init__(
        self,
        split: str,
        load_rgb: bool = False,
        campaign: str = None,
        return_full_rgb: bool = False,
    ):
        self.train = split == "train"
        # RGB is only needed for the generation reference grid, not for CFM
        # training/eval. Default off so we don't pay the (large) RGB read per step.
        self.load_rgb = load_rgb
        # When on, each item additionally returns the *uncropped* 518x518 uint8 RGB
        # plus the (y0, x0) crop offsets that were applied, so an eval can recompute
        # protocol-matched quantities (e.g. DAv2 depth over the full view, then the
        # exact same crop). Free I/O-wise: the full RGB slice is read anyway.
        self.return_full_rgb = return_full_rgb
        self.records = []
        self._depth_mmap_paths = {}
        self._dino_mmap_paths = {}
        self._rgb_mmap_paths = {}
        self._worker_mmaps = {}

        csv_path = PROJECT_ROOT / "data_split" / f"{split}.csv"
        depth_key_cache = {}

        available_campaigns = set()

        with csv_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                # NB: use row-local names here. Reusing `campaign` would shadow
                # the constructor's `campaign` filter argument and leave it set
                # to the last row's campaign, silently breaking per-campaign
                # selection (every campaign would resolve to the same subset).
                row_campaign, deployment, rel_path = (
                    row.get("campaign"),
                    row.get("deployment"),
                    row.get("path"),
                )
                if not all([row_campaign, deployment, rel_path]):
                    raise ValueError(f"Missing required fields in CSV row: {row}")

                available_campaigns.add(row_campaign)
                dep_key = (row_campaign, deployment)
                depth_npy = DEPTH_ROOT / row_campaign / f"{deployment}.npy"
                keys_npy = DEPTH_ROOT / row_campaign / f"{deployment}_keys.npy"
                dino_npy = FEAT_ROOT / row_campaign / f"{deployment}.npy"
                rgb_npy = RGB_ROOT / row_campaign / f"{deployment}.npy"

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
                            row_campaign,
                            deployment,
                            rel_path,
                            depth_key_cache[dep_key][filename],
                        )
                    )

        # Optionally restrict to a single campaign (per-campaign conditional FID).
        if campaign is not None:
            self.records = [r for r in self.records if r[0] == campaign]

        # print(f"Available campaigns are {available_campaigns}, split '{split}' has {len(self.records)} records, load_rgb={self.load_rgb}")

        self.available_campaigns = available_campaigns
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
        full_rgb = torch.empty(0)
        if self.load_rgb or self.return_full_rgb:
            rgb_slice = np.array(rgb_arr[idx], copy=True)  # [S, S, 3] uint8
            if self.load_rgb:
                rgb = torch.from_numpy(rgb_slice).permute(2, 0, 1).float().div_(255.0)
            if self.return_full_rgb:
                # Uncropped native view, kept uint8 (PIL-ready, 4x lighter to collate).
                full_rgb = torch.from_numpy(rgb_slice)  # [S, S, 3] uint8

        # Grid-Snapped Crop logic
        if self.train:
            y_patch = int(torch.randint(0, NATIVE_GRID - TRAIN_GRID + 1, (1,)).item())
            x_patch = int(torch.randint(0, NATIVE_GRID - TRAIN_GRID + 1, (1,)).item())
            if torch.rand(1).item() < 0.5:
                depth = TF.hflip(depth)
                dino = torch.flip(dino, dims=[1])
                if self.load_rgb:
                    rgb = TF.hflip(rgb)
                if self.return_full_rgb:
                    # Flip the full view too so the crop offsets below stay valid
                    # in the returned image's frame.
                    full_rgb = torch.flip(full_rgb, dims=[1])
        else:
            y_patch = x_patch = (NATIVE_GRID - TRAIN_GRID) // 2

        y0, x0 = y_patch * PATCH_SIZE, x_patch * PATCH_SIZE
        depth = TF.crop(depth, y0, x0, TRAIN_SIZE, TRAIN_SIZE)
        dino = dino[y_patch : y_patch + TRAIN_GRID, x_patch : x_patch + TRAIN_GRID, :]
        if self.load_rgb:
            rgb = TF.crop(rgb, y0, x0, TRAIN_SIZE, TRAIN_SIZE)

        if self.return_full_rgb:
            # Crop offsets (in the returned full_rgb's frame) so evals can apply
            # the *exact* crop this sample got to any full-view recomputation.
            return rgb, depth, dino, full_rgb, torch.tensor([y0, x0])
        return rgb, depth, dino


class _DINOv2CLS(torch.nn.Module):
    """Thin wrapper so torchmetrics FID can use DINOv2 CLS tokens as features.

    torchmetrics probes the extractor at init with a 299x299 dummy image.
    DINOv2 requires H and W to be multiples of its patch size (14), so we
    resize any off-grid input before forwarding.
    """

    _PATCH = 14

    def __init__(self, model):
        super().__init__()
        self.model = model

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # probe dummy from torchmetrics is uint8 on CPU; real images are float on DEVICE
        x = x.float().to(next(self.model.parameters()).device)
        h, w = x.shape[-2:]
        h14, w14 = (h // self._PATCH) * self._PATCH, (w // self._PATCH) * self._PATCH
        if h14 != h or w14 != w:
            x = torch.nn.functional.interpolate(
                x, size=(h14, w14), mode="bilinear", align_corners=False
            )
        mean = self._MEAN.to(x.device)
        std = self._STD.to(x.device)
        return self.model.forward_features((x - mean) / std)["x_norm_clstoken"]


@torch.no_grad()
def dino_patch_embed(model, x):
    """DINOv2 patch tokens for a [B, 3, H, W] float batch in [0,1].
    Mirrors _DINOv2CLS's preprocessing (resize to a multiple of the patch size,
    ImageNet normalise) but returns the per-patch tokens x_norm_patchtokens
    ([B, num_patches, embed_dim]) instead of the CLS token. Callers mean-pool
    these to one [D] vector per image for the content-fidelity cosine similarity
    """
    _PATCH = 14
    x = x.float().to(next(model.parameters()).device)
    h, w = x.shape[-2:]
    h14, w14 = (h // _PATCH) * _PATCH, (w // _PATCH) * _PATCH
    if h14 != h or w14 != w:
        x = torch.nn.functional.interpolate(
            x, size=(h14, w14), mode="bilinear", align_corners=False
        )
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return model.forward_features((x - mean) / std)["x_norm_patchtokens"]


class DepthConsistency:
    """Self-consistency between the generated depth channel and Depth-Anything-V2's
    depth estimate on the *generated* RGB image.
    The generator emits an RGB-D sample; a physically coherent sample should have
    a depth channel that agrees with what a monocular depth estimator would infer
    from the RGB alone. We run DAv2 (the same variant/output space used to build
    """

    def __init__(self, dav2_run, size=TRAIN_SIZE):
        self.run = dav2_run
        self.size = size
        self.reset()

    def reset(self):
        self.si_rmse_sum = 0.0
        self.pearson_sum = 0.0
        self.target_std_sum = 0.0
        self.n = 0

    def _predict(self, gen_rgb):
        """DAv2 raw predicted_depth for a [B, 3, H, W] float batch, as [B, S, S]."""
        pil = [TF.to_pil_image(img.detach().cpu().clamp(0, 1)) for img in gen_rgb]
        preds = self.run(pil)  # list of HxW float32 np arrays (raw predicted_depth)
        out = []
        for p in preds:
            p = np.squeeze(
                np.ascontiguousarray(p)
            )  # drop any leading singleton dims -> [H, W]
            t = torch.from_numpy(np.ascontiguousarray(p)).float()[None, None]
            t = torch.nn.functional.interpolate(
                t, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
            out.append(t[0, 0])
        return torch.stack(out)

    @torch.no_grad()
    def update(self, gen_rgb, gen_depth, pred=None):
        """Score gen_depth against DAv2 on gen_rgb.

        `pred` ([B, S, S] or [B, 1, S, S]) overrides the internal DAv2 forward:
        pass it to reuse an already-computed prediction, or to score against a
        reference depth obtained under a different protocol (e.g. the full-view
        recomputed targets from dav2_fullview_depth_crop). With `pred` given,
        gen_rgb is ignored.
        """
        if pred is None:
            pred = self._predict(gen_rgb)
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        pred = pred.to(gen_depth.device)  # [B, S, S]
        target = gen_depth.detach().squeeze(1).to(pred.dtype)  # [B, S, S]
        b = pred.shape[0]
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)

        eps = 1e-8
        mp = p.mean(dim=1, keepdim=True)
        mt = t.mean(dim=1, keepdim=True)
        pc, tc = p - mp, t - mt
        var_p = (pc * pc).sum(dim=1)
        var_t = (tc * tc).sum(dim=1)
        cov = (pc * tc).sum(dim=1)

        # Least-squares affine alignment of pred onto target, then RMSE.
        s = cov / var_p.clamp_min(eps)  # [B]
        shift = mt.squeeze(1) - s * mp.squeeze(1)  # [B]
        aligned = s[:, None] * p + shift[:, None]
        rmse = torch.sqrt(((aligned - t) ** 2).mean(dim=1))  # [B]

        pearson = cov / (var_p.sqrt() * var_t.sqrt() + eps)  # [B] in [-1, 1]

        self.si_rmse_sum += float(rmse.sum().item())
        self.pearson_sum += float(pearson.sum().item())
        self.target_std_sum += float((var_t / t.shape[1]).sqrt().sum().item())
        self.n += b

    def compute(self, suffix=""):
        """{'depth_si_rmse', 'depth_pearson', 'depth_std'} (+suffix on each key);
        None values if nothing was seen. Pass suffix='_real' for the real-pair
        ceiling so it can live beside the generated scores in one dict."""
        keys = [
            f"depth_si_rmse{suffix}",
            f"depth_pearson{suffix}",
            f"depth_std{suffix}",
        ]
        if self.n == 0:
            return dict.fromkeys(keys)
        return dict(
            zip(
                keys,
                [
                    self.si_rmse_sum / self.n,
                    self.pearson_sum / self.n,
                    self.target_std_sum / self.n,
                ],
            )
        )


@torch.no_grad()
def dav2_fullview_depth_crop(dav2_run, full_rgb_uint8, crop_yx):
    """Recompute the training depth targets on the fly, protocol-matched.
    Reproduces extract_depth.py's target pipeline exactly: DAv2 on the *entire*
    518x518 view, per-image min-max normalisation over the full view
    (normalize_depth, i.e. normalise-before-crop as in training), then the same
    crop the dataloader applied to this sample (crop_yx from
    """
    pil = [Image.fromarray(img.numpy()) for img in full_rgb_uint8]
    preds = dav2_run(pil)  # list of HxW float32 arrays (raw predicted_depth)
    out = []
    for p, (y0, x0) in zip(preds, crop_yx.tolist()):
        d = normalize_depth(np.squeeze(np.asarray(p)))  # [518, 518] in [0, 1]
        t = torch.from_numpy(d)[y0 : y0 + TRAIN_SIZE, x0 : x0 + TRAIN_SIZE]
        out.append(t)
    return torch.stack(out)


def generate_batch_unconditionally(cfm_model, rae_model, batch_size, n_steps):
    null = cfm_model.null_cond.unsqueeze(0).expand(batch_size, -1)  # [n, 768]
    z_cond = cfm_sample_cfg(cfm_model, null, n_steps=n_steps)
    gen_image_batch, gen_depth_batch = decode(z_cond, cfm_model, rae_model)
    return gen_image_batch, gen_depth_batch


def generate_batch_conditionally(cfm_model, rae_model, cond, n_steps):
    # cond: [B, 768] global DINO vector, the train-time conditioning
    # (dino grid mean-pooled over space; see train_cfm_cfg.py).
    z_cond = cfm_sample_cfg(cfm_model, cond, n_steps=n_steps)
    gen_image_batch, gen_depth_batch = decode(z_cond, cfm_model, rae_model)
    return gen_image_batch, gen_depth_batch


class GenerationMetrics:
    """FID + (optionally) KID sharing a single feature extractor.
    Both torchmetrics metrics take the same `feature` argument (an int for the
    stock InceptionV3, or an nn.Module for a custom extractor), so we build them
    from the same extractor and drive them through one reset/update/compute API.
    This keeps the eval loops metric-agnostic: they update once and get both
    """

    def __init__(
        self, feature, extractor_label, normalize, with_kid=True, kid_subset_size=50
    ):
        self.extractor_label = extractor_label
        self.kid_subset_size = kid_subset_size
        self.fid = FrechetInceptionDistance(feature=feature, normalize=normalize).to(
            DEVICE
        )
        self.kid = (
            KernelInceptionDistance(
                feature=feature, normalize=normalize, subset_size=kid_subset_size
            ).to(DEVICE)
            if with_kid
            else None
        )

    def reset(self):
        self.fid.reset()
        if self.kid is not None:
            self.kid.reset()

    def update(self, imgs, real):
        self.fid.update(imgs, real=real)
        if self.kid is not None:
            self.kid.update(imgs, real=real)

    def compute(self, n_real=None, n_fake=None):
        """Return a JSON-friendly dict of scores + sample counts.

        Schema mirrors flux_test_generation.py: {'fid', 'kid_mean', 'kid_std',
        'kid_subset_size', 'n_real', 'n_fake'} with None where a metric could
        not be computed. KID's subset_size is clamped to the available sample
        count (KID needs `subset_size <= min(#real,#fake)`) so a small campaign
        can't crash the run; below 2 usable samples KID is skipped entirely.
        """
        out = {
            "fid": None,
            "kid_mean": None,
            "kid_std": None,
            "kid_subset_size": None,
            "n_real": n_real,
            "n_fake": n_fake,
        }
        try:
            out["fid"] = float(self.fid.compute().item())
        except Exception as e:
            print(f"[warn] FID compute failed: {e}")

        if self.kid is not None:
            subset = self.kid_subset_size
            if n_real is not None and n_fake is not None:
                subset = min(subset, n_real, n_fake)
            out["kid_subset_size"] = subset
            if subset >= 2:
                self.kid.subset_size = subset
                try:
                    m, s = self.kid.compute()
                    out["kid_mean"], out["kid_std"] = float(m.item()), float(s.item())
                except Exception as e:
                    print(f"[warn] KID compute failed: {e}")
            else:
                print(f"[warn] KID skipped (only {subset} usable samples)")
        return out


def format_metrics(m):
    """One-line 'FID x | KID mean ± std' summary from a compute() dict."""
    fid = "n/a" if m.get("fid") is None else f"{m['fid']:.4f}"
    parts = [f"FID {fid}"]
    if m.get("kid_mean") is not None:
        parts.append(f"KID {m['kid_mean']:.5f} ± {m['kid_std']:.5f}")
    if m.get("dino_pooled_cossim") is not None:
        parts.append(f"DINO_pooled_cossim {m['dino_pooled_cossim']:.5f}")
    if m.get("dino_pooled_mse") is not None:
        parts.append(f"DINO_pooled_mse {m['dino_pooled_mse']:.5f}")
    if m.get("depth_si_rmse") is not None:
        parts.append(f"depth_SIRMSE {m['depth_si_rmse']:.4f}")
    if m.get("depth_pearson") is not None:
        parts.append(f"depth_r {m['depth_pearson']:.4f}")
    if m.get("depth_std") is not None:
        parts.append(f"depth_std {m['depth_std']:.4f}")
    # Real-pair protocol ceiling (DAv2 on real RGB vs stored depth): the best
    # score achievable under this protocol -- compare depth_r against it, not 1.0.
    if m.get("depth_pearson_real") is not None:
        parts.append(f"depth_r_real {m['depth_pearson_real']:.4f}")
    # Extraction-pipeline sanity (recomputed full-view reference vs stored crop).
    if m.get("depth_pearson_ref_check") is not None:
        parts.append(f"depth_r_refcheck {m['depth_pearson_ref_check']:.4f}")
    return "  |  ".join(parts)


def print_campaign_summary(title, results):
    """Print a per-campaign table plus per-metric means across campaigns."""
    print(f"\n=== {title} ===")
    if not results:
        print("(no campaigns evaluated)")
        return
    width = max(len(c) for c in results)
    for campaign, m in sorted(results.items()):
        print(f"{campaign:>{width}}: {format_metrics(m)}")
    mean_parts = []
    fids = [m["fid"] for m in results.values() if m.get("fid") is not None]
    kids = [m["kid_mean"] for m in results.values() if m.get("kid_mean") is not None]
    cossims = [
        m["dino_pooled_cossim"]
        for m in results.values()
        if m.get("dino_pooled_cossim") is not None
    ]
    mses = [
        m["dino_pooled_mse"]
        for m in results.values()
        if m.get("dino_pooled_mse") is not None
    ]
    sirmses = [
        m["depth_si_rmse"]
        for m in results.values()
        if m.get("depth_si_rmse") is not None
    ]
    depth_rs = [
        m["depth_pearson"]
        for m in results.values()
        if m.get("depth_pearson") is not None
    ]
    depth_r_reals = [
        m["depth_pearson_real"]
        for m in results.values()
        if m.get("depth_pearson_real") is not None
    ]
    if fids:
        mean_parts.append(f"FID {sum(fids) / len(fids):.4f}")
    if kids:
        mean_parts.append(f"KID {sum(kids) / len(kids):.5f}")
    if cossims:
        mean_parts.append(f"DINO_pooled_cossim {sum(cossims) / len(cossims):.5f}")
    if mses:
        mean_parts.append(f"DINO_pooled_mse {sum(mses) / len(mses):.5f}")
    if sirmses:
        mean_parts.append(f"depth_SIRMSE {sum(sirmses) / len(sirmses):.4f}")
    if depth_rs:
        mean_parts.append(f"depth_r {sum(depth_rs) / len(depth_rs):.4f}")
    if depth_r_reals:
        mean_parts.append(f"depth_r_real {sum(depth_r_reals) / len(depth_r_reals):.4f}")
    print(f"{'MEAN':>{width}}: {'  |  '.join(mean_parts)}")


def update_results_json(path, run_key, config, sections):
    """Merge this run's results into the JSON file, preserving other runs.

    `sections` holds the freshly computed pieces (e.g. {"unconditional": {...}},
    {"conditional": {...}}, {"gold": {...}}); only those keys are overwritten
    within run_key, so running --mode unconditional then --mode gold accumulates
    into one entry. Mirrors flux_test_generation.py's writer.
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


def build_metrics(args):
    """Build the FID(+KID) bundle with the chosen feature extractor."""
    with_kid = not args.skip_kid
    if args.feature_extractor == "dino":
        dino_judge = load_dinov2()
        return GenerationMetrics(
            feature=_DINOv2CLS(dino_judge),
            extractor_label="DINOv2",
            normalize=False,
            with_kid=with_kid,
            kid_subset_size=args.kid_subset_size,
        )
    return GenerationMetrics(
        feature=2048,
        extractor_label="InceptionV3",
        normalize=True,
        with_kid=with_kid,
        kid_subset_size=args.kid_subset_size,
    )


def evaluate_unconditional(model, rae, metrics, args, depth_consistency=None):
    """FID/KID between unconditional CFM samples and a matched sample of the test split.
    Both the real reference and the generated set are capped at --sample_size
    (the same global cap gold uses), so this no longer scans the whole test split;
    a matched batch of samples is generated for each real batch until the budget
    is hit.
    """
    metrics.reset()
    real_depth_consistency = None
    if depth_consistency is not None:
        depth_consistency.reset()
        real_depth_consistency = DepthConsistency(depth_consistency.run)
    test_dataset = CSVSplitDataset(split="test", load_rgb=True)
    # shuffle=True: we cap at sample_size and break early, so the subset must be
    # representative across the split rather than the first-N (a single deployment
    # in CSV order). The capped count is small, so the random-read cost is minor.
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    gen_time, update_time = 0.0, 0.0
    n_real = n_fake = 0
    with torch.no_grad():
        n_batches = (args.sample_size + args.batch_size - 1) // args.batch_size
        pbar = tqdm(test_loader, desc="Unconditional FID", total=n_batches)
        for rgb_real, depth_real, _ in pbar:
            # Trim the last batch so real + generated land exactly on sample_size.
            take = min(rgb_real.shape[0], args.sample_size - n_real)
            real_rgb = rgb_real[:take].to(DEVICE)
            real_depth = depth_real[:take].to(DEVICE)

            torch.cuda.synchronize()
            t0 = time.time()
            gen_rgb, gen_depth = generate_batch_unconditionally(
                model, rae, batch_size=real_rgb.shape[0], n_steps=args.n_steps
            )
            torch.cuda.synchronize()
            t1 = time.time()

            metrics.update(real_rgb, real=True)
            metrics.update(gen_rgb, real=False)
            if depth_consistency is not None:
                depth_consistency.update(gen_rgb, gen_depth)
                # Real-pair ceiling: DAv2 on the real crop vs its stored depth.
                real_depth_consistency.update(real_rgb, real_depth)
            torch.cuda.synchronize()
            t2 = time.time()

            n_real += real_rgb.shape[0]
            n_fake += gen_rgb.shape[0]
            gen_time += t1 - t0
            update_time += t2 - t1
            pbar.set_postfix(gen=f"{gen_time:.1f}s", update=f"{update_time:.1f}s")
            if n_real >= args.sample_size:
                break

    score = metrics.compute(n_real, n_fake)
    if depth_consistency is not None:
        score.update(depth_consistency.compute())
        score.update(real_depth_consistency.compute(suffix="_real"))
    print(f"Unconditional -- {format_metrics(score)}  (real={n_real}, fake={n_fake})")
    print(
        f"Total time -- generation: {gen_time:.1f}s, metric update "
        f"({metrics.extractor_label} + stats): {update_time:.1f}s"
    )
    return score


def save_conditional_debug(campaign, rgb_src, gen_rgb, n_samples):
    """Save [conditioning source | generated] pairs for eyeballing.

    Top row: the train RGB crop whose (mean-pooled) DINO grid conditioned
    generation. Bottom row: the generated image for that condition. RGB and
    DINO share the same crop/flip, so the source is what the condition "saw".
    """
    n = min(n_samples, rgb_src.shape[0], gen_rgb.shape[0])
    out_dir = FIG_ROOT / "cfm" / "test_generation" / campaign
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5), squeeze=False)
    for j in range(n):
        axes[0, j].imshow(rgb_src[j].permute(1, 2, 0).cpu().numpy())
        axes[1, j].imshow(gen_rgb[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Conditioning\n(train RGB)", fontsize=10)
    axes[1, 0].set_ylabel("Generated", fontsize=10)
    fig.suptitle(f"Conditional generation — {campaign}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "conditional_samples.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{campaign}] saved debug samples to {out_path}")


def save_rae_reconstruction_debug(
    campaign, rgb_src, recon_rgb, depth_src, recon_depth, n_samples
):
    """Save [source | RAE reconstruction] columns (RGB and depth) for eyeballing.

    Rows: source RGB, RAE-reconstructed RGB, source depth, RAE-reconstructed depth.
    This is the RAE round-trip with no flow matching, so the columns show how
    faithfully the frozen autoencoder reproduces each modality -- the visual
    counterpart of the RAE reconstruction FID/KID and dino_pooled_mse. Depth uses
    the repo's turbo/[0,1] convention (see visualize_depth.py).
    """
    n = min(n_samples, rgb_src.shape[0])
    out_dir = FIG_ROOT / "cfm" / "test_generation" / campaign
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12.5), squeeze=False)
    for j in range(n):
        axes[0, j].imshow(rgb_src[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[1, j].imshow(recon_rgb[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[2, j].imshow(depth_src[j, 0].cpu().numpy(), cmap="turbo", vmin=0, vmax=1)
        axes[3, j].imshow(
            recon_depth[j, 0].clamp(0, 1).cpu().numpy(), cmap="turbo", vmin=0, vmax=1
        )
        for r in range(4):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Source RGB", fontsize=10)
    axes[1, 0].set_ylabel("RAE RGB", fontsize=10)
    axes[2, 0].set_ylabel("Source depth", fontsize=10)
    axes[3, 0].set_ylabel("RAE depth", fontsize=10)
    fig.suptitle(f"RAE reconstruction (no flow matching) — {campaign}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "rae_reconstruction_samples.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{campaign}] saved RAE reconstruction samples to {out_path}")


def select_campaigns(args):
    """Campaigns to evaluate: those passed via --campaigns, else all in the test split.

    Sorted for a deterministic, reproducible campaign order (available_campaigns
    is a set, whose iteration order varies per process under hash randomization).
    """
    available = CSVSplitDataset(split="test", load_rgb=False).available_campaigns
    print(f"Available campaigns: {sorted(available)}")
    if args.campaigns:
        unknown = [c for c in args.campaigns if c not in available]
        if unknown:
            raise ValueError(
                f"Unknown campaign(s) {unknown}. Available: {sorted(available)}"
            )
        return sorted(c for c in available if c in args.campaigns)
    return sorted(available)


def evaluate_conditional_per_campaign(
    model, rae, metrics, args, depth_consistency=None
):
    """Per-campaign conditional FID/KID, plus the folded-in RAE-reconstruction floor.
    For each campaign: condition generation on that campaign's *train* DINO
    features and compare the generated images against that campaign's *test*
    images. A high per-campaign FID relative to the unconditional FID signals
    the model is not honouring the campaign conditioning.
    """
    campaigns = select_campaigns(args)
    cap_str = (
        "full split"
        if args.per_campaign_size is None
        else f"<= {args.per_campaign_size}"
    )
    print(
        f"\nConditional + RAE-reconstruction per-campaign FID/KID over {len(campaigns)} "
        f"campaigns ({cap_str} samples each): {campaigns}"
    )

    # One DINOv2 patch model serves both content metrics: the conditional
    # dino_pooled_cossim (source vs generation) and the RAE dino_pooled_mse
    # (source vs reconstruction).
    dino_patch_model = load_dinov2().to(DEVICE)

    # Second FID/KID bundle for the RAE round-trip (real = source, fake = recon):
    # it accumulates a different real/fake distribution than the conditional
    # bundle, so it cannot share `metrics`. The DAv2 estimator is reused.
    rae_metrics = build_metrics(args)
    # Three depth scorers beyond the CFM self-consistency one (all share the DAv2 run):
    #  * rae_depth_consistency -- RAE recon depth vs the protocol-matched reference
    #    (DAv2 on the full 518 source view, normalised over the full view, then the
    #    dataloader's exact crop; see dav2_fullview_depth_crop). Same FOV and
    #    normalisation as training targets, so read this against ~1.0.
    #  * ref_check_depth_consistency -- recomputed reference vs the stored depth
    #    crop from disk. Pure pipeline sanity check: expect r ~= 1; anything lower
    #    means extraction/preprocessing drift between DEPTH_ROOT and RGB_ROOT.
    #  * real_depth_consistency -- DAv2 on the real 224 crop vs the stored depth
    #    (crop-FOV protocol ceiling). This is the reference level for the CFM
    #    *generated* scores, which can only be checked crop-FOV-wise because a
    #    generated sample has no full 518 view.
    rae_depth_consistency = (
        DepthConsistency(depth_consistency.run)
        if depth_consistency is not None
        else None
    )
    ref_check_depth_consistency = (
        DepthConsistency(depth_consistency.run)
        if depth_consistency is not None
        else None
    )
    real_depth_consistency = (
        DepthConsistency(depth_consistency.run)
        if depth_consistency is not None
        else None
    )

    results, rae_results = {}, {}
    for campaign in campaigns:
        # Train drives the conditioning (dino); test is the real reference. Both
        # resolve to this campaign's test split; the source RGB doubles as the RAE
        # reconstruction reference and the debug-grid / content-metric source.
        # return_full_rgb: the gen+rae pass also gets the uncropped 518 view and the
        # crop offsets, to recompute the protocol-matched DAv2 depth reference.
        train_ds = CSVSplitDataset(
            split="test", load_rgb=True, campaign=campaign, return_full_rgb=True
        )
        test_ds = CSVSplitDataset(split="test", load_rgb=True, campaign=campaign)
        if len(train_ds) == 0 or len(test_ds) < 2:
            print(
                f"[skip] {campaign}: train={len(train_ds)} test={len(test_ds)} (need train>0, test>=2)"
            )
            continue
        # shuffle=False: the full campaign split is scanned and FID/KID are
        # order-independent; sequential reads keep the big RGB mmaps cheap.
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            pin_memory=True,
        )

        metrics.reset()
        rae_metrics.reset()
        if depth_consistency is not None:
            depth_consistency.reset()
        if rae_depth_consistency is not None:
            rae_depth_consistency.reset()
            ref_check_depth_consistency.reset()
        if real_depth_consistency is not None:
            real_depth_consistency.reset()
        with torch.no_grad():
            # ---- real reference for the conditional bundle ----
            # Split DataLoader-wait from GPU time so a slow real phase can be
            # attributed to I/O vs feature extraction (see flux_test_generation).
            n_real = 0
            data_wait = gpu_time = 0.0
            t_prev = time.perf_counter()
            pbar = tqdm(test_loader, desc=f"{campaign} real")
            for rgb_real, depth_real, _ in pbar:
                t0 = time.perf_counter()
                data_wait += t0 - t_prev
                rgb_real = rgb_real.to(DEVICE, non_blocking=True)
                metrics.update(rgb_real, real=True)
                if real_depth_consistency is not None:
                    # Real-pair depth-consistency ceiling (stored depth target).
                    real_depth_consistency.update(
                        rgb_real, depth_real.to(DEVICE, non_blocking=True)
                    )
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t_prev = time.perf_counter()
                gpu_time += t_prev - t0
                n_real += rgb_real.shape[0]
                pbar.set_postfix(data=f"{data_wait:.1f}s", gpu=f"{gpu_time:.1f}s")
                if (
                    args.per_campaign_size is not None
                    and n_real >= args.per_campaign_size
                ):
                    break
            print(
                f"[{campaign}] real: {n_real} imgs -- dataloader wait {data_wait:.1f}s, "
                f"gpu update {gpu_time:.1f}s"
            )

            # ---- one source pass: CFM generation + RAE reconstruction ----
            n_fake = 0
            debug_saved = False
            # Content-fidelity metrics, both mean-pooling the source's DINOv2 patch
            # tokens once: cosine similarity vs the CFM generation (higher = more
            # faithful to the condition) and MSE vs the RAE reconstruction.
            pooled_cossim_sum, pooled_cossim_n = 0.0, 0
            pooled_cossim_sum_rae, pooled_cossim_n_rae = 0.0, 0
            for rgb_src, depth, dino, full_rgb, crop_yx in tqdm(
                train_loader, desc=f"{campaign} gen+rae"
            ):
                rgb_src = rgb_src.to(DEVICE, non_blocking=True)
                depth = depth.to(DEVICE, non_blocking=True)
                dino = dino.to(DEVICE, non_blocking=True)  # [B, 16, 16, 768]
                src_pooled = dino_patch_embed(dino_patch_model, rgb_src).mean(
                    dim=1
                )  # [B, D], shared

                # Protocol-matched depth reference: DAv2 over the full 518 view,
                # normalised over the full view, then this sample's exact crop --
                # the same space as the stored targets and the decoded depth.
                ref_depth = None
                if depth_consistency is not None:
                    ref_depth = dav2_fullview_depth_crop(
                        depth_consistency.run, full_rgb, crop_yx
                    ).to(
                        DEVICE
                    )  # [B, 224, 224]
                    # Pipeline sanity: recomputed reference vs the stored crop.
                    # Expect r ~= 1; lower means extraction drift on disk.
                    ref_check_depth_consistency.update(rgb_src, depth, pred=ref_depth)

                # -- CFM conditional generation --
                cond = dino.mean(dim=(1, 2))  # [B, 768] pooled DINO grid
                gen_rgb, gen_depth = generate_batch_conditionally(
                    model, rae, cond, args.n_steps
                )
                metrics.update(gen_rgb, real=False)
                if depth_consistency is not None:
                    depth_consistency.update(gen_rgb, gen_depth)
                gen_pooled = dino_patch_embed(dino_patch_model, gen_rgb).mean(
                    dim=1
                )  # [B, D]
                cos_per_img = torch.nn.functional.cosine_similarity(
                    src_pooled, gen_pooled, dim=1
                )  # [B] in [-1, 1]
                pooled_cossim_sum += float(cos_per_img.sum().item())
                pooled_cossim_n += cos_per_img.shape[0]

                # -- RAE round-trip (no flow matching) on the same source batch --
                rgbd, _ = rae(dino, depth)  # [B, 4, 224, 224]
                recon_rgb = rgbd[:, :3].clamp(0, 1)
                recon_depth = rgbd[:, 3:4].clamp(0, 1)
                rae_metrics.update(rgb_src, real=True)
                rae_metrics.update(recon_rgb, real=False)
                if rae_depth_consistency is not None:
                    # Recon depth vs the full-view recomputed reference: same FOV
                    # and normalisation as training, so a perfect decoder scores
                    # ~1 here (no DAv2 pass on the recon needed).
                    rae_depth_consistency.update(recon_rgb, recon_depth, pred=ref_depth)
                recon_pooled = dino_patch_embed(dino_patch_model, recon_rgb).mean(
                    dim=1
                )  # [B, D]
                cos_per_img = torch.nn.functional.cosine_similarity(
                    src_pooled, recon_pooled, dim=1
                )  # [B]
                pooled_cossim_sum_rae += float(cos_per_img.sum().item())
                pooled_cossim_n_rae += cos_per_img.shape[0]

                if args.debug_samples > 0 and not debug_saved:
                    save_conditional_debug(
                        campaign, rgb_src, gen_rgb, args.debug_samples
                    )
                    save_rae_reconstruction_debug(
                        campaign,
                        rgb_src,
                        recon_rgb,
                        depth,
                        recon_depth,
                        args.debug_samples,
                    )
                    debug_saved = True

                n_fake += gen_rgb.shape[0]
                if (
                    args.per_campaign_size is not None
                    and n_fake >= args.per_campaign_size
                ):
                    break

        # -- conditional score (real = test images, fake = CFM generations) --
        score = metrics.compute(n_real, n_fake)
        score["dino_pooled_cossim"] = (
            pooled_cossim_sum / pooled_cossim_n if pooled_cossim_n > 0 else None
        )
        if depth_consistency is not None:
            score.update(depth_consistency.compute())
            score.update(real_depth_consistency.compute(suffix="_real"))
        results[campaign] = score
        print(
            f"Conditional [{campaign}]: {format_metrics(score)}  (real={n_real}, fake={n_fake})"
        )

        # -- RAE reconstruction score (real = source, fake = reconstructions; equal
        # counts, one reconstruction per source image) --
        rae_score = rae_metrics.compute(n_fake, n_fake)
        rae_score["dino_pooled_cossim"] = (
            pooled_cossim_sum_rae / pooled_cossim_n_rae
            if pooled_cossim_n_rae > 0
            else None
        )
        if rae_depth_consistency is not None:
            # depth_* here = recon vs full-view protocol-matched reference (read vs ~1);
            # depth_*_ref_check = recomputed reference vs stored crop (pipeline sanity).
            rae_score.update(rae_depth_consistency.compute())
            rae_score.update(ref_check_depth_consistency.compute(suffix="_ref_check"))
        rae_results[campaign] = rae_score
        print(
            f"RAE reconstruction [{campaign}]: {format_metrics(rae_score)}  (real={n_fake}, fake={n_fake})"
        )

    print_campaign_summary("Conditional per-campaign FID/KID", results)
    print_campaign_summary("RAE reconstruction per-campaign FID/KID", rae_results)
    return results, rae_results


def _update_from_loader(loader, metrics, real, size, desc):
    """Feed up to `size` images from `loader` into the metric bundle.

    `size=None` means no cap: consume the whole loader (used when
    --sample_size / --per_campaign_size are left unset -> full split).
    """
    n = 0
    for rgb, _, _ in tqdm(loader, desc=desc):
        metrics.update(rgb.to(DEVICE), real=real)
        n += rgb.shape[0]
        if size is not None and n >= size:
            break
    return n


def evaluate_gold_standard(metrics, args):
    """FID/KID between real *train* and real *test* images -- the achievable floor.
    No generation is involved: this measures the train/test distribution gap
    (plus finite-sample bias) itself, so any generative FID above it reflects
    model error rather than dataset shift. Reported globally and per campaign so
    it sits directly beside the unconditional and conditional numbers.
    """

    def _loader(split, campaign=None):
        ds = CSVSplitDataset(split=split, load_rgb=True, campaign=campaign)
        # shuffle=True here (unlike the other modes): gold caps at sample_size /
        # per_campaign_size and breaks early, so it needs a representative sample
        # rather than the first-N (which, in CSV order, is a single deployment).
        # The capped count is small, so the random-read cost is negligible.
        return ds, DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=True,
        )

    # ---- global ----
    metrics.reset()
    _, test_loader = _loader("test")
    _, train_loader = _loader("train")
    with torch.no_grad():
        n_test = _update_from_loader(
            test_loader, metrics, True, args.sample_size, "Gold global test"
        )
        n_train = _update_from_loader(
            train_loader, metrics, False, args.sample_size, "Gold global train"
        )
    global_score = metrics.compute(n_test, n_train)
    print(
        f"Gold-standard (train vs test) global -- {format_metrics(global_score)}  "
        f"(test={n_test}, train={n_train})"
    )

    # ---- per campaign ----
    campaigns = select_campaigns(args)
    cap_str = (
        "full split"
        if args.per_campaign_size is None
        else f"<= {args.per_campaign_size}"
    )
    print(
        f"\nGold-standard per-campaign FID/KID over {len(campaigns)} campaigns "
        f"({cap_str} samples each): {campaigns}"
    )

    results = {}
    for campaign in campaigns:
        test_ds, test_loader = _loader("test", campaign)
        train_ds, train_loader = _loader("train", campaign)
        if len(test_ds) < 2 or len(train_ds) < 2:
            print(
                f"[skip] {campaign}: train={len(train_ds)} test={len(test_ds)} (need >=2 each)"
            )
            continue

        metrics.reset()
        with torch.no_grad():
            n_test = _update_from_loader(
                test_loader, metrics, True, args.per_campaign_size, f"{campaign} test"
            )
            n_train = _update_from_loader(
                train_loader,
                metrics,
                False,
                args.per_campaign_size,
                f"{campaign} train",
            )
        score = metrics.compute(n_test, n_train)
        results[campaign] = score
        print(
            f"Gold-standard [{campaign}]: {format_metrics(score)}  (test={n_test}, train={n_train})"
        )

    print_campaign_summary(
        "Gold-standard per-campaign FID/KID (train vs test)", results
    )
    return {"global": global_score, "per_campaign": results}


def main(args):
    metrics = build_metrics(args)
    kid_note = "off" if args.skip_kid else f"on (subset_size={args.kid_subset_size})"
    print(
        f"Feature extractor: {metrics.extractor_label}, KID {kid_note}, "
        f"fid device: {metrics.fid.real_features_sum.device}"
    )

    # Gold standard is real-vs-real, so it needs no CFM/RAE checkpoints.
    if args.mode != "gold":
        model = UNetCFM().to(DEVICE)
        model.load_state_dict(
            torch.load(args.cfm_ckpt, map_location=DEVICE, weights_only=True)
        )
        model.eval()
        rae = load_rae_frozen(args.rae_ckpt, DEVICE)

    # Depth self-consistency uses DAv2 on the *generated* RGB, so it only applies
    # to the generative modes (gold is real-vs-real with no generated depth). Load
    # the estimator once and reuse it across unconditional + conditional.
    depth_consistency = None
    if not args.skip_depth_consistency and args.mode in (
        "unconditional",
        "conditional",
        "both",
        "all",
    ):
        dav2_run = load_dav2(variant=args.dav2_variant, metric=False, processed=False)
        depth_consistency = DepthConsistency(dav2_run)

    sections = {}
    if args.mode in ("unconditional", "both", "all"):
        sections["unconditional"] = evaluate_unconditional(
            model, rae, metrics, args, depth_consistency
        )

    if args.mode in ("conditional", "both", "all"):
        # One pass over each campaign yields both the conditional generation FID/KID
        # and the RAE-reconstruction floor (real vs RAE round-trip, no flow matching).
        cond_res, rae_res = evaluate_conditional_per_campaign(
            model, rae, metrics, args, depth_consistency
        )
        sections["conditional"] = {"per_campaign": cond_res}
        sections["rae_reconstruction"] = {"per_campaign": rae_res}

    if args.mode in ("gold", "all"):
        sections["gold"] = evaluate_gold_standard(metrics, args)

    config = {
        "feature_extractor": metrics.extractor_label,
        "n_steps": args.n_steps,
        "sample_size": args.sample_size,
        "per_campaign_size": args.per_campaign_size,
        "kid_subset_size": args.kid_subset_size,
        "skip_kid": args.skip_kid,
        "cfm_ckpt": args.cfm_ckpt,
        "rae_ckpt": args.rae_ckpt,
        "depth_consistency": depth_consistency is not None,
        "dav2_variant": args.dav2_variant if depth_consistency is not None else None,
    }
    ckpt_tag = Path(args.cfm_ckpt).stem if args.cfm_ckpt else "no_ckpt"
    run_key = f"{ckpt_tag}_{metrics.extractor_label}_steps{args.n_steps}"
    update_results_json(RESULTS_JSON, run_key, config, sections)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description="Test generative quality of CFM with different metric."
    )

    argparser.add_argument("--rae_ckpt", type=str, help="Path to the RAE checkpoint.")
    argparser.add_argument("--cfm_ckpt", type=str, help="Path to the CFM checkpoint.")
    argparser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Global sample cap: max real/generated images for the unconditional "
        "and gold-global evals (per-campaign uses --per_campaign_size).",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for generation."
    )
    argparser.add_argument(
        "--n_steps", type=int, default=50, help="Number of steps for CFM sampling."
    )
    argparser.add_argument(
        "--n_workers", type=int, default=4, help="Number of DataLoader workers."
    )
    argparser.add_argument(
        "--feature_extractor",
        type=str,
        default="dino",
        choices=["dino", "inception"],
        help="Feature extractor for FID/KID: DINOv2 CLS token (dino) or standard InceptionV3 (inception).",
    )
    argparser.add_argument(
        "--skip_kid",
        action="store_true",
        help="Only compute FID (skip the complementary KID metric).",
    )
    argparser.add_argument(
        "--skip_depth_consistency",
        action="store_true",
        help="Skip the depth self-consistency metric (DAv2 on the generated RGB "
        "vs the generated depth channel). Off by default; skip it to avoid "
        "loading the Depth-Anything-V2 model during eval.",
    )
    argparser.add_argument(
        "--dav2_variant",
        type=str,
        default="Large",
        choices=["Small", "Base", "Large"],
        help="Depth-Anything-V2 variant for the depth-consistency metric. Default "
        "Large to match the training depth targets (see extract_depth.py).",
    )
    argparser.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        help="KID MMD subset size per resample; clamped down to the available "
        "sample count per comparison so small campaigns don't crash.",
    )
    argparser.add_argument(
        "--mode",
        type=str,
        default="unconditional",
        choices=["unconditional", "conditional", "gold", "both", "all"],
        help="unconditional: FID vs whole test set; conditional: per-campaign "
        "FID (train DINO conditions generation, test images are the reference); "
        "gold: real train-vs-test FID floor (global + per-campaign, no generation); "
        "both: unconditional+conditional; all: everything.",
    )
    argparser.add_argument(
        "--per_campaign_size",
        type=int,
        default=None,
        help="Max real/generated samples per campaign for the conditional, RAE "
        "reconstruction and gold per-campaign evals. Default None = the "
        "campaign's full test split; pass a small value to limit per-campaign "
        "generation for debugging. For a real (non-debug) run keep it well above "
        "the feature dim (768 DINO / 2048 inception) or FID is unstable.",
    )
    argparser.add_argument(
        "--campaigns",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated campaigns to evaluate (conditional/gold modes). "
        "Default: all campaigns in the test split.",
    )
    argparser.add_argument(
        "--debug_samples",
        type=int,
        default=4,
        help="Per campaign (conditional mode), save this many [conditioning source | "
        "generated] pairs to figs/cfm/test_generation/<campaign>/. 0 disables (and "
        "skips the extra source-RGB read).",
    )

    args = argparser.parse_args()
    main(args)
