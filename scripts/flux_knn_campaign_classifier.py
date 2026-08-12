# scripts/flux_knn_campaign_classifier.py
"""Campaign classification with a k-NN over pooled DINOv2 embeddings -- FLUX.2 variant.

The FLUX.2 counterpart of knn_campaign_classifier.py (the CFM version): same
protocol, same metric, same test split -- only the generator changes. Two
classifiers are trained and evaluated on the *same* real test split, so the only
thing that differs is where the training embeddings come from:

  * real      -- k-NN fit on pooled DINO embeddings of real *train* images.
  * generated -- k-NN fit on pooled DINO embeddings of images FLUX.2-dev
                 *generates* when reference-conditioned on each train image.
                 Each generated sample inherits the campaign label of its
                 reference image.

Comparing the two accuracies answers: do the FLUX.2 reference-conditioned images
carry the same campaign-discriminative content as the real images they were
conditioned on? If the generated-trained k-NN classifies the real test split
about as well as the real-trained one, FLUX honours the campaign it was shown.

CFM vs FLUX conditioning
------------------------
The CFM is conditioned on a [768] pooled-DINO *vector* and decodes a 224 image.
FLUX.2-dev instead takes the *reference image itself* (image=[seed]), generated
at --gen_size (512 by default) then resized to --eval_size (224) for scoring --
exactly as flux_test_generation.py conditions and scores it, and using the same
full-frame (uncropped) reference the FLUX eval found works best. So here the
"conditioning" is the train image's RGB, and all three embedding sets (real
train, generated train, real test) are pooled from DINOv2 at --eval_size, which
is the common footing that keeps the comparison fair.

Distributed
-----------
FLUX generation is the bottleneck (one image per reference, no batching of
distinct references), so -- like flux_test_generation.py -- the embedding passes
are data-parallel: each rank processes a disjoint stride of the dataset and the
per-rank embeddings are all-gathered before the (cheap) k-NN fit / predict /
report, which run on rank 0. Launch with torchrun for multi-GPU; it also runs
single-process unchanged.

Feature protocol (shared by every embedding here): DINOv2 patch tokens on the
--eval_size image, mean-pooled over the grid to one [768] vector (the same
pooled DINO used elsewhere in the repo).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import datetime
import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless; set before test_generation imports pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# DINOv2's layers each warn once that xFormers is available (SwiGLU/Attention/
# Block) when torch.hub.load imports the model. It's informational -- xFormers
# *is* being used -- so filter it out to keep the run logs clean. Installed here,
# before load_dinov2() triggers the dinov2 import.
warnings.filterwarnings("ignore", message="xFormers is available")

from benthicflow import PROJECT_ROOT, FIG_ROOT
from scripts.test_generation import dino_patch_embed
from scripts.extract_features import load_dinov2
# The FLUX pipeline, generator, resize helper, full-frame dataset and the
# distributed plumbing all live in the FLUX eval script; reuse them verbatim so
# this classifier conditions/scores FLUX identically to flux_test_generation.py.
import scripts.flux_test_generation as ftg

RESULTS_JSON = FIG_ROOT / "flux" / "knn_campaign" / "knn_results.json"
EMB_DIM = 768   # DINOv2 vitb14 embedding width (pooled feature length)

# Set by setup_distributed() in main(); default to single-process values so the
# module-level helpers work when launched without torchrun.
RANK, WORLD_SIZE, LOCAL_RANK = 0, 1, 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_main() -> bool:
    return RANK == 0


def rprint(*a, **k):
    """print only on rank 0, to avoid WORLD_SIZE-fold duplicated logs."""
    if is_main():
        print(*a, **k)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def campaigns_in_split(split: str) -> list[str]:
    """Sorted unique campaign names in a split CSV (cheap column scan)."""
    path = PROJECT_ROOT / "data_split" / f"{split}.csv"
    seen = set()
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            c = row.get("campaign")
            if c:
                seen.add(c)
    return sorted(seen)


def campaign_to_class(campaign: str) -> str:
    """Unify same-site campaigns into one class by stripping the trailing YYYYMM:
    ScottReef200907/201108/201503 -> 'ScottReef', Batemans* -> 'Batemans',
    Hawaii201801 -> 'Hawaii'. Campaigns from the same reef look alike on average
    (same biome), so keeping them as separate classes just splits (dilutes) their
    recall between each other; the site is the label we actually want to predict.
    """
    return campaign.rstrip("0123456789")


class LabeledFluxDataset(Dataset):
    """Wrap a (per-campaign or full) FLUX CSVSplitDataset to also emit a class label.

    ftg.CSVSplitDataset yields (rgb, dino) full-frame with no label; FLUX conditions
    on the RGB image directly (not on the DINO grid the CFM used), so the grid is
    dropped here. We recover the campaign from its record table, collapse it to its
    site via campaign_to_class, and map that through class_to_idx (a site -> index
    map). `indices` optionally restricts to a subset (used to cap each campaign at N
    train images). Returns (rgb, label): rgb is both the real-image embedding source
    and the FLUX reference image.
    """

    def __init__(self, base: "ftg.CSVSplitDataset", class_to_idx: dict[str, int],
                 indices: list[int] | None = None):
        self.base = base
        self.class_to_idx = class_to_idx
        self.indices = list(range(len(base))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        rgb, _dino = self.base[j]
        campaign = self.base.records[j][0]
        return rgb, self.class_to_idx[campaign_to_class(campaign)]


def build_train_dataset(campaigns: list[str], class_to_idx: dict[str, int],
                        n_per_class: int, seed: int) -> ConcatDataset:
    """Class-balanced train set: n_per_class images per class, concatenated.

    Each class's quota is split as evenly as possible across its campaigns, so a
    multi-campaign site (ScottReef, Batemans) still spans all its deployments but
    contributes the *same* total as a single-campaign site (Hawaii). A balanced
    bank matters because the k-NN majority-votes over neighbours -- unequal class
    counts would bias the vote toward the larger classes.

    A random (seeded) subset is taken per campaign rather than the first N: CSV
    order groups by deployment, so the first N would be a single deployment
    instead of a representative sample. The seed is shared across ranks, so every
    rank builds the identical dataset (a prerequisite for the strided sharding).
    """
    rng = np.random.default_rng(seed)
    class_campaigns: dict[str, list[str]] = {}
    for c in campaigns:
        class_campaigns.setdefault(campaign_to_class(c), []).append(c)

    parts = []
    for cls in sorted(class_campaigns):
        camps = class_campaigns[cls]
        m = len(camps)
        # Even split with the remainder spread over the first few campaigns.
        quotas = [n_per_class // m + (1 if i < n_per_class % m else 0) for i in range(m)]
        got = 0
        for campaign, quota in zip(camps, quotas):
            base = ftg.CSVSplitDataset(split="train", load_rgb=True, campaign=campaign)
            n = len(base)
            if n == 0:
                rprint(f"[warn] campaign '{campaign}' has 0 train records; skipping.")
                continue
            take = min(quota, n)
            idx = rng.choice(n, size=take, replace=False).tolist()
            parts.append(LabeledFluxDataset(base, class_to_idx, idx))
            got += take
        note = "" if got == n_per_class else f"  [warn] short of {n_per_class}"
        rprint(f"  class '{cls}': {got} train images from {m} campaign(s){note}")
    return ConcatDataset(parts)


def build_test_dataset(test_base: "ftg.CSVSplitDataset", class_to_idx: dict[str, int],
                       per_campaign: int | None, max_total: int | None,
                       seed: int) -> LabeledFluxDataset:
    """The full test split, with optional debug caps (applied in this order):

      per_campaign -- keep at most this many random test images *per campaign*.
        A balanced subset that keeps every class represented -- better for a quick
        debug pass than a raw total cap, whose random draw can starve the small
        campaigns (Hawaii is ~40% of the test split).
      max_total    -- keep at most this many test images overall (random subset).

    Both None -> the entire test split. Indices come from the in-memory record
    table (records[j][0] is the campaign), so capping costs no image I/O.
    """
    if per_campaign is None and max_total is None:
        return LabeledFluxDataset(test_base, class_to_idx)

    rng = np.random.default_rng(seed)
    if per_campaign is None:
        idx = list(range(len(test_base)))
    else:
        by_campaign: dict[str, list[int]] = {}
        for j in range(len(test_base)):
            by_campaign.setdefault(test_base.records[j][0], []).append(j)
        idx = []
        for js in by_campaign.values():
            idx.extend(rng.choice(js, size=min(per_campaign, len(js)), replace=False).tolist())

    if max_total is not None and len(idx) > max_total:
        idx = rng.choice(idx, size=max_total, replace=False).tolist()
    return LabeledFluxDataset(test_base, class_to_idx, idx)


# --------------------------------------------------------------------------- #
# Distributed embedding computation
# --------------------------------------------------------------------------- #
def _shard_loader(dataset, args) -> DataLoader:
    """DataLoader over this rank's disjoint stride of `dataset`.

    A plain strided subset (rank, rank+WS, rank+2WS, ...) rather than
    DistributedSampler: the union is *exactly* the full set with no padded
    duplicates, so the all-gathered banks below are the dataset once-through.
    Order within a shard is sequential (shuffle=False), so the three per-rank
    tensors (real, gen, labels) stay aligned index-for-index.
    """
    idx = list(range(RANK, len(dataset), WORLD_SIZE))
    shard = Subset(dataset, idx)
    return DataLoader(shard, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.n_workers, pin_memory=True)


def _all_gather_cat(t: torch.Tensor) -> torch.Tensor:
    """All-gather a per-rank CPU tensor and concat in rank order (identity if 1 rank)."""
    if WORLD_SIZE == 1:
        return t
    buf: list = [None] * WORLD_SIZE
    dist.all_gather_object(buf, t)
    return torch.cat(buf, dim=0)


@torch.no_grad()
def pooled_dino(dino_model, rgb: torch.Tensor) -> torch.Tensor:
    """Average-pooled DINOv2 patch embedding for [B, 3, H, W] RGB in [0, 1]."""
    return dino_patch_embed(dino_model, rgb).mean(dim=1)  # [B, 768]


@torch.no_grad()
def compute_train_banks(train_ds, dino_model, pipe, args, want_real, want_gen):
    """One data-parallel pass over the train set -> (real_bank, gen_bank, labels).

    Each rank embeds its stride: the real embedding is the pooled DINO of the real
    RGB (resized to eval_size); the generated embedding is the pooled DINO of the
    image FLUX generates when reference-conditioned on that same RGB. Both share
    the batch's labels, so the two banks are aligned index-for-index. The per-rank
    tensors are all-gathered so every rank holds the full banks. `pipe` is None
    when the generated classifier is disabled (real embeddings only).
    """
    loader = _shard_loader(train_ds, args)
    real_feats, gen_feats, labels = [], [], []
    for rgb, label in tqdm(loader, desc="train embeddings", disable=not is_main()):
        rgb = rgb.to(DEVICE, non_blocking=True)
        labels.append(label.clone())
        if want_real:
            real_ev = ftg.resize_batch(rgb, args.eval_size)          # full frame -> eval_size
            real_feats.append(pooled_dino(dino_model, real_ev).cpu())
        if want_gen:
            gen_rgb = ftg.flux_generate(pipe, rgb, args)             # [B, 3, eval, eval]
            gen_feats.append(pooled_dino(dino_model, gen_rgb).cpu())

    labels = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
    real_local = torch.cat(real_feats) if real_feats else torch.empty(0, EMB_DIM)
    gen_local = torch.cat(gen_feats) if gen_feats else torch.empty(0, EMB_DIM)

    labels = _all_gather_cat(labels)
    real_bank = _all_gather_cat(real_local) if want_real else None
    gen_bank = _all_gather_cat(gen_local) if want_gen else None
    return real_bank, gen_bank, labels


@torch.no_grad()
def compute_test_embeddings(test_ds, dino_model, args):
    """Data-parallel pooled-DINO embeddings of the real test split -> (emb, labels).

    Cheap (no generation) but sharded anyway so it scales with the ranks already
    spun up for generation; per-rank tensors are all-gathered to the full set.
    """
    loader = _shard_loader(test_ds, args)
    feats, labels = [], []
    for rgb, label in tqdm(loader, desc="test embeddings", disable=not is_main()):
        rgb = rgb.to(DEVICE, non_blocking=True)
        emb = pooled_dino(dino_model, ftg.resize_batch(rgb, args.eval_size))
        feats.append(emb.cpu())
        labels.append(label.clone())
    emb = torch.cat(feats) if feats else torch.empty(0, EMB_DIM)
    labels = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
    return _all_gather_cat(emb), _all_gather_cat(labels)


# --------------------------------------------------------------------------- #
# k-NN
# --------------------------------------------------------------------------- #
class TorchKNN:
    """Plain GPU k-NN classifier (cosine or L2), majority vote over neighbours."""

    def __init__(self, k: int, num_classes: int, metric: str = "cosine"):
        self.k = k
        self.num_classes = num_classes
        self.metric = metric

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X = X.to(DEVICE).float()
        self.X = F.normalize(X, dim=1) if self.metric == "cosine" else X
        self.y = y.to(DEVICE).long()
        self.k = min(self.k, self.X.shape[0])
        return self

    @torch.no_grad()
    def predict(self, Q: torch.Tensor) -> torch.Tensor:
        Q = Q.to(DEVICE).float()
        if self.metric == "cosine":
            sims = F.normalize(Q, dim=1) @ self.X.t()          # [B, N], higher = closer
            nn_idx = sims.topk(self.k, dim=1).indices
        else:
            dists = torch.cdist(Q, self.X)                     # [B, N]
            nn_idx = dists.topk(self.k, dim=1, largest=False).indices
        votes = F.one_hot(self.y[nn_idx], self.num_classes).sum(dim=1)  # [B, C]
        return votes.argmax(dim=1)                             # ties -> lowest class idx


@torch.no_grad()
def predict_all(knn: TorchKNN, emb: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Predict a whole embedding matrix in row-chunks (bounds the N_test x N_train matmul)."""
    out = []
    for i in range(0, emb.shape[0], chunk):
        out.append(knn.predict(emb[i:i + chunk]).cpu())
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Evaluation / reporting
# --------------------------------------------------------------------------- #
def classification_report(preds: torch.Tensor, labels: torch.Tensor,
                          idx_to_class: dict[int, str]) -> dict:
    """Per-class precision/recall/F1/support + micro (overall) and macro accuracy."""
    C = len(idx_to_class)
    per_class = {}
    recalls, f1s = [], []
    for c in range(C):
        pred_pos = preds == c
        true_pos = labels == c
        tp = int((pred_pos & true_pos).sum())
        fp = int((pred_pos & ~true_pos).sum())
        fn = int((~pred_pos & true_pos).sum())
        support = int(true_pos.sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[idx_to_class[c]] = {"precision": precision, "recall": recall,
                                      "f1": f1, "support": support}
        recalls.append(recall)
        f1s.append(f1)
    overall_acc = float((preds == labels).float().mean())
    return {
        "accuracy": overall_acc,                       # micro / overall
        "balanced_accuracy": sum(recalls) / C,         # macro recall (class-balanced)
        "macro_f1": sum(f1s) / C,
        "n_test": int(labels.numel()),
        "per_class": per_class,
    }


def print_report(name: str, report: dict, idx_to_class: dict[int, str]):
    print(f"\n=== {name} classifier ===")
    print(f"overall accuracy : {report['accuracy']:.4f}")
    print(f"balanced acc     : {report['balanced_accuracy']:.4f}")
    print(f"macro F1         : {report['macro_f1']:.4f}   (n_test={report['n_test']})")
    width = max(len(c) for c in idx_to_class.values())
    print(f"{'campaign':>{width}}  {'prec':>6} {'recall':>6} {'f1':>6} {'support':>8}")
    for c in sorted(report["per_class"]):
        m = report["per_class"][c]
        print(f"{c:>{width}}  {m['precision']:6.3f} {m['recall']:6.3f} "
              f"{m['f1']:6.3f} {m['support']:8d}")


def save_results(config: dict, reports: dict[str, dict]):
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if RESULTS_JSON.exists():
        try:
            data = json.loads(RESULTS_JSON.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {RESULTS_JSON} was not valid JSON; starting fresh.")
    run_key = (f"flux2dev_k{config['k']}_{config['metric']}_n{config['n_per_class']}"
               f"_sz{config['gen_size']}_ev{config['eval_size']}"
               f"_steps{config['steps']}_cfg{config['guidance_scale']}")
    data[run_key] = {
        "config": config,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "reports": reports,
    }
    RESULTS_JSON.write_text(json.dumps(data, indent=2))
    print(f"\nResults written to {RESULTS_JSON} under run key '{run_key}'")


# --------------------------------------------------------------------------- #
# Confusion-matrix figure (publication-ready vector output)
# --------------------------------------------------------------------------- #
# Single-hue blue sequential ramp (light -> dark): the validated dataviz default
# for a magnitude heatmap. Near-zero recedes toward the light surface; the
# darkest step marks the strongest cell (the correct-classification diagonal).
_SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_INK = "#0b0b0b"     # primary ink
_MUTED = "#898781"   # axis / muted labels
_CBAR_LABEL = "Row-normalized frequency (recall on diagonal)"


def _pub_style():
    """Publication rcParams: Times-like serif (STIX, bundled), embedded fonts.

    STIX matches Times (the ECCV/LNCS body font) and ships with matplotlib, so
    the figure text blends with the paper without needing system Times or LaTeX.
    fonttype 42 embeds TrueType glyphs rather than Type-3 (some venues reject Type-3).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep SVG text as selectable text
    })


def _pretty_campaign(name: str) -> tuple[str, str]:
    """'ScottReef201108' -> ('ScottReef', '2011-08'); trailing YYYYMM -> 'YYYY-MM'."""
    i = len(name)
    while i > 0 and name[i - 1].isdigit():
        i -= 1
    site, digits = name[:i], name[i:]
    if len(digits) == 6:
        return site, f"{digits[:4]}-{digits[4:]}"
    return site, digits


def confusion_counts(preds: torch.Tensor, labels: torch.Tensor, C: int) -> np.ndarray:
    """[C, C] integer confusion counts; rows = true class, cols = predicted class."""
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (labels.numpy(), preds.numpy()), 1)
    return cm


def _draw_confusion(ax, cm: np.ndarray, cmap, idx_to_class: dict[int, str],
                    show_ylabels: bool = True, title: str | None = None):
    """Draw one row-normalized confusion matrix (recall on the diagonal) on `ax`."""
    C = cm.shape[0]
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.nan_to_num(cm / support[:, None])

    im = ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    # Unified classes are site names with no trailing date, so `d` is empty and
    # the labels collapse to one line; the two-line form still works if a caller
    # ever passes full campaign names (e.g. ScottReef / 2011-08).
    pretty = [_pretty_campaign(idx_to_class[i]) for i in range(C)]
    x_labels = [f"{s}\n{d}" if d else s for s, d in pretty]
    y_labels = [f"{s} {d}".strip() + f"\n(n={support[i]:,})" for i, (s, d) in enumerate(pretty)]
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor", color=_INK)
    ax.set_yticklabels(y_labels if show_ylabels else [""] * C, color=_INK)
    ax.set_xlabel("Predicted class", color=_INK)
    if show_ylabels:
        ax.set_ylabel("True class", color=_INK)
    if title:
        ax.set_title(title, color=_INK, pad=6)

    # Hairline white gaps between cells (the surface-gap spec); drop the frame.
    ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, C, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell text: ink chosen by cell luminance so it always reads; diagonal
    # (the recall values) in bold to draw the eye.
    for i in range(C):
        for j in range(C):
            v = norm[i, j]
            r, g, b, _ = cmap(v)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if lum < 0.5 else _INK,
                    fontweight="bold" if i == j else "normal")
    return im


def _add_cbar(fig, im, ax):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(_CBAR_LABEL, color=_INK)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2, colors=_MUTED)
    return cbar


def save_confusion_figures(preds: dict[str, torch.Tensor], labels: torch.Tensor,
                           idx_to_class: dict[int, str], fig_dir: Path,
                           formats: list[str]):
    """Per-classifier confusion matrices + (when both exist) a stacked comparison.

    Row-normalized so the diagonal is per-campaign recall (prior-invariant); the
    true-class (y) labels carry each campaign's test support so the class
    imbalance is legible on the figure itself. Vector output (PDF/SVG) for the paper.
    """
    _pub_style()
    cmap = LinearSegmentedColormap.from_list("reef_seq_blue", _SEQ_BLUE)
    C = len(idx_to_class)
    fig_dir.mkdir(parents=True, exist_ok=True)
    titles = {"real": "Trained on real images",
              "generated": "Trained on FLUX.2 generations"}

    def _save(fig, stem):
        for fmt in formats:
            path = fig_dir / f"{stem}.{fmt}"
            fig.savefig(path, facecolor="white")
            print(f"  saved {path}")
        plt.close(fig)

    # ---- individual (each self-contained with its own colorbar) ----
    for name, p in preds.items():
        cm = confusion_counts(p, labels, C)
        fig, ax = plt.subplots(figsize=(5.4, 4.4), layout="constrained")
        im = _draw_confusion(ax, cm, cmap, idx_to_class)
        _add_cbar(fig, im, ax)
        _save(fig, f"confusion_{name}")

    # ---- stacked comparison (real over generated, shared colorbar) ----
    if "real" in preds and "generated" in preds:
        fig, axes = plt.subplots(2, 1, figsize=(5.2, 8.6), layout="constrained")
        im = None
        for ax, name in zip(axes, ("real", "generated")):
            cm = confusion_counts(preds[name], labels, C)
            im = _draw_confusion(ax, cm, cmap, idx_to_class, title=titles[name])
        _add_cbar(fig, im, list(axes))
        _save(fig, "confusion_comparison")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    global RANK, WORLD_SIZE, LOCAL_RANK, DEVICE
    # Reuse the FLUX eval's distributed setup so ftg.DEVICE (which ftg.flux_generate
    # reads) is set for this rank, then mirror the values into our own globals.
    ftg.setup_distributed()
    RANK, WORLD_SIZE, LOCAL_RANK, DEVICE = ftg.RANK, ftg.WORLD_SIZE, ftg.LOCAL_RANK, ftg.DEVICE
    rprint(f"Distributed: WORLD_SIZE={WORLD_SIZE} (one FLUX.2 pipeline per GPU)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    campaigns = sorted(set(campaigns_in_split("train")) | set(campaigns_in_split("test")))
    classes = sorted({campaign_to_class(c) for c in campaigns})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    n_classes = len(classes)
    rprint(f"{len(campaigns)} campaigns -> {n_classes} classes: {classes}")
    for cls in classes:
        members = [c for c in campaigns if campaign_to_class(c) == cls]
        rprint(f"  {cls}: {members}")
    rprint(f"mode={args.mode}  k={args.k}  metric={args.metric}  "
           f"n_per_class={args.n_per_class}  chance={1.0 / n_classes:.3f}")

    dino_model = load_dinov2().to(DEVICE).eval()

    # FLUX.2-dev (4-bit) pipeline only needed for the generated-image classifier.
    pipe = None
    want_real = args.mode in ("real", "both")
    want_gen = args.mode in ("generated", "both")
    if want_gen:
        rprint("Loading FLUX.2-dev (4-bit) pipeline, resident on each GPU...")
        pipe = ftg.build_flux2_pipeline(LOCAL_RANK)

    # ---- training banks (class-balanced: n_per_class per class), data-parallel ----
    train_ds = build_train_dataset(campaigns, class_to_idx, args.n_per_class, args.seed)
    rprint(f"train set: {len(train_ds)} images, {n_classes} classes x "
           f"{args.n_per_class} (class-balanced)")
    real_bank, gen_bank, train_labels = compute_train_banks(
        train_ds, dino_model, pipe, args, want_real, want_gen)

    # ---- real test split (full by default; optional debug caps), data-parallel ----
    test_base = ftg.CSVSplitDataset(split="test", load_rgb=True)
    test_ds = build_test_dataset(test_base, class_to_idx,
                                 args.test_per_campaign, args.max_test, args.seed)
    full = args.test_per_campaign is None and args.max_test is None
    rprint(f"test set: {len(test_ds)} images"
           + ("" if full else f" (debug cap; from {len(test_base)} full)"))
    test_emb, test_labels = compute_test_embeddings(test_ds, dino_model, args)

    # ---- fit / predict / report (rank 0; banks are identical across ranks) ----
    if is_main():
        classifiers = {}
        if real_bank is not None:
            classifiers["real"] = TorchKNN(args.k, n_classes, args.metric).fit(real_bank, train_labels)
        if gen_bank is not None:
            classifiers["generated"] = TorchKNN(args.k, n_classes, args.metric).fit(gen_bank, train_labels)

        preds = {name: predict_all(knn, test_emb) for name, knn in classifiers.items()}

        reports = {}
        for name in classifiers:
            reports[name] = classification_report(preds[name], test_labels, idx_to_class)
            print_report(name, reports[name], idx_to_class)

        if "real" in reports and "generated" in reports:
            # Compare on the imbalance-robust headline metrics, not micro-accuracy
            # (which the large campaigns dominate); see per-class recall invariance.
            for key, tag in (("balanced_accuracy", "balanced acc"), ("macro_f1", "macro F1")):
                r, g = reports["real"][key], reports["generated"][key]
                print(f"real vs generated -- {tag}: {r:.4f} vs {g:.4f} "
                      f"(generated retains {100 * g / r:.1f}% of real)")

        if not args.no_figures:
            fig_dir = Path(args.fig_dir) if args.fig_dir else (FIG_ROOT / "flux" / "knn_campaign")
            print(f"\nWriting confusion-matrix figures to {fig_dir} ...")
            save_confusion_figures(preds, test_labels, idx_to_class, fig_dir, args.fig_formats)

        config = {
            "model": "FLUX.2-dev-4bit",
            "mode": args.mode, "k": args.k, "metric": args.metric,
            "n_per_class": args.n_per_class, "n_train": int(train_labels.numel()),
            "steps": args.steps, "guidance_scale": args.guidance_scale,
            "gen_size": args.gen_size, "eval_size": args.eval_size, "seed": args.seed,
            "test_per_campaign": args.test_per_campaign, "max_test": args.max_test,
            "n_test_eval": int(test_labels.numel()),
            "classes": classes, "n_classes": n_classes,
            "chance_accuracy": 1.0 / n_classes, "world_size": WORLD_SIZE,
        }
        save_results(config, reports)

    if WORLD_SIZE > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["real", "generated", "both"], default="both",
                   help="Which k-NN(s) to train: real images, FLUX generations, or both.")
    p.add_argument("--k", type=int, default=20, help="Number of neighbours for the k-NN vote.")
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine",
                   help="k-NN distance (cosine is the standard DINO k-NN metric).")
    p.add_argument("--n_per_class", type=int, default=500,
                   help="Class-balanced train bank: images per unified class (site), split "
                        "evenly across that class's campaigns. Equal per class so the k-NN "
                        "vote isn't biased by class size. Kept modest by default because each "
                        "generated image is a full FLUX.2 generation.")
    p.add_argument("--test_per_campaign", type=int, default=None,
                   help="Debug cap: max test images PER campaign (balanced subset; keeps every "
                        "class represented). Default None = full test split.")
    p.add_argument("--max_test", type=int, default=None,
                   help="Debug cap: max TOTAL test images (random subset; composes with "
                        "--test_per_campaign). Default None = no total cap.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="DataLoader/embedding batch. FLUX.2 still generates one image per "
                        "reference regardless (no batching of distinct references).")
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=28, help="FLUX.2 denoising steps (generated mode).")
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="FLUX.2 guidance scale (generated mode; FLUX.2 default 4.0).")
    p.add_argument("--gen_size", type=int, default=512,
                   help="Square resolution FLUX.2 generates at (multiple of 16). 512 is the "
                        "speed/quality sweet spot; FLUX.2 degrades well below ~512.")
    p.add_argument("--eval_size", type=int, default=224,
                   help="Resolution at which every image is pooled through DINOv2 (real train, "
                        "FLUX generations, and real test), so all three share one feature space.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_figures", action="store_true",
                   help="Skip writing the confusion-matrix figures.")
    p.add_argument("--fig_dir", type=str, default=None,
                   help="Output dir for figures (default figs/flux/knn_campaign).")
    p.add_argument("--fig_formats", nargs="+", default=["pdf", "svg"],
                   help="Vector formats to write; 'png' also allowed for a quick preview.")
    args = p.parse_args()

    main(args)
