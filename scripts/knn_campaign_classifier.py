"""k-NN campaign classification evaluation.

Evaluates campaign classification accuracy using k-nearest neighbors on latent representations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import datetime
import json
import warnings

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")  # headless; set before test_generation imports pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# DINOv2's layers each warn once that xFormers is available (SwiGLU/Attention/
# Block) when torch.hub.load imports the model. It's informational -- xFormers
# *is* being used -- so filter it out to keep the run logs clean. Installed here,
# before load_dinov2() triggers the dinov2 import.
warnings.filterwarnings("ignore", message="xFormers is available")

from benthicflow import FIG_ROOT, PROJECT_ROOT
from models.unet_cfm import UNetCFM, cfm_sample_cfg
from scripts.extract_features import load_dinov2
from scripts.test_generation import DEVICE, CSVSplitDataset, decode, dino_patch_embed
from scripts.train_cfm_cfg import load_rae_frozen

RESULTS_JSON = FIG_ROOT / "cfm" / "knn_campaign" / "knn_results.json"


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


class LabeledFeatureDataset(Dataset):
    """Wrap a (per-campaign or full) CSVSplitDataset to also emit a class label.
    CSVSplitDataset yields (rgb, depth, dino) with no label; we recover the
    campaign from its record table, collapse it to its site via campaign_to_class,
    and map that through class_to_idx (a site -> index map). `indices` optionally
    restricts to a subset (used to cap each campaign at N train images). Returns
    """

    def __init__(
        self,
        base: CSVSplitDataset,
        class_to_idx: dict[str, int],
        indices: list[int] | None = None,
    ):
        self.base = base
        self.class_to_idx = class_to_idx
        self.indices = list(range(len(base))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        rgb, _depth, dino = self.base[j]
        campaign = self.base.records[j][0]
        return rgb, dino, self.class_to_idx[campaign_to_class(campaign)]


def build_train_dataset(
    campaigns: list[str], class_to_idx: dict[str, int], n_per_class: int, seed: int
) -> ConcatDataset:
    """Class-balanced train set: n_per_class images per class, concatenated.
    Each class's quota is split as evenly as possible across its campaigns, so a
    multi-campaign site (ScottReef, Batemans) still spans all its deployments but
    contributes the *same* total as a single-campaign site (Hawaii). A balanced
    bank matters because the k-NN majority-votes over neighbours -- unequal class
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
        quotas = [
            n_per_class // m + (1 if i < n_per_class % m else 0) for i in range(m)
        ]
        got = 0
        for campaign, quota in zip(camps, quotas):
            base = CSVSplitDataset(split="train", load_rgb=True, campaign=campaign)
            n = len(base)
            if n == 0:
                print(f"[warn] campaign '{campaign}' has 0 train records; skipping.")
                continue
            take = min(quota, n)
            idx = rng.choice(n, size=take, replace=False).tolist()
            parts.append(LabeledFeatureDataset(base, class_to_idx, idx))
            got += take
        note = "" if got == n_per_class else f"  [warn] short of {n_per_class}"
        print(f"  class '{cls}': {got} train images from {m} campaign(s){note}")
    return ConcatDataset(parts)


def build_test_dataset(
    test_base: CSVSplitDataset,
    class_to_idx: dict[str, int],
    per_campaign: int | None,
    max_total: int | None,
    seed: int,
) -> LabeledFeatureDataset:
    """The full test split, with optional debug caps (applied in this order):
    per_campaign -- keep at most this many random test images *per campaign*.
    A balanced subset that keeps every class represented -- better for a quick
    debug pass than a raw total cap, whose random draw can starve the small
    campaigns (Hawaii is ~40% of the test split).
    """
    if per_campaign is None and max_total is None:
        return LabeledFeatureDataset(test_base, class_to_idx)

    rng = np.random.default_rng(seed)
    if per_campaign is None:
        idx = list(range(len(test_base)))
    else:
        by_campaign: dict[str, list[int]] = {}
        for j in range(len(test_base)):
            by_campaign.setdefault(test_base.records[j][0], []).append(j)
        idx = []
        for js in by_campaign.values():
            idx.extend(
                rng.choice(js, size=min(per_campaign, len(js)), replace=False).tolist()
            )

    if max_total is not None and len(idx) > max_total:
        idx = rng.choice(idx, size=max_total, replace=False).tolist()
    return LabeledFeatureDataset(test_base, class_to_idx, idx)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pooled_dino(dino_model, rgb: torch.Tensor) -> torch.Tensor:
    """Average-pooled DINOv2 patch embedding for [B, 3, H, W] RGB in [0, 1]."""
    return dino_patch_embed(dino_model, rgb).mean(dim=1)  # [B, 768]


@torch.no_grad()
def generate_conditional(
    cfm, rae, cond: torch.Tensor, n_steps: int, cfg_scale: float
) -> torch.Tensor:
    """CFM images conditioned on a [B, 768] pooled-DINO vector; returns RGB [B,3,224,224]."""
    z = cfm_sample_cfg(cfm, cond, n_steps=n_steps, cfg_scale=cfg_scale)
    gen_rgb, _gen_depth = decode(z, cfm, rae)
    return gen_rgb


def build_train_banks(loader, dino_model, cfm, rae, args):
    """One pass over the train loader -> (real_bank, gen_bank, labels).
    Each batch yields the real RGB and the stored DINO grid for the same images.
    The real embedding is the pooled DINO of the real RGB; the generated
    embedding is the pooled DINO of the image the CFM produces when conditioned
    on that grid's mean (the train-time conditioning vector). Both share the
    """
    want_gen = cfm is not None
    real_feats, gen_feats, all_labels = [], [], []
    for rgb, dino, label in tqdm(loader, desc="train embeddings"):
        rgb = rgb.to(DEVICE, non_blocking=True)
        all_labels.append(label.clone())

        if args.mode in ("real", "both"):
            real_feats.append(pooled_dino(dino_model, rgb).cpu())

        if want_gen:
            cond = dino.to(DEVICE, non_blocking=True).mean(dim=(1, 2))  # [B, 768]
            gen_rgb = generate_conditional(cfm, rae, cond, args.n_steps, args.cfg_scale)
            gen_feats.append(pooled_dino(dino_model, gen_rgb).cpu())

    labels = torch.cat(all_labels)
    real_bank = torch.cat(real_feats) if real_feats else None
    gen_bank = torch.cat(gen_feats) if gen_feats else None
    return real_bank, gen_bank, labels


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
            sims = F.normalize(Q, dim=1) @ self.X.t()  # [B, N], higher = closer
            nn_idx = sims.topk(self.k, dim=1).indices
        else:
            dists = torch.cdist(Q, self.X)  # [B, N]
            nn_idx = dists.topk(self.k, dim=1, largest=False).indices
        votes = F.one_hot(self.y[nn_idx], self.num_classes).sum(dim=1)  # [B, C]
        return votes.argmax(dim=1)  # ties -> lowest class idx


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(test_loader, dino_model, classifiers: dict[str, TorchKNN]):
    """Stream the test split once; predict every classifier from shared embeddings.

    Every classifier is trained in the same pooled-DINO space, so the real test
    embedding is computed once per batch and reused by all of them.
    """
    labels = []
    preds = {name: [] for name in classifiers}
    for rgb, _dino, label in tqdm(test_loader, desc="test predict"):
        rgb = rgb.to(DEVICE, non_blocking=True)
        emb = pooled_dino(dino_model, rgb)
        labels.append(label.clone())
        for name, knn in classifiers.items():
            preds[name].append(knn.predict(emb).cpu())
    labels = torch.cat(labels)
    preds = {name: torch.cat(p) for name, p in preds.items()}
    return preds, labels


def classification_report(
    preds: torch.Tensor, labels: torch.Tensor, idx_to_class: dict[int, str]
) -> dict:
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
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[idx_to_class[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        recalls.append(recall)
        f1s.append(f1)
    overall_acc = float((preds == labels).float().mean())
    return {
        "accuracy": overall_acc,  # micro / overall
        "balanced_accuracy": sum(recalls) / C,  # macro recall (class-balanced)
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
        print(
            f"{c:>{width}}  {m['precision']:6.3f} {m['recall']:6.3f} "
            f"{m['f1']:6.3f} {m['support']:8d}"
        )


def save_results(config: dict, reports: dict[str, dict]):
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if RESULTS_JSON.exists():
        try:
            data = json.loads(RESULTS_JSON.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {RESULTS_JSON} was not valid JSON; starting fresh.")
    run_key = f"k{config['k']}_{config['metric']}_n{config['n_per_class']}"
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
_SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
_INK = "#0b0b0b"  # primary ink
_MUTED = "#898781"  # axis / muted labels
_CBAR_LABEL = "Row-normalized frequency (recall on diagonal)"


def _pub_style():
    """Publication rcParams: Times-like serif (STIX, bundled), embedded fonts.

    STIX matches Times (the ECCV/LNCS body font) and ships with matplotlib, so
    the figure text blends with the paper without needing system Times or LaTeX.
    fonttype 42 embeds TrueType glyphs rather than Type-3 (some venues reject Type-3).
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",  # keep SVG text as selectable text
        }
    )


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


def _draw_confusion(
    ax,
    cm: np.ndarray,
    cmap,
    idx_to_class: dict[int, str],
    show_ylabels: bool = True,
    title: str | None = None,
):
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
    y_labels = [
        f"{s} {d}".strip() + f"\n(n={support[i]:,})" for i, (s, d) in enumerate(pretty)
    ]
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(
        x_labels, rotation=45, ha="right", rotation_mode="anchor", color=_INK
    )
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
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#ffffff" if lum < 0.5 else _INK,
                fontweight="bold" if i == j else "normal",
            )
    return im


def _add_cbar(fig, im, ax):
    cbar = fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.25, 0.5, 0.75, 1.0]
    )
    cbar.set_label(_CBAR_LABEL, color=_INK)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(length=2, colors=_MUTED)
    return cbar


def save_confusion_figures(
    preds: dict[str, torch.Tensor],
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    fig_dir: Path,
    formats: list[str],
):
    """Per-classifier confusion matrices + (when both exist) a stacked comparison.

    Row-normalized so the diagonal is per-campaign recall (prior-invariant); the
    true-class (y) labels carry each campaign's test support so the class
    imbalance is legible on the figure itself. Vector output (PDF/SVG) for the paper.
    """
    _pub_style()
    cmap = LinearSegmentedColormap.from_list("reef_seq_blue", _SEQ_BLUE)
    C = len(idx_to_class)
    fig_dir.mkdir(parents=True, exist_ok=True)
    titles = {
        "real": "Trained on real images",
        "generated": "Trained on generated images",
    }

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    campaigns = sorted(
        set(campaigns_in_split("train")) | set(campaigns_in_split("test"))
    )
    classes = sorted({campaign_to_class(c) for c in campaigns})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    n_classes = len(classes)
    print(f"{len(campaigns)} campaigns -> {n_classes} classes: {classes}")
    for cls in classes:
        members = [c for c in campaigns if campaign_to_class(c) == cls]
        print(f"  {cls}: {members}")
    print(
        f"mode={args.mode}  k={args.k}  metric={args.metric}  "
        f"n_per_class={args.n_per_class}  chance={1.0 / n_classes:.3f}"
    )

    dino_model = load_dinov2().to(DEVICE).eval()

    # CFM + RAE only needed for the generated-image classifier.
    cfm = rae = None
    if args.mode in ("generated", "both"):
        cfm = UNetCFM().to(DEVICE)
        cfm.load_state_dict(
            torch.load(args.cfm_ckpt, map_location=DEVICE, weights_only=True)
        )
        cfm.eval()
        rae = load_rae_frozen(args.rae_ckpt, DEVICE)

    # ---- training banks (class-balanced: n_per_class per class) ----
    train_ds = build_train_dataset(campaigns, class_to_idx, args.n_per_class, args.seed)
    print(
        f"train set: {len(train_ds)} images, {n_classes} classes x "
        f"{args.n_per_class} (class-balanced)"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
    )
    real_bank, gen_bank, train_labels = build_train_banks(
        train_loader, dino_model, cfm, rae, args
    )

    classifiers = {}
    if real_bank is not None:
        classifiers["real"] = TorchKNN(args.k, n_classes, args.metric).fit(
            real_bank, train_labels
        )
    if gen_bank is not None:
        classifiers["generated"] = TorchKNN(args.k, n_classes, args.metric).fit(
            gen_bank, train_labels
        )

    # ---- test on the test split (full by default; optional debug caps) ----
    test_base = CSVSplitDataset(split="test", load_rgb=True)
    test_ds = build_test_dataset(
        test_base, class_to_idx, args.test_per_campaign, args.max_test, args.seed
    )
    full = args.test_per_campaign is None and args.max_test is None
    print(
        f"test set: {len(test_ds)} images"
        + ("" if full else f" (debug cap; from {len(test_base)} full)")
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    preds, labels = evaluate(test_loader, dino_model, classifiers)

    reports = {}
    for name in classifiers:
        reports[name] = classification_report(preds[name], labels, idx_to_class)
        print_report(name, reports[name], idx_to_class)

    if "real" in reports and "generated" in reports:
        # Compare on the imbalance-robust headline metrics, not micro-accuracy
        # (which the large campaigns dominate); see per-class recall invariance.
        for key, tag in (
            ("balanced_accuracy", "balanced acc"),
            ("macro_f1", "macro F1"),
        ):
            r, g = reports["real"][key], reports["generated"][key]
            print(
                f"real vs generated -- {tag}: {r:.4f} vs {g:.4f} "
                f"(generated retains {100 * g / r:.1f}% of real)"
            )

    if not args.no_figures:
        fig_dir = (
            Path(args.fig_dir) if args.fig_dir else (FIG_ROOT / "cfm" / "knn_campaign")
        )
        print(f"\nWriting confusion-matrix figures to {fig_dir} ...")
        save_confusion_figures(preds, labels, idx_to_class, fig_dir, args.fig_formats)

    config = {
        "mode": args.mode,
        "k": args.k,
        "metric": args.metric,
        "n_per_class": args.n_per_class,
        "n_train": int(train_labels.numel()),
        "n_steps": args.n_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "cfm_ckpt": args.cfm_ckpt,
        "rae_ckpt": args.rae_ckpt,
        "test_per_campaign": args.test_per_campaign,
        "max_test": args.max_test,
        "n_test_eval": int(labels.numel()),
        "classes": classes,
        "n_classes": n_classes,
        "chance_accuracy": 1.0 / n_classes,
    }
    save_results(config, reports)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=["real", "generated", "both"],
        default="both",
        help="Which k-NN(s) to train: real images, generated images, or both.",
    )
    p.add_argument(
        "--k", type=int, default=20, help="Number of neighbours for the k-NN vote."
    )
    p.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="k-NN distance (cosine is the standard DINO k-NN metric).",
    )
    p.add_argument(
        "--n_per_class",
        type=int,
        default=1500,
        help="Class-balanced train bank: images per unified class (site), split "
        "evenly across that class's campaigns. Equal per class so the k-NN "
        "vote isn't biased by class size.",
    )
    p.add_argument(
        "--test_per_campaign",
        type=int,
        default=None,
        help="Debug cap: max test images PER campaign (balanced subset; keeps every "
        "class represented). Default None = full test split.",
    )
    p.add_argument(
        "--max_test",
        type=int,
        default=None,
        help="Debug cap: max TOTAL test images (random subset; composes with "
        "--test_per_campaign). Default None = no total cap.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument(
        "--n_steps", type=int, default=50, help="CFM sampling steps (generated mode)."
    )
    p.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
        help="CFM guidance scale (generated mode).",
    )
    p.add_argument(
        "--cfm_ckpt", type=str, default=None, help="CFM checkpoint (generated mode)."
    )
    p.add_argument(
        "--rae_ckpt", type=str, default=None, help="RAE checkpoint (generated mode)."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip writing the confusion-matrix figures.",
    )
    p.add_argument(
        "--fig_dir",
        type=str,
        default=None,
        help="Output dir for figures (default figs/cfm/knn_campaign).",
    )
    p.add_argument(
        "--fig_formats",
        nargs="+",
        default=["pdf", "svg"],
        help="Vector formats to write; 'png' also allowed for a quick preview.",
    )
    args = p.parse_args()

    if args.mode in ("generated", "both") and (not args.cfm_ckpt or not args.rae_ckpt):
        p.error("--cfm_ckpt and --rae_ckpt are required for --mode generated/both")

    main(args)
