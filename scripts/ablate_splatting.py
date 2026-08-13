import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from benthicflow import CKPT_ROOT, FIG_ROOT
from models.unet_cfm import UNetCFM
from scripts.lift_panorama_3d import (
    CFM_RUN_NAME,
    RAE_CKPT,
    build_surfels,
    generate_canvas,
    psnr,
    refine_surfels_color_perceptual,
    refine_surfels_source_view,
    render_view,
    ssim,
    unproject,
    write_gaussian_ply,
)
from scripts.test_panorama import sample_conditioning
from scripts.train_cfm_cfg import load_rae_frozen

# ============================================================================= #
#  Ablation grid                                                                #
# ============================================================================= #
# thin_ratio=1.0 -> isotropic 3D Gaussians; hero thin_ratio -> 2D surfels.
ROWS = [
    dict(key="A_iso_init", surfel=False, refine=False, perceptual=False),
    dict(key="B_surfel_init", surfel=True, refine=False, perceptual=False),
    dict(key="C_surfel_ref", surfel=True, refine=True, perceptual=False),
    dict(key="D_full", surfel=True, refine=True, perceptual=True),
    dict(key="E_iso_full", surfel=False, refine=True, perceptual=True),
]
REFERENCE_ROW = "D_full"  # paired deltas are reported against this row


# ============================================================================= #
#  LPIPS (lazy singleton; pip install lpips)                                    #
# ============================================================================= #
_LPIPS = None


def lpips_fn(a_hwc, b_hwc, device):
    """LPIPS(AlexNet) between two HxWx3 [0,1] tensors. Returns float or None
    if the lpips package is unavailable."""
    global _LPIPS
    if _LPIPS is False:
        return None
    if _LPIPS is None:
        try:
            import lpips

            _LPIPS = lpips.LPIPS(net="alex").to(device).eval()
            for p in _LPIPS.parameters():
                p.requires_grad_(False)
        except ImportError:
            print("WARNING: `pip install lpips` for LPIPS; reporting without it.")
            _LPIPS = False
            return None
    to_nchw = lambda t: (t.permute(2, 0, 1)[None] * 2 - 1).to(device)
    with torch.no_grad():
        return float(_LPIPS(to_nchw(a_hwc), to_nchw(b_hwc)))


# ============================================================================= #
#  Per-sample evaluation                                                        #
# ============================================================================= #
def eval_row(row, points, rgb, focal, bg, args, device):
    """Build (+ optionally refine) primitives for one ablation row and render
    the front view. Returns (metrics dict, front render, coverage).

    The perceptual polish returns a small info dict; we capture its
    'perceptual_backend' onto the metrics so a silent backend fallback -- which
    would otherwise look like 'the polish hurts the metrics' -- is auditable
    from the raw journal. Rows that do not run the polish report backend
    'n/a'."""
    t0 = time.time()
    thin = 1.0 if not row["surfel"] else args.thin_ratio
    g = build_surfels(
        points,
        rgb,
        k_overlap=args.k_overlap,
        thin_ratio=thin,
        max_scale_mult=args.max_scale_mult,
        normal_smooth=args.normal_smooth,
        opacity=args.opacity,
    )
    if row["refine"] and args.refine_steps > 0:
        g = refine_surfels_source_view(
            g, rgb, focal, bg, args.rasterize_mode, steps=args.refine_steps
        )
    backend = "n/a"
    if row["perceptual"] and args.perceptual_steps > 0:
        g, pinfo = refine_surfels_color_perceptual(
            g, rgb, focal, bg, args.rasterize_mode, steps=args.perceptual_steps
        )
        backend = str(pinfo.get("perceptual_backend", "unknown"))
    lift_s = time.time() - t0

    W = rgb.shape[1]
    front, alpha = render_view(
        g, torch.eye(4, device=device), focal, W, W, args.ss, bg, args.rasterize_mode
    )

    out = dict(
        psnr=round(psnr(front, rgb), 4),
        ssim=round(ssim(front, rgb), 5),
        l1=round(float((front - rgb).abs().mean()), 5),
        lift_seconds=round(lift_s, 3),
        n_primitives=int(g["means"].shape[0]),
        perceptual_backend=backend,
    )
    lp = lpips_fn(front, rgb, device)
    if lp is not None:
        out["lpips"] = round(lp, 5)
    return out, front, float((alpha > 0.5).float().mean()), g


def generate_sample(model, rae, campaigns, rng, args, device):
    """Draw one conditioning seed (campaign chosen by `rng`) and generate a
    single-window RGB-D sample. The four canvas corners share the SAME
    conditioning vector, so generate_canvas degenerates to a plain conditional
    sample -- keeping the exact generation code path of the hero figure without
    paying for a 4-corner panorama per ablation sample."""
    camp = campaigns[int(rng.integers(len(campaigns)))]
    _, cond = sample_conditioning(camp, rng, device)
    rgb, depth = generate_canvas(
        model,
        rae,
        (cond, cond, cond, cond),
        n=args.canvas_n,
        n_steps=args.steps,
        stride=args.stride,
        cfg_scale=args.cfg,
        temperature=args.temperature,
        device=device,
        verbose=False,
    )
    return rgb, depth, camp


# ============================================================================= #
#  Aggregation + tables                                                         #
# ============================================================================= #
METRIC_ORDER = ["psnr", "ssim", "lpips", "l1", "lift_seconds"]
HIGHER_BETTER = {"psnr", "ssim"}


def aggregate(records):
    """records: list of dicts with keys sample_id, row, <metrics>."""
    by_row = {}
    for r in records:
        by_row.setdefault(r["row"], {}).setdefault(r["sample_id"], r)

    agg = {}
    for row_key, samples in by_row.items():
        agg[row_key] = {"n": len(samples)}
        for m in METRIC_ORDER:
            vals = [v[m] for v in samples.values() if m in v]
            if vals:
                agg[row_key][m] = dict(
                    mean=float(np.mean(vals)), std=float(np.std(vals))
                )
        # majority perceptual backend actually used (audit trail)
        backends = [v.get("perceptual_backend", "n/a") for v in samples.values()]
        non_na = [b for b in backends if b != "n/a"]
        if non_na:
            uniq, cnt = np.unique(non_na, return_counts=True)
            agg[row_key]["perceptual_backend"] = str(uniq[int(cnt.argmax())])
            if len(uniq) > 1:
                agg[row_key]["perceptual_backend_mixed"] = {
                    str(u): int(c) for u, c in zip(uniq, cnt)
                }

    # paired deltas vs the reference row, on the intersection of sample ids
    ref = by_row.get(REFERENCE_ROW, {})
    deltas = {}
    for row_key, samples in by_row.items():
        if row_key == REFERENCE_ROW:
            continue
        common = sorted(set(samples) & set(ref))
        d = {}
        for m in METRIC_ORDER:
            pairs = [
                (samples[i][m], ref[i][m])
                for i in common
                if m in samples[i] and m in ref[i]
            ]
            if pairs:
                diff = np.array([a - b for a, b in pairs])
                if m in HIGHER_BETTER:
                    win = float((diff > 0).mean())
                else:
                    win = float((diff < 0).mean())
                d[m] = dict(
                    mean=float(diff.mean()),
                    std=float(diff.std()),
                    n=len(diff),
                    win_rate=round(win, 3),
                )
        deltas[row_key] = d
    return agg, deltas


CHECK_TEX = {True: r"\cmark", False: r"\xmark"}
CHECK_MD = {True: "\u2713", False: "\u2717"}


def fmt(agg_row, m, decimals):
    if m not in agg_row:
        return "--"
    return f"{agg_row[m]['mean']:.{decimals}f} $\\pm$ {agg_row[m]['std']:.{decimals}f}"


def fmt_md(agg_row, m, decimals):
    if m not in agg_row:
        return "--"
    return f"{agg_row[m]['mean']:.{decimals}f} \u00b1 {agg_row[m]['std']:.{decimals}f}"


def write_tables(agg, deltas, out_dir, n_samples):
    cols = [
        ("psnr", "PSNR $\\uparrow$", 2),
        ("ssim", "SSIM $\\uparrow$", 4),
        ("lpips", "LPIPS $\\downarrow$", 4),
        ("l1", "L1 $\\downarrow$", 4),
    ]
    have_lpips = any("lpips" in agg[r["key"]] for r in ROWS if r["key"] in agg)
    if not have_lpips:
        cols = [c for c in cols if c[0] != "lpips"]

    # ---- LaTeX ----
    lines = [
        "% requires \\usepackage{booktabs,pifont}",
        "% \\newcommand{\\cmark}{\\ding{51}} \\newcommand{\\xmark}{\\ding{55}}",
        "\\begin{table}[t]\\centering",
        f"\\caption{{Ablation of the generate-then-splat rendering over "
        f"{n_samples} generated samples. Metrics score the front re-projection "
        f"against the (single-view) generated reference; mean $\\pm$ std with "
        f"paired inputs across rows. The two optimisation phases are isolated "
        f"in rows B--D; rows A/E vary only the primitive (isotropic vs.\\ 2D "
        f"surfel) under matched settings.}}",
        "\\label{tab:splat_ablation}",
        "\\begin{tabular}{ccc" + "c" * len(cols) + "}",
        "\\toprule",
        "2D surfels & Anchored refine. & Perceptual & "
        + " & ".join(h for _, h, _ in cols)
        + " \\\\",
        "\\midrule",
    ]
    for row in ROWS:
        a = agg.get(row["key"])
        if not a:
            continue
        cells = [
            CHECK_TEX[row["surfel"]],
            CHECK_TEX[row["refine"]],
            CHECK_TEX[row["perceptual"]],
        ]
        cells += [fmt(a, m, d) for m, _, d in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out_dir / "table.tex").write_text("\n".join(lines))

    # ---- Markdown ----
    md = [
        "| 2D surfels | Anchored refine | Perceptual | "
        + " | ".join(
            h.replace("$\\uparrow$", "\u2191").replace("$\\downarrow$", "\u2193")
            for _, h, _ in cols
        )
        + " | n |",
        "|" + "---|" * (3 + len(cols) + 1),
    ]
    for row in ROWS:
        a = agg.get(row["key"])
        if not a:
            continue
        cells = [
            CHECK_MD[row["surfel"]],
            CHECK_MD[row["refine"]],
            CHECK_MD[row["perceptual"]],
        ]
        cells += [fmt_md(a, m, d) for m, _, d in cols]
        cells.append(str(a["n"]))
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append(
        f"Paired deltas vs `{REFERENCE_ROW}` (mean of per-sample diffs, "
        f"with win-rate) are in results.json."
    )
    for row in ROWS:
        a = agg.get(row["key"])
        if a and "perceptual_backend" in a:
            note = a["perceptual_backend"]
            if "perceptual_backend_mixed" in a:
                note += f"  (mixed: {a['perceptual_backend_mixed']})"
            md.append(f"- `{row['key']}` perceptual backend: {note}")
    (out_dir / "table.md").write_text("\n".join(md))


def resolve_ply_sample(spec, records):
    """Turn --ply-sample into a concrete sample id. 'none' -> None (disabled);
    an integer string -> that id; 'best' -> the sample whose D_full LPIPS is
    lowest (or highest PSNR if LPIPS was unavailable)."""
    if spec is None or str(spec).lower() == "none":
        return None
    if str(spec).lower() == "best":
        ref = [r for r in records if r["row"] == REFERENCE_ROW]
        if not ref:
            return None
        if all("lpips" in r for r in ref):
            best = min(ref, key=lambda r: r["lpips"])
        else:
            best = max(ref, key=lambda r: r["psnr"])
        print(
            f"  [ply] 'best' -> sample {best['sample_id']} "
            f"(D_full lpips={best.get('lpips', 'n/a')}, psnr={best['psnr']})"
        )
        return int(best["sample_id"])
    try:
        return int(spec)
    except ValueError:
        print(
            f"  [ply] WARNING: could not parse --ply-sample '{spec}'; "
            f"skipping export"
        )
        return None


def export_example_plys(sid, ply_rows, model, rae, args, bg, device, out_dir):
    """Regenerate sample `sid` (deterministic under the per-sample seed) and
    write one .ply per requested ablation row into
    out_dir/example_ply/sample_XXXX/, for the qualitative 3D figure
    (fig:qualitative-3d). Exporting D_full and E_iso_full from the SAME sample
    lets both be framed at one tilt in SuperSplat for a paired
    """
    pdir = out_dir / "example_ply" / f"sample_{sid:04d}"
    pdir.mkdir(parents=True, exist_ok=True)
    # reproduce the exact sample-`sid` generation stream
    rng = np.random.default_rng(args.seed + 1000 * sid)
    torch.manual_seed(args.seed + 1000 * sid)
    rgb, depth, camp = generate_sample(model, rae, args.campaigns, rng, args, device)
    W = depth.shape[1]
    focal = args.focal or float(W)
    points = unproject(depth, focal, args.z_near, args.z_span)

    # ---- STAGE 1: the generated depth tensor (layout matters!) ----
    dd = depth.detach().float()
    print(
        f"  [depth-dbg] depth: shape={tuple(dd.shape)} "
        f"min={float(dd.min()):.5f} max={float(dd.max()):.5f} "
        f"std={float(dd.std()):.5f}"
    )
    if dd.dim() == 3 and dd.shape[0] in (3, 4):
        print(
            f"  [depth-dbg] !! depth tensor has {dd.shape[0]} leading "
            f"channels -- this looks like an RGB(D) stack, not a single depth "
            f"map. `W = depth.shape[1]` and unproject may be reading the "
            f"wrong axis; the depth channel is likely depth[3] or depth[-1]."
        )
    if float(dd.std()) < 1e-4:
        print(
            f"  [depth-dbg] !! depth is CONSTANT (std<1e-4): the flatness is "
            f"UPSTREAM of unproject -- generate_canvas is returning a flat or "
            f"wrong depth channel for this sample."
        )

    # ---- STAGE 2: the unprojected point cloud ----
    _pts = points.detach().float().reshape(-1, points.shape[-1])
    _z = _pts[:, 2]
    print(
        f"  [depth-dbg] points: shape={tuple(points.shape)}  Z "
        f"min={float(_z.min()):.5f} max={float(_z.max()):.5f} "
        f"std={float(_z.std()):.5f}  (expect Z in "
        f"[{args.z_near:.2f}, {args.z_near + args.z_span:.2f}])"
    )
    if float(_z.std()) < 1e-4 and float(dd.std()) >= 1e-4:
        print(
            f"  [depth-dbg] !! Z is FLAT after unproject though depth had "
            f"relief -- unproject is dropping the depth axis (wrong input "
            f"layout/orientation, or z_span={args.z_span} applied to the "
            f"wrong dim)."
        )

    # relief diagnostics: how much depth variation is there to show off-axis?
    d = depth.detach().float().cpu()
    d_std = float(d.std())
    lo, hi = float(torch.quantile(d, 0.05)), float(torch.quantile(d, 0.95))
    d_span = hi - lo  # normalised [0,1] depth span
    z_span_units = d_span * args.z_span  # same span in lifted camera-range
    relief = {
        "depth_std": round(d_std, 4),
        "depth_span_p5_p95": round(d_span, 4),
        "range_span_p5_p95": round(z_span_units, 4),
    }
    # heuristic: p5-p95 span below ~8% of the [0,1] range is effectively flat
    low_relief = d_span < 0.08

    plt.imsave(pdir / "reference_rgb.png", np.clip(rgb.detach().cpu().numpy(), 0, 1))
    plt.imsave(pdir / "reference_depth.png", d.numpy(), cmap="viridis")

    print(
        f"  [ply] sample {sid} ({camp})  depth std={d_std:.4f} "
        f"p5-p95 span={d_span:.4f} (range {z_span_units:.4f})"
    )
    if low_relief:
        print(
            f"  [ply] WARNING: this sample is nearly flat (depth span "
            f"{d_span:.3f} < 0.08). Surfels and isotropic Gaussians will look "
            f"almost identical off-axis -- pick a different --ply-sample with "
            f"more relief rather than expecting this one to show the "
            f"difference."
        )

    row_by_key = {r["key"]: r for r in ROWS}
    manifest = {
        "sample_id": sid,
        "campaign": camp,
        "relief": relief,
        "low_relief_flag": low_relief,
        "rows": {},
    }
    for key in ply_rows:
        metrics, _front, _cov, g = eval_row(
            row_by_key[key], points, rgb, focal, bg, args, device
        )
        # ---- STAGE 3: the surfel means that get written to the .ply ----
        _mz = g["means"].detach().float()[:, 2]
        print(
            f"  [depth-dbg] {key}: means Z min={float(_mz.min()):.5f} "
            f"max={float(_mz.max()):.5f} std={float(_mz.std()):.5f}"
        )
        if float(_mz.std()) < 1e-4 and float(_z.std()) >= 1e-4:
            print(
                f"  [depth-dbg] !! {key}: surfel Z flat although the point "
                f"cloud had relief -- build_surfels is collapsing positions."
            )
        ply_path = pdir / f"example_{key}.ply"
        write_gaussian_ply(ply_path, g)
        size_mb = ply_path.stat().st_size / 1e6
        manifest["rows"][key] = {**metrics, "ply_mb": round(size_mb, 3)}
        print(
            f"  [ply] wrote {ply_path.name} ({size_mb:.2f} MB)  "
            f"psnr={metrics['psnr']} lpips={metrics.get('lpips', 'n/a')}"
        )
    (pdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  [ply] example export for fig:qualitative-3d -> {pdir}")
    print(
        f"  [ply] open the .ply files in SuperSplat, frame both at the SAME "
        f"tilt/azimuth, and screenshot for the paired comparison."
    )


def save_qualitative(sample_id, rgb, fronts, out_dir):
    qdir = out_dir / "qualitative"
    qdir.mkdir(exist_ok=True)
    n = 1 + len(fronts)
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
    ax[0].imshow(rgb.detach().cpu().numpy())
    ax[0].set_title("generated reference", fontsize=9)
    for a, (key, img) in zip(ax[1:], fronts.items()):
        a.imshow(img.detach().cpu().numpy())
        a.set_title(key, fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(qdir / f"sample_{sample_id:04d}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================= #
#  Driver                                                                       #
# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--campaigns",
        nargs="+",
        required=True,
        help="campaign pool; each sample draws its conditioning "
        "seed from a uniformly random campaign in this list",
    )
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument(
        "--rows",
        nargs="+",
        default=None,
        choices=[r["key"] for r in ROWS],
        help="subset of ablation rows (default: all)",
    )
    # generation (reduced ODE budget: the generator is not under ablation)
    p.add_argument(
        "--canvas-n",
        type=int,
        default=1,
        help="canvas size in windows; 1 = plain conditional sample",
    )
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.0)
    # geometry / lift (hero defaults)
    p.add_argument("--focal", type=float, default=None)
    p.add_argument("--z-near", type=float, default=2.0)
    p.add_argument("--z-span", type=float, default=0.35)
    p.add_argument("--k-overlap", type=float, default=0.75)
    p.add_argument("--thin-ratio", type=float, default=0.1)
    p.add_argument("--normal-smooth", type=int, default=1)
    p.add_argument("--opacity", type=float, default=0.999)
    p.add_argument("--max-scale-mult", type=float, default=3.0)
    p.add_argument("--refine-steps", type=int, default=800)
    p.add_argument("--perceptual-steps", type=int, default=300)
    # rendering
    p.add_argument("--ss", type=int, default=3)
    p.add_argument(
        "--rasterize-mode", choices=["classic", "antialiased"], default="antialiased"
    )
    p.add_argument("--bg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    # bookkeeping
    p.add_argument("--n-qualitative", type=int, default=8)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--out", type=Path, default=None)
    # example .ply export for the qualitative 3D figure (fig:qualitative-3d).
    # Exports the FULL-PIPELINE (D) surfels and, from the SAME sample, the
    # isotropic (E) blobs, so both can be framed at one tilt in SuperSplat for
    # a paired surfel-vs-isotropic screenshot.
    p.add_argument(
        "--ply-sample",
        type=str,
        default="0",
        help="which sample to export as example .ply files: an "
        "integer sample id, or 'best' to pick the sample whose "
        "D_full LPIPS is lowest (falls back to PSNR if LPIPS is "
        "unavailable), or 'none' to disable export",
    )
    p.add_argument(
        "--ply-rows",
        nargs="+",
        default=["D_full", "E_iso_full"],
        choices=[r["key"] for r in ROWS],
        help="which ablation rows to export for the example sample "
        "(default: the shipped surfel pipeline D and its "
        "isotropic counterpart E, for a paired figure)",
    )
    p.add_argument(
        "--ply-only",
        action="store_true",
        help="skip the sweep entirely: load an existing "
        "per_sample.jsonl from --out, regenerate just the "
        "--ply-sample sample, write the example .ply files, and "
        "exit. Use this to (re)make the fig:qualitative-3d "
        "assets from a finished (or partial) run in seconds "
        "rather than re-running the whole ablation. 'best' "
        "still resolves against the loaded records; with an "
        "integer --ply-sample no journal is even required.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in ROWS if args.rows is None or r["key"] in args.rows]
    out_dir = args.out or (FIG_ROOT / CFM_RUN_NAME / "ablation_splatting")
    out_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.tensor(args.bg, dtype=torch.float32, device=device)

    (out_dir / "config.json").write_text(
        json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            | {
                "rows": [r["key"] for r in rows],
                "reference_row": REFERENCE_ROW,
                "single_view_only": True,
            },
            indent=2,
        )
    )

    # ---- resume journal ----
    journal = out_dir / "per_sample.jsonl"
    records = []
    if journal.exists():
        records = [json.loads(l) for l in journal.read_text().splitlines() if l]
        print(f"Resuming: {len(records)} records found in {journal.name}")
    done_ids = {
        r["sample_id"]
        for r in records
        if sum(x["sample_id"] == r["sample_id"] for x in records) >= len(rows)
    }

    print("Loading CFM + RAE ...")
    model = UNetCFM().to(device)
    model.load_state_dict(
        torch.load(
            CKPT_ROOT / CFM_RUN_NAME / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    rae = load_rae_frozen(RAE_CKPT, device)
    rae.eval()

    # ---- fast path: just (re)generate the example .ply and exit ----
    if args.ply_only:
        ply_sid = resolve_ply_sample(args.ply_sample, records)
        if ply_sid is None:
            sys.exit(
                "--ply-only needs a concrete --ply-sample: pass an integer "
                "id, or 'best' with an existing per_sample.jsonl in --out."
            )
        print(
            f"[ply-only] exporting example .ply for fig:qualitative-3d "
            f"(sample {ply_sid}, rows {args.ply_rows}) ..."
        )
        export_example_plys(
            ply_sid, args.ply_rows, model, rae, args, bg, device, out_dir
        )
        print("[ply-only] done; skipping the ablation sweep.")
        return

    jf = journal.open("a")
    t_start = time.time()
    for sid in range(args.n_samples):
        # per-sample RNGs -> identical sample stream regardless of resume point
        rng = np.random.default_rng(args.seed + 1000 * sid)
        torch.manual_seed(args.seed + 1000 * sid)
        if sid in done_ids:
            continue

        rgb, depth, camp = generate_sample(
            model, rae, args.campaigns, rng, args, device
        )
        W = depth.shape[1]
        focal = args.focal or float(W)
        points = unproject(depth, focal, args.z_near, args.z_span)

        fronts, coverage = {}, None
        for row in rows:
            metrics, front, cov, _g = eval_row(
                row, points, rgb, focal, bg, args, device
            )
            coverage = cov  # primitive-independent at full coverage; last wins
            rec = dict(
                sample_id=sid,
                campaign=camp,
                row=row["key"],
                coverage=round(cov, 5),
                **metrics,
            )
            records.append(rec)
            jf.write(json.dumps(rec) + "\n")
            fronts[row["key"]] = front
        jf.flush()

        if sid < args.n_qualitative:
            save_qualitative(sid, rgb, fronts, out_dir)

        n_done = len({r["sample_id"] for r in records})
        rate = (time.time() - t_start) / max(1, n_done - len(done_ids))
        eta_h = rate * (args.n_samples - n_done) / 3600
        cov_flag = (
            ""
            if coverage is None or coverage > 0.99
            else f"  COVERAGE={coverage:.3f}(!)"
        )
        print(
            f"[{n_done}/{args.n_samples}] sample {sid} ({camp})  "
            f"~{rate:.1f}s/sample  ETA {eta_h:.2f}h{cov_flag}"
        )

        # refresh tables every 25 samples so partial results are inspectable
        if n_done % 25 == 0 or n_done == args.n_samples:
            agg, deltas = aggregate(records)
            write_tables(agg, deltas, out_dir, n_done)
            (out_dir / "results.json").write_text(
                json.dumps(
                    dict(
                        n_samples=n_done,
                        single_view_only=True,
                        aggregate=agg,
                        paired_deltas_vs_reference=deltas,
                        reference_row=REFERENCE_ROW,
                    ),
                    indent=2,
                )
            )
    jf.close()

    agg, deltas = aggregate(records)
    n_final = len({r["sample_id"] for r in records})
    write_tables(agg, deltas, out_dir, n_final)
    (out_dir / "results.json").write_text(
        json.dumps(
            dict(
                n_samples=n_final,
                single_view_only=True,
                aggregate=agg,
                paired_deltas_vs_reference=deltas,
                reference_row=REFERENCE_ROW,
            ),
            indent=2,
        )
    )

    # ---- example .ply export for the qualitative 3D figure ----
    ply_sid = resolve_ply_sample(args.ply_sample, records)
    if ply_sid is not None:
        print(
            f"Exporting example .ply for fig:qualitative-3d "
            f"(sample {ply_sid}, rows {args.ply_rows}) ..."
        )
        export_example_plys(
            ply_sid, args.ply_rows, model, rae, args, bg, device, out_dir
        )

    print((out_dir / "table.md").read_text())
    print(
        f"\nDone. Table: {out_dir / 'table.tex'}  "
        f"Raw: {journal}  Aggregates: {out_dir / 'results.json'}"
    )


if __name__ == "__main__":
    main()
