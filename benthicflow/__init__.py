from pathlib import Path
import os

# Absolute path detection (PROJECT_ROOT env, set by env.sh, wins over detection)
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
SCRATCH_ROOT = Path(os.environ.get("BENTHICFLOW_SCRATCH_ROOT", os.environ.get("REEF_SCRATCH_ROOT", PROJECT_ROOT / "scratch")))
SCRATCH_NODE_ROOT = Path(os.environ.get("BENTHICFLOW_SCRATCH_NODE_ROOT", os.environ.get("REEF_SCRATCH_NODE_ROOT", SCRATCH_ROOT / "node")))

# Pipeline variant. BENTHICFLOW_VARIANT=dpf or REEF_VARIANT=dpf selects the BenthicFlow-DPF data roots
# ("-dpf" suffix everywhere): DPF-Net-processed imagery, 266px features, and
# unbounded metric depth. The scripts/*_dpf.py entry points force this
# themselves; job scripts additionally export it for any subprocesses.
VARIANT = os.environ.get("BENTHICFLOW_VARIANT", os.environ.get("REEF_VARIANT", "")).lower()
_SFX = "-dpf" if VARIANT == "dpf" else ""

DATA_ROOT = SCRATCH_ROOT / "data"
FEAT_ROOT = SCRATCH_NODE_ROOT / f"features{_SFX}"
FEATURE_ROOT = FEAT_ROOT  # Alias for backward compatibility
PCA_ROOT  = PROJECT_ROOT / f"pca{_SFX}"
FIG_ROOT  = PROJECT_ROOT / f"figs{_SFX}"
# Base variant keeps normalized imagery in the repo; the DPF variant stores its
# processed imagery on shared scratch (it is produced there by
# dpf_net/clean_and_extract.py).
if VARIANT == "dpf":
    DATA_NORM_ROOT = SCRATCH_ROOT / "data_normalized-dpf"
else:
    DATA_NORM_ROOT = PROJECT_ROOT / "data_normalized"
DEPTH_ROOT     = SCRATCH_NODE_ROOT / f"depth{_SFX}"
RGB_ROOT       = SCRATCH_NODE_ROOT / f"rgb{_SFX}"   # precomputed per-deployment uint8 RGB arrays
CKPT_ROOT      = SCRATCH_ROOT / f"checkpoints{_SFX}"

# Debug print: This will show up in your SLURM .out file
print(f"--- BenthicFlow initialized at {PROJECT_ROOT} (variant: {VARIANT or 'base'}) ---")

# Create directories
for folder in [FEAT_ROOT, PCA_ROOT, FIG_ROOT, DATA_NORM_ROOT, DEPTH_ROOT, RGB_ROOT, CKPT_ROOT]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Verified directory: {folder}")
    except Exception as e:
        print(f"Failed to create {folder}: {e}")

from .rgbd_loader import RGBDImageLoader, NATIVE_SIZE, resize_shorter_then_center_crop
from .rgbd_feature_dataset import RGBDFeatureDataset

