#!/bin/bash
#SBATCH --job-name=test_generation_dpf
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/dpf/train_rae_ddp.sh" >&2; exit 1; }
cd "$PROJECT_ROOT"

# Select the BenthicFlow-DPF data roots (features-dpf/, depth-dpf/, ...)
export REEF_VARIANT=dpf


echo "Hello from DPF-Net test_generation!"

# Eval is light (a capped sample of the test split), so read the depth/feature/RGB
# arrays straight from shared scratch instead of staging them node-local: point the
# node root (which REEF/__init__ uses for DEPTH_ROOT/FEAT_ROOT/RGB_ROOT) at the
# shared root. (Training uses a real node-local copy + --constraint=scratch-node.)
export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

# DINOv2 / InceptionV3 feature extractors download weights on first use.
export SSL_CERT_FILE=$(python -m certifi)

export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8:$LD_PRELOAD"

# Reference depth for the depth-consistency metric now uses the DPF-Net mechanism
# (DPEM x_d), so --dpem_ckpt / --depth_anything_dir replace reef's --dav2_variant.
# Both default to these PROJECT_ROOT-relative paths, but are passed explicitly for
# reproducibility. Set --skip_depth_consistency to drop the metric (and the models).
python -u scripts/test_generation_dpf.py \
    --cfm_ckpt "$REEF_SCRATCH_ROOT/checkpoints-dpf/unet_cfm_final_dpf/best_model.pt" \
    --rae_ckpt "$REEF_SCRATCH_ROOT/checkpoints-dpf/rae/last.pt" \
    --batch_size 32 \
    --n_steps 50 \
    --n_workers 4 \
    --feature_extractor inception \
    --mode both \
    --sample_size 2000
