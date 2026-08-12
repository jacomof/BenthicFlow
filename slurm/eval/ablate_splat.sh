#!/bin/bash
#SBATCH --job-name=ablate_splatting
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/train/train_rae_ddp.sh" >&2; exit 1; }

echo "Starting script"

cd "$PROJECT_ROOT"

export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

echo "Activating conda environment: $CURRENT_ENV"

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

echo "Activated conda environment: $CURRENT_ENV"

# LPIPS is required for the perceptual metric column (no-op if already present)
pip install --quiet lpips

python -u scripts/ablate_splatting.py \
--campaigns Hawaii201801 Batemans201211 Batemans201011 ScottReef201108 \
--n-samples 500 \
--steps 250 \
--refine-steps 800 \
--perceptual-steps 300 \
--ss 3 \
--rasterize-mode antialiased \
--seed 44 \
--ply-only \
--ply-sample 44