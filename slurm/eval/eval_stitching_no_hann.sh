#!/bin/bash
#SBATCH --job-name=eval_stitching_no_hann
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/train/train_rae_ddp.sh" >&2; exit 1; }

echo "Starting script"

# Ensure we are in the right directory before running scripts
cd "$PROJECT_ROOT"

export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

echo "Activating conda environment: $CURRENT_ENV"

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

echo "Activated conda environment: $CURRENT_ENV"

python -u scripts/eval_stitching.py --scale 5 \
    --n_images 512 \
    --n_steps 50 \
    --strides 2 4 8 16 \
    --batch_size 64 \
    --decode_batch_size 4 \
    --no_hann \