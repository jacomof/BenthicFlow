#!/bin/bash
#SBATCH --job-name=flux_generation
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/train/train_rae_ddp.sh" >&2; exit 1; }

echo "Hello!"


export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

export HF_HOME="/scratch-shared/$USER/hf_models"
mkdir -p "$HF_HOME"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"


python scripts/flux_generation.py --model flux1-dev --strength 0.35 --steps 25