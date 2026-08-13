#!/bin/bash
#SBATCH --job-name=generate_3d
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

echo "Hello! from $(hostname) at $(date)"

mkdir -p logs
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

echo "REEF_DATA_SCRATCH_ROOT: $REEF_DATA_SCRATCH_ROOT"


python -u scripts/generate_3d.py --rgb "$PROJECT_ROOT"/data_normalized/Hawaii201801/r20180206_012603_SS15_waikoloa_legs_mid_central/PR_20180206_045531_107_LC16.jpg \
    --out "$REEF_SCRATCH_ROOT"/PR_20180206_045531_107_LC16/ \