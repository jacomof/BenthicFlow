#!/bin/bash
#SBATCH --job-name=reef_pca
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

set -euo pipefail

mkdir -p logs
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

CAMPAIGNS=(
    ScottReef200907
)

# python -u scripts/extract_features.py \
#     --campaigns "${CAMPAIGNS[@]}" \
#     --variant   dinov2_vitl14 \
#     --batch     64 \
#     --workers   8

# python -u scripts/fit_pca.py


python -u scripts/plot_latent_maps.py --per-deployment

echo "PCA analysis complete"