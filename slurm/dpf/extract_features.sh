#!/bin/bash
#SBATCH --job-name=reef_features_dpf
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/dpf/train_rae_ddp.sh" >&2; exit 1; }
cd "$PROJECT_ROOT"

# Select the BenthicFlow-DPF data roots (features-dpf/, depth-dpf/, ...)
export REEF_VARIANT=dpf


mkdir -p "$PROJECT_ROOT/logs"
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

CAMPAIGNS=(
  "Batemans201011"
  "Batemans201211"
  "Batemans201411"
  "Hawaii201801"
  "ScottReef200907"
  "ScottReef201108"
  "ScottReef201503"
)

python -u scripts/extract_features_dpf.py \
  --batch 32 \
  --workers 4
