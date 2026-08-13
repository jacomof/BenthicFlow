#!/bin/bash
#SBATCH --job-name=reef_rgb_extraction_dpf
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/dpf/train_rae_ddp.sh" >&2; exit 1; }
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

python -u scripts/precompute_rgb_dpf.py \
  --workers 18 \
  --deployments "Batemans201011/r20101119_204443_site6guz_10_densegrids" \
  "Batemans201211/r20121127_204111_Durras_site3guz_01_dense" \
  "Batemans201411/r20141117_053229_06_Burrewarra_BU_DG3_dense" \
  "Hawaii201801/r20180203_182731_SS12_waikoloa_broad_cross_200_150_c_shallow" \
  "ScottReef200907/r20090728_004240_scott_08_long_transect_auv4"
