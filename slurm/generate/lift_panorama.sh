#!/bin/bash
#SBATCH --job-name=generate_panorama
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=scratch-node

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/train/train_rae_ddp.sh" >&2; exit 1; }

# Ensure we are in the right directory before running scripts
cd "$PROJECT_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

# python -u scripts/test_panorama.py
# python -u scripts/lift_panorama_3d.py --seed 43 --bilinear
python -u scripts/lift_panorama_single.py --image "$PROJECT_ROOT/assets/eccv_method_input_rgb.jpg" \
  --max-side 384 \
  --render-max-side 768 \
  --ss 1 \
  --rasterize-mode classic \
  --z-span 0.55

