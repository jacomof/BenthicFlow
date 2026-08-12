#!/bin/bash
#SBATCH --job-name=generate_hero_hawaii
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=scratch-node

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/train/train_rae_ddp.sh" >&2; exit 1; }


export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"
export SSL_CERT_FILE=$(python -m certifi)


python scripts/lift_panorama_hero.py \
  --save-rgbd hero_rgbd.pt --save-all-angles \
  --campaigns Hawaii201801 Hawaii201801 Hawaii201801 Hawaii201801 \
  --steps 500 --stride 4 --cfg 2.0 --temperature 1.0 \
  --n 6 --pin-corners --cond-gamma 2.5 \
  --experiment-label HawaiiLuck
  #--cond-jitter 0.0 --jitter-seed 0

#   --campaigns Hawaii201801 Batemans201211 Batemans201011 ScottReef201108 \
