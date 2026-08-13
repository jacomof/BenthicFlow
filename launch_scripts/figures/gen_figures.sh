#!/bin/bash
#SBATCH --job-name=gen_figures
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

mkdir -p logs
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

python -u scripts/gen_figures.py --rae-ckpt "$REEF_SCRATCH_ROOT/checkpoints/rae/last.pt" \
    --cfm-ckpt "$REEF_SCRATCH_ROOT/checkpoints/cfm/best_model.pt" \
    --window-overlap 0.8 \
    --avg-end-t 1.0 \
    --num-images 10 \
    #--cache-dir "$PROJECT_ROOT/checkpoints/latent_cache" \ 
