#!/bin/bash
#SBATCH --job-name=test_generation_inception_both_with_depth
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=scratch-node

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

echo "Hello!"


export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"
export SSL_CERT_FILE=$(python -m certifi)


python -u scripts/test_generation.py \
    --cfm_ckpt "$REEF_SCRATCH_ROOT/checkpoints/cfm/best_model.pt" \
    --rae_ckpt "$REEF_SCRATCH_ROOT/checkpoints/rae/last.pt" \
    --batch_size 32 \
    --n_steps 50 \
    --n_workers 4 \
    --feature_extractor inception \
    --mode both \
    --sample_size 2000 \
