#!/bin/bash
#SBATCH --job-name=find_colorful_refs
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

# CPU/I-O bound scan of the precomputed RGB arrays on shared scratch; no GPU.
# Full pass (--frame-stride 1) over all ScottReef campaigns is ~4 h; drop to
# --frame-stride 5 (~50 min) for a quicker sweep -- adjacent AUV frames are
# near-duplicates, so little is lost.
python -u scripts/find_colorful_refs.py \
    --frame-stride 5 \
    --top 100 \
    --min-sep 50 \
    --prefix Batemans
