#!/bin/bash
#SBATCH --job-name=test_reconstruction_inception
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=scratch-node

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

# Ensure we are in the right directory before running scripts
cd "$PROJECT_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Fail fast: if any copy below fails (e.g. node-local disk runs out of space),
# stop the job instead of running an eval that will crash with FileNotFoundError.
set -e

if [ "$REEF_SCRATCH_ROOT" != "$REEF_SCRATCH_NODE_ROOT" ]; then
    mkdir -p "$REEF_SCRATCH_NODE_ROOT/features"
    mkdir -p "$REEF_SCRATCH_NODE_ROOT/depth"
    mkdir -p "$REEF_SCRATCH_NODE_ROOT/rgb"

    # --- Node-local disk space guard ---
    TMPDIR_TARGET="${TMPDIR:-/tmp}"
    echo "Checking node-local disk space..."
    REQUIRED=$(du -sbc "$REEF_SCRATCH_ROOT/depth" "$REEF_SCRATCH_ROOT/features" "$REEF_SCRATCH_ROOT/rgb" | tail -1 | awk '{print $1}')
    NEEDED=$(( REQUIRED + REQUIRED / 10 ))
    AVAIL=$(df -B1 --output=avail "$TMPDIR_TARGET" | tail -1 | tr -d ' ')
    echo "  Required (data + 10% margin): $(( NEEDED  / 1024 / 1024 / 1024 )) GiB"
    echo "  Available in $TMPDIR_TARGET:         $(( AVAIL   / 1024 / 1024 / 1024 )) GiB"
    if (( AVAIL < NEEDED )); then
        echo "ERROR: Not enough node-local disk space in $TMPDIR_TARGET. Aborting before copy." >&2
        exit 1
    fi
    echo "Disk space OK. Proceeding with copy to node-local disk."

    COPY_T0=$(date +%s)
    echo "[$(date '+%F %T')] Copying depth/features/rgb to node-local disk..."
    cp -rT "$REEF_SCRATCH_ROOT/depth/"    "$REEF_SCRATCH_NODE_ROOT/depth/"
    cp -rT "$REEF_SCRATCH_ROOT/features/" "$REEF_SCRATCH_NODE_ROOT/features/"
    cp -rT "$REEF_SCRATCH_ROOT/rgb/"      "$REEF_SCRATCH_NODE_ROOT/rgb/"
    COPY_DT=$(( $(date +%s) - COPY_T0 ))
    echo "[$(date '+%F %T')] Copy to node-local disk done in ${COPY_DT}s ($(( COPY_DT / 60 ))m $(( COPY_DT % 60 ))s)"
fi

python -u scripts/test_reconstruction.py \
    --rae_ckpt "$REEF_SCRATCH_ROOT/checkpoints/rae/last.pt" \
    --batch_size 32 \
    --n_workers 15 \
    --feature_extractor inception \
    --mode both \
    --debug_samples 4
