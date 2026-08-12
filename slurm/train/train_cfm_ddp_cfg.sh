#!/bin/bash
#SBATCH --job-name=reef_cfm_cfg_ddp
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=72:00:00
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

# Prevent CPU thread thrashing when using DDP
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8



# Fail fast: if any copy below fails (e.g. node-local disk runs out of space),
# stop the job instead of launching training that will crash with FileNotFoundError.
set -e

mkdir -p "$REEF_SCRATCH_NODE_ROOT/features"
mkdir -p "$REEF_SCRATCH_NODE_ROOT/depth"
mkdir -p "$REEF_SCRATCH_NODE_ROOT/rgb"

# --- Node-local disk space guard ---
# Abort in seconds if $TMPDIR can't hold the dataset, instead of dying partway
# through a multi-hour copy. Measures actual source size + 10% headroom.
echo "Checking node-local disk space (this reads source metadata, may take a minute)..."
REQUIRED=$(du -sbc "$REEF_SCRATCH_ROOT/depth" "$REEF_SCRATCH_ROOT/features" "$REEF_SCRATCH_ROOT/rgb" | tail -1 | awk '{print $1}')
NEEDED=$(( REQUIRED + REQUIRED / 10 ))
AVAIL=$(df -B1 --output=avail "$TMPDIR" | tail -1 | tr -d ' ')
echo "  Required (data + 10% margin): $(( NEEDED  / 1024 / 1024 / 1024 )) GiB"
echo "  Available in \$TMPDIR:         $(( AVAIL   / 1024 / 1024 / 1024 )) GiB"
if (( AVAIL < NEEDED )); then
    echo "ERROR: Not enough node-local disk space in \$TMPDIR. Aborting before copy." >&2
    exit 1
fi
echo "Disk space OK. Proceeding with copy."

# --- Timed copy to node-local disk (so we can compare against reading from shared) ---
COPY_T0=$(date +%s)

echo "[$(date '+%F %T')] Copying depth data from $REEF_SCRATCH_ROOT/depth/ to $REEF_SCRATCH_NODE_ROOT/depth/..."
DEPTH_T0=$(date +%s)
cp -rT "$REEF_SCRATCH_ROOT/depth/" "$REEF_SCRATCH_NODE_ROOT/depth/"
DEPTH_DT=$(( $(date +%s) - DEPTH_T0 ))
DEPTH_BYTES=$(du -sb "$REEF_SCRATCH_NODE_ROOT/depth" | awk '{print $1}')
echo "[$(date '+%F %T')] Depth copy done in ${DEPTH_DT}s ($(( DEPTH_BYTES / 1024 / 1024 / 1024 )) GiB, ~$(( DEPTH_DT > 0 ? DEPTH_BYTES / 1024 / 1024 / DEPTH_DT : 0 )) MiB/s)"

echo "[$(date '+%F %T')] Copying features from $REEF_SCRATCH_ROOT/features/ to $REEF_SCRATCH_NODE_ROOT/features/..."
FEAT_T0=$(date +%s)
cp -rT "$REEF_SCRATCH_ROOT/features/" "$REEF_SCRATCH_NODE_ROOT/features/"
FEAT_DT=$(( $(date +%s) - FEAT_T0 ))
FEAT_BYTES=$(du -sb "$REEF_SCRATCH_NODE_ROOT/features" | awk '{print $1}')
echo "[$(date '+%F %T')] Feature copy done in ${FEAT_DT}s ($(( FEAT_BYTES / 1024 / 1024 / 1024 )) GiB, ~$(( FEAT_DT > 0 ? FEAT_BYTES / 1024 / 1024 / FEAT_DT : 0 )) MiB/s)"

echo "[$(date '+%F %T')] Copying RGB from $REEF_SCRATCH_ROOT/rgb/ to $REEF_SCRATCH_NODE_ROOT/rgb/..."
RGB_T0=$(date +%s)
cp -rT "$REEF_SCRATCH_ROOT/rgb/" "$REEF_SCRATCH_NODE_ROOT/rgb/"
RGB_DT=$(( $(date +%s) - RGB_T0 ))
RGB_BYTES=$(du -sb "$REEF_SCRATCH_NODE_ROOT/rgb" | awk '{print $1}')
echo "[$(date '+%F %T')] RGB copy done in ${RGB_DT}s ($(( RGB_BYTES / 1024 / 1024 / 1024 )) GiB, ~$(( RGB_DT > 0 ? RGB_BYTES / 1024 / 1024 / RGB_DT : 0 )) MiB/s)"

COPY_DT=$(( $(date +%s) - COPY_T0 ))
echo "[$(date '+%F %T')] Total copy to node-local disk: ${COPY_DT}s ($(( COPY_DT / 60 ))m $(( COPY_DT % 60 ))s)"

RAE_CKPT="/scratch-shared/$USER/reef/checkpoints/rae/last.pt"

torchrun --standalone --nproc_per_node=4 scripts/train_cfm_cfg.py \
    --rae-ckpt "$RAE_CKPT" \
    --run-name unet_cfm_final \
    --epochs 200 \
    --batch 256 \
    --workers 7 \
    --p-uncond 0.1 \
    --gen-every 10 \
    --amp \
    --compile