#!/bin/bash
#SBATCH --job-name=flux_test_generation_inception_conditional
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
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
export SSL_CERT_FILE=$(python -m certifi)

# Prevent CPU thread thrashing when using DDP
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Download / cache HuggingFace models (FLUX.2-dev 4-bit, ~20 GB) on shared
# scratch rather than in $HOME (small quota). diffusers / huggingface_hub read
# HF_HOME and place the hub cache under $HF_HOME/hub.
export HF_HOME="/scratch-shared/$USER/hf_models"
mkdir -p "$HF_HOME"


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
# Only the depth *_keys.npy index is copied (not the ~700 GB of depth arrays),
# so size that against just those files; features and rgb are copied in full.
DEPTH_KEYS_BYTES=$(find "$REEF_SCRATCH_ROOT/depth" -name '*_keys.npy' -print0 | du -scb --files0-from=- | tail -1 | awk '{print $1}')
FEAT_RGB_BYTES=$(du -sbc "$REEF_SCRATCH_ROOT/features" "$REEF_SCRATCH_ROOT/rgb" | tail -1 | awk '{print $1}')
REQUIRED=$(( DEPTH_KEYS_BYTES + FEAT_RGB_BYTES ))
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

# flux_test_generation.py only reads the *_keys.npy index from the depth dir
# (it maps filenames to the shared DINO/RGB row order); the actual depth arrays
# are never loaded. Copy just the index files, preserving the <campaign>/ layout.
echo "[$(date '+%F %T')] Copying depth *_keys.npy index from $REEF_SCRATCH_ROOT/depth/ to $REEF_SCRATCH_NODE_ROOT/depth/..."
DEPTH_T0=$(date +%s)
rsync -a --include='*/' --include='*_keys.npy' --exclude='*' "$REEF_SCRATCH_ROOT/depth/" "$REEF_SCRATCH_NODE_ROOT/depth/"
DEPTH_DT=$(( $(date +%s) - DEPTH_T0 ))
DEPTH_BYTES=$(du -sb "$REEF_SCRATCH_NODE_ROOT/depth" | awk '{print $1}')
echo "[$(date '+%F %T')] Depth keys copy done in ${DEPTH_DT}s ($(( DEPTH_BYTES / 1024 / 1024 )) MiB)"

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

# Data-parallel across all GPUs on the node: torchrun spawns one process per
# GPU, each running a full FLUX pipeline on its own shard. FID/KID sync across
# ranks in .compute(). --standalone handles single-node rendezvous.
NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[$(date '+%F %T')] Launching flux eval on ${NGPU} GPU(s) via torchrun"
torchrun --standalone --nproc_per_node="${NGPU}" scripts/flux_test_generation.py \
    --mode both \
    --batch_size 16 \
    --n_workers 3 \
    --steps 28 \
    --feature_extractor inception \
    --kid_subset_size 500 \
    --debug_samples 4 \
    --n_samples 1000 \
    --gen_size 512 \
    --eval_size 224 \
    --per_campaign_size 2000 \

