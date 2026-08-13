#!/bin/bash
#SBATCH --job-name=flux_knn_campaign_classifier
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

echo "Hello!"

# Read features/rgb/depth-keys straight from shared scratch instead of copying
# them node-local: FLUX generation (seconds per image) dominates runtime, so the
# data reads are not the bottleneck here. Mirrors knn_campaign_classifier.sh.
export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

# Ensure we are in the right directory before running scripts
cd "$PROJECT_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"
export SSL_CERT_FILE=$(python -m certifi)
export PYTHONUNBUFFERED=1

# Prevent CPU thread thrashing under data-parallel (one process per GPU).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Data-parallel across all GPUs on the node: torchrun spawns one process per GPU,
# each running a full FLUX pipeline on its own shard of the train/test sets. The
# per-rank pooled-DINO embeddings are all-gathered before the (rank-0) k-NN fit /
# predict / report. --standalone handles single-node rendezvous.
NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[$(date '+%F %T')] Launching FLUX k-NN campaign classifier on ${NGPU} GPU(s) via torchrun"
torchrun --standalone --nproc_per_node="${NGPU}" scripts/flux_knn_campaign_classifier.py \
    --mode both \
    --k 20 \
    --metric cosine \
    --n_per_class 2000 \
    --batch_size 8 \
    --n_workers 8 \
    --steps 28 \
    --guidance_scale 4.0 \
    --gen_size 512 \
    --eval_size 224 \
