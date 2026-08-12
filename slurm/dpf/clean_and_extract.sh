#!/bin/bash
#SBATCH --job-name=net-clean-dpf
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch slurm/dpf/train_rae_ddp.sh" >&2; exit 1; }
cd "$PROJECT_ROOT"

# Select the BenthicFlow-DPF data roots (features-dpf/, depth-dpf/, ...)
export REEF_VARIANT=dpf


mkdir -p "./logs"
eval "$($CONDA_EXE shell.bash hook)"
echo "Activating conda environment: $CURRENT_ENV"
conda activate "$CURRENT_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8:$LD_PRELOAD"


pip list | grep timm

# Runs DPF-Net enhancement over one raw deployment (example below), writing the
# processed imagery consumed by the DPF pipeline (REEF_VARIANT=dpf reads it as
# data_normalized-dpf). Repeat / array over deployments to cover the dataset.
python -u dpf_net/clean_and_extract.py \
  --raw_image_path "$REEF_DATA_SCRATCH_ROOT/Batemans201011/r20101117_001912_batemans_03_site1sz/images" \
  --output_path "$REEF_SCRATCH_ROOT/data_normalized-dpf/" \
  --batch_size 8 \
  --device "cuda:0" \
  --split_path "./data_split/"
