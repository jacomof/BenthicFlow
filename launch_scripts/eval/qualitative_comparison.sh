#!/bin/bash
#SBATCH --job-name=qualitative_comparison
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"
export SSL_CERT_FILE=$(python -m certifi)

# 1) reef CFM row + Reference depth.
python -u scripts/qualitative_comparison.py --stage cfm

# 2) BenthicFlow-DPF row (same script since the DPF-Net unification -- no
#    separate-repo subprocess or manifest handoff anymore; the stage resolves
#    the -dpf checkpoints/data itself and stays base-variant).
#    libjpeg preload: the cloned env's libtiff needs the env's libjpeg symbols.
LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8:$LD_PRELOAD" \
  python -u scripts/qualitative_comparison.py --stage dpf

# 3) FLUX.2-dev row (4-bit; heaviest -- own process so CFM memory is released).
python -u scripts/qualitative_comparison.py --stage flux

# 4) Assemble the 5-row ECCV figure (CPU; can also be re-run on a login node).
python -u scripts/qualitative_comparison.py --stage assemble
