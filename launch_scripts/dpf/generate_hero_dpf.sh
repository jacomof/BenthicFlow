#!/bin/bash
#SBATCH --job-name=generate_hero_no_jitter_dpf
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/dpf/train_rae_ddp.sh" >&2; exit 1; }
cd "$PROJECT_ROOT"

# Select the BenthicFlow-DPF data roots (features-dpf/, depth-dpf/, ...)
export REEF_VARIANT=dpf

# Features/depth/rgb arrays live under REEF_SCRATCH_NODE_ROOT; point it at the
# shared scratch where they were written (mirrors test_generation.sh).
export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"
export SSL_CERT_FILE=$(python -m certifi)


python scripts/lift_panorama_hero.py \
  --save-rgbd hero_rgbd.pt --save-all-angles \
  --seeds \
  "$PROJECT_ROOT/data_normalized-dpf/ScottReef201503/r20150330_225013_09_scott_grids_deep_auv2/PR_20150330_225452_705_LC16.png" \
  "$PROJECT_ROOT/data_normalized-dpf/Batemans201411/r20141112_004500_02_Tollgates_site4sz_dense/PR_20141112_005105_839_LC16.png" \
  "$PROJECT_ROOT/data_normalized-dpf/Hawaii201801/r20180203_182731_SS12_waikoloa_broad_cross_200_150_c_shallow/PR_20180203_183824_386_LC16.png" \
  "$PROJECT_ROOT/data_normalized-dpf/Batemans201411/r20141116_225628_04_Durras_site3guz_dense/PR_20141116_230515_102_LC16.png" \
  --steps 500 --stride 6 --cfg 3.0 --temperature 1.0 \
  --n 6 --pin-corners --cond-gamma 2 \
  --experiment-label DPFSandyRocks \
  --z-span 0.2