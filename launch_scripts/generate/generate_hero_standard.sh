#!/bin/bash
#SBATCH --job-name=generate_hero_scott_rocky_blues
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --constraint=scratch-node

source env.sh 2>/dev/null || { echo "ERROR: env.sh not found - submit jobs from the repo root, e.g. sbatch launch_scripts/train/train_rae_ddp.sh" >&2; exit 1; }


export REEF_SCRATCH_NODE_ROOT="$REEF_SCRATCH_ROOT"

mkdir -p logs

# Initialize conda
eval "$($CONDA_EXE shell.bash hook)"
conda activate "$CURRENT_ENV"
export SSL_CERT_FILE=$(python -m certifi)


python scripts/lift_panorama_hero.py \
  --save-rgbd hero_rgbd.pt --save-all-angles \
  --seeds \
  "$PROJECT_ROOT/data_normalized/ScottReef201503/r20150328_000850_03_scott_grids_deep_auv4_shifted/PR_20150328_032748_933_LC16.jpg" \
  "$PROJECT_ROOT/data_normalized/ScottReef201503/r20150328_000850_03_scott_grids_deep_auv4_shifted/PR_20150328_015302_331_LC16.jpg" \
  "$PROJECT_ROOT/data_normalized/ScottReef201503/r20150329_231742_07_scott_grids_deep_auv3/PR_20150330_024337_271_LC16.jpg" \
  "$PROJECT_ROOT/data_normalized/ScottReef201503/r20150329_231742_07_scott_grids_deep_auv3/PR_20150330_030032_274_LC16.jpg" \
  --steps 500 --stride 6 --cfg 3 --temperature 1.0 \
  --n 6 --pin-corners --cond-gamma 2 \
  --experiment-label ScottReefRockyBlues \