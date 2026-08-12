#!/bin/bash
#SBATCH --job-name=gsplat_build
#SBATCH --partition=gpu_h100        
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=gsplat_build_%j.log

set -eo pipefail

# ----------------------------- CONFIG ---------------------------------------
CONDA_ENV="ocean2"               # conda env that has your torch
CUDA_MODULE="CUDA/12.8.0"       # MAJOR must match torch.version.cuda
GPU_ARCH="9.0"                  # H100=9.0  (A100=8.0)
GSPLAT_SRC="git+https://github.com/nerfstudio-project/gsplat.git"
DEFJOBS=$(nproc); [ "$DEFJOBS" -gt 16 ] && DEFJOBS=16
: "${MAX_JOBS:=$DEFJOBS}"       # parallel nvcc jobs (~2GB RAM each); override via env
# ----------------------------------------------------------------------------

banner() { echo; echo "==== $* ===================================================="; }
SECONDS=0

banner "1/6  activate conda env: $CONDA_ENV"
source "${HOME}/miniforge3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "python: $(which python)"

banner "2/6  load CUDA toolkit: $CUDA_MODULE"
module load 2025
module load "$CUDA_MODULE"
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found after module load"; exit 1; }
nvcc --version | tail -1

banner "3/6  check torch <-> CUDA major match"
python -c "import torch; print('torch', torch.__version__, '| torch.cuda', torch.version.cuda, '| GPU OK', torch.cuda.is_available())"
TORCH_MAJ=$(python -c "import torch; print((torch.version.cuda or '0').split('.')[0])")
NVCC_MAJ=$(nvcc --version | grep -oP 'release \K[0-9]+')
echo "torch CUDA major=$TORCH_MAJ   nvcc major=$NVCC_MAJ"
if [ "$TORCH_MAJ" != "$NVCC_MAJ" ]; then
  echo "ERROR: CUDA major mismatch. Load a $CUDA_MODULE whose major == $TORCH_MAJ.x"; exit 1
fi

banner "4/6  set build env"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export MAX_JOBS
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST   MAX_JOBS=$MAX_JOBS"

banner "5/6  clean + build gsplat from source  (the slow part: be patient)"
pip uninstall -y gsplat || true
pip install -U ninja setuptools wheel
# --no-build-isolation: compile against the torch already in this env (critical)
# --no-cache-dir:       never reuse the bad wheel
# -v:                   stream ninja's [N/M] progress so it's clearly alive
pip install -v --no-build-isolation --no-cache-dir "$GSPLAT_SRC"

banner "6/6  verify the CUDA backend actually loaded"
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[skip] no visible GPU in this shell -> re-run the check below on a GPU node:"
  echo "       python -c \"from gsplat import rasterization; print('import ok')\""
  echo "BUILD FINISHED in ${SECONDS}s (verification skipped, no GPU here)."
  exit 0
fi
set +e
python - <<'PY'
import torch
from gsplat import rasterization
dev = "cuda"; N = 10
m = torch.randn(N, 3, device=dev); q = torch.randn(N, 4, device=dev)
s = torch.rand(N, 3, device=dev) * 0.1; o = torch.rand(N, device=dev)
c = torch.rand(N, 3, device=dev)
vm = torch.eye(4, device=dev)[None]
K = torch.tensor([[100., 0, 16], [0, 100., 16], [0, 0, 1.]], device=dev)[None]
out, alpha, _ = rasterization(m, q, s, o, c, vm, K, 32, 32, render_mode="RGB+ED")
assert tuple(out.shape) == (1, 32, 32, 4), out.shape
print("gsplat OK", tuple(out.shape))
PY
RC=$?
set -e

echo
if [ $RC -eq 0 ]; then
  echo "================  SUCCESS in ${SECONDS}s  ================"
  echo "gsplat builds and its CUDA backend loads. Your original command now runs:"
  echo "  python scripts/lift_panorama_3d.py --n 2 --seed 0"
else
  echo "================  BUILD OK BUT BACKEND STILL FAILS  ================"
  echo "_C did not load even after a clean source build. This is no longer an"
  echo "install issue -> switch to the pure-torch renderer (no compiled extension):"
  echo "  run lift_panorama_3d.py with --methods mesh now, and ask Claude to add"
  echo "  the --renderer torch fallback for the Gaussian methods."
  exit 1
fi