#!/bin/bash
# install_gsplat_local.sh -- Install gsplat CUDA extension for local / non-SLURM environments.

export CUDAHOSTCXX=/usr/bin/g++-12
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export MAX_JOBS=2

set -eo pipefail

# ----------------------------- CONFIG ---------------------------------------
CONDA_ENV="${CURRENT_ENV:-ocean2}"
GSPLAT_SRC="git+https://github.com/nerfstudio-project/gsplat.git"
DEFJOBS=$(nproc 2>/dev/null || echo 4); [ "$DEFJOBS" -gt 16 ] && DEFJOBS=16
: "${MAX_JOBS:=$DEFJOBS}"
# ----------------------------------------------------------------------------

banner() { echo; echo "==== $* ===================================================="; }
SECONDS=0

banner "1/5  Source environment & activate conda env: $CONDA_ENV"
if [ -f "env.sh" ]; then
    source env.sh
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

echo "Python binary: $(which python)"

banner "2/5  Check PyTorch & CUDA availability"
if ! python -c "import torch; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| PyTorch CUDA:', torch.version.cuda)"; then
    echo "ERROR: Failed to import PyTorch in environment $CONDA_ENV" >&2
    exit 1
fi

banner "3/5  Check NVCC (CUDA Compiler)"
if command -v nvcc >/dev/null 2>&1; then
    echo "NVCC version: $(nvcc --version | tail -1)"
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    echo "CUDA_HOME set to: $CUDA_HOME"
else
    echo "WARNING: 'nvcc' not found on PATH. If compilation fails, ensure CUDA toolkit is installed and nvcc is on PATH."
fi

# Detect GPU architecture if possible
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_ARCH=$(python -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')")
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$GPU_ARCH}"
    echo "Detected GPU Architecture: $GPU_ARCH (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"
fi

export MAX_JOBS

banner "4/5  Compile + install gsplat from source"
# 1. Clone gsplat
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
# 2. Fix the error directly in ProjectionEWA3DGSFused.cu
# We replace 'cuda::ceil_div(a, b)' with standard integer division '(a + b - 1) / b'
sed -i 's/::cuda::ceil_div<int64_t>(\([^,]*\),\s*\([^)]*\))/(\1 + \2 - 1) \/ \2/g' gsplat/cuda/csrc/*.cu
# 4. Install local fixed repo
pip install . --no-build-isolation

banner "5/5  Verify gsplat installation"
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[skip] No visible CUDA GPU in current shell. Verify after launching with GPU:"
    echo "       python -c \"from gsplat import rasterization; print('gsplat import OK')\""
    echo "BUILD FINISHED in ${SECONDS}s."
    exit 0
fi

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
print("gsplat rasterization backend check: OK", tuple(out.shape))
PY

echo
echo "================  SUCCESS in ${SECONDS}s  ================"
echo "gsplat build verified! You can now run 3D lifting scripts."

rm -rf gsplat  # Clean up source directory after successful build
