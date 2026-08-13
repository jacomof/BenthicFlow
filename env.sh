# env.sh -- shared environment configuration sourced by all scripts and jobs.
#
# Convention: run scripts or submit SLURM jobs FROM THE REPOSITORY ROOT
# so that this file resolves and PROJECT_ROOT defaults to the repo root.

# Conda/mamba environment name (create it from environment.yml).
export CURRENT_ENV="ocean2"

# Repository root. Defaults to the directory jobs are submitted from.
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$PROJECT_ROOT"

# Fast shared scratch / repo root holding the preprocessed dataset (features/, depth/,
# rgb/) and checkpoints/. Defaults to the repository root.
export BENTHICFLOW_SCRATCH_ROOT="${BENTHICFLOW_SCRATCH_ROOT:-$PROJECT_ROOT}"
export BENTHICFLOW_DATA_SCRATCH_ROOT="${BENTHICFLOW_DATA_SCRATCH_ROOT:-$PROJECT_ROOT/data}"
export REEF_SCRATCH_ROOT="$BENTHICFLOW_SCRATCH_ROOT"
export REEF_DATA_SCRATCH_ROOT="$BENTHICFLOW_DATA_SCRATCH_ROOT"

# Node-local scratch: defaults to REEF_SCRATCH_ROOT for local execution, or custom scratch if set.
export BENTHICFLOW_SCRATCH_NODE_ROOT="${BENTHICFLOW_SCRATCH_NODE_ROOT:-$BENTHICFLOW_SCRATCH_ROOT}"
export REEF_SCRATCH_NODE_ROOT="$BENTHICFLOW_SCRATCH_NODE_ROOT"

# Model caches, stored in .cache/ inside the repository root by default.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"

# Secrets live in an untracked env.local.sh, NEVER in this file:
#   HF_TOKEN       -- HuggingFace token for gated models (FLUX baselines)
#   SQUIDLE_TOKEN  -- optional Squidle+ API token (higher download rate limits)
# e.g.  echo 'export HF_TOKEN=hf_...' >> env.local.sh
if [ -f "$PROJECT_ROOT/env.local.sh" ]; then
    source "$PROJECT_ROOT/env.local.sh"
fi
