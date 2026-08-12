# env.sh -- shared configuration sourced by every SLURM job script.
#
# Convention: submit all jobs FROM THE REPOSITORY ROOT, e.g.
#   sbatch slurm/train/train_rae_ddp.sh
# so that this file resolves and PROJECT_ROOT defaults to the repo root.

# Conda/mamba environment name (create it from environment.yml).
export CURRENT_ENV="ocean2"

# Repository root. Defaults to the directory jobs are submitted from.
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$PROJECT_ROOT"

# Fast shared scratch holding the preprocessed dataset (features/, depth/,
# rgb/) and checkpoints/. EDIT THESE for your cluster / username.
export BENTHICFLOW_SCRATCH_ROOT="/scratch-shared/$USER/benthicflow/"
export BENTHICFLOW_DATA_SCRATCH_ROOT="/scratch-shared/$USER/benthicflow/data/"
export REEF_SCRATCH_ROOT="$BENTHICFLOW_SCRATCH_ROOT"
export REEF_DATA_SCRATCH_ROOT="$BENTHICFLOW_DATA_SCRATCH_ROOT"

# Node-local scratch: jobs submitted with --constraint=scratch-node copy the
# dataset here for fast sequential reads (jobs that read from shared scratch
# instead simply re-point this at BENTHICFLOW_SCRATCH_ROOT).
export BENTHICFLOW_SCRATCH_NODE_ROOT="${TMPDIR:-/tmp}/$USER/benthicflow/node/"
export REEF_SCRATCH_NODE_ROOT="$BENTHICFLOW_SCRATCH_NODE_ROOT"

# Model caches, kept off $HOME because of its small quota. EDIT for your user.
export HF_HOME="/scratch-shared/$USER/hf_models"
export TORCH_HOME="/scratch-shared/$USER/torch_models"

# Secrets live in an untracked env.local.sh, NEVER in this file:
#   HF_TOKEN       -- HuggingFace token for gated models (FLUX baselines)
#   SQUIDLE_TOKEN  -- optional Squidle+ API token (higher download rate limits)
# e.g.  echo 'export HF_TOKEN=hf_...' >> env.local.sh
if [ -f "$PROJECT_ROOT/env.local.sh" ]; then
    source "$PROJECT_ROOT/env.local.sh"
fi
