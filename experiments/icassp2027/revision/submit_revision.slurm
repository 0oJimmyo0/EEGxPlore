#!/usr/bin/env bash
#SBATCH --job-name=ICASSP_REV
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/icassp_logs/revision_%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/icassp_logs/revision_%x_%j.err

set -euo pipefail

# Submit one condition/seed per job. Use multiple sbatch jobs to occupy both
# A6000 GPUs; each job remains isolated in its own result directory.
if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch --export=ALL,SEED=42,DATASET_DIR=...,FOUNDATION_DIR=... $0 DATASET CONDITION [SEED] [PROTOCOL] [EPOCHS]" >&2
  exit 2
fi

# Slurm executes a submitted script from a spool path, so BASH_SOURCE is not
# a reliable way to locate the repository. Submit from the repository root or
# provide REPO_DIR explicitly; SLURM_SUBMIT_DIR is the canonical fallback.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_DIR="$(realpath -m "$REPO_DIR")"
SCRIPT_DIR="$REPO_DIR/experiments/icassp2027/revision"
if [[ ! -f "$SCRIPT_DIR/run_revision.sh" ]]; then
  echo "Revision launcher not found under REPO_DIR: $REPO_DIR" >&2
  echo "Submit from the repository root or export REPO_DIR explicitly." >&2
  exit 2
fi
SEED="${3:-${SEED:-42}}"
PROTOCOL="${4:-${REVISION_PROTOCOL:-cbramod_benchmark}}"
REQUESTED_EPOCHS="${5:-${EPOCHS:-}}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

if [[ -z "$EXPECTED_COMMIT" && "${RUN_MODE:-paper}" != "internal" ]]; then
  echo "EXPECTED_COMMIT must be supplied for paper/smoke jobs; submit with --export=ALL,EXPECTED_COMMIT=<immutable-snapshot-sha>" >&2
  exit 2
fi

USE_ACCRE_STACK="${USE_ACCRE_STACK:-0}"
if [[ "$USE_ACCRE_STACK" == "1" ]]; then
  set +u
  export USER_IS_ROOT="${USER_IS_ROOT-}"
  if ! type module >/dev/null 2>&1; then
    source /etc/accre/lmod.sh
  fi
  set -u
  module --force purge
  if type setup_accre_software_stack >/dev/null 2>&1; then
    setup_accre_software_stack
  elif [[ -f /etc/accre/bashrc ]]; then
    source /etc/accre/bashrc
    setup_accre_software_stack
  fi
  module load python/3.11.5
  module load scipy-stack/2026a
fi

VENV_PREFIX="${VENV_PREFIX:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre}"
if [[ -x "$VENV_PREFIX/bin/python" ]]; then
  export PYTHON_BIN="$VENV_PREFIX/bin/python"
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
  source "$VENV_PREFIX/bin/activate"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p "$REPO_DIR/icassp_logs"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" - <<'PY'
import os
import sys

import einops
import lmdb
import numpy
import torch
import yaml

expected = os.path.realpath(os.environ.get('VIRTUAL_ENV', ''))
actual = os.path.realpath(sys.prefix)
if expected and actual != expected:
    raise RuntimeError(f'Activated environment mismatch: VIRTUAL_ENV={expected} sys.prefix={actual}')
print(f'python={sys.executable}', flush=True)
print(f'sys_prefix={sys.prefix}', flush=True)
print(f'numpy={numpy.__version__}', flush=True)
print(f'torch={torch.__version__}', flush=True)
print(f'cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}', flush=True)
print(f'einops={einops.__version__}', flush=True)
print(f'lmdb={lmdb.__file__}', flush=True)
print(f'pyyaml={yaml.__version__}', flush=True)
PY
nvidia-smi || true

# A Slurm-visible GPU can still contain an unrelated process (for example,
# a service launched outside this allocation).  Catch that before model
# initialization rather than spending a paper run on an avoidable OOM.  The
# default threshold leaves ample room for the paper models, whose recorded
# peak is below 9 GB on an A6000, while rejecting the observed ~41 GB VLLM
# contamination.  Set GPU_PREFLIGHT_MAX_USED_MB=0 only for an intentional
# diagnostic override.
GPU_PREFLIGHT_MAX_USED_MB="${GPU_PREFLIGHT_MAX_USED_MB:-12000}"
if [[ "$GPU_PREFLIGHT_MAX_USED_MB" != "0" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os
import sys

import torch

limit_mb = int(os.environ.get("GPU_PREFLIGHT_MAX_USED_MB", "12000"))
if not torch.cuda.is_available():
    raise SystemExit("GPU preflight failed: CUDA is unavailable")
free, total = torch.cuda.mem_get_info(0)
used_mb = (total - free) / (1024 ** 2)
total_mb = total / (1024 ** 2)
print(
    f"gpu_preflight=cuda:0 used_mb={used_mb:.0f} total_mb={total_mb:.0f} "
    f"limit_mb={limit_mb}",
    flush=True,
)
if used_mb > limit_mb:
    raise SystemExit(
        f"GPU preflight failed: assigned GPU already uses {used_mb:.0f} MB "
        f"> limit {limit_mb} MB"
    )
PY
fi

if [[ -n "$REQUESTED_EPOCHS" ]]; then
  exec bash "$SCRIPT_DIR/run_revision.sh" "$1" "$2" "$SEED" "$PROTOCOL" "$REQUESTED_EPOCHS" "${MODEL_ROOT:-}" "$EXPECTED_COMMIT"
fi
exec bash "$SCRIPT_DIR/run_revision.sh" "$1" "$2" "$SEED" "$PROTOCOL" "" "${MODEL_ROOT:-}" "$EXPECTED_COMMIT"
