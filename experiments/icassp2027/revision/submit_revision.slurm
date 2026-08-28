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

VENV_PREFIX="${VENV_PREFIX:-/data/neurogroup/mingyangjiang/venvs/eegxplore-accre}"
if [[ -x "$VENV_PREFIX/bin/python" ]]; then
  export PYTHON_BIN="$VENV_PREFIX/bin/python"
  source "$VENV_PREFIX/bin/activate"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p "$REPO_DIR/icassp_logs"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

if [[ -n "$REQUESTED_EPOCHS" ]]; then
  exec bash "$SCRIPT_DIR/run_revision.sh" "$1" "$2" "$SEED" "$PROTOCOL" "$REQUESTED_EPOCHS"
fi
exec bash "$SCRIPT_DIR/run_revision.sh" "$1" "$2" "$SEED" "$PROTOCOL"
