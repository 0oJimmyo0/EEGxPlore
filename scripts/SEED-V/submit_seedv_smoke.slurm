#!/bin/bash
# ACCRE interactive GPU smoke test for SEED-V.
# Submit with your interactive GPU account, for example:
#   sbatch /path/to/submit_seedv_smoke.slurm
# If your group does not have an interactive GPU QOS, use the batch launcher instead.
# ACCRE reference: https://help.accre.vanderbilt.edu/index.php?title=GPUs_at_ACCRE
#SBATCH --job-name=SEEDV_SMOKE
#SBATCH --account=neurogroup_acc
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/logs/SEED-V/out/seedv_smoke_%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/logs/SEED-V/err/seedv_smoke_%x_%j.err
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

set +u
export USER_IS_ROOT="${USER_IS_ROOT-}"
if ! type module >/dev/null 2>&1; then
  source /etc/accre/lmod.sh
fi
set -u
module --force purge
module load StdEnvACCRE/2023
module load python/3.11.5
module load scipy-stack/2026a
VENV_PREFIX="${VENV_PREFIX:-/data/neurogroup/mingyangjiang/venvs/eegxplore-accre}"
source "$VENV_PREFIX/bin/activate"
export PYTHONNOUSERSITE=1

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore"
LOG_ROOT="$REPO_DIR/logs/SEED-V"
mkdir -p "$LOG_ROOT/out" "$LOG_ROOT/err"

if [[ ! -x "$VENV_PREFIX/bin/python" ]]; then
  echo "[seedv-smoke] venv python not found: $VENV_PREFIX/bin/python" >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/output/seedv_smoke_${SLURM_JOB_ID:-manual}}"

mkdir -p "$MODEL_DIR"

SMOKE_TEST=1 \
EPOCHS="${EPOCHS:-1}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
USE_MOE="${USE_MOE:-0}" \
RUN_MODE="${RUN_MODE:-dense}" \
DATASET_DIR="$DATASET_DIR" \
FOUNDATION_DIR="$FOUNDATION_DIR" \
MODEL_DIR="$MODEL_DIR" \
bash "$REPO_DIR/scripts/run_seedv.sh"
