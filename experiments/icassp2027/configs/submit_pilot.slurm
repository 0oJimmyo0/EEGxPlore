#!/bin/bash
# One ICASSP pilot on one A6000. Submit from the EEGxPlore repository root:
#   sbatch --export=ALL,DATASET=SEED-V,METHOD=depth_aggregation experiments/icassp2027/configs/submit_pilot.slurm
#
# The experiment itself remains in run_pilot.sh so local dry-runs and Slurm
# runs use exactly the same command and causal settings.
#SBATCH --job-name=icassp_pilot
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/logs/slurm/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore}"
ENV_DIR="${ENV_DIR:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre}"
DATASET="${DATASET:?DATASET must be exported, e.g. DATASET=SEED-V}"
METHOD="${METHOD:?METHOD must be exported, e.g. METHOD=static}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
REQUIRE_CLEAN_TREE="${REQUIRE_CLEAN_TREE:-1}"
EPOCHS="${EPOCHS:-20}"
if [[ -z "${MODEL_ROOT+x}" ]]; then
  if [[ "$METHOD" == "static" || "$METHOD" == "routed" ]]; then
    MODEL_ROOT="$REPO_DIR/output/icassp2027_pilots"
  else
    MODEL_ROOT="$REPO_DIR/output/icassp2027_depth/health20"
  fi
fi
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
export EPOCHS MODEL_ROOT SEED NUM_WORKERS

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "ERROR: Python environment is unavailable: $ENV_DIR" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: repository is unavailable: $REPO_DIR" >&2
  exit 2
fi

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$EXPECTED_COMMIT" && "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "ERROR: expected commit $EXPECTED_COMMIT but found $ACTUAL_COMMIT" >&2
  exit 1
fi
if [[ "$REQUIRE_CLEAN_TREE" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "ERROR: repository is dirty; commit the locked experiment before submission" >&2
  git status --short >&2
  exit 1
fi

mkdir -p "$REPO_DIR/logs/slurm"
echo "ICASSP pilot: dataset=$DATASET method=$METHOD" >&2
echo "python: $(command -v python)" >&2
echo "sys_prefix: $(python -c 'import sys; print(sys.prefix)')" >&2
echo "git_commit: $ACTUAL_COMMIT" >&2
echo "data_root defaults: /data/neurogroup/mingyangjiang/data" >&2
python - <<'PY'
import os
import sys
import torch

expected = os.path.realpath(os.environ['VIRTUAL_ENV'])
actual = os.path.realpath(sys.prefix)
if actual != expected:
    raise RuntimeError(f'Activated environment mismatch: VIRTUAL_ENV={expected} sys.prefix={actual}')
print('python:', sys.executable)
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError('ICASSP pilot requires CUDA; Slurm allocation did not expose a GPU')
print('gpu0:', torch.cuda.get_device_name(0))
for name in ('lmdb', 'einops'):
    module = __import__(name)
    print(name + ':', getattr(module, '__version__', module.__file__))
PY
nvidia-smi

bash "$REPO_DIR/experiments/icassp2027/configs/run_pilot.sh" "$DATASET" "$METHOD"
