#!/usr/bin/env bash
#SBATCH --job-name=ICASSP_ROUTING_VAL
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/icassp_logs/routing_val_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/icassp_logs/routing_val_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_DIR="$(realpath -m "$REPO_DIR")"
PYTHON_BIN="${PYTHON_BIN:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
OUTPUT="${OUTPUT:-$REPO_DIR/output/icassp2027_revision/full_validation_routing_diagnostics.json}"
AUDIT_CSV="${AUDIT_CSV:-$REPO_DIR/output/icassp2027_frozen_20260830/confirmatory_seed_results.csv}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
[[ -f "$AUDIT_CSV" ]] || { echo "frozen evidence CSV not found: $AUDIT_CSV" >&2; exit 2; }

unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONNOUSERSITE=1
mkdir -p "$REPO_DIR/icassp_logs" "$(dirname "$OUTPUT")"

echo "[routing-val] repo=$REPO_DIR cuda=$CUDA_ID"
echo "[routing-val] audit_csv=$AUDIT_CSV output=$OUTPUT"
echo "[routing-val] code_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
exec "$PYTHON_BIN" "$REPO_DIR/experiments/icassp2027/analysis/full_validation_routing_diagnostics.py" \
  --audit-csv "$AUDIT_CSV" \
  --split val \
  --cuda "$CUDA_ID" \
  --output "$OUTPUT"
