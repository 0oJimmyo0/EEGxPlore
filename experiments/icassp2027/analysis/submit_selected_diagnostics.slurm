#!/usr/bin/env bash
#SBATCH --job-name=ICASSP_DIAG
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/icassp_logs/diagnostics_%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/icassp_logs/diagnostics_%x_%j.err

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch $0 ISRUC|PhysioNet-MI" >&2
  exit 2
fi
if [[ "$1" != "ISRUC" && "$1" != "PhysioNet-MI" ]]; then
  echo "Usage: sbatch $0 ISRUC|PhysioNet-MI" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REPO_DIR="$(realpath -m "$REPO_DIR")"
PYTHON_BIN="${PYTHON_BIN:-/data/neurogroup/mingyangjiang/venvs/cbramod-accre/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/output/icassp2027_revision/selected_diagnostics}"
DATASET="$1"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
export PYTHONNOUSERSITE=1
mkdir -p "$REPO_DIR/icassp_logs" "$OUTPUT_ROOT"

BASE_FROZEN="$REPO_DIR/output/icassp2027_frozen_20260830"
BASE_REVISION="$REPO_DIR/output/icassp2027_revision"
RUN_ROOTS=()
case "$DATASET" in
  ISRUC)
    RUN_ROOTS+=(
      "$BASE_FROZEN/isruc/full/seed_3407"
      "$BASE_FROZEN/isruc/full/seed_2024"
      "$BASE_FROZEN/isruc/full/seed_2027"
      "$BASE_REVISION/isruc/full_attnres_only/seed_3407"
      "$BASE_REVISION/isruc/full_attnres_only/seed_2024"
      "$BASE_REVISION/isruc/full_attnres_only/seed_2027"
      "$BASE_FROZEN/isruc/specialist_augmented_full/seed_3407"
      "$BASE_FROZEN/isruc/specialist_augmented_full/seed_2024"
      "$BASE_FROZEN/isruc/specialist_augmented_full/seed_2027"
    )
    OUTPUT="$OUTPUT_ROOT/isruc_selected_diagnostics.json"
    ;;
  PhysioNet-MI)
    RUN_ROOTS+=(
      "$BASE_FROZEN/physionet_mi/full/seed_3407"
      "$BASE_FROZEN/physionet_mi/full/seed_2024"
      "$BASE_FROZEN/physionet_mi/full/seed_2027"
      "$BASE_REVISION/physionet_mi/full_attnres_only/seed_3407"
      "$BASE_REVISION/physionet_mi/full_attnres_only/seed_2024"
      "$BASE_REVISION/physionet_mi/full_attnres_only/seed_2027"
    )
    RUN_ROOTS+=(
      "$BASE_FROZEN/physionet_mi/specialist_augmented_full/seed_3407"
      "$BASE_FROZEN/physionet_mi/specialist_augmented_full/seed_2024"
      "$BASE_FROZEN/physionet_mi/specialist_augmented_full/seed_2027"
    )
    OUTPUT="$OUTPUT_ROOT/physionet_mi_selected_diagnostics.json"
    ;;
esac

for root in "${RUN_ROOTS[@]}"; do
  [[ -d "$root" ]] || { echo "missing run root: $root" >&2; exit 2; }
done

ARGS=(
  "$REPO_DIR/experiments/icassp2027/analysis/replay_selected_diagnostics.py"
  --cuda "$CUDA_ID"
  --output "$OUTPUT"
)
for root in "${RUN_ROOTS[@]}"; do
  ARGS+=(--run-root "$root")
done

echo "[diagnostics] dataset=$DATASET cuda=$CUDA_ID output=$OUTPUT"
echo "[diagnostics] code_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
exec "$PYTHON_BIN" "${ARGS[@]}"
