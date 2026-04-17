#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd scripts/Mumtaz
#   bash submit_tuning_sweep.sh
# Optional overrides:
#   DATASET_DIR=/... MODEL_ROOT=/... SEEDS="42 3407" bash submit_tuning_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/Mumtaz2016}"
MODEL_ROOT="${MODEL_ROOT:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/mumtaz_tuning}"
SEEDS="${SEEDS:-42 3407}"

profiles=(
  dense_small_head
  linear_probe_avgpool
  attnres_nomoe
  selective_light_moe
)

submit_one() {
  local profile="$1"
  local seed="$2"
  local run_name="mumtaz_${profile}_s${seed}"
  local model_dir="$MODEL_ROOT/${run_name}"

  echo "[sweep] submit profile=$profile seed=$seed run_name=$run_name" >&2
  sbatch --export=ALL,DATASET_DIR="$DATASET_DIR",TUNE_PROFILE="$profile",RUN_SEED="$seed",MODEL_DIR="$model_dir",RUN_NAME="$run_name" submit_train.slurm
}

mkdir -p "$MODEL_ROOT"

for seed in $SEEDS; do
  for profile in "${profiles[@]}"; do
    submit_one "$profile" "$seed"
  done
done

echo "[sweep] submitted ${#profiles[@]} profiles x $(wc -w <<<"$SEEDS") seeds" >&2
