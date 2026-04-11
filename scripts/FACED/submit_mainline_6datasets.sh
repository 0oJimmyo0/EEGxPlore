#!/bin/bash
set -euo pipefail

# Submit the same stable mainline recipe across six datasets.
# Minimal retuning policy: only DATASET, DATASET_DIR, NUM_CLASSES differ.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_ENV=(
  "ATTNRES_VARIANT=pre_attn"
  "DEPTH_ROUTER=on"
  "DEPTH_ROUTER_DIM=15"
  "DEPTH_SUMMARY_MODE=attn_delta4"
  "DEPTH_SUMMARY_GRAD_MODE=detached"
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"
  "ROUTER_ARCH=mlp"
  "ROUTER_MLP_HIDDEN=128"
  "ROUTER_DISPATCH_MODE=soft"
  "ROUTER_TEMPERATURE=1.8"
  "ROUTER_ENTROPY_COEF=0.01"
  "ROUTER_BALANCE_KL_COEF=0.01"
  "ROUTER_Z_LOSS_COEF=0.001"
  "ROUTER_JITTER_STD=0.02"
  "ROUTER_JITTER_FINAL_STD=0.005"
  "ROUTER_JITTER_ANNEAL_EPOCHS=30"
  "ROUTER_SOFT_WARMUP_EPOCHS=15"
)

submit_ds() {
  local ds="$1"
  local ddir="$2"
  local ncls="$3"
  local run_name="$4"
  echo "[submit] ds=$ds classes=$ncls run=$run_name" >&2
  env "${BASE_ENV[@]}" \
    "DATASET=$ds" \
    "DATASET_DIR=$ddir" \
    "NUM_CLASSES=$ncls" \
    "RUN_NAME=$run_name" \
    sbatch submit_train.slurm
}

# Set dataset directories for your cluster before use.
# The class counts below are placeholders to reduce manual mistakes;
# override if your processed label setup differs.

submit_ds "FACED" "/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED" "9" "mainline_FACED"
submit_ds "SEED-V" "${SEEDV_DIR:-/path/to/SEED-V/processed}" "5" "mainline_SEEDV"
submit_ds "PhysioNet-MI" "${PHYSIO_DIR:-/path/to/PhysioNet-MI/processed}" "4" "mainline_PhysioMI"
submit_ds "SHU-MI" "${SHU_DIR:-/path/to/SHU-MI/processed}" "1" "mainline_SHUMI"
submit_ds "ISRUC" "${ISRUC_DIR:-/path/to/ISRUC/processed}" "5" "mainline_ISRUC"
submit_ds "BCIC-IV-2a" "${BCI2A_DIR:-/path/to/BCIC-IV-2a/processed}" "4" "mainline_BCI2A"

echo "[done] submitted 6-dataset mainline recipe; replace placeholder paths as needed" >&2
