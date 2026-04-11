#!/bin/bash
set -euo pipefail

# Controlled ablations for the stable mainline:
# pre_attn + 1 MoE layer + mlp router + depth dim 15 + soft dispatch + long warmup/jitter.
# Only one factor changes per block.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_ENV=(
  "ATTNRES_VARIANT=pre_attn"
  "DEPTH_ROUTER=on"
  "DEPTH_ROUTER_DIM=15"
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

submit_one() {
  local run_name="$1"
  shift
  echo "[submit] $run_name" >&2
  env "${BASE_ENV[@]}" "RUN_NAME=$run_name" "$@" sbatch submit_train.slurm
}

# A) Depth-summary gradient path ablation (content fixed: attn_delta4)
submit_one "ablate_grad_detached" \
  "DEPTH_SUMMARY_MODE=attn_delta4" \
  "DEPTH_SUMMARY_GRAD_MODE=detached" \
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"

submit_one "ablate_grad_delayed_unfreeze" \
  "DEPTH_SUMMARY_MODE=attn_delta4" \
  "DEPTH_SUMMARY_GRAD_MODE=delayed_unfreeze" \
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"

submit_one "ablate_grad_trainable" \
  "DEPTH_SUMMARY_MODE=attn_delta4" \
  "DEPTH_SUMMARY_GRAD_MODE=trainable" \
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"

# B) Depth-summary content ablation at fixed dim=15 (grad path fixed: detached)
submit_one "ablate_mode_attn_delta4" \
  "DEPTH_SUMMARY_MODE=attn_delta4" \
  "DEPTH_SUMMARY_GRAD_MODE=detached" \
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"

submit_one "ablate_mode_attn_mlp_balanced" \
  "DEPTH_SUMMARY_MODE=attn_mlp_balanced" \
  "DEPTH_SUMMARY_GRAD_MODE=detached" \
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"

submit_one "ablate_mode_attn_mlp_latemix" \
  "DEPTH_SUMMARY_MODE=attn_mlp_latemix" \
  "DEPTH_SUMMARY_GRAD_MODE=detached" \
  "DEPTH_SUMMARY_UNFREEZE_EPOCH=16"

echo "[done] submitted controlled depth-profile ablations" >&2
