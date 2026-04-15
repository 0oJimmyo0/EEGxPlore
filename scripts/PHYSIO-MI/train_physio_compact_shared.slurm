#!/bin/bash
#SBATCH --job-name=PHYSIO_MOE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/Phy_MI/out/physio_moe_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/Phy_MI/err/physio_moe_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00

set -euo pipefail

module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

REPO_DIR="/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore"
LOG_ROOT="$REPO_DIR/logs/Phy_MI"
mkdir -p "$LOG_ROOT/out" "$LOG_ROOT/err"

# Required placeholders (override via env when launching sbatch).
DATASET_DIR="${DATASET_DIR:-/path/to/physionet_mi_lmdb}"
MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/Physio_MI_moe}"
NUM_CLASSES="${NUM_CLASSES:-4}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"

# Routing mode switch (single script for compact_shared and dual_query).
DEPTH_CONTEXT_MODE="${DEPTH_CONTEXT_MODE:-compact_shared}"  # compact_shared | dual_query_block_typed_proj

RUN_TAG="${RUN_TAG:-${DEPTH_CONTEXT_MODE}}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_physio_${RUN_TAG}}"

# Core training knobs.
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"
CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.75}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.5}"
EFFECTIVE_NUM_BETA="${EFFECTIVE_NUM_BETA:-0.999}"

# LR / optimizer knobs.
LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-2}"

# EMA knobs.
USE_EMA="${USE_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
EMA_EVAL_ONLY="${EMA_EVAL_ONLY:-1}"

# AttnRes + MoE base recipe (best-style downstream defaults).
ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
ATTNRES_START_LAYER="${ATTNRES_START_LAYER:-0}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_ROUTE_MODE="${MOE_ROUTE_MODE:-typed_capacity_domain}"
ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"

# Warmup blend knobs.
MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS="${MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS:-3}"
MOE_SHARED_BLEND_WARMUP_EPOCHS="${MOE_SHARED_BLEND_WARMUP_EPOCHS:-5}"
MOE_SHARED_BLEND_START="${MOE_SHARED_BLEND_START:-1.0}"
MOE_SHARED_BLEND_END="${MOE_SHARED_BLEND_END:-0.0}"

# Depth-summary settings (resolved by mode like ISRUC script).
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-}"
DEPTH_ROUTER_DIM="${DEPTH_ROUTER_DIM:-15}"
DEPTH_BLOCK_COUNT="${DEPTH_BLOCK_COUNT:-4}"

case "$DEPTH_CONTEXT_MODE" in
  compact_shared)
    if [[ -z "$DEPTH_SUMMARY_MODE" ]]; then
      DEPTH_SUMMARY_MODE="attn_delta4"
    fi
    if [[ -z "$DEPTH_PROBE_MLP_FOR_ROUTER" ]]; then
      DEPTH_PROBE_MLP_FOR_ROUTER="1"
    fi
    ;;
  dual_query_block_typed_proj)
    if [[ -z "$DEPTH_SUMMARY_MODE" ]]; then
      DEPTH_SUMMARY_MODE="auto"
    fi
    if [[ -z "$DEPTH_PROBE_MLP_FOR_ROUTER" ]]; then
      DEPTH_PROBE_MLP_FOR_ROUTER="0"
    fi
    ;;
  *)
    echo "[physio-slurm] invalid DEPTH_CONTEXT_MODE=$DEPTH_CONTEXT_MODE (expected compact_shared or dual_query_block_typed_proj)" >&2
    exit 2
    ;;
esac

EMA_ARGS=()
if [[ "$USE_EMA" == "1" ]]; then
  EMA_ARGS+=(--use_ema)
else
  EMA_ARGS+=(--no-use_ema)
fi
EMA_ARGS+=(--ema_decay "$EMA_DECAY")
EMA_ARGS+=(--ema_warmup_steps "$EMA_WARMUP_STEPS")
if [[ "$EMA_EVAL_ONLY" == "1" ]]; then
  EMA_ARGS+=(--ema_eval_only)
else
  EMA_ARGS+=(--no-ema_eval_only)
fi

DEPTH_ARGS=(
  --moe_use_attnres_depth_router_features
  --moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM"
  --moe_attnres_depth_context_mode "$DEPTH_CONTEXT_MODE"
  --moe_attnres_depth_block_count "$DEPTH_BLOCK_COUNT"
)
if [[ "$DEPTH_CONTEXT_MODE" == "dual_query_block_typed_proj" ]]; then
  DEPTH_ARGS+=(--moe_attnres_depth_summary_mode auto)
else
  DEPTH_ARGS+=(--moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE")
  if [[ "$DEPTH_PROBE_MLP_FOR_ROUTER" == "1" ]]; then
    DEPTH_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
fi

CLASS_WEIGHT_ARGS=(
  --class_weight_mode "$CLASS_WEIGHT_MODE"
  --class_weight_clip_min "$CLASS_WEIGHT_CLIP_MIN"
  --class_weight_clip_max "$CLASS_WEIGHT_CLIP_MAX"
  --effective_num_beta "$EFFECTIVE_NUM_BETA"
)

echo "[physio-slurm] mode=$DEPTH_CONTEXT_MODE dataset_dir=$DATASET_DIR num_classes=$NUM_CLASSES" >&2
echo "[physio-slurm] lr/min_lr/wd=($LR/$MIN_LR/$WEIGHT_DECAY)" >&2
echo "[physio-slurm] ema(use/decay/warmup)=($USE_EMA/$EMA_DECAY/$EMA_WARMUP_STEPS)" >&2
echo "[physio-slurm] depth_summary_mode=$DEPTH_SUMMARY_MODE depth_probe_mlp_for_router=$DEPTH_PROBE_MLP_FOR_ROUTER" >&2
echo "[physio-slurm] model_dir=$MODEL_DIR run_name=$RUN_NAME" >&2

python "$REPO_DIR/finetune_main.py" \
  --seed 42 \
  --cuda "$CUDA_DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout 0.3 \
  --classifier all_patch_reps \
  --downstream_dataset PhysioNet-MI \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes "$NUM_CLASSES" \
  --model_dir "$MODEL_DIR" \
  --num_workers "$NUM_WORKERS" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --attnres_variant "$ATTNRES_VARIANT" \
  --attnres_start_layer "$ATTNRES_START_LAYER" \
  --use_pretrained_weights \
  --foundation_dir "$FOUNDATION_DIR" \
  --moe \
  --moe_diagnostics \
  --moe_num_layers "$MOE_NUM_LAYERS" \
  --moe_num_experts "$MOE_NUM_EXPERTS" \
  --moe_route_mode "$MOE_ROUTE_MODE" \
  --moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE" \
  "${DEPTH_ARGS[@]}" \
  --moe_uniform_dispatch_warmup_epochs "$MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS" \
  --moe_shared_blend_warmup_epochs "$MOE_SHARED_BLEND_WARMUP_EPOCHS" \
  --moe_shared_blend_start "$MOE_SHARED_BLEND_START" \
  --moe_shared_blend_end "$MOE_SHARED_BLEND_END" \
  --routing_run_name "$RUN_NAME" \
  "${EMA_ARGS[@]}" \
  "${CLASS_WEIGHT_ARGS[@]}" \
  --no-tqdm
