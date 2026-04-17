#!/bin/bash
#SBATCH --job-name=PHYSIO_MOE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/Phy_MI/out/physio_moe_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/Phy_MI/err/physio_moe_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
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

DATASET="${DATASET:-PhysioNet-MI}"
if [[ "$DATASET" != "PhysioNet-MI" ]]; then
  echo "[physio-slurm] DATASET must be PhysioNet-MI (got: $DATASET)" >&2
  exit 2
fi

# Required placeholders (override via env when launching sbatch).
DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/physionet_mi}"
MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/Physio_MI_moe}"
NUM_CLASSES="${NUM_CLASSES:-4}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[physio-slurm] expected PhysioNet-MI LMDB directory missing: $DATASET_DIR" >&2
  exit 2
fi

apply_bank_reg_first_try() {
  if [[ "$MOE_ROUTER_ENTROPY_COEF_SPATIAL" == "-1.0" ]]; then
    MOE_ROUTER_ENTROPY_COEF_SPATIAL="0.02"
  fi
  if [[ "$MOE_ROUTER_ENTROPY_COEF_SPECTRAL" == "-1.0" ]]; then
    MOE_ROUTER_ENTROPY_COEF_SPECTRAL="0.01"
  fi
  if [[ "$MOE_ROUTER_BALANCE_KL_COEF_SPATIAL" == "-1.0" ]]; then
    MOE_ROUTER_BALANCE_KL_COEF_SPATIAL="0.3"
  fi
  if [[ "$MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL" == "-1.0" ]]; then
    MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL="0.015"
  fi
}

# Routing mode switch. Default baseline mirrors the stronger compact-shared EMA run.
DEPTH_CONTEXT_MODE="${DEPTH_CONTEXT_MODE:-compact_shared}"  # compact_shared | block_shared_typed_proj | dual_query_block_typed_proj

# Core training knobs.
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.02}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"  # none | inv_freq_clip | effective_num
CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.85}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.15}"
EFFECTIVE_NUM_BETA="${EFFECTIVE_NUM_BETA:-0.999}"

# LR / optimizer knobs.
LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-2}"

# EMA knobs.
USE_EMA="${USE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
EMA_EVAL_ONLY="${EMA_EVAL_ONLY:-0}"

# AttnRes + MoE base recipe (best-style downstream defaults).
ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
ATTNRES_START_LAYER="${ATTNRES_START_LAYER:-0}"
DEPTH_ROUTER="${DEPTH_ROUTER:-on}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_ROUTE_MODE="${MOE_ROUTE_MODE:-typed_capacity_domain}"
ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"
if [[ -z "${DEPTH_ROUTER_DIM:-}" ]]; then
  if [[ "$ATTNRES_VARIANT" == "pre_attn" ]]; then
    DEPTH_ROUTER_DIM=15
  else
    DEPTH_ROUTER_DIM=26
  fi
fi
ROUTER_ARCH="${ROUTER_ARCH:-mlp}"
ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-128}"

# Warmup blend knobs.
MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS="${MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS:-3}"
MOE_SHARED_BLEND_WARMUP_EPOCHS="${MOE_SHARED_BLEND_WARMUP_EPOCHS:-5}"
MOE_SHARED_BLEND_START="${MOE_SHARED_BLEND_START:-1.0}"
MOE_SHARED_BLEND_END="${MOE_SHARED_BLEND_END:-0.0}"

# Depth-summary settings (resolved by mode like ISRUC script).
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-}"
DEPTH_BLOCK_COUNT="${DEPTH_BLOCK_COUNT:-4}"
DEPTH_SUMMARY_GRAD_MODE="${DEPTH_SUMMARY_GRAD_MODE:-delayed_unfreeze}"
DEPTH_SUMMARY_UNFREEZE_EPOCH="${DEPTH_SUMMARY_UNFREEZE_EPOCH:-8}"

ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.5}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.0}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.0}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-1e-4}"
ROUTER_JITTER_STD="${ROUTER_JITTER_STD:-0.01}"
ROUTER_JITTER_FINAL_STD="${ROUTER_JITTER_FINAL_STD:-0.003}"
ROUTER_JITTER_ANNEAL_EPOCHS="${ROUTER_JITTER_ANNEAL_EPOCHS:-20}"
ROUTER_SOFT_WARMUP_EPOCHS="${ROUTER_SOFT_WARMUP_EPOCHS:-8}"
COMPACT_FEATURE_MODE="${COMPACT_FEATURE_MODE:-none}"  # none | eeg_summary | psd_summary
COMPACT_FEATURE_DIM="${COMPACT_FEATURE_DIM:-8}"
COMPACT_ROUTER_WARMUP_EPOCHS="${COMPACT_ROUTER_WARMUP_EPOCHS:-8}"
COMPACT_ROUTER_GATE_INIT="${COMPACT_ROUTER_GATE_INIT:-1.0}"

# Bank-specific router regularization switches. Keep -1.0 to fall back to globals.
MOE_ROUTER_ENTROPY_COEF_SPATIAL="${MOE_ROUTER_ENTROPY_COEF_SPATIAL:--1.0}"
MOE_ROUTER_ENTROPY_COEF_SPECTRAL="${MOE_ROUTER_ENTROPY_COEF_SPECTRAL:--1.0}"
MOE_ROUTER_BALANCE_KL_COEF_SPATIAL="${MOE_ROUTER_BALANCE_KL_COEF_SPATIAL:--1.0}"
MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL="${MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL:--1.0}"

COMPONENT_LR="${COMPONENT_LR:-off}"  # on | off
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-0.75}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-1.5}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.25}"
LR_CLASSIFIER_MULT="${LR_CLASSIFIER_MULT:-1.0}"
LR_OTHER_MULT="${LR_OTHER_MULT:-1.0}"
DEPTH_BLOCK_SEPARATION_COEF="${DEPTH_BLOCK_SEPARATION_COEF:-0.0}"
DEPTH_BLOCK_SEPARATION_TARGET_JS="${DEPTH_BLOCK_SEPARATION_TARGET_JS:-0.03}"

# Dedicated ablation switch for the three new tests:
#   bank_reg
#   mild_component_lr
#   bank_reg_plus_mild_component_lr
REG_LR_ABLATION="${REG_LR_ABLATION:-bank_reg_plus_mild_component_lr}"  # none | bank_reg | mild_component_lr | bank_reg_plus_mild_component_lr

case "$DEPTH_CONTEXT_MODE" in
  compact_shared)
    if [[ -z "$DEPTH_SUMMARY_MODE" ]]; then
      DEPTH_SUMMARY_MODE="attn_delta4"
    fi
    if [[ -z "$DEPTH_PROBE_MLP_FOR_ROUTER" ]]; then
      DEPTH_PROBE_MLP_FOR_ROUTER="on"
    fi
    ;;
  block_shared_typed_proj)
    if [[ -z "$DEPTH_SUMMARY_MODE" ]]; then
      DEPTH_SUMMARY_MODE="auto"
    fi
    if [[ -z "$DEPTH_PROBE_MLP_FOR_ROUTER" ]]; then
      DEPTH_PROBE_MLP_FOR_ROUTER="off"
    fi
    ;;
  dual_query_block_typed_proj)
    if [[ -z "$DEPTH_SUMMARY_MODE" ]]; then
      DEPTH_SUMMARY_MODE="auto"
    fi
    if [[ -z "$DEPTH_PROBE_MLP_FOR_ROUTER" ]]; then
      DEPTH_PROBE_MLP_FOR_ROUTER="off"
    fi
    ;;
  *)
    echo "[physio-slurm] invalid DEPTH_CONTEXT_MODE=$DEPTH_CONTEXT_MODE (expected compact_shared|block_shared_typed_proj|dual_query_block_typed_proj)" >&2
    exit 2
    ;;
esac

ABLATION_GROUP="${ABLATION_GROUP:-none}"  # none | ema | grad | context
ABLATION_VALUE="${ABLATION_VALUE:-}"
ABLATION_TAG="stable"
case "$ABLATION_GROUP" in
  none)
    ;;
  ema)
    case "$ABLATION_VALUE" in
      on)
        USE_EMA=1
        ABLATION_TAG="ema_on"
        ;;
      off)
        USE_EMA=0
        ABLATION_TAG="ema_off"
        ;;
      *)
        echo "[physio-slurm] ABLATION_GROUP=ema requires ABLATION_VALUE in {on,off}" >&2
        exit 2
        ;;
    esac
    ;;
  grad)
    case "$ABLATION_VALUE" in
      detached|delayed_unfreeze|trainable)
        DEPTH_SUMMARY_GRAD_MODE="$ABLATION_VALUE"
        ABLATION_TAG="grad_${ABLATION_VALUE}"
        ;;
      *)
        echo "[physio-slurm] ABLATION_GROUP=grad requires ABLATION_VALUE in {detached,delayed_unfreeze,trainable}" >&2
        exit 2
        ;;
    esac
    ;;
  context)
    case "$ABLATION_VALUE" in
      compact_shared|block_shared_typed_proj|dual_query_block_typed_proj)
        DEPTH_CONTEXT_MODE="$ABLATION_VALUE"
        if [[ "$DEPTH_CONTEXT_MODE" == "compact_shared" ]]; then
          DEPTH_SUMMARY_MODE="attn_delta4"
          DEPTH_PROBE_MLP_FOR_ROUTER="on"
        else
          DEPTH_SUMMARY_MODE="auto"
          DEPTH_PROBE_MLP_FOR_ROUTER="off"
        fi
        ABLATION_TAG="context_${ABLATION_VALUE}"
        ;;
      *)
        echo "[physio-slurm] ABLATION_GROUP=context requires ABLATION_VALUE in {compact_shared,block_shared_typed_proj,dual_query_block_typed_proj}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "[physio-slurm] invalid ABLATION_GROUP: $ABLATION_GROUP (expected: none|ema|grad|context)" >&2
    exit 2
    ;;
esac

REG_LR_TAG="baseline"
case "$REG_LR_ABLATION" in
  none)
    ;;
  bank_reg)
    apply_bank_reg_first_try
    REG_LR_TAG="bank_reg"
    ;;
  mild_component_lr)
    COMPONENT_LR="on"
    REG_LR_TAG="component_lr_mild"
    ;;
  bank_reg_plus_mild_component_lr)
    COMPONENT_LR="on"
    apply_bank_reg_first_try
    REG_LR_TAG="bank_reg_component_lr_mild"
    ;;
  *)
    echo "[physio-slurm] invalid REG_LR_ABLATION=$REG_LR_ABLATION (expected: none|bank_reg|mild_component_lr|bank_reg_plus_mild_component_lr)" >&2
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

DEPTH_ROUTER_ARGS=()
if [[ "$DEPTH_ROUTER" == "on" ]]; then
  DEPTH_ROUTER_ARGS+=(--moe_use_attnres_depth_router_features --moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_context_mode "$DEPTH_CONTEXT_MODE")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_block_count "$DEPTH_BLOCK_COUNT")
  if [[ "$DEPTH_CONTEXT_MODE" == "block_shared_typed_proj" || "$DEPTH_CONTEXT_MODE" == "dual_query_block_typed_proj" ]]; then
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_mode auto)
  else
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE")
    if [[ "$DEPTH_PROBE_MLP_FOR_ROUTER" == "on" ]]; then
      DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
    fi
  fi
fi

CLASS_WEIGHT_ARGS=(
  --class_weight_mode "$CLASS_WEIGHT_MODE"
  --class_weight_clip_min "$CLASS_WEIGHT_CLIP_MIN"
  --class_weight_clip_max "$CLASS_WEIGHT_CLIP_MAX"
  --effective_num_beta "$EFFECTIVE_NUM_BETA"
)

COMPONENT_LR_ARGS=()
if [[ "$COMPONENT_LR" == "on" ]]; then
  COMPONENT_LR_ARGS+=(
    --use_component_lr
    --lr_backbone_mult "$LR_BACKBONE_MULT"
    --lr_router_mult "$LR_ROUTER_MULT"
    --lr_expert_mult "$LR_EXPERT_MULT"
    --lr_classifier_mult "$LR_CLASSIFIER_MULT"
    --lr_other_mult "$LR_OTHER_MULT"
  )
else
  COMPONENT_LR_ARGS+=(--no-use_component_lr)
fi

RUN_TAG_SUFFIX="$REG_LR_TAG"
if [[ "$ABLATION_GROUP" != "none" ]]; then
  RUN_TAG_SUFFIX="${ABLATION_TAG}_${REG_LR_TAG}"
fi
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX}_compact-${COMPACT_FEATURE_MODE}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_physio_${RUN_TAG_SUFFIX}_${ATTNRES_VARIANT}_${DEPTH_ROUTER}}"
mkdir -p "$MODEL_DIR"

echo "[physio-slurm] dataset=$DATASET dir=$DATASET_DIR num_classes=$NUM_CLASSES" >&2
echo "[physio-slurm] ablation_group=$ABLATION_GROUP ablation_value=${ABLATION_VALUE:-NA} ablation_tag=$ABLATION_TAG" >&2
echo "[physio-slurm] reg_lr_ablation=$REG_LR_ABLATION reg_lr_tag=$REG_LR_TAG" >&2
echo "[physio-slurm] depth_context_mode=$DEPTH_CONTEXT_MODE depth_summary_grad_mode=$DEPTH_SUMMARY_GRAD_MODE" >&2
echo "[physio-slurm] lr/min_lr/wd=($LR/$MIN_LR/$WEIGHT_DECAY)" >&2
echo "[physio-slurm] ema(use/decay/warmup/eval_only)=($USE_EMA/$EMA_DECAY/$EMA_WARMUP_STEPS/$EMA_EVAL_ONLY)" >&2
echo "[physio-slurm] reg(global_entropy/global_kl)=($ROUTER_ENTROPY_COEF/$ROUTER_BALANCE_KL_COEF)" >&2
echo "[physio-slurm] reg(bank_sp_entropy/bank_sc_entropy/bank_sp_kl/bank_sc_kl)=($MOE_ROUTER_ENTROPY_COEF_SPATIAL/$MOE_ROUTER_ENTROPY_COEF_SPECTRAL/$MOE_ROUTER_BALANCE_KL_COEF_SPATIAL/$MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL)" >&2
echo "[physio-slurm] component_lr=$COMPONENT_LR mults=($LR_BACKBONE_MULT/$LR_ROUTER_MULT/$LR_EXPERT_MULT/$LR_CLASSIFIER_MULT/$LR_OTHER_MULT)" >&2
echo "[physio-slurm] compact_summary(mode/dim/warmup/gate)=($COMPACT_FEATURE_MODE/$COMPACT_FEATURE_DIM/$COMPACT_ROUTER_WARMUP_EPOCHS/$COMPACT_ROUTER_GATE_INIT)" >&2
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
  --downstream_dataset "$DATASET" \
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
  --moe_router_arch "$ROUTER_ARCH" \
  --moe_router_mlp_hidden "$ROUTER_MLP_HIDDEN" \
  "${DEPTH_ROUTER_ARGS[@]}" \
  --moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE" \
  --moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH" \
  --moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE" \
  --moe_router_temperature "$ROUTER_TEMPERATURE" \
  --moe_router_entropy_coef "$ROUTER_ENTROPY_COEF" \
  --moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF" \
  --moe_router_entropy_coef_spatial "$MOE_ROUTER_ENTROPY_COEF_SPATIAL" \
  --moe_router_entropy_coef_spectral "$MOE_ROUTER_ENTROPY_COEF_SPECTRAL" \
  --moe_router_balance_kl_coef_spatial "$MOE_ROUTER_BALANCE_KL_COEF_SPATIAL" \
  --moe_router_balance_kl_coef_spectral "$MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL" \
  --moe_router_z_loss_coef "$ROUTER_Z_LOSS_COEF" \
  --moe_router_jitter_std "$ROUTER_JITTER_STD" \
  --moe_router_jitter_final_std "$ROUTER_JITTER_FINAL_STD" \
  --moe_router_jitter_anneal_epochs "$ROUTER_JITTER_ANNEAL_EPOCHS" \
  --moe_router_soft_warmup_epochs "$ROUTER_SOFT_WARMUP_EPOCHS" \
  --moe_router_compact_feature_mode "$COMPACT_FEATURE_MODE" \
  --moe_router_compact_feature_dim "$COMPACT_FEATURE_DIM" \
  --moe_router_compact_warmup_epochs "$COMPACT_ROUTER_WARMUP_EPOCHS" \
  --moe_router_compact_gate_init "$COMPACT_ROUTER_GATE_INIT" \
  --moe_uniform_dispatch_warmup_epochs "$MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS" \
  --moe_shared_blend_warmup_epochs "$MOE_SHARED_BLEND_WARMUP_EPOCHS" \
  --moe_shared_blend_start "$MOE_SHARED_BLEND_START" \
  --moe_shared_blend_end "$MOE_SHARED_BLEND_END" \
  --routing_run_name "$RUN_NAME" \
  "${COMPONENT_LR_ARGS[@]}" \
  --moe_attnres_depth_block_separation_coef "$DEPTH_BLOCK_SEPARATION_COEF" \
  --moe_attnres_depth_block_separation_target_js "$DEPTH_BLOCK_SEPARATION_TARGET_JS" \
  "${EMA_ARGS[@]}" \
  "${CLASS_WEIGHT_ARGS[@]}" \
  --no-tqdm
