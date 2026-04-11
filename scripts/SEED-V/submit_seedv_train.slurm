#!/bin/bash
#SBATCH --job-name=SEEDV_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/SEED-V/out/seedv_%x_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/SEED-V/err/seedv_%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00

set -euo pipefail

module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

REPO_DIR="/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore"
mkdir -p "$REPO_DIR/logs/SEED-V/out" "$REPO_DIR/logs/SEED-V/err"

###############################################################################
# Main knobs (single slurm entrypoint; no wrapper .sh needed)
#
# PROFILE:
#   baseline      -> dense baseline (no MoE)
#   best          -> current best guess (pre_attn + 1 MoE + depth router)
#   conservative  -> sparse conservative (linear router + hard dispatch)
#
# ABLATION_GROUP/ABLATION_VALUE (applied on top of PROFILE, usually PROFILE=best):
#   none
#   attnres       : pre_attn | full
#   moe_layers    : 1 | 2
#   router        : linear | mlp
#   depth_dim     : 15 | 26
#   depth_summary : attn_delta4 | attn_mlp_balanced | attn_mlp_latemix | auto
#   grad          : detached | delayed_unfreeze | trainable
#   branch        : both | spatial_only | spectral_only
#   compact       : none | eeg_summary | psd_summary
#   uniform_warmup: <int>=0
#   shared_blend  : <int>=0 (warmup epochs)
#   component_lr  : off | tuned
#
# Backward compatibility:
#   CASE_NAME supports legacy values:
#   baseline_dense, sparse_conservative, selective_best,
#   ablate_full, ablate_two_moe_layers, ablate_linear_router, ablate_depth_dim26
###############################################################################

CASE_NAME="${CASE_NAME:-}"
PROFILE="${PROFILE:-best}"
ABLATION_GROUP="${ABLATION_GROUP:-none}"
ABLATION_VALUE="${ABLATION_VALUE:-}"

DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb}"
MODEL_ROOT="${MODEL_ROOT:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/SEEDV_sparse_adaptation}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"
SEEDV_SPLIT_MANIFEST="${SEEDV_SPLIT_MANIFEST:-}"

CUDA_ID="${CUDA_ID:-0}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Legacy CASE_NAME mapping to PROFILE + ABLATION_* for one-command parity.
LEGACY_DEPTH_SUMMARY_MODE=""
if [[ -n "$CASE_NAME" ]]; then
  case "$CASE_NAME" in
    baseline_dense)
      PROFILE="baseline"
      ABLATION_GROUP="none"
      ABLATION_VALUE=""
      ;;
    sparse_conservative)
      PROFILE="conservative"
      ABLATION_GROUP="none"
      ABLATION_VALUE=""
      ;;
    selective_best)
      PROFILE="best"
      ABLATION_GROUP="none"
      ABLATION_VALUE=""
      ;;
    ablate_full)
      PROFILE="best"
      ABLATION_GROUP="attnres"
      ABLATION_VALUE="full"
      ;;
    ablate_two_moe_layers)
      PROFILE="best"
      ABLATION_GROUP="moe_layers"
      ABLATION_VALUE="2"
      ;;
    ablate_linear_router)
      PROFILE="best"
      ABLATION_GROUP="router"
      ABLATION_VALUE="linear"
      ;;
    ablate_depth_dim26)
      PROFILE="best"
      ABLATION_GROUP="depth_dim"
      ABLATION_VALUE="26"
      LEGACY_DEPTH_SUMMARY_MODE="auto"
      ;;
    *)
      echo "[seedv-slurm] unknown CASE_NAME=$CASE_NAME" >&2
      exit 2
      ;;
  esac
fi

# Base recipe defaults (PROFILE=best).
ATTNRES_VARIANT="pre_attn"
USE_MOE="1"
MOE_NUM_LAYERS="1"
MOE_NUM_EXPERTS="4"
MOE_CAPACITY_FACTOR="1.0"
MOE_LOAD_BALANCE="0.005"
ROUTER_ARCH="mlp"
ROUTER_MLP_HIDDEN="128"
DEPTH_ROUTER="on"
DEPTH_ROUTER_DIM="15"
DEPTH_SUMMARY_MODE="attn_delta4"
DEPTH_PROBE_MLP_FOR_ROUTER="on"
DEPTH_SUMMARY_GRAD_MODE="detached"
DEPTH_SUMMARY_UNFREEZE_EPOCH="16"
ROUTER_DISPATCH_MODE="soft"
ROUTER_TEMPERATURE="1.5"
ROUTER_ENTROPY_COEF="0.005"
ROUTER_BALANCE_KL_COEF="0.01"
ROUTER_Z_LOSS_COEF="0.001"
ROUTER_JITTER_STD="0.02"
ROUTER_JITTER_FINAL_STD="0.005"
ROUTER_JITTER_ANNEAL_EPOCHS="30"
ROUTER_SOFT_WARMUP_EPOCHS="15"

# New selective/compact knobs (no-op defaults preserve previous behavior).
MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS="${MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS:-0}"
MOE_SHARED_BLEND_WARMUP_EPOCHS="${MOE_SHARED_BLEND_WARMUP_EPOCHS:-0}"
MOE_SHARED_BLEND_START="${MOE_SHARED_BLEND_START:-1.0}"
MOE_SHARED_BLEND_END="${MOE_SHARED_BLEND_END:-0.0}"
MOE_ROUTER_ENTROPY_COEF_SPATIAL="${MOE_ROUTER_ENTROPY_COEF_SPATIAL:--1.0}"
MOE_ROUTER_ENTROPY_COEF_SPECTRAL="${MOE_ROUTER_ENTROPY_COEF_SPECTRAL:--1.0}"
MOE_ROUTER_BALANCE_KL_COEF_SPATIAL="${MOE_ROUTER_BALANCE_KL_COEF_SPATIAL:--1.0}"
MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL="${MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL:--1.0}"
MOE_SPECIALIST_BRANCH_MODE="${MOE_SPECIALIST_BRANCH_MODE:-both}"
MOE_ROUTER_COMPACT_FEATURE_MODE="${MOE_ROUTER_COMPACT_FEATURE_MODE:-none}"
MOE_ROUTER_COMPACT_FEATURE_DIM="${MOE_ROUTER_COMPACT_FEATURE_DIM:-8}"
MOE_EXPERT_INIT_NOISE_STD="${MOE_EXPERT_INIT_NOISE_STD:-0.0}"

# Optional component-wise LR controls.
USE_COMPONENT_LR="${USE_COMPONENT_LR:-0}"
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-1.0}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-1.0}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.0}"
LR_CLASSIFIER_MULT="${LR_CLASSIFIER_MULT:-1.0}"
LR_OTHER_MULT="${LR_OTHER_MULT:-1.0}"

# Escape hatch for future args without editing this script again.
EXTRA_SEEDV_ARGS="${EXTRA_SEEDV_ARGS:-}"

case "$PROFILE" in
  baseline)
    ATTNRES_VARIANT="none"
    USE_MOE="0"
    DEPTH_ROUTER="off"
    ;;
  conservative)
    ATTNRES_VARIANT="pre_attn"
    USE_MOE="1"
    MOE_NUM_LAYERS="1"
    ROUTER_ARCH="linear"
    ROUTER_DISPATCH_MODE="hard_capacity"
    DEPTH_ROUTER="off"
    ;;
  best)
    ;;
  *)
    echo "[seedv-slurm] invalid PROFILE=$PROFILE (expected: baseline|best|conservative)" >&2
    exit 2
    ;;
esac

if [[ -n "$LEGACY_DEPTH_SUMMARY_MODE" ]]; then
  DEPTH_SUMMARY_MODE="$LEGACY_DEPTH_SUMMARY_MODE"
fi

if [[ "$ABLATION_GROUP" != "none" ]]; then
  case "$ABLATION_GROUP" in
    attnres)
      case "$ABLATION_VALUE" in
        pre_attn|full) ATTNRES_VARIANT="$ABLATION_VALUE" ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for attnres: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    moe_layers)
      case "$ABLATION_VALUE" in
        1|2) MOE_NUM_LAYERS="$ABLATION_VALUE" ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for moe_layers: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    router)
      case "$ABLATION_VALUE" in
        linear|mlp) ROUTER_ARCH="$ABLATION_VALUE" ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for router: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    depth_dim)
      case "$ABLATION_VALUE" in
        15|26)
          DEPTH_ROUTER="on"
          DEPTH_ROUTER_DIM="$ABLATION_VALUE"
          ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for depth_dim: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    depth_summary)
      case "$ABLATION_VALUE" in
        attn_delta4|attn_mlp_balanced|attn_mlp_latemix|auto)
          DEPTH_ROUTER="on"
          DEPTH_SUMMARY_MODE="$ABLATION_VALUE"
          ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for depth_summary: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    grad)
      case "$ABLATION_VALUE" in
        detached|delayed_unfreeze|trainable)
          DEPTH_ROUTER="on"
          DEPTH_SUMMARY_GRAD_MODE="$ABLATION_VALUE"
          ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for grad: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    branch)
      case "$ABLATION_VALUE" in
        both|spatial_only|spectral_only)
          MOE_SPECIALIST_BRANCH_MODE="$ABLATION_VALUE"
          ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for branch: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    compact)
      case "$ABLATION_VALUE" in
        none|eeg_summary|psd_summary)
          MOE_ROUTER_COMPACT_FEATURE_MODE="$ABLATION_VALUE"
          ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for compact: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    uniform_warmup)
      if [[ "$ABLATION_VALUE" =~ ^[0-9]+$ ]]; then
        MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS="$ABLATION_VALUE"
      else
        echo "[seedv-slurm] invalid ABLATION_VALUE for uniform_warmup: $ABLATION_VALUE" >&2
        exit 2
      fi
      ;;
    shared_blend)
      if [[ "$ABLATION_VALUE" =~ ^[0-9]+$ ]]; then
        MOE_SHARED_BLEND_WARMUP_EPOCHS="$ABLATION_VALUE"
      else
        echo "[seedv-slurm] invalid ABLATION_VALUE for shared_blend: $ABLATION_VALUE" >&2
        exit 2
      fi
      ;;
    component_lr)
      case "$ABLATION_VALUE" in
        off)
          USE_COMPONENT_LR="0"
          ;;
        tuned)
          USE_COMPONENT_LR="1"
          LR_BACKBONE_MULT="0.6"
          LR_ROUTER_MULT="1.8"
          LR_EXPERT_MULT="1.4"
          LR_CLASSIFIER_MULT="1.2"
          LR_OTHER_MULT="1.0"
          ;;
        *)
          echo "[seedv-slurm] invalid ABLATION_VALUE for component_lr: $ABLATION_VALUE" >&2
          exit 2
          ;;
      esac
      ;;
    *)
      echo "[seedv-slurm] invalid ABLATION_GROUP=$ABLATION_GROUP" >&2
      exit 2
      ;;
  esac
fi

RUN_TAG="$PROFILE"
if [[ "$ABLATION_GROUP" != "none" ]]; then
  RUN_TAG="${RUN_TAG}_ablate_${ABLATION_GROUP}_${ABLATION_VALUE}"
fi
if [[ -n "$CASE_NAME" ]]; then
  RUN_TAG="$CASE_NAME"
fi

MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/$RUN_TAG}"
RUN_NAME="${RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID:-seedv_attnres_full}}"

COMMON_ARGS=(
  --seed 42
  --cuda "$CUDA_ID"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr 2e-4
  --weight_decay 2e-2
  --optimizer AdamW
  --clip_value 1.0
  --dropout 0.3
  --classifier all_patch_reps
  --downstream_dataset SEED-V
  --datasets_dir "$DATASET_DIR"
  --num_of_classes 5
  --label_smoothing 0.05
  --num_workers "$NUM_WORKERS"
  --use_pretrained_weights
  --foundation_dir "$FOUNDATION_DIR"
  --no-tqdm
)

if [[ -n "$SEEDV_SPLIT_MANIFEST" ]]; then
  COMMON_ARGS+=(--seedv_split_manifest "$SEEDV_SPLIT_MANIFEST")
fi

EXTRA_ARGS=()
EXTRA_ARGS+=(--attnres_variant "$ATTNRES_VARIANT")

if [[ "$USE_MOE" == "1" ]]; then
  EXTRA_ARGS+=(
    --moe
    --moe_diagnostics
    --moe_num_layers "$MOE_NUM_LAYERS"
    --moe_num_experts "$MOE_NUM_EXPERTS"
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR"
    --moe_load_balance "$MOE_LOAD_BALANCE"
    --moe_router_arch "$ROUTER_ARCH"
    --moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE"
    --moe_router_temperature "$ROUTER_TEMPERATURE"
    --moe_router_entropy_coef "$ROUTER_ENTROPY_COEF"
    --moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF"
    --moe_router_z_loss_coef "$ROUTER_Z_LOSS_COEF"
    --moe_router_jitter_std "$ROUTER_JITTER_STD"
    --moe_router_jitter_final_std "$ROUTER_JITTER_FINAL_STD"
    --moe_router_jitter_anneal_epochs "$ROUTER_JITTER_ANNEAL_EPOCHS"
    --moe_router_soft_warmup_epochs "$ROUTER_SOFT_WARMUP_EPOCHS"
    --moe_uniform_dispatch_warmup_epochs "$MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS"
    --moe_shared_blend_warmup_epochs "$MOE_SHARED_BLEND_WARMUP_EPOCHS"
    --moe_shared_blend_start "$MOE_SHARED_BLEND_START"
    --moe_shared_blend_end "$MOE_SHARED_BLEND_END"
    --moe_router_entropy_coef_spatial "$MOE_ROUTER_ENTROPY_COEF_SPATIAL"
    --moe_router_entropy_coef_spectral "$MOE_ROUTER_ENTROPY_COEF_SPECTRAL"
    --moe_router_balance_kl_coef_spatial "$MOE_ROUTER_BALANCE_KL_COEF_SPATIAL"
    --moe_router_balance_kl_coef_spectral "$MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL"
    --moe_specialist_branch_mode "$MOE_SPECIALIST_BRANCH_MODE"
    --moe_router_compact_feature_mode "$MOE_ROUTER_COMPACT_FEATURE_MODE"
    --moe_router_compact_feature_dim "$MOE_ROUTER_COMPACT_FEATURE_DIM"
    --moe_expert_init_noise_std "$MOE_EXPERT_INIT_NOISE_STD"
  )

  if [[ "$ROUTER_ARCH" == "mlp" ]]; then
    EXTRA_ARGS+=(--moe_router_mlp_hidden "$ROUTER_MLP_HIDDEN")
  fi

  if [[ "$DEPTH_ROUTER" == "on" ]]; then
    EXTRA_ARGS+=(
      --moe_use_attnres_depth_router_features
      --moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM"
      --moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE"
      --moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE"
      --moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH"
    )
    if [[ "$DEPTH_PROBE_MLP_FOR_ROUTER" == "on" ]]; then
      EXTRA_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
    fi
  fi

  if [[ "$USE_COMPONENT_LR" == "1" ]]; then
    EXTRA_ARGS+=(
      --use_component_lr
      --lr_backbone_mult "$LR_BACKBONE_MULT"
      --lr_router_mult "$LR_ROUTER_MULT"
      --lr_expert_mult "$LR_EXPERT_MULT"
      --lr_classifier_mult "$LR_CLASSIFIER_MULT"
      --lr_other_mult "$LR_OTHER_MULT"
    )
  fi
fi

if [[ -n "$EXTRA_SEEDV_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SEEDV_ARR=($EXTRA_SEEDV_ARGS)
  EXTRA_ARGS+=("${EXTRA_SEEDV_ARR[@]}")
fi

mkdir -p "$MODEL_DIR"

echo "[seedv-slurm] profile=$PROFILE case_name=${CASE_NAME:-<none>} model_dir=$MODEL_DIR" >&2
echo "[seedv-slurm] ablation_group=$ABLATION_GROUP ablation_value=${ABLATION_VALUE:-<none>}" >&2
echo "[seedv-slurm] attnres_variant=$ATTNRES_VARIANT use_moe=$USE_MOE moe_layers=$MOE_NUM_LAYERS" >&2
echo "[seedv-slurm] router_arch=$ROUTER_ARCH dispatch=$ROUTER_DISPATCH_MODE temp=$ROUTER_TEMPERATURE" >&2
echo "[seedv-slurm] depth_router=$DEPTH_ROUTER depth_dim=$DEPTH_ROUTER_DIM depth_summary=$DEPTH_SUMMARY_MODE" >&2
echo "[seedv-slurm] depth_grad_mode=$DEPTH_SUMMARY_GRAD_MODE unfreeze_epoch=$DEPTH_SUMMARY_UNFREEZE_EPOCH" >&2
echo "[seedv-slurm] branch_mode=$MOE_SPECIALIST_BRANCH_MODE compact_mode=$MOE_ROUTER_COMPACT_FEATURE_MODE compact_dim=$MOE_ROUTER_COMPACT_FEATURE_DIM" >&2
echo "[seedv-slurm] uniform_warmup=$MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS shared_blend_warmup=$MOE_SHARED_BLEND_WARMUP_EPOCHS blend=($MOE_SHARED_BLEND_START->$MOE_SHARED_BLEND_END)" >&2
echo "[seedv-slurm] expert_init_noise_std=$MOE_EXPERT_INIT_NOISE_STD component_lr=$USE_COMPONENT_LR" >&2
echo "[seedv-slurm] dataset_dir=$DATASET_DIR split_manifest=${SEEDV_SPLIT_MANIFEST:-<none>}" >&2

time python "$REPO_DIR/finetune_main.py" \
  "${COMMON_ARGS[@]}" \
  --model_dir "$MODEL_DIR" \
  --routing_run_name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}"
