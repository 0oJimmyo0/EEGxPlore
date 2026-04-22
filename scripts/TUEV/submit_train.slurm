#!/bin/bash
#SBATCH --job-name=TUEV_FT
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/TUEV/out/tuev_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/TUEV/err/tuev_%j.err
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
LOG_ROOT="$REPO_DIR/logs/TUEV"
mkdir -p "$LOG_ROOT/out" "$LOG_ROOT/err"

DATASET="${DATASET:-TUEV}"
DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/TUEV/processed}"
NUM_CLASSES="${NUM_CLASSES:-6}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"
MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/tuev_baseline}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_tuev}"

if [[ "$DATASET" != "TUEV" ]]; then
  echo "[tuev-slurm] DATASET must be TUEV (got: $DATASET)" >&2
  exit 2
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[tuev-slurm] datasets dir missing: $DATASET_DIR" >&2
  exit 2
fi

for split_dir in processed_train processed_eval processed_test; do
  if [[ ! -d "$DATASET_DIR/$split_dir" ]]; then
    echo "[tuev-slurm] missing split directory: $DATASET_DIR/$split_dir" >&2
    exit 2
  fi
done

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

is_false() {
  case "${1:-}" in
    0|false|FALSE|False|no|NO|No|off|OFF|Off) return 0 ;;
    *) return 1 ;;
  esac
}

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-5e-5}"
MIN_LR="${MIN_LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-2}"
DROPOUT="${DROPOUT:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
CLASSIFIER="${CLASSIFIER:-all_patch_reps}"
SEED="${SEED:-42}"
SELECTION_METRIC="${SELECTION_METRIC:-balanced_accuracy}"
USE_EMA="${USE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
EMA_EVAL_ONLY="${EMA_EVAL_ONLY:-1}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-inv_freq_clip}"
CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.75}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.5}"
EFFECTIVE_NUM_BETA="${EFFECTIVE_NUM_BETA:-0.999}"
USE_COMPONENT_LR="${USE_COMPONENT_LR:-0}"
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-0.5}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-2.0}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.5}"
LR_CLASSIFIER_MULT="${LR_CLASSIFIER_MULT:-1.0}"
LR_OTHER_MULT="${LR_OTHER_MULT:-1.0}"

ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
USE_MOE="${USE_MOE:-1}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
ROUTER_ARCH="${ROUTER_ARCH:-mlp}"
ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-128}"
ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.5}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.005}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.01}"
ROUTER_ENTROPY_COEF_SPATIAL="${ROUTER_ENTROPY_COEF_SPATIAL:--1.0}"
ROUTER_ENTROPY_COEF_SPECTRAL="${ROUTER_ENTROPY_COEF_SPECTRAL:--1.0}"
ROUTER_BALANCE_KL_COEF_SPATIAL="${ROUTER_BALANCE_KL_COEF_SPATIAL:--1.0}"
ROUTER_BALANCE_KL_COEF_SPECTRAL="${ROUTER_BALANCE_KL_COEF_SPECTRAL:--1.0}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.001}"
ROUTER_JITTER_STD="${ROUTER_JITTER_STD:-0.02}"
ROUTER_JITTER_FINAL_STD="${ROUTER_JITTER_FINAL_STD:-0.005}"
ROUTER_JITTER_ANNEAL_EPOCHS="${ROUTER_JITTER_ANNEAL_EPOCHS:-30}"
ROUTER_SOFT_WARMUP_EPOCHS="${ROUTER_SOFT_WARMUP_EPOCHS:-15}"
DEPTH_ROUTER_DIM="${DEPTH_ROUTER_DIM:-15}"
DEPTH_CONTEXT_MODE="${DEPTH_CONTEXT_MODE:-compact_shared}"
DEPTH_BLOCK_COUNT="${DEPTH_BLOCK_COUNT:-4}"
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-attn_delta4}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-on}"
DEPTH_SUMMARY_GRAD_MODE="${DEPTH_SUMMARY_GRAD_MODE:-delayed_unfreeze}"
DEPTH_SUMMARY_UNFREEZE_EPOCH="${DEPTH_SUMMARY_UNFREEZE_EPOCH:-8}"
DRY_RUN="${DRY_RUN:-0}"

case "$CLASS_WEIGHT_MODE" in
  none|inv_freq_clip|effective_num) ;;
  *)
    echo "[tuev-slurm] invalid CLASS_WEIGHT_MODE: $CLASS_WEIGHT_MODE" >&2
    exit 2
    ;;
esac

case "$ATTNRES_VARIANT" in
  none|final|pre_attn|pre_mlp|full) ;;
  *)
    echo "[tuev-slurm] invalid ATTNRES_VARIANT: $ATTNRES_VARIANT" >&2
    exit 2
    ;;
esac

case "$DEPTH_SUMMARY_GRAD_MODE" in
  detached|delayed_unfreeze|trainable) ;;
  *)
    echo "[tuev-slurm] invalid DEPTH_SUMMARY_GRAD_MODE: $DEPTH_SUMMARY_GRAD_MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$MODEL_DIR"

PY_ARGS=(
  --seed "$SEED"
  --cuda 0
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --min_lr "$MIN_LR"
  --weight_decay "$WEIGHT_DECAY"
  --optimizer AdamW
  --clip_value 1.0
  --dropout "$DROPOUT"
  --classifier "$CLASSIFIER"
  --downstream_dataset "$DATASET"
  --datasets_dir "$DATASET_DIR"
  --num_of_classes "$NUM_CLASSES"
  --model_dir "$MODEL_DIR"
  --label_smoothing "$LABEL_SMOOTHING"
  --num_workers "$NUM_WORKERS"
  --selection_metric "$SELECTION_METRIC"
  --class_weight_mode "$CLASS_WEIGHT_MODE"
  --class_weight_clip_min "$CLASS_WEIGHT_CLIP_MIN"
  --class_weight_clip_max "$CLASS_WEIGHT_CLIP_MAX"
  --effective_num_beta "$EFFECTIVE_NUM_BETA"
  --attnres_variant "$ATTNRES_VARIANT"
  --attnres_start_layer 0
  --use_pretrained_weights
  --foundation_dir "$FOUNDATION_DIR"
  --lr_backbone_mult "$LR_BACKBONE_MULT"
  --lr_router_mult "$LR_ROUTER_MULT"
  --lr_expert_mult "$LR_EXPERT_MULT"
  --lr_classifier_mult "$LR_CLASSIFIER_MULT"
  --lr_other_mult "$LR_OTHER_MULT"
  --ema_decay "$EMA_DECAY"
  --ema_warmup_steps "$EMA_WARMUP_STEPS"
  --no-tqdm
)

if is_true "$USE_EMA"; then
  PY_ARGS+=(--use_ema)
fi
if is_false "$EMA_EVAL_ONLY"; then
  PY_ARGS+=(--no-ema_eval_only)
else
  PY_ARGS+=(--ema_eval_only)
fi
if is_true "$USE_COMPONENT_LR"; then
  PY_ARGS+=(--use_component_lr)
fi

MOE_ARGS=()
if [[ "$USE_MOE" == "1" ]]; then
  MOE_ARGS+=(
    --moe
    --moe_diagnostics
    --moe_num_layers "$MOE_NUM_LAYERS"
    --moe_num_experts "$MOE_NUM_EXPERTS"
    --moe_router_arch "$ROUTER_ARCH"
    --moe_router_mlp_hidden "$ROUTER_MLP_HIDDEN"
    --moe_use_attnres_depth_router_features
    --moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM"
    --moe_attnres_depth_context_mode "$DEPTH_CONTEXT_MODE"
    --moe_attnres_depth_block_count "$DEPTH_BLOCK_COUNT"
    --moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE"
    --moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE"
    --moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH"
    --moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE"
    --moe_router_temperature "$ROUTER_TEMPERATURE"
    --moe_router_entropy_coef "$ROUTER_ENTROPY_COEF"
    --moe_router_entropy_coef_spatial "$ROUTER_ENTROPY_COEF_SPATIAL"
    --moe_router_entropy_coef_spectral "$ROUTER_ENTROPY_COEF_SPECTRAL"
    --moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF"
    --moe_router_balance_kl_coef_spatial "$ROUTER_BALANCE_KL_COEF_SPATIAL"
    --moe_router_balance_kl_coef_spectral "$ROUTER_BALANCE_KL_COEF_SPECTRAL"
    --moe_router_z_loss_coef "$ROUTER_Z_LOSS_COEF"
    --moe_router_jitter_std "$ROUTER_JITTER_STD"
    --moe_router_jitter_final_std "$ROUTER_JITTER_FINAL_STD"
    --moe_router_jitter_anneal_epochs "$ROUTER_JITTER_ANNEAL_EPOCHS"
    --moe_router_soft_warmup_epochs "$ROUTER_SOFT_WARMUP_EPOCHS"
    --moe_uniform_dispatch_warmup_epochs 3
    --moe_shared_blend_warmup_epochs 5
    --moe_shared_blend_start 1.0
    --moe_shared_blend_end 0.0
  )
  if [[ "$DEPTH_PROBE_MLP_FOR_ROUTER" == "on" ]]; then
    MOE_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
fi

echo "[tuev-slurm] dataset_dir=$DATASET_DIR num_classes=$NUM_CLASSES" >&2
echo "[tuev-slurm] run_name=$RUN_NAME model_dir=$MODEL_DIR selection_metric=$SELECTION_METRIC" >&2
echo "[tuev-slurm] attnres_variant=$ATTNRES_VARIANT use_moe=$USE_MOE" >&2
echo "[tuev-slurm] ema use=$USE_EMA decay=$EMA_DECAY warmup=$EMA_WARMUP_STEPS eval_only=$EMA_EVAL_ONLY" >&2
echo "[tuev-slurm] class_weight_mode=$CLASS_WEIGHT_MODE clip=[$CLASS_WEIGHT_CLIP_MIN,$CLASS_WEIGHT_CLIP_MAX] beta=$EFFECTIVE_NUM_BETA" >&2
echo "[tuev-slurm] component_lr use=$USE_COMPONENT_LR backbone=$LR_BACKBONE_MULT router=$LR_ROUTER_MULT experts=$LR_EXPERT_MULT classifier=$LR_CLASSIFIER_MULT other=$LR_OTHER_MULT" >&2
echo "[tuev-slurm] router temp=$ROUTER_TEMPERATURE entropy=$ROUTER_ENTROPY_COEF balance_kl=$ROUTER_BALANCE_KL_COEF entropy_spatial=$ROUTER_ENTROPY_COEF_SPATIAL entropy_spectral=$ROUTER_ENTROPY_COEF_SPECTRAL balance_kl_spatial=$ROUTER_BALANCE_KL_COEF_SPATIAL balance_kl_spectral=$ROUTER_BALANCE_KL_COEF_SPECTRAL" >&2
echo "[tuev-slurm] depth_summary grad_mode=$DEPTH_SUMMARY_GRAD_MODE unfreeze_epoch=$DEPTH_SUMMARY_UNFREEZE_EPOCH probe_mlp=$DEPTH_PROBE_MLP_FOR_ROUTER" >&2

PY_ARGS+=("${MOE_ARGS[@]}")

if is_true "$DRY_RUN"; then
  printf '[tuev-slurm] DRY_RUN command:' >&2
  printf ' %q' python "$REPO_DIR/finetune_main.py" "${PY_ARGS[@]}" >&2
  printf '\n' >&2
  exit 0
fi

python "$REPO_DIR/finetune_main.py" "${PY_ARGS[@]}"
