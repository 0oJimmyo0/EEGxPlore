#!/bin/bash
#SBATCH --job-name=MUMTAZ_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/Mumtaz/out/mumtaz_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/Mumtaz/err/mumtaz_hier_%j.err
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
LOG_ROOT="$REPO_DIR/logs/Mumtaz"
mkdir -p "$LOG_ROOT/out" "$LOG_ROOT/err"

DATASET="${DATASET:-Mumtaz2016}"
if [[ "$DATASET" != "Mumtaz2016" ]]; then
  echo "[mumtaz-slurm] DATASET must be Mumtaz2016 (got: $DATASET)" >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/Mumtaz2016}"
NUM_CLASSES="${NUM_CLASSES:-2}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[mumtaz-slurm] expected Mumtaz LMDB directory missing: $DATASET_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET_DIR/data.mdb" || ! -f "$DATASET_DIR/lock.mdb" ]]; then
  echo "[mumtaz-slurm] expected LMDB files missing under $DATASET_DIR (need data.mdb and lock.mdb)" >&2
  exit 2
fi

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-2}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.02}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"  # none | inv_freq_clip | effective_num
CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.85}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.15}"
EFFECTIVE_NUM_BETA="${EFFECTIVE_NUM_BETA:-0.999}"

USE_EMA="${USE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
EMA_EVAL_ONLY="${EMA_EVAL_ONLY:-1}"

ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
DEPTH_ROUTER="${DEPTH_ROUTER:-on}"
DEPTH_CONTEXT_MODE="${DEPTH_CONTEXT_MODE:-compact_shared}"  # compact_shared | block_shared_typed_proj | dual_query_block_typed_proj
DEPTH_BLOCK_COUNT="${DEPTH_BLOCK_COUNT:-4}"
if [[ -z "${DEPTH_ROUTER_DIM:-}" ]]; then
  if [[ "$ATTNRES_VARIANT" == "pre_attn" ]]; then
    DEPTH_ROUTER_DIM=15
  else
    DEPTH_ROUTER_DIM=26
  fi
fi
ROUTER_ARCH="${ROUTER_ARCH:-mlp}"
ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-128}"
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-attn_delta4}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-on}"
DEPTH_SUMMARY_GRAD_MODE="${DEPTH_SUMMARY_GRAD_MODE:-delayed_unfreeze}"  # detached | delayed_unfreeze | trainable
DEPTH_SUMMARY_UNFREEZE_EPOCH="${DEPTH_SUMMARY_UNFREEZE_EPOCH:-8}"

ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.5}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.005}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.01}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.001}"
ROUTER_JITTER_STD="${ROUTER_JITTER_STD:-0.02}"
ROUTER_JITTER_FINAL_STD="${ROUTER_JITTER_FINAL_STD:-0.005}"
ROUTER_JITTER_ANNEAL_EPOCHS="${ROUTER_JITTER_ANNEAL_EPOCHS:-30}"
ROUTER_SOFT_WARMUP_EPOCHS="${ROUTER_SOFT_WARMUP_EPOCHS:-15}"

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
        echo "[mumtaz-slurm] ABLATION_GROUP=ema requires ABLATION_VALUE in {on,off}" >&2
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
        echo "[mumtaz-slurm] ABLATION_GROUP=grad requires ABLATION_VALUE in {detached,delayed_unfreeze,trainable}" >&2
        exit 2
        ;;
    esac
    ;;
  context)
    case "$ABLATION_VALUE" in
      compact_shared|block_shared_typed_proj|dual_query_block_typed_proj)
        DEPTH_CONTEXT_MODE="$ABLATION_VALUE"
        ABLATION_TAG="context_${ABLATION_VALUE}"
        ;;
      *)
        echo "[mumtaz-slurm] ABLATION_GROUP=context requires ABLATION_VALUE in {compact_shared,block_shared_typed_proj,dual_query_block_typed_proj}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "[mumtaz-slurm] invalid ABLATION_GROUP: $ABLATION_GROUP (expected: none|ema|grad|context)" >&2
    exit 2
    ;;
esac

MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/Mumtaz_typed_moe_A1_base}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_mumtaz_${ABLATION_TAG}_${ATTNRES_VARIANT}_${DEPTH_ROUTER}}"
mkdir -p "$MODEL_DIR"

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

CLASS_WEIGHT_ARGS=(
  --class_weight_mode "$CLASS_WEIGHT_MODE"
  --class_weight_clip_min "$CLASS_WEIGHT_CLIP_MIN"
  --class_weight_clip_max "$CLASS_WEIGHT_CLIP_MAX"
  --effective_num_beta "$EFFECTIVE_NUM_BETA"
)

echo "[mumtaz-slurm] dataset=$DATASET dir=$DATASET_DIR num_classes=$NUM_CLASSES" >&2
echo "[mumtaz-slurm] ablation_group=$ABLATION_GROUP ablation_value=${ABLATION_VALUE:-NA} ablation_tag=$ABLATION_TAG" >&2
echo "[mumtaz-slurm] depth_context_mode=$DEPTH_CONTEXT_MODE depth_summary_grad_mode=$DEPTH_SUMMARY_GRAD_MODE" >&2
echo "[mumtaz-slurm] ema(use/decay/warmup/eval_only)=($USE_EMA/$EMA_DECAY/$EMA_WARMUP_STEPS/$EMA_EVAL_ONLY)" >&2
echo "[mumtaz-slurm] model_dir=$MODEL_DIR run_name=$RUN_NAME" >&2

python "$REPO_DIR/finetune_main.py" \
  --seed 42 \
  --cuda 0 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout 0.1 \
  --classifier all_patch_reps \
  --downstream_dataset "$DATASET" \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes "$NUM_CLASSES" \
  --model_dir "$MODEL_DIR" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --num_workers "$NUM_WORKERS" \
  --attnres_variant "$ATTNRES_VARIANT" \
  --attnres_start_layer 0 \
  --use_pretrained_weights \
  --foundation_dir "$FOUNDATION_DIR" \
  --moe \
  --moe_diagnostics \
  --moe_num_layers 1 \
  --moe_num_experts 4 \
  --moe_router_arch "$ROUTER_ARCH" \
  --moe_router_mlp_hidden "$ROUTER_MLP_HIDDEN" \
  "${DEPTH_ROUTER_ARGS[@]}" \
  --moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE" \
  --moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH" \
  --moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE" \
  --moe_router_temperature "$ROUTER_TEMPERATURE" \
  --moe_router_entropy_coef "$ROUTER_ENTROPY_COEF" \
  --moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF" \
  --moe_router_z_loss_coef "$ROUTER_Z_LOSS_COEF" \
  --moe_router_jitter_std "$ROUTER_JITTER_STD" \
  --moe_router_jitter_final_std "$ROUTER_JITTER_FINAL_STD" \
  --moe_router_jitter_anneal_epochs "$ROUTER_JITTER_ANNEAL_EPOCHS" \
  --moe_router_soft_warmup_epochs "$ROUTER_SOFT_WARMUP_EPOCHS" \
  --moe_uniform_dispatch_warmup_epochs 3 \
  --moe_shared_blend_warmup_epochs 5 \
  --moe_shared_blend_start 1.0 \
  --moe_shared_blend_end 0.0 \
  --routing_run_name "$RUN_NAME" \
  "${EMA_ARGS[@]}" \
  "${CLASS_WEIGHT_ARGS[@]}" \
  --no-tqdm
