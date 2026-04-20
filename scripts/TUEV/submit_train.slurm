#!/bin/bash
#SBATCH --job-name=TUEV_FT
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/TUEV/out/tuev_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/TUEV/err/tuev_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00

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

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-2}"
DROPOUT="${DROPOUT:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
CLASSIFIER="${CLASSIFIER:-all_patch_reps}"
SEED="${SEED:-42}"
SELECTION_METRIC="${SELECTION_METRIC:-balanced_accuracy}"

ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
USE_MOE="${USE_MOE:-0}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
ROUTER_ARCH="${ROUTER_ARCH:-mlp}"
ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-128}"
ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.5}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.005}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.01}"
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

mkdir -p "$MODEL_DIR"

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
    --moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF"
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

python "$REPO_DIR/finetune_main.py" \
  --seed "$SEED" \
  --cuda 0 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout "$DROPOUT" \
  --classifier "$CLASSIFIER" \
  --downstream_dataset "$DATASET" \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes "$NUM_CLASSES" \
  --model_dir "$MODEL_DIR" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --num_workers "$NUM_WORKERS" \
  --selection_metric "$SELECTION_METRIC" \
  --attnres_variant "$ATTNRES_VARIANT" \
  --attnres_start_layer 0 \
  --use_pretrained_weights \
  --foundation_dir "$FOUNDATION_DIR" \
  "${MOE_ARGS[@]}" \
  --no-tqdm
