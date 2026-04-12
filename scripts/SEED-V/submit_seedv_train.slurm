#!/bin/bash
#SBATCH --job-name=SEEDV_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/SEED-V/out/seedv_%x_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/SEED-V/err/seedv_%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
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
LOG_ROOT="$REPO_DIR/logs/SEED-V"
mkdir -p "$LOG_ROOT/out" "$LOG_ROOT/err"

# Keep this launcher focused on SEED-V -> model_for_seedv only.
DATASET="${DATASET:-SEED-V}"
if [[ "$DATASET" != "SEED-V" ]]; then
  echo "[seedv-slurm] DATASET must be SEED-V (got: $DATASET)" >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/data/SEED-V/processed_raw_eeg_200hz_1to75_4x200_lmdb_v2}"
NUM_CLASSES="${NUM_CLASSES:-5}"
SEEDV_SPLIT_MANIFEST="${SEEDV_SPLIT_MANIFEST:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/data/SEED-V/subject_disjoint_manifest_balanced_v2.json}"

MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/SEEDV_manual_${SLURM_JOB_ID:-manual}}"
RUN_NAME="${RUN_NAME:-seedv_${SLURM_JOB_ID:-manual}}"

FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"

CUDA_ID="${CUDA_ID:-0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-2e-2}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
CLIP_VALUE="${CLIP_VALUE:-1.0}"
DROPOUT="${DROPOUT:-0.3}"
CLASSIFIER="${CLASSIFIER:-all_patch_reps}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"

RUN_MODE="${RUN_MODE:-selective}"  # dense | selective | custom

ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
ATTNRES_START_LAYER="${ATTNRES_START_LAYER:-0}"

USE_MOE="${USE_MOE:-1}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-1.0}"
MOE_LOAD_BALANCE="${MOE_LOAD_BALANCE:-0.005}"

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

DEPTH_ROUTER="${DEPTH_ROUTER:-on}"
DEPTH_ROUTER_DIM="${DEPTH_ROUTER_DIM:-15}"
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-attn_delta4}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-on}"
DEPTH_SUMMARY_GRAD_MODE="${DEPTH_SUMMARY_GRAD_MODE:-detached}"
DEPTH_SUMMARY_UNFREEZE_EPOCH="${DEPTH_SUMMARY_UNFREEZE_EPOCH:-16}"

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

USE_COMPONENT_LR="${USE_COMPONENT_LR:-0}"
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-1.0}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-1.0}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.0}"
LR_CLASSIFIER_MULT="${LR_CLASSIFIER_MULT:-1.0}"
LR_OTHER_MULT="${LR_OTHER_MULT:-1.0}"

# Escape hatch for future flags without editing this script again.
EXTRA_SEEDV_ARGS="${EXTRA_SEEDV_ARGS:-}"

case "$RUN_MODE" in
  dense)
    USE_MOE="0"
    ATTNRES_VARIANT="none"
    ;;
  selective)
    # Keep defaults above.
    ;;
  custom)
    # Caller controls USE_MOE / ATTNRES_VARIANT and other knobs explicitly.
    ;;
  *)
    echo "[seedv-slurm] invalid RUN_MODE=$RUN_MODE (expected: dense|selective|custom)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$SEEDV_SPLIT_MANIFEST" ]]; then
  echo "[seedv-slurm] subject-disjoint manifest not found: $SEEDV_SPLIT_MANIFEST" >&2
  echo "[seedv-slurm] build it with: python preprocessing/build_seedv_subject_manifest.py --lmdb_path $DATASET_DIR --output_json $DATASET_DIR/subject_disjoint_manifest.json --seed 42" >&2
  exit 2
fi

if [[ "$NUM_CLASSES" != "5" ]]; then
  echo "[seedv-slurm][warn] NUM_CLASSES is usually 5 for SEED-V (got: $NUM_CLASSES)" >&2
fi

DEPTH_ROUTER_ARGS=()
if [[ "$DEPTH_ROUTER" == "on" ]]; then
  DEPTH_ROUTER_ARGS+=(--moe_use_attnres_depth_router_features)
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH")
  if [[ "$DEPTH_PROBE_MLP_FOR_ROUTER" == "on" ]]; then
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
fi

SPLIT_ARGS=()
if [[ -n "$SEEDV_SPLIT_MANIFEST" ]]; then
  SPLIT_ARGS+=(--seedv_split_manifest "$SEEDV_SPLIT_MANIFEST")
fi

COMPONENT_LR_ARGS=()
if [[ "$USE_COMPONENT_LR" == "1" ]]; then
  COMPONENT_LR_ARGS+=(--use_component_lr)
  COMPONENT_LR_ARGS+=(--lr_backbone_mult "$LR_BACKBONE_MULT")
  COMPONENT_LR_ARGS+=(--lr_router_mult "$LR_ROUTER_MULT")
  COMPONENT_LR_ARGS+=(--lr_expert_mult "$LR_EXPERT_MULT")
  COMPONENT_LR_ARGS+=(--lr_classifier_mult "$LR_CLASSIFIER_MULT")
  COMPONENT_LR_ARGS+=(--lr_other_mult "$LR_OTHER_MULT")
fi

MOE_ARGS=()
if [[ "$USE_MOE" == "1" ]]; then
  MOE_ARGS+=(--moe --moe_diagnostics)
  MOE_ARGS+=(--moe_num_layers "$MOE_NUM_LAYERS")
  MOE_ARGS+=(--moe_num_experts "$MOE_NUM_EXPERTS")
  MOE_ARGS+=(--moe_capacity_factor "$MOE_CAPACITY_FACTOR")
  MOE_ARGS+=(--moe_load_balance "$MOE_LOAD_BALANCE")
  MOE_ARGS+=(--moe_router_arch "$ROUTER_ARCH")
  MOE_ARGS+=(--moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE")
  MOE_ARGS+=(--moe_router_temperature "$ROUTER_TEMPERATURE")
  MOE_ARGS+=(--moe_router_entropy_coef "$ROUTER_ENTROPY_COEF")
  MOE_ARGS+=(--moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF")
  MOE_ARGS+=(--moe_router_z_loss_coef "$ROUTER_Z_LOSS_COEF")
  MOE_ARGS+=(--moe_router_jitter_std "$ROUTER_JITTER_STD")
  MOE_ARGS+=(--moe_router_jitter_final_std "$ROUTER_JITTER_FINAL_STD")
  MOE_ARGS+=(--moe_router_jitter_anneal_epochs "$ROUTER_JITTER_ANNEAL_EPOCHS")
  MOE_ARGS+=(--moe_router_soft_warmup_epochs "$ROUTER_SOFT_WARMUP_EPOCHS")
  MOE_ARGS+=(--moe_uniform_dispatch_warmup_epochs "$MOE_UNIFORM_DISPATCH_WARMUP_EPOCHS")
  MOE_ARGS+=(--moe_shared_blend_warmup_epochs "$MOE_SHARED_BLEND_WARMUP_EPOCHS")
  MOE_ARGS+=(--moe_shared_blend_start "$MOE_SHARED_BLEND_START")
  MOE_ARGS+=(--moe_shared_blend_end "$MOE_SHARED_BLEND_END")
  MOE_ARGS+=(--moe_router_entropy_coef_spatial "$MOE_ROUTER_ENTROPY_COEF_SPATIAL")
  MOE_ARGS+=(--moe_router_entropy_coef_spectral "$MOE_ROUTER_ENTROPY_COEF_SPECTRAL")
  MOE_ARGS+=(--moe_router_balance_kl_coef_spatial "$MOE_ROUTER_BALANCE_KL_COEF_SPATIAL")
  MOE_ARGS+=(--moe_router_balance_kl_coef_spectral "$MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL")
  MOE_ARGS+=(--moe_specialist_branch_mode "$MOE_SPECIALIST_BRANCH_MODE")
  MOE_ARGS+=(--moe_router_compact_feature_mode "$MOE_ROUTER_COMPACT_FEATURE_MODE")
  MOE_ARGS+=(--moe_router_compact_feature_dim "$MOE_ROUTER_COMPACT_FEATURE_DIM")
  MOE_ARGS+=(--moe_expert_init_noise_std "$MOE_EXPERT_INIT_NOISE_STD")
  if [[ "$ROUTER_ARCH" == "mlp" ]]; then
    MOE_ARGS+=(--moe_router_mlp_hidden "$ROUTER_MLP_HIDDEN")
  fi
  MOE_ARGS+=("${DEPTH_ROUTER_ARGS[@]}")
  MOE_ARGS+=("${COMPONENT_LR_ARGS[@]}")
fi

EXTRA_ARGS=()
if [[ -n "$EXTRA_SEEDV_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($EXTRA_SEEDV_ARGS)
fi

mkdir -p "$MODEL_DIR"

echo "[seedv-slurm] dataset=$DATASET dataset_dir=$DATASET_DIR num_classes=$NUM_CLASSES" >&2
echo "[seedv-slurm] run_mode=$RUN_MODE" >&2
echo "[seedv-slurm] model_dir=$MODEL_DIR run_name=$RUN_NAME" >&2
echo "[seedv-slurm] attnres_variant=$ATTNRES_VARIANT use_moe=$USE_MOE moe_layers=$MOE_NUM_LAYERS" >&2
echo "[seedv-slurm] router_arch=$ROUTER_ARCH dispatch=$ROUTER_DISPATCH_MODE temp=$ROUTER_TEMPERATURE" >&2
echo "[seedv-slurm] depth_router=$DEPTH_ROUTER depth_dim=$DEPTH_ROUTER_DIM depth_summary=$DEPTH_SUMMARY_MODE" >&2
echo "[seedv-slurm] depth_grad_mode=$DEPTH_SUMMARY_GRAD_MODE unfreeze_epoch=$DEPTH_SUMMARY_UNFREEZE_EPOCH" >&2
echo "[seedv-slurm] split_manifest=${SEEDV_SPLIT_MANIFEST:-<none>} component_lr=$USE_COMPONENT_LR" >&2

time python "$REPO_DIR/finetune_main.py" \
  --seed 42 \
  --cuda "$CUDA_ID" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --optimizer "$OPTIMIZER" \
  --clip_value "$CLIP_VALUE" \
  --dropout "$DROPOUT" \
  --classifier "$CLASSIFIER" \
  --downstream_dataset "$DATASET" \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes "$NUM_CLASSES" \
  --model_dir "$MODEL_DIR" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --num_workers "$NUM_WORKERS" \
  --attnres_variant "$ATTNRES_VARIANT" \
  --attnres_start_layer "$ATTNRES_START_LAYER" \
  --use_pretrained_weights \
  --foundation_dir "$FOUNDATION_DIR" \
  --routing_run_name "$RUN_NAME" \
  --no-tqdm \
  "${SPLIT_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
