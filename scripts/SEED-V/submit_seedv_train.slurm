#!/bin/bash
#SBATCH --job-name=SEEDV_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/SEED-V/out/seedv_%x_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/SEED-V/err/seedv_%x_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
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

SEEDV_PROTOCOL="${SEEDV_PROTOCOL:-cbramod_benchmark}"  # cbramod_benchmark | legacy_subject_disjoint_4x200
case "$SEEDV_PROTOCOL" in
  cbramod_benchmark)
    DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb}"
    SEEDV_SPLIT_MANIFEST="${SEEDV_SPLIT_MANIFEST:-}"
    ;;
  legacy_subject_disjoint_4x200)
    DATASET_DIR="${DATASET_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/data/SEED-V/processed_raw_eeg_200hz_1to75_4x200_lmdb_v2}"
    SEEDV_SPLIT_MANIFEST="${SEEDV_SPLIT_MANIFEST:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/data/SEED-V/subject_disjoint_manifest_balanced_v2.json}"
    ;;
  *)
    echo "[seedv-slurm] invalid SEEDV_PROTOCOL=$SEEDV_PROTOCOL (expected: cbramod_benchmark|legacy_subject_disjoint_4x200)" >&2
    exit 2
    ;;
esac

NUM_CLASSES="${NUM_CLASSES:-5}"

MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/SEEDV_manual_${SLURM_JOB_ID:-manual}}"
RUN_NAME="${RUN_NAME:-seedv_${SLURM_JOB_ID:-manual}}"

FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"

CUDA_ID="${CUDA_ID:-0}"
EPOCHS="${EPOCHS:-25}"
SEED_LIST="${SEED_LIST:-42}"  # space/comma separated
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-2}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
CLIP_VALUE="${CLIP_VALUE:-1.0}"
DROPOUT="${DROPOUT:-0.1}"
CLASSIFIER="${CLASSIFIER:-all_patch_reps}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
USE_EMA="${USE_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.9995}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
EMA_EVAL_ONLY="${EMA_EVAL_ONLY:-1}"

RUN_MODE="${RUN_MODE:-custom}"  # dense | selective | custom
A1_STRICT_BLOCK_ABLATION="${A1_STRICT_BLOCK_ABLATION:-1}"  # 1 => keep A1 as one clean block-summary change

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
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.001}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.002}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.001}"
ROUTER_JITTER_STD="${ROUTER_JITTER_STD:-0.005}"
ROUTER_JITTER_FINAL_STD="${ROUTER_JITTER_FINAL_STD:-0.005}"
ROUTER_JITTER_ANNEAL_EPOCHS="${ROUTER_JITTER_ANNEAL_EPOCHS:-30}"
ROUTER_SOFT_WARMUP_EPOCHS="${ROUTER_SOFT_WARMUP_EPOCHS:-15}"

DEPTH_ROUTER="${DEPTH_ROUTER:-on}"
DEPTH_ROUTER_DIM="${DEPTH_ROUTER_DIM:-15}"
DEPTH_CONTEXT_MODE="${DEPTH_CONTEXT_MODE:-block_shared_typed_proj}"
DEPTH_QUERY_MODE="${DEPTH_QUERY_MODE:-dual}"  # shared | dual
DEPTH_BLOCK_COUNT="${DEPTH_BLOCK_COUNT:-8}"
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-attn_delta4}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-on}"
DEPTH_ROUTER_INIT="${DEPTH_ROUTER_INIT:-xavier}"
DEPTH_ROUTER_NORM_GATE="${DEPTH_ROUTER_NORM_GATE:-1}"
DEPTH_ROUTER_NORM_EPS="${DEPTH_ROUTER_NORM_EPS:-1e-6}"
DEPTH_ROUTER_GATE_INIT="${DEPTH_ROUTER_GATE_INIT:-0.075}"
DEPTH_BLOCK_SEP_COEF="${DEPTH_BLOCK_SEP_COEF:-0.000}"
DEPTH_BLOCK_SEP_TARGET_JS="${DEPTH_BLOCK_SEP_TARGET_JS:-0.03}"
DEPTH_SUMMARY_GRAD_MODE="${DEPTH_SUMMARY_GRAD_MODE:-delayed_unfreeze}"
DEPTH_SUMMARY_UNFREEZE_EPOCH="${DEPTH_SUMMARY_UNFREEZE_EPOCH:-1}"

case "$DEPTH_QUERY_MODE" in
  shared|dual)
    ;;
  *)
    echo "[seedv-slurm] invalid DEPTH_QUERY_MODE=$DEPTH_QUERY_MODE (expected: shared|dual)" >&2
    exit 2
    ;;
esac

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

USE_COMPONENT_LR="${USE_COMPONENT_LR:-1}"
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-0.5}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-3.0}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.5}"
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

if [[ "$A1_STRICT_BLOCK_ABLATION" == "1" ]]; then
  if [[ "$SEEDV_PROTOCOL" != "cbramod_benchmark" ]]; then
    echo "[seedv-slurm][warn] A1_STRICT_BLOCK_ABLATION=1 expects CBraMod cohort; forcing SEEDV_PROTOCOL=cbramod_benchmark" >&2
    SEEDV_PROTOCOL="cbramod_benchmark"
    DATASET_DIR="/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb"
    SEEDV_SPLIT_MANIFEST=""
  fi
  if [[ -n "$SEEDV_SPLIT_MANIFEST" ]]; then
    echo "[seedv-slurm][warn] A1_STRICT_BLOCK_ABLATION=1 uses LMDB __keys__ cohort; ignoring SEEDV_SPLIT_MANIFEST override" >&2
    SEEDV_SPLIT_MANIFEST=""
  fi
  if [[ "$USE_MOE" == "1" && "$DEPTH_ROUTER" == "on" ]]; then
    if [[ "$DEPTH_QUERY_MODE" == "dual" ]]; then
      DEPTH_CONTEXT_MODE="dual_query_block_typed_proj"
    else
      DEPTH_CONTEXT_MODE="block_shared_typed_proj"
    fi
    DEPTH_BLOCK_COUNT="4"
    DEPTH_ROUTER_DIM="15"
    DEPTH_SUMMARY_MODE="auto"
    DEPTH_PROBE_MLP_FOR_ROUTER="off"
  else
    echo "[seedv-slurm][warn] A1_STRICT_BLOCK_ABLATION=1 is active, but USE_MOE=$USE_MOE DEPTH_ROUTER=$DEPTH_ROUTER; strict block context is not applied." >&2
  fi
fi

if [[ -n "$SEEDV_SPLIT_MANIFEST" && ! -f "$SEEDV_SPLIT_MANIFEST" ]]; then
  echo "[seedv-slurm] subject-disjoint manifest not found: $SEEDV_SPLIT_MANIFEST" >&2
  echo "[seedv-slurm] build it with: python preprocessing/build_seedv_subject_manifest.py --lmdb_path $DATASET_DIR --output_json $DATASET_DIR/subject_disjoint_manifest.json --seed 42 (or scripts/SEED-V/build_seedv_subject_disjoint_manifest.py)" >&2
  exit 2
fi

if [[ "$NUM_CLASSES" != "5" ]]; then
  echo "[seedv-slurm][warn] NUM_CLASSES is usually 5 for SEED-V (got: $NUM_CLASSES)" >&2
fi

DEPTH_ROUTER_ARGS=()
if [[ "$DEPTH_ROUTER" == "on" ]]; then
  DEPTH_ROUTER_ARGS+=(--moe_use_attnres_depth_router_features)
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_context_mode "$DEPTH_CONTEXT_MODE")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_block_count "$DEPTH_BLOCK_COUNT")
  if [[ "$DEPTH_CONTEXT_MODE" == "block_shared_typed_proj" || "$DEPTH_CONTEXT_MODE" == "dual_query_block_typed_proj" ]]; then
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_mode auto)
  else
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE")
  fi
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_router_init "$DEPTH_ROUTER_INIT")
  if [[ "$DEPTH_ROUTER_NORM_GATE" == "1" ]]; then
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_router_norm_gate)
  else
    DEPTH_ROUTER_ARGS+=(--no-moe_attnres_depth_router_norm_gate)
  fi
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_router_norm_eps "$DEPTH_ROUTER_NORM_EPS")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_router_gate_init "$DEPTH_ROUTER_GATE_INIT")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_block_separation_coef "$DEPTH_BLOCK_SEP_COEF")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_block_separation_target_js "$DEPTH_BLOCK_SEP_TARGET_JS")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH")
  if [[ "$DEPTH_CONTEXT_MODE" != "block_shared_typed_proj" && "$DEPTH_CONTEXT_MODE" != "dual_query_block_typed_proj" && "$DEPTH_PROBE_MLP_FOR_ROUTER" == "on" ]]; then
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
fi

SPLIT_ARGS=()
if [[ -n "$SEEDV_SPLIT_MANIFEST" ]]; then
  SPLIT_ARGS+=(--seedv_split_manifest "$SEEDV_SPLIT_MANIFEST")
fi

COMPONENT_LR_ARGS=()
COMPONENT_LR_ARGS+=(--lr_backbone_mult "$LR_BACKBONE_MULT")
COMPONENT_LR_ARGS+=(--lr_router_mult "$LR_ROUTER_MULT")
COMPONENT_LR_ARGS+=(--lr_expert_mult "$LR_EXPERT_MULT")
COMPONENT_LR_ARGS+=(--lr_classifier_mult "$LR_CLASSIFIER_MULT")
COMPONENT_LR_ARGS+=(--lr_other_mult "$LR_OTHER_MULT")
if [[ "$USE_COMPONENT_LR" == "1" ]]; then
  COMPONENT_LR_ARGS+=(--use_component_lr)
else
  COMPONENT_LR_ARGS+=(--no-use_component_lr)
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
fi

EXTRA_ARGS=()
if [[ -n "$EXTRA_SEEDV_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($EXTRA_SEEDV_ARGS)
fi

mkdir -p "$MODEL_DIR"

# Parse once so a single sbatch can run a fixed 3-seed sweep.
SEED_LIST_SANITIZED="${SEED_LIST//,/ }"
# shellcheck disable=SC2206
SEEDS=($SEED_LIST_SANITIZED)
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "[seedv-slurm] SEED_LIST is empty after parsing: '$SEED_LIST'" >&2
  exit 2
fi

DEPTH_SUMMARY_LABEL="$DEPTH_SUMMARY_MODE"
if [[ "$DEPTH_CONTEXT_MODE" == "block_shared_typed_proj" ]]; then
  DEPTH_SUMMARY_LABEL="block_typed_learned"
elif [[ "$DEPTH_CONTEXT_MODE" == "dual_query_block_typed_proj" ]]; then
  DEPTH_SUMMARY_LABEL="dual_query_block_typed_learned"
fi

echo "[seedv-slurm] dataset=$DATASET dataset_dir=$DATASET_DIR num_classes=$NUM_CLASSES" >&2
echo "[seedv-slurm] seedv_protocol=$SEEDV_PROTOCOL" >&2
echo "[seedv-slurm] run_mode=$RUN_MODE" >&2
echo "[seedv-slurm] a1_strict_block_ablation=$A1_STRICT_BLOCK_ABLATION" >&2
echo "[seedv-slurm] model_dir=$MODEL_DIR run_name=$RUN_NAME" >&2
echo "[seedv-slurm] seeds=${SEEDS[*]}" >&2
echo "[seedv-slurm] attnres_variant=$ATTNRES_VARIANT use_moe=$USE_MOE moe_layers=$MOE_NUM_LAYERS" >&2
echo "[seedv-slurm] lr=$LR min_lr=$MIN_LR" >&2
echo "[seedv-slurm] ema(use/decay/warmup/eval_only)=($USE_EMA/$EMA_DECAY/$EMA_WARMUP_STEPS/$EMA_EVAL_ONLY)" >&2
echo "[seedv-slurm] router_arch=$ROUTER_ARCH dispatch=$ROUTER_DISPATCH_MODE temp=$ROUTER_TEMPERATURE" >&2
echo "[seedv-slurm] router_soft_warmup_epochs=$ROUTER_SOFT_WARMUP_EPOCHS (soft-dispatch runs use warmup blending in MoE router)" >&2
echo "[seedv-slurm] depth_router=$DEPTH_ROUTER depth_dim=$DEPTH_ROUTER_DIM context_mode=$DEPTH_CONTEXT_MODE block_count=$DEPTH_BLOCK_COUNT depth_summary=$DEPTH_SUMMARY_LABEL depth_proj_init=$DEPTH_ROUTER_INIT depth_norm_gate=$DEPTH_ROUTER_NORM_GATE depth_norm_eps=$DEPTH_ROUTER_NORM_EPS depth_gate_init=$DEPTH_ROUTER_GATE_INIT depth_sep_coef=$DEPTH_BLOCK_SEP_COEF depth_sep_target_js=$DEPTH_BLOCK_SEP_TARGET_JS" >&2
echo "[seedv-slurm] depth_grad_mode=$DEPTH_SUMMARY_GRAD_MODE unfreeze_epoch=$DEPTH_SUMMARY_UNFREEZE_EPOCH" >&2
echo "[seedv-slurm] split_manifest=${SEEDV_SPLIT_MANIFEST:-<none>} component_lr=$USE_COMPONENT_LR lr_mults=(bb:$LR_BACKBONE_MULT,router:$LR_ROUTER_MULT,expert:$LR_EXPERT_MULT,clf:$LR_CLASSIFIER_MULT,other:$LR_OTHER_MULT)" >&2

for SEED in "${SEEDS[@]}"; do
  RUN_MODEL_DIR="${MODEL_DIR}/seed_${SEED}"
  RUN_RUN_NAME="${RUN_NAME}_s${SEED}"
  mkdir -p "$RUN_MODEL_DIR"

  echo "[seedv-slurm] starting seed=$SEED model_dir=$RUN_MODEL_DIR run_name=$RUN_RUN_NAME" >&2
  time python "$REPO_DIR/finetune_main.py" \
    --seed "$SEED" \
    --cuda "$CUDA_ID" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --min_lr "$MIN_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --optimizer "$OPTIMIZER" \
    --clip_value "$CLIP_VALUE" \
    --dropout "$DROPOUT" \
    --classifier "$CLASSIFIER" \
    --downstream_dataset "$DATASET" \
    --datasets_dir "$DATASET_DIR" \
    --num_of_classes "$NUM_CLASSES" \
    --model_dir "$RUN_MODEL_DIR" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --num_workers "$NUM_WORKERS" \
    --attnres_variant "$ATTNRES_VARIANT" \
    --attnres_start_layer "$ATTNRES_START_LAYER" \
    --use_pretrained_weights \
    --foundation_dir "$FOUNDATION_DIR" \
    --routing_run_name "$RUN_RUN_NAME" \
    --no-tqdm \
    "${SPLIT_ARGS[@]}" \
    "${EMA_ARGS[@]}" \
    "${COMPONENT_LR_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
done
