#!/bin/bash
#SBATCH --job-name=FACED_LABRAM
#SBATCH --account=neurogroup_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/logs/FACED/out/%x_%j.out
#SBATCH --error=/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore/logs/FACED/err/%x_%j.err

set -euo pipefail

REPO_DIR="/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore"
ENV_DIR="/data/neurogroup/mingyangjiang/venvs/labram-accre"
DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/FACED}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/checkpoints/labram-base.pth}"
LABRAM_REPO_DIR="${LABRAM_REPO_DIR:-/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM}"
FACED_META_CSV="${FACED_META_CSV:-/data/neurogroup/mingyangjiang/data/metadata/Recording_info.csv}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/output/faced_labram_dense_${SLURM_JOB_ID:-manual}}"
RUN_NAME="${RUN_NAME:-faced_labram_dense_${SLURM_JOB_ID:-manual}}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-6e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DROPOUT="${DROPOUT:-0.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
CLASSIFIER="${CLASSIFIER:-labram_pooled_linear}"
BACKBONE="${BACKBONE:-labram}"
# Current EEGxPlore FACED LaBraM sweeps perform best with the LaBraM-style
# scale100 setting, which also matches the original LaBraM finetune path that
# divides LMDB amplitudes by 100 before patchification.
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
LABRAM_TOKEN_POOL_NO_ADAPTER="${LABRAM_TOKEN_POOL_NO_ADAPTER:-0}"
LABRAM_FORCE_ADAPTER="${LABRAM_FORCE_ADAPTER:-0}"
FROZEN="${FROZEN:-0}"
LABRAM_ADAPTER_LAYERS="${LABRAM_ADAPTER_LAYERS:-0}"
ATTNRES_VARIANT="${ATTNRES_VARIANT:-none}"
ATTNRES_GATED="${ATTNRES_GATED:-0}"
ATTNRES_GATE_INIT="${ATTNRES_GATE_INIT:-0.0}"
ATTNRES_START_LAYER="${ATTNRES_START_LAYER:-0}"
USE_MOE="${USE_MOE:-0}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_ROUTER_ARCH="${MOE_ROUTER_ARCH:-mlp}"
MOE_ROUTER_MLP_HIDDEN="${MOE_ROUTER_MLP_HIDDEN:-128}"
MOE_ROUTE_MODE="${MOE_ROUTE_MODE:-typed_capacity_domain}"
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-1.0}"
MOE_LOAD_BALANCE="${MOE_LOAD_BALANCE:-0.005}"
MOE_DOMAIN_BIAS="${MOE_DOMAIN_BIAS:-0}"
MOE_DOMAIN_EMB_DIM="${MOE_DOMAIN_EMB_DIM:-16}"
MOE_DOMAIN_BIAS_REG="${MOE_DOMAIN_BIAS_REG:-0.0}"
MOE_USE_ATTNRES_DEPTH_ROUTER_FEATURES="${MOE_USE_ATTNRES_DEPTH_ROUTER_FEATURES:-0}"
MOE_ATTNRES_DEPTH_ROUTER_DIM="${MOE_ATTNRES_DEPTH_ROUTER_DIM:-15}"
MOE_ATTNRES_DEPTH_CONTEXT_MODE="${MOE_ATTNRES_DEPTH_CONTEXT_MODE:-compact_shared}"
MOE_ATTNRES_DEPTH_BLOCK_COUNT="${MOE_ATTNRES_DEPTH_BLOCK_COUNT:-4}"
MOE_ATTNRES_DEPTH_SUMMARY_MODE="${MOE_ATTNRES_DEPTH_SUMMARY_MODE:-attn_delta4}"
MOE_ATTNRES_DEPTH_PROBE_MLP_FOR_ROUTER="${MOE_ATTNRES_DEPTH_PROBE_MLP_FOR_ROUTER:-0}"
MOE_ATTNRES_DEPTH_SUMMARY_GRAD_MODE="${MOE_ATTNRES_DEPTH_SUMMARY_GRAD_MODE:-detached}"
MOE_ATTNRES_DEPTH_SUMMARY_UNFREEZE_EPOCH="${MOE_ATTNRES_DEPTH_SUMMARY_UNFREEZE_EPOCH:-16}"
MOE_ROUTER_DISPATCH_MODE="${MOE_ROUTER_DISPATCH_MODE:-soft}"
MOE_ROUTER_TEMPERATURE="${MOE_ROUTER_TEMPERATURE:-1.5}"
MOE_ROUTER_ENTROPY_COEF="${MOE_ROUTER_ENTROPY_COEF:-0.005}"
MOE_ROUTER_BALANCE_KL_COEF="${MOE_ROUTER_BALANCE_KL_COEF:-0.01}"
MOE_ROUTER_Z_LOSS_COEF="${MOE_ROUTER_Z_LOSS_COEF:-0.001}"
MOE_ROUTER_JITTER_STD="${MOE_ROUTER_JITTER_STD:-0.02}"
MOE_ROUTER_JITTER_FINAL_STD="${MOE_ROUTER_JITTER_FINAL_STD:-0.005}"
MOE_ROUTER_JITTER_ANNEAL_EPOCHS="${MOE_ROUTER_JITTER_ANNEAL_EPOCHS:-30}"
MOE_ROUTER_SOFT_WARMUP_EPOCHS="${MOE_ROUTER_SOFT_WARMUP_EPOCHS:-15}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-none}"
CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.75}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.5}"
EFFECTIVE_NUM_BETA="${EFFECTIVE_NUM_BETA:-0.999}"
USE_EMA="${USE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.9995}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
SELECTION_METRIC="${SELECTION_METRIC:-balanced_accuracy}"
PIN_MEMORY="${PIN_MEMORY:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
USE_COMPONENT_LR="${USE_COMPONENT_LR:-0}"
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-0.5}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-2.0}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.5}"
LR_CLASSIFIER_MULT="${LR_CLASSIFIER_MULT:-1.0}"
LR_OTHER_MULT="${LR_OTHER_MULT:-1.0}"
LABRAM_LAYER_DECAY="${LABRAM_LAYER_DECAY:-0.65}"

mkdir -p "$REPO_DIR/logs/FACED/out" "$REPO_DIR/logs/FACED/err" "$MODEL_DIR"

unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
module --force purge >/dev/null 2>&1
module load gentoo/2023 gcc/12.3 python/3.11.5 scipy-stack/2026a >/dev/null 2>&1
export PYTHONNOUSERSITE=1

source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

python - <<'PY'
import sys, torch, lmdb
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("lmdb:", lmdb.__file__)
PY
nvidia-smi || true
echo "EEGxPlore FACED LaBraM launch: RUN_NAME=$RUN_NAME BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS LR=$LR EPOCHS=$EPOCHS" >&2
echo "BACKBONE=$BACKBONE ADAPTER_LAYERS=$LABRAM_ADAPTER_LAYERS ATTNRES_VARIANT=$ATTNRES_VARIANT ATTNRES_GATED=$ATTNRES_GATED ATTNRES_GATE_INIT=$ATTNRES_GATE_INIT USE_MOE=$USE_MOE LABRAM_LAYER_DECAY=$LABRAM_LAYER_DECAY TOKEN_POOL_NO_ADAPTER=$LABRAM_TOKEN_POOL_NO_ADAPTER FORCE_ADAPTER=$LABRAM_FORCE_ADAPTER" >&2

EXTRA_ARGS=()
PYTHON_ARGS=()
if [[ "$USE_MOE" == "1" ]]; then
  EXTRA_ARGS+=(--moe --moe_diagnostics)
  EXTRA_ARGS+=(--moe_num_layers "$MOE_NUM_LAYERS")
  EXTRA_ARGS+=(--moe_num_experts "$MOE_NUM_EXPERTS")
  EXTRA_ARGS+=(--moe_router_arch "$MOE_ROUTER_ARCH")
  EXTRA_ARGS+=(--moe_router_mlp_hidden "$MOE_ROUTER_MLP_HIDDEN")
  EXTRA_ARGS+=(--moe_route_mode "$MOE_ROUTE_MODE")
  EXTRA_ARGS+=(--moe_capacity_factor "$MOE_CAPACITY_FACTOR")
  EXTRA_ARGS+=(--moe_load_balance "$MOE_LOAD_BALANCE")
  EXTRA_ARGS+=(--moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM")
  EXTRA_ARGS+=(--moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG")
  EXTRA_ARGS+=(--moe_attnres_depth_router_dim "$MOE_ATTNRES_DEPTH_ROUTER_DIM")
  EXTRA_ARGS+=(--moe_attnres_depth_context_mode "$MOE_ATTNRES_DEPTH_CONTEXT_MODE")
  EXTRA_ARGS+=(--moe_attnres_depth_block_count "$MOE_ATTNRES_DEPTH_BLOCK_COUNT")
  EXTRA_ARGS+=(--moe_attnres_depth_summary_mode "$MOE_ATTNRES_DEPTH_SUMMARY_MODE")
  EXTRA_ARGS+=(--moe_attnres_depth_summary_grad_mode "$MOE_ATTNRES_DEPTH_SUMMARY_GRAD_MODE")
  EXTRA_ARGS+=(--moe_attnres_depth_summary_unfreeze_epoch "$MOE_ATTNRES_DEPTH_SUMMARY_UNFREEZE_EPOCH")
  EXTRA_ARGS+=(--moe_router_dispatch_mode "$MOE_ROUTER_DISPATCH_MODE")
  EXTRA_ARGS+=(--moe_router_temperature "$MOE_ROUTER_TEMPERATURE")
  EXTRA_ARGS+=(--moe_router_entropy_coef "$MOE_ROUTER_ENTROPY_COEF")
  EXTRA_ARGS+=(--moe_router_balance_kl_coef "$MOE_ROUTER_BALANCE_KL_COEF")
  EXTRA_ARGS+=(--moe_router_z_loss_coef "$MOE_ROUTER_Z_LOSS_COEF")
  EXTRA_ARGS+=(--moe_router_jitter_std "$MOE_ROUTER_JITTER_STD")
  EXTRA_ARGS+=(--moe_router_jitter_final_std "$MOE_ROUTER_JITTER_FINAL_STD")
  EXTRA_ARGS+=(--moe_router_jitter_anneal_epochs "$MOE_ROUTER_JITTER_ANNEAL_EPOCHS")
  EXTRA_ARGS+=(--moe_router_soft_warmup_epochs "$MOE_ROUTER_SOFT_WARMUP_EPOCHS")
  if [[ "$MOE_DOMAIN_BIAS" == "1" ]]; then
    EXTRA_ARGS+=(--moe_domain_bias)
  fi
  if [[ "$MOE_USE_ATTNRES_DEPTH_ROUTER_FEATURES" == "1" ]]; then
    EXTRA_ARGS+=(--moe_use_attnres_depth_router_features)
  fi
  if [[ "$MOE_ATTNRES_DEPTH_PROBE_MLP_FOR_ROUTER" == "1" ]]; then
    EXTRA_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
fi

if [[ "$USE_EMA" == "1" ]]; then
  PYTHON_ARGS+=(--use_ema --ema_decay "$EMA_DECAY" --ema_warmup_steps "$EMA_WARMUP_STEPS" --ema_eval_only)
fi
if [[ "$USE_COMPONENT_LR" == "1" ]]; then
  PYTHON_ARGS+=(--use_component_lr)
  PYTHON_ARGS+=(--lr_backbone_mult "$LR_BACKBONE_MULT")
  PYTHON_ARGS+=(--lr_router_mult "$LR_ROUTER_MULT")
  PYTHON_ARGS+=(--lr_expert_mult "$LR_EXPERT_MULT")
  PYTHON_ARGS+=(--lr_classifier_mult "$LR_CLASSIFIER_MULT")
  PYTHON_ARGS+=(--lr_other_mult "$LR_OTHER_MULT")
fi
if [[ "$FROZEN" == "1" ]]; then
  PYTHON_ARGS+=(--frozen)
fi
if [[ "$ATTNRES_GATED" == "1" ]]; then
  PYTHON_ARGS+=(--attnres_gated)
fi
if [[ "$PIN_MEMORY" == "1" ]]; then
  PYTHON_ARGS+=(--pin_memory)
fi
if [[ "$PERSISTENT_WORKERS" == "1" ]]; then
  PYTHON_ARGS+=(--persistent_workers)
fi
if [[ "$LABRAM_TOKEN_POOL_NO_ADAPTER" == "1" ]]; then
  PYTHON_ARGS+=(--labram_token_pool_no_adapter)
fi
if [[ "$LABRAM_FORCE_ADAPTER" == "1" ]]; then
  PYTHON_ARGS+=(--labram_force_adapter)
fi

python "$REPO_DIR/finetune_main.py" \
  --seed "$SEED" \
  --cuda 0 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --min_lr "$MIN_LR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --warmup_start_factor "$WARMUP_START_FACTOR" \
  --weight_decay "$WEIGHT_DECAY" \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout "$DROPOUT" \
  --classifier "$CLASSIFIER" \
  --downstream_dataset FACED \
  --datasets_dir "$DATASET_DIR" \
  --input_scale_divisor "$INPUT_SCALE_DIVISOR" \
  --num_of_classes 9 \
  --model_dir "$MODEL_DIR" \
  --num_workers "$NUM_WORKERS" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --class_weight_mode "$CLASS_WEIGHT_MODE" \
  --class_weight_clip_min "$CLASS_WEIGHT_CLIP_MIN" \
  --class_weight_clip_max "$CLASS_WEIGHT_CLIP_MAX" \
  --effective_num_beta "$EFFECTIVE_NUM_BETA" \
  --selection_metric "$SELECTION_METRIC" \
  --use_pretrained_weights \
  --foundation_dir "$FOUNDATION_DIR" \
  --backbone "$BACKBONE" \
  --labram_repo_dir "$LABRAM_REPO_DIR" \
  --labram_model_name labram_base_patch200_200 \
  --labram_adapter_layers "$LABRAM_ADAPTER_LAYERS" \
  --labram_drop_path_rate 0.1 \
  --labram_layer_scale_init_value 0.1 \
  --labram_layer_decay "$LABRAM_LAYER_DECAY" \
  --attnres_variant "$ATTNRES_VARIANT" \
  --attnres_gate_init "$ATTNRES_GATE_INIT" \
  --attnres_start_layer "$ATTNRES_START_LAYER" \
  --faced_meta_csv "$FACED_META_CSV" \
  --routing_run_name "$RUN_NAME" \
  "${PYTHON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --no-tqdm
