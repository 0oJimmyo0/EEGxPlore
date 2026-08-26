#!/usr/bin/env bash
set -euo pipefail

# LEGACY ARTIFACT-REUSE PATH: use the ICASSP revision launcher for new paper runs.
# Minimal SEED-V run script aligned to CBraMod benchmark cohort:
# shared processed_lmdb with LMDB __keys__ train/val/test split.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_ID="${CUDA_ID:-0}"
DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
SEEDV_SPLIT_MANIFEST="${SEEDV_SPLIT_MANIFEST:-}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/output/seedv_refactor_anchor}"
SMOKE_TEST="${SMOKE_TEST:-0}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-5e-6}"
ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
USE_MOE="${USE_MOE:-1}"
A1_STRICT_BLOCK_ABLATION="${A1_STRICT_BLOCK_ABLATION:-1}"
DEPTH_CONTEXT_MODE="${DEPTH_CONTEXT_MODE:-block_shared_typed_proj}"
DEPTH_QUERY_MODE="${DEPTH_QUERY_MODE:-shared}"
DEPTH_BLOCK_COUNT="${DEPTH_BLOCK_COUNT:-4}"
DEPTH_ROUTER_DIM="${DEPTH_ROUTER_DIM:-15}"
DEPTH_ROUTER_NORM_GATE="${DEPTH_ROUTER_NORM_GATE:-1}"
DEPTH_ROUTER_NORM_EPS="${DEPTH_ROUTER_NORM_EPS:-1e-6}"
DEPTH_ROUTER_GATE_INIT="${DEPTH_ROUTER_GATE_INIT:-0.075}"
DEPTH_BLOCK_SEP_COEF="${DEPTH_BLOCK_SEP_COEF:-0.001}"
DEPTH_BLOCK_SEP_TARGET_JS="${DEPTH_BLOCK_SEP_TARGET_JS:-0.03}"
USE_COMPONENT_LR="${USE_COMPONENT_LR:-1}"
LR_BACKBONE_MULT="${LR_BACKBONE_MULT:-1.0}"
LR_ROUTER_MULT="${LR_ROUTER_MULT:-1.0}"
LR_EXPERT_MULT="${LR_EXPERT_MULT:-1.0}"
LR_CLASSIFIER_MULT="${LR_CLASSIFIER_MULT:-1.0}"
LR_OTHER_MULT="${LR_OTHER_MULT:-1.0}"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[run_seedv] dataset dir not found: $DATASET_DIR" >&2
  exit 2
fi

if [[ ! -f "$FOUNDATION_DIR" ]]; then
  echo "[run_seedv] foundation checkpoint not found: $FOUNDATION_DIR" >&2
  exit 2
fi

PYTHON_BIN=(python)
if [[ -n "$CONDA_ENV_PREFIX" ]]; then
  if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    echo "[run_seedv] conda env python not found: $CONDA_ENV_PREFIX/bin/python" >&2
    exit 2
  fi
  PYTHON_BIN=(conda run --no-capture-output -p "$CONDA_ENV_PREFIX" python)
fi

if [[ "$SMOKE_TEST" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
fi

case "$DEPTH_QUERY_MODE" in
  shared|dual)
    ;;
  *)
    echo "[run_seedv] invalid DEPTH_QUERY_MODE=$DEPTH_QUERY_MODE (expected shared|dual)" >&2
    exit 2
    ;;
esac

if [[ "$A1_STRICT_BLOCK_ABLATION" == "1" ]]; then
  if [[ "$DEPTH_QUERY_MODE" == "dual" ]]; then
    DEPTH_CONTEXT_MODE="dual_query_block_typed_proj"
  else
    DEPTH_CONTEXT_MODE="block_shared_typed_proj"
  fi
  DEPTH_BLOCK_COUNT="4"
  DEPTH_ROUTER_DIM="15"
fi

mkdir -p "$MODEL_DIR"

echo "[run_seedv] cohort=CBraMod benchmark (LMDB __keys__)" >&2
echo "[run_seedv] dataset_dir=$DATASET_DIR" >&2
echo "[run_seedv] foundation_dir=$FOUNDATION_DIR" >&2
echo "[run_seedv] split_manifest=${SEEDV_SPLIT_MANIFEST:-<none>}" >&2
echo "[run_seedv] smoke_test=$SMOKE_TEST epochs=$EPOCHS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS" >&2
echo "[run_seedv] attnres_variant=$ATTNRES_VARIANT use_moe=$USE_MOE" >&2
echo "[run_seedv] lr=$LR min_lr=$MIN_LR component_lr=$USE_COMPONENT_LR lr_mults=(bb:$LR_BACKBONE_MULT,router:$LR_ROUTER_MULT,expert:$LR_EXPERT_MULT,clf:$LR_CLASSIFIER_MULT,other:$LR_OTHER_MULT)" >&2
echo "[run_seedv] router_soft_warmup_epochs=15 (soft-dispatch runs use warmup blending in MoE router)" >&2
echo "[run_seedv] a1_strict=$A1_STRICT_BLOCK_ABLATION depth_query_mode=$DEPTH_QUERY_MODE depth_context_mode=$DEPTH_CONTEXT_MODE block_count=$DEPTH_BLOCK_COUNT depth_dim=$DEPTH_ROUTER_DIM depth_norm_gate=$DEPTH_ROUTER_NORM_GATE depth_norm_eps=$DEPTH_ROUTER_NORM_EPS depth_gate_init=$DEPTH_ROUTER_GATE_INIT depth_sep_coef=$DEPTH_BLOCK_SEP_COEF depth_sep_target_js=$DEPTH_BLOCK_SEP_TARGET_JS" >&2

CMD=(
  "${PYTHON_BIN[@]}" "$REPO_DIR/finetune_main.py"
  --seed 42
  --cuda "$CUDA_ID"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --min_lr "$MIN_LR"
  --weight_decay 2e-2
  --optimizer AdamW
  --clip_value 1.0
  --dropout 0.3
  --classifier all_patch_reps
  --downstream_dataset SEED-V
  --datasets_dir "$DATASET_DIR"
  --num_of_classes 5
  --model_dir "$MODEL_DIR"
  --label_smoothing 0.05
  --num_workers "$NUM_WORKERS"
  --use_pretrained_weights
  --foundation_dir "$FOUNDATION_DIR"
  --attnres_variant "$ATTNRES_VARIANT"
  --no-tqdm
)

CMD+=(--lr_backbone_mult "$LR_BACKBONE_MULT")
CMD+=(--lr_router_mult "$LR_ROUTER_MULT")
CMD+=(--lr_expert_mult "$LR_EXPERT_MULT")
CMD+=(--lr_classifier_mult "$LR_CLASSIFIER_MULT")
CMD+=(--lr_other_mult "$LR_OTHER_MULT")
if [[ "$USE_COMPONENT_LR" == "1" ]]; then
  CMD+=(--use_component_lr)
else
  CMD+=(--no-use_component_lr)
fi

if [[ "$USE_MOE" == "1" ]]; then
  CMD+=(--moe)
  CMD+=(--moe_diagnostics)
  CMD+=(--moe_num_layers 1)
  CMD+=(--moe_num_experts 4)
  CMD+=(--moe_router_arch mlp)
  CMD+=(--moe_router_mlp_hidden 128)
  CMD+=(--moe_use_attnres_depth_router_features)
  CMD+=(--moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM")
  CMD+=(--moe_attnres_depth_context_mode "$DEPTH_CONTEXT_MODE")
  CMD+=(--moe_attnres_depth_block_count "$DEPTH_BLOCK_COUNT")
  if [[ "$DEPTH_CONTEXT_MODE" == "block_shared_typed_proj" || "$DEPTH_CONTEXT_MODE" == "dual_query_block_typed_proj" ]]; then
    CMD+=(--moe_attnres_depth_summary_mode auto)
  else
    CMD+=(--moe_attnres_depth_summary_mode attn_delta4)
    CMD+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
  if [[ "$DEPTH_ROUTER_NORM_GATE" == "1" ]]; then
    CMD+=(--moe_attnres_depth_router_norm_gate)
  else
    CMD+=(--no-moe_attnres_depth_router_norm_gate)
  fi
  CMD+=(--moe_attnres_depth_router_norm_eps "$DEPTH_ROUTER_NORM_EPS")
  CMD+=(--moe_attnres_depth_router_gate_init "$DEPTH_ROUTER_GATE_INIT")
  CMD+=(--moe_attnres_depth_block_separation_coef "$DEPTH_BLOCK_SEP_COEF")
  CMD+=(--moe_attnres_depth_block_separation_target_js "$DEPTH_BLOCK_SEP_TARGET_JS")
  CMD+=(--moe_attnres_depth_summary_grad_mode detached)
  CMD+=(--moe_attnres_depth_summary_unfreeze_epoch 16)
  CMD+=(--moe_router_dispatch_mode soft)
  CMD+=(--moe_router_temperature 1.5)
  CMD+=(--moe_router_entropy_coef 0.005)
  CMD+=(--moe_router_balance_kl_coef 0.01)
  CMD+=(--moe_router_z_loss_coef 0.001)
  CMD+=(--moe_router_jitter_std 0.02)
  CMD+=(--moe_router_jitter_final_std 0.005)
  CMD+=(--moe_router_jitter_anneal_epochs 30)
  CMD+=(--moe_router_soft_warmup_epochs 15)
fi

if [[ -n "$SEEDV_SPLIT_MANIFEST" ]]; then
  CMD+=(--seedv_split_manifest "$SEEDV_SPLIT_MANIFEST")
fi

"${CMD[@]}"
