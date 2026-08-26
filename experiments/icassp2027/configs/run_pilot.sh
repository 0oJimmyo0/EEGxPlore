#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT_EXPLICIT=0
if [[ -n "${MODEL_ROOT+x}" ]]; then
  MODEL_ROOT_EXPLICIT=1
fi
source "$CONFIG_DIR/common.sh"

DRY_RUN=0
if [[ $# -eq 3 && "$3" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <SEED-V|FACED|ISRUC|PhysioNet-MI> <static|routed|upper4|frozen|full|depth_aggregation> [--dry-run]" >&2
  exit 2
fi

DATASET="$1"
METHOD="$2"
case "$METHOD" in
  static|routed|upper4|frozen|full|depth_aggregation) ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac

# Static/Routed are archived experiments. Keep their implicit output root out
# of the active DepthAgg evidence hierarchy; an explicit MODEL_ROOT still
# allows deliberate historical reruns.
if [[ ("$METHOD" == "static" || "$METHOD" == "routed") && "$MODEL_ROOT_EXPLICIT" == "0" ]]; then
  MODEL_ROOT="$ICASSP_REPO_DIR/output/icassp2027_pilots"
fi

if [[ ! "$MOE_EXPERT_INIT_NOISE_STD" =~ ^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$ ]]; then
  echo "MOE_EXPERT_INIT_NOISE_STD must be a nonnegative numeric value, got: $MOE_EXPERT_INIT_NOISE_STD" >&2
  exit 2
fi
if [[ "$ICASSP_ROUTING_DIAGNOSTIC" == "0" || "$ICASSP_ROUTING_DIAGNOSTIC" == "false" ]]; then
  if ! awk -v value="$MOE_EXPERT_INIT_NOISE_STD" 'BEGIN { exit !((value + 0) == 0) }'; then
    echo "Nonzero MOE_EXPERT_INIT_NOISE_STD requires ICASSP_ROUTING_DIAGNOSTIC=1/true" >&2
    exit 2
  fi
fi

case "$DATASET" in
  SEED-V)
    DATASET_DIR="$SEEDV_DATA_DIR"
    NUM_CLASSES=5
    MANIFEST="$ICASSP_REPO_DIR/experiments/icassp2027/manifests/seedv/split_manifest.json"
    DATASET_TAG="seedv"
    DEFAULT_EPOCHS=20
    ;;
  FACED)
    DATASET_DIR="$FACED_DATA_DIR"
    NUM_CLASSES=9
    MANIFEST="$ICASSP_REPO_DIR/experiments/icassp2027/manifests/faced/split_manifest.json"
    DATASET_TAG="faced"
    DEFAULT_EPOCHS=20
    ;;
  ISRUC)
    DATASET_DIR="$ISRUC_DATA_DIR"
    NUM_CLASSES=5
    MANIFEST="$ICASSP_REPO_DIR/experiments/icassp2027/manifests/isruc/split_manifest.json"
    DATASET_TAG="isruc"
    DEFAULT_EPOCHS=20
    ;;
  PhysioNet-MI)
    DATASET_DIR="$PHYSIONET_DATA_DIR"
    NUM_CLASSES=4
    MANIFEST="$ICASSP_REPO_DIR/experiments/icassp2027/manifests/physionet_mi/split_manifest.json"
    DATASET_TAG="physionet_mi"
    DEFAULT_EPOCHS=20
    ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

if [[ -z "$DATASET_DIR" || ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory is unavailable: ${DATASET_DIR:-<unset>}" >&2
  exit 2
fi
if [[ ! -f "$FOUNDATION_DIR" ]]; then
  echo "Foundation checkpoint is unavailable: $FOUNDATION_DIR" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" || ! -f "$(dirname "$MANIFEST")/split_manifest.sha256" ]]; then
  echo "Manifest or hash sidecar is unavailable: $MANIFEST" >&2
  exit 2
fi

RUN_EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
MODEL_DIR="$MODEL_ROOT/$DATASET_TAG/$METHOD/seed_$SEED"
if [[ "$DRY_RUN" == "0" ]]; then
  mkdir -p "$MODEL_DIR"
fi

if [[ "$METHOD" == "static" || "$METHOD" == "routed" ]]; then
  TRAINABILITY_MODE="typed_conditional"
else
  TRAINABILITY_MODE="$METHOD"
fi

ATTNRES_VARIANT="none"
ATTNRES_START_LAYER=0
if [[ "$METHOD" == "depth_aggregation" ]]; then
  ATTNRES_VARIANT="pre_attn"
  ATTNRES_START_LAYER=8
fi

CMD=(
  python "$ICASSP_REPO_DIR/finetune_main.py"
  --seed "$SEED"
  --cuda "$CUDA_ID"
  --epochs "$RUN_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --min_lr "$MIN_LR"
  --weight_decay "$WEIGHT_DECAY"
  --optimizer AdamW
  --clip_value 1.0
  --dropout "$DROPOUT"
  --classifier all_patch_reps
  --downstream_dataset "$DATASET"
  --datasets_dir "$DATASET_DIR"
  --num_of_classes "$NUM_CLASSES"
  --model_dir "$MODEL_DIR"
  --input_scale_divisor "$INPUT_SCALE_DIVISOR"
  --num_workers "$NUM_WORKERS"
  --warmup_epochs "$WARMUP_EPOCHS"
  --warmup_start_factor "$WARMUP_START_FACTOR"
  --label_smoothing "$LABEL_SMOOTHING"
  --class_weight_mode "$CLASS_WEIGHT_MODE"
  --use_pretrained_weights
  --foundation_dir "$FOUNDATION_DIR"
  --experiment_profile icassp2027
  --icassp_split_manifest "$MANIFEST"
  --selection_metric kappa
  --attnres_variant "$ATTNRES_VARIANT"
  --attnres_start_layer "$ATTNRES_START_LAYER"
  --trainability_mode "$TRAINABILITY_MODE"
  --no-tqdm
)

if [[ "$USE_COMPONENT_LR" == "1" || "$USE_COMPONENT_LR" == "true" ]]; then
  CMD+=(
    --use_component_lr
    --lr_backbone_mult "$LR_BACKBONE_MULT"
    --lr_router_mult "$LR_ROUTER_MULT"
    --lr_expert_mult "$LR_EXPERT_MULT"
    --lr_classifier_mult "$LR_CLASSIFIER_MULT"
    --lr_other_mult "$LR_OTHER_MULT"
    --lr_depth_mult "$LR_DEPTH_MULT"
  )
elif [[ "$USE_COMPONENT_LR" != "0" && "$USE_COMPONENT_LR" != "false" ]]; then
  echo "USE_COMPONENT_LR must be 0/1 or false/true, got: $USE_COMPONENT_LR" >&2
  exit 2
fi

if [[ "$SELECTED_CHECKPOINT_DIAGNOSTICS" == "1" || "$SELECTED_CHECKPOINT_DIAGNOSTICS" == "true" ]]; then
  CMD+=(--selected_checkpoint_diagnostics)
elif [[ "$SELECTED_CHECKPOINT_DIAGNOSTICS" != "0" && "$SELECTED_CHECKPOINT_DIAGNOSTICS" != "false" ]]; then
  echo "SELECTED_CHECKPOINT_DIAGNOSTICS must be 0/1 or false/true, got: $SELECTED_CHECKPOINT_DIAGNOSTICS" >&2
  exit 2
fi

if [[ "$ICASSP_ROUTING_DIAGNOSTIC" == "1" || "$ICASSP_ROUTING_DIAGNOSTIC" == "true" ]]; then
  CMD+=(--icassp_routing_diagnostic)
elif [[ "$ICASSP_ROUTING_DIAGNOSTIC" != "0" && "$ICASSP_ROUTING_DIAGNOSTIC" != "false" ]]; then
  echo "ICASSP_ROUTING_DIAGNOSTIC must be 0/1 or false/true, got: $ICASSP_ROUTING_DIAGNOSTIC" >&2
  exit 2
fi

if [[ "$METHOD" == "frozen" ]]; then
  CMD+=(--frozen)
fi

if [[ "$METHOD" == "static" || "$METHOD" == "routed" ]]; then
  if [[ "$METHOD" == "static" ]]; then
    ROUTER_POLICY="static"
  else
    ROUTER_POLICY="sample"
  fi
  CMD+=(
    --moe
    --moe_num_layers 4
    --moe_num_experts 4
    --moe_route_mode typed_conditional
    --moe_router_policy "$ROUTER_POLICY"
    --moe_router_arch mlp
    --moe_router_mlp_hidden 128
    --moe_router_temperature 1.0
    --moe_shared_output_scale 1.0
    --moe_expert_output_scale 1.0
    --moe_router_dispatch_mode soft
    --moe_attnres_depth_context_mode compact_shared
    --moe_specialist_branch_mode both
    --moe_router_compact_feature_mode none
    --moe_load_balance 0
    --moe_router_entropy_coef 0
    --moe_router_balance_kl_coef 0
    --moe_router_z_loss_coef 0
    --moe_router_jitter_std 0
    --moe_router_jitter_final_std 0
    --moe_router_soft_warmup_epochs 0
    --moe_uniform_dispatch_warmup_epochs 0
    --moe_shared_blend_warmup_epochs 0
    --moe_expert_init_noise_std "$MOE_EXPERT_INIT_NOISE_STD"
  )
  if [[ "$DATASET" == "SEED-V" ]]; then
    CMD+=(
      --routing_export_dir "$MODEL_DIR/routing_export"
      --routing_export_splits test
      --routing_run_name "${DATASET_TAG}_${METHOD}_seed${SEED}"
    )
  fi
elif [[ "$METHOD" == "upper4" ]]; then
  : # The common command already carries --trainability_mode upper4.
elif [[ "$METHOD" == "depth_aggregation" ]]; then
  : # DepthAgg uses the method-specific AttnRes settings above.
fi

if [[ "$DATASET" == "FACED" ]]; then
  if [[ ! -f "$FACED_META_CSV" ]]; then
    echo "FACED metadata CSV is unavailable: $FACED_META_CSV" >&2
    exit 2
  fi
  CMD+=(--faced_meta_csv "$FACED_META_CSV")
fi

echo "[icassp-pilot] dataset=$DATASET method=$METHOD seed=$SEED epochs=$RUN_EPOCHS" >&2
echo "[icassp-pilot] model_dir=$MODEL_DIR" >&2
printf '[icassp-pilot] command:' >&2
printf ' %q' "${CMD[@]}" >&2
printf '\n' >&2
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
exec "${CMD[@]}"
