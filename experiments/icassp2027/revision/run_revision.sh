#!/usr/bin/env bash
set -euo pipefail

# Single active interface for new ICASSP revision runs.
# Usage: run_revision.sh DATASET CONDITION SEED [PROTOCOL] [EPOCHS] [MODEL_ROOT] [EXPECTED_COMMIT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 DATASET CONDITION SEED [PROTOCOL] [EPOCHS] [MODEL_ROOT] [EXPECTED_COMMIT]" >&2
  exit 2
fi

DATASET="$1"
CONDITION="$2"
SEED="$3"
PROTOCOL="${4:-${REVISION_PROTOCOL:-cbramod_benchmark}}"
RUN_MODE="${RUN_MODE:-paper}"
REQUESTED_EPOCHS="${5:-${EPOCHS:-}}"
REQUESTED_MODEL_ROOT="${6:-${MODEL_ROOT:-}}"
EXPECTED_COMMIT="${7:-${EXPECTED_COMMIT:-$(git -C "$REPO_DIR" rev-parse HEAD)}}"
CUDA_ID="${CUDA_ID:-0}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUESTED_BATCH_SIZE="${BATCH_SIZE:-}"
REQUESTED_LR="${LR:-}"
REQUESTED_MIN_LR="${MIN_LR:-}"
REQUESTED_WEIGHT_DECAY="${WEIGHT_DECAY:-}"
REQUESTED_DROPOUT="${DROPOUT:-}"
REQUESTED_LABEL_SMOOTHING="${LABEL_SMOOTHING:-}"
REQUESTED_USE_EMA="${USE_EMA:-}"
REQUESTED_EMA_DECAY="${EMA_DECAY:-}"
REQUESTED_EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-}"
REQUESTED_EMA_EVAL_ONLY="${EMA_EVAL_ONLY:-}"
REQUESTED_USE_COMPONENT_LR="${USE_COMPONENT_LR:-}"
REQUESTED_NUM_WORKERS="${NUM_WORKERS:-}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
FRESH_SELECTIVE_RECIPE_PATH="${FRESH_SELECTIVE_RECIPE_PATH:-$SCRIPT_DIR/fresh_selective_recipe.json}"
HISTORICAL_CANDIDATE_RECIPE_PATH="${HISTORICAL_CANDIDATE_RECIPE_PATH:-${HISTORICAL_CANDIDATE_RECIPE:-}}"
PAPER_METHOD_RECIPE_PATH="${PAPER_METHOD_RECIPE_PATH:-$SCRIPT_DIR/paper_method_specialist_augmented_full_v1.json}"
HISTORICAL_CANDIDATE_STAGE="${HISTORICAL_CANDIDATE_STAGE:-route}"
HISTORICAL_CANDIDATE_SMOKE="${HISTORICAL_CANDIDATE_SMOKE:-0}"
HISTORICAL_CANDIDATE_HISTORICAL_FAMILY_ID="${HISTORICAL_CANDIDATE_HISTORICAL_FAMILY_ID:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" && "$DRY_RUN" != "false" && "$DRY_RUN" != "true" ]]; then
  echo "DRY_RUN must be 0/1 or false/true, got: $DRY_RUN" >&2
  exit 2
fi

case "$RUN_MODE" in
  paper|smoke|internal) ;;
  *)
    echo "RUN_MODE must be paper, smoke, or internal (got: $RUN_MODE)" >&2
    exit 2
    ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$CONDITION" == "historical_selective" ]]; then
  echo "historical_selective is permanently locked: historical checkpoint and complete recipe are unavailable" >&2
  echo "use selective_paper for the paper-derived run or selective_fresh for the independent recipe" >&2
  exit 2
fi
if [[ "$CONDITION" == "specialist_augmented_full" ]]; then
  if [[ ! -f "$PAPER_METHOD_RECIPE_PATH" ]]; then
    echo "paper method recipe is unavailable: $PAPER_METHOD_RECIPE_PATH" >&2
    exit 2
  fi
  PAPER_METHOD_SHELL_EXPORTS="$($PYTHON_BIN "$SCRIPT_DIR/verify_paper_method.py" --recipe "$PAPER_METHOD_RECIPE_PATH" --emit-shell)"
  eval "$PAPER_METHOD_SHELL_EXPORTS"
fi
if [[ "$CONDITION" == "historical_candidate" ]]; then
  if [[ "$RUN_MODE" != "internal" ]]; then
    echo "historical_candidate is development-only and requires RUN_MODE=internal" >&2
    exit 2
  fi
  if [[ "$HISTORICAL_CANDIDATE_STAGE" != "opt" && "$HISTORICAL_CANDIDATE_STAGE" != "route" ]]; then
    echo "HISTORICAL_CANDIDATE_STAGE must be opt or route" >&2
    exit 2
  fi
  if [[ -z "$HISTORICAL_CANDIDATE_RECIPE_PATH" || ! -f "$HISTORICAL_CANDIDATE_RECIPE_PATH" ]]; then
    echo "historical candidate recipe is unavailable: $HISTORICAL_CANDIDATE_RECIPE_PATH" >&2
    exit 2
  fi
  CANDIDATE_VERIFY_ARGS=(
    --recipe "$HISTORICAL_CANDIDATE_RECIPE_PATH"
    --dataset "$DATASET"
    --stage "$HISTORICAL_CANDIDATE_STAGE"
    --emit-shell
  )
  if [[ "$HISTORICAL_CANDIDATE_SMOKE" == "1" || "$HISTORICAL_CANDIDATE_SMOKE" == "true" ]]; then
    CANDIDATE_VERIFY_ARGS+=(--smoke)
  fi
  CANDIDATE_SHELL_EXPORTS="$($PYTHON_BIN "$SCRIPT_DIR/verify_historical_candidate.py" "${CANDIDATE_VERIFY_ARGS[@]}")"
  eval "$CANDIDATE_SHELL_EXPORTS"
fi
HISTORICAL_FAMILY_ID="${HISTORICAL_CANDIDATE_HISTORICAL_FAMILY_ID:-}"
if [[ "$CONDITION" == "selective_fresh" && ! -f "$FRESH_SELECTIVE_RECIPE_PATH" ]]; then
  echo "fresh selective recipe is unavailable: $FRESH_SELECTIVE_RECIPE_PATH" >&2
  exit 2
fi

case "$DATASET" in
  SEED-V)
    DATASET_TAG="seedv"
    NUM_CLASSES=5
    DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb}"
    ;;
  FACED)
    DATASET_TAG="faced"
    NUM_CLASSES=9
    DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/FACED}"
    FACED_META_CSV="${FACED_META_CSV:-/data/neurogroup/mingyangjiang/data/metadata/Recording_info.csv}"
    ;;
  ISRUC)
    DATASET_TAG="isruc"
    NUM_CLASSES=5
    DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
    ;;
  PhysioNet-MI)
    DATASET_TAG="physionet_mi"
    NUM_CLASSES=4
    DATASET_DIR="${DATASET_DIR:-/data/neurogroup/mingyangjiang/data/PHYSIO_MI}"
    PHYSIO_SPLIT_MANIFEST="${PHYSIO_SPLIT_MANIFEST:-$REPO_DIR/experiments/icassp2027/manifests/physionet_mi/split_manifest.json}"
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac

if [[ "$PROTOCOL" != "cbramod_benchmark" ]]; then
  echo "Unsupported revision protocol: $PROTOCOL" >&2
  exit 2
fi

if [[ "$RUN_MODE" == "paper" || "$RUN_MODE" == "smoke" ]]; then
  case "$DATASET:$CONDITION" in
    SEED-V:upper1|SEED-V:full|SEED-V:specialist_augmented_full|FACED:full|FACED:selective_paper|FACED:specialist_augmented_full|ISRUC:full|ISRUC:selective_paper|ISRUC:specialist_augmented_full|PhysioNet-MI:full|PhysioNet-MI:specialist_augmented_full) ;;
    *)
      echo "Paper-facing run is not in the frozen new-run matrix: $DATASET/$CONDITION" >&2
      echo "Allowed: SEED-V/{upper1,full,specialist_augmented_full}, FACED/{full,selective_paper,specialist_augmented_full}, ISRUC/{full,selective_paper,specialist_augmented_full}, PhysioNet-MI/{full,specialist_augmented_full}" >&2
      echo "Use an archived launcher or RUN_MODE=internal for non-paper diagnostics." >&2
      exit 2
      ;;
  esac
  if [[ "$CONDITION" == "specialist_augmented_full" || "$CONDITION" == "full" ]]; then
    case "$SEED" in
      3407|2024|2027) ;;
      *)
        echo "Final full/specialist_augmented_full seed must be one of 3407, 2024, 2027 (got: $SEED)" >&2
        exit 2
        ;;
    esac
  else
    case "$SEED" in
      42|3407|2024) ;;
      *)
        echo "Paper-facing seed must be one of 42, 3407, 2024 (got: $SEED)" >&2
        exit 2
        ;;
    esac
  fi
fi

PAPER_PROTOCOL_PATH=""
PAPER_PROTOCOL_ID=""
PAPER_PROTOCOL_SHA256=""
PAPER_PROTOCOL_USE_COMPONENT_LR=""
PAPER_METHOD_RECIPE_ID="${PAPER_METHOD_RECIPE_ID:-}"
PAPER_METHOD_RECIPE_SHA256="${PAPER_METHOD_RECIPE_SHA256:-}"
USE_PAPER_PROTOCOL=1
if [[ "$RUN_MODE" == "internal" && "$CONDITION" == "selective_fresh" ]]; then
  USE_PAPER_PROTOCOL=0
fi
if [[ "$CONDITION" == "historical_candidate" ]]; then
  USE_PAPER_PROTOCOL=0
fi
if [[ "$USE_PAPER_PROTOCOL" == "1" ]]; then
  case "$DATASET_TAG" in
    seedv) PAPER_PROTOCOL_PATH="$SCRIPT_DIR/paper_protocol_seedv_v1.json" ;;
    faced) PAPER_PROTOCOL_PATH="$SCRIPT_DIR/paper_protocol_faced_v1.json" ;;
    isruc) PAPER_PROTOCOL_PATH="$SCRIPT_DIR/paper_protocol_isruc_v1.json" ;;
    physionet_mi) PAPER_PROTOCOL_PATH="$SCRIPT_DIR/paper_protocol_physionet_mi_v1.json" ;;
  esac
  if [[ ! -f "$PAPER_PROTOCOL_PATH" ]]; then
    echo "Paper protocol is unavailable: $PAPER_PROTOCOL_PATH" >&2
    exit 2
  fi
  eval "$("$PYTHON_BIN" "$SCRIPT_DIR/verify_paper_protocol.py" --dataset "$DATASET" --protocol "$PAPER_PROTOCOL_PATH" --emit-shell)"

  check_numeric_override() {
    local name="$1"
    local requested="$2"
    local expected="$3"
    if [[ -z "$requested" ]]; then
      return 0
    fi
    if ! "$PYTHON_BIN" - "$name" "$requested" "$expected" <<'PY'
import math
import sys

name, requested, expected = sys.argv[1:]
if name in {'use_ema', 'use_component_lr', 'ema_eval_only'}:
    requested_bool = requested.lower() in {'1', 'true', 'yes', 'on'}
    expected_bool = expected == '1'
    equal = requested_bool == expected_bool
else:
    equal = math.isclose(float(requested), float(expected), rel_tol=1e-12, abs_tol=1e-12)
if not equal:
    raise SystemExit(1)
PY
    then
      echo "$name is locked to $expected for $DATASET (got: $requested)" >&2
      exit 2
    fi
  }

  if [[ "$RUN_MODE" != "smoke" ]]; then
    check_numeric_override epochs "$REQUESTED_EPOCHS" "$PAPER_PROTOCOL_EPOCHS"
  fi
  check_numeric_override batch_size "$REQUESTED_BATCH_SIZE" "$PAPER_PROTOCOL_BATCH_SIZE"
  check_numeric_override lr "$REQUESTED_LR" "$PAPER_PROTOCOL_LR"
  check_numeric_override min_lr "$REQUESTED_MIN_LR" "$PAPER_PROTOCOL_MIN_LR"
  check_numeric_override weight_decay "$REQUESTED_WEIGHT_DECAY" "$PAPER_PROTOCOL_WEIGHT_DECAY"
  check_numeric_override dropout "$REQUESTED_DROPOUT" "$PAPER_PROTOCOL_DROPOUT"
  check_numeric_override label_smoothing "$REQUESTED_LABEL_SMOOTHING" "$PAPER_PROTOCOL_LABEL_SMOOTHING"
  check_numeric_override use_ema "$REQUESTED_USE_EMA" "$PAPER_PROTOCOL_USE_EMA"
  check_numeric_override use_component_lr "$REQUESTED_USE_COMPONENT_LR" "$PAPER_PROTOCOL_USE_COMPONENT_LR"
  check_numeric_override ema_decay "$REQUESTED_EMA_DECAY" "$PAPER_PROTOCOL_EMA_DECAY"
  check_numeric_override ema_warmup_steps "$REQUESTED_EMA_WARMUP_STEPS" "$PAPER_PROTOCOL_EMA_WARMUP_STEPS"
  check_numeric_override ema_eval_only "$REQUESTED_EMA_EVAL_ONLY" "$PAPER_PROTOCOL_EMA_EVAL_ONLY"
  check_numeric_override num_workers "$REQUESTED_NUM_WORKERS" "$PAPER_PROTOCOL_NUM_WORKERS"

  if [[ "$RUN_MODE" == "smoke" ]]; then
    if [[ -n "$REQUESTED_EPOCHS" && "$REQUESTED_EPOCHS" != "1" ]]; then
      echo "Smoke runs require EPOCHS=1 (got: $REQUESTED_EPOCHS)" >&2
      exit 2
    fi
    EPOCHS=1
  else
    EPOCHS="$PAPER_PROTOCOL_EPOCHS"
  fi
  BATCH_SIZE="$PAPER_PROTOCOL_BATCH_SIZE"
  NUM_WORKERS="$PAPER_PROTOCOL_NUM_WORKERS"
  LR="$PAPER_PROTOCOL_LR"
  MIN_LR="$PAPER_PROTOCOL_MIN_LR"
  WEIGHT_DECAY="$PAPER_PROTOCOL_WEIGHT_DECAY"
  DROPOUT="$PAPER_PROTOCOL_DROPOUT"
  LABEL_SMOOTHING="$PAPER_PROTOCOL_LABEL_SMOOTHING"
  USE_EMA="$PAPER_PROTOCOL_USE_EMA"
  EMA_DECAY="$PAPER_PROTOCOL_EMA_DECAY"
  EMA_WARMUP_STEPS="$PAPER_PROTOCOL_EMA_WARMUP_STEPS"
  EMA_EVAL_ONLY="$PAPER_PROTOCOL_EMA_EVAL_ONLY"
  CLASSIFIER="$PAPER_PROTOCOL_CLASSIFIER"
else
  if [[ "$CONDITION" == "historical_candidate" ]]; then
    EPOCHS="$HISTORICAL_CANDIDATE_EPOCHS"
    if [[ "$HISTORICAL_CANDIDATE_SMOKE" == "1" || "$HISTORICAL_CANDIDATE_SMOKE" == "true" ]]; then
      EPOCHS=1
    fi
    BATCH_SIZE="$HISTORICAL_CANDIDATE_BATCH_SIZE"
    NUM_WORKERS="$HISTORICAL_CANDIDATE_NUM_WORKERS"
    LR="$HISTORICAL_CANDIDATE_LR"
    MIN_LR="$HISTORICAL_CANDIDATE_MIN_LR"
    WEIGHT_DECAY="$HISTORICAL_CANDIDATE_WEIGHT_DECAY"
    DROPOUT="$HISTORICAL_CANDIDATE_DROPOUT"
    LABEL_SMOOTHING="$HISTORICAL_CANDIDATE_LABEL_SMOOTHING"
    USE_EMA="$HISTORICAL_CANDIDATE_USE_EMA"
    EMA_DECAY="$HISTORICAL_CANDIDATE_EMA_DECAY"
    EMA_WARMUP_STEPS="$HISTORICAL_CANDIDATE_EMA_WARMUP_STEPS"
    EMA_EVAL_ONLY="$HISTORICAL_CANDIDATE_EMA_EVAL_ONLY"
    CLASSIFIER="$HISTORICAL_CANDIDATE_CLASSIFIER"
  else
    EPOCHS="${REQUESTED_EPOCHS:-40}"
    BATCH_SIZE="${REQUESTED_BATCH_SIZE:-64}"
    NUM_WORKERS="${REQUESTED_NUM_WORKERS:-4}"
    LR="${REQUESTED_LR:-3e-5}"
    MIN_LR="${REQUESTED_MIN_LR:-5e-6}"
    WEIGHT_DECAY="${REQUESTED_WEIGHT_DECAY:-3e-2}"
    DROPOUT="${REQUESTED_DROPOUT:-0.1}"
    LABEL_SMOOTHING="${REQUESTED_LABEL_SMOOTHING:-0.05}"
    USE_EMA="${REQUESTED_USE_EMA:-1}"
    EMA_DECAY="${REQUESTED_EMA_DECAY:-0.9995}"
    EMA_WARMUP_STEPS="${REQUESTED_EMA_WARMUP_STEPS:-1000}"
    EMA_EVAL_ONLY="${REQUESTED_EMA_EVAL_ONLY:-1}"
    CLASSIFIER="all_patch_reps"
  fi
fi

if [[ -n "$REQUESTED_MODEL_ROOT" ]]; then
  MODEL_ROOT="$REQUESTED_MODEL_ROOT"
elif [[ "$CONDITION" == "historical_candidate" ]]; then
  MODEL_ROOT="$REPO_DIR/output/icassp2027_historical_candidate"
elif [[ "$RUN_MODE" == "smoke" ]]; then
  MODEL_ROOT="$REPO_DIR/output/icassp2027_smoke"
else
  MODEL_ROOT="$REPO_DIR/output/icassp2027_revision"
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory is unavailable: $DATASET_DIR" >&2
  exit 2
fi
if [[ ! -f "$FOUNDATION_DIR" ]]; then
  echo "Foundation checkpoint is unavailable: $FOUNDATION_DIR" >&2
  exit 2
fi
if [[ "$DATASET" == "FACED" && ! -f "$FACED_META_CSV" ]]; then
  echo "FACED metadata CSV is unavailable: $FACED_META_CSV" >&2
  exit 2
fi

SPLIT_ARGS=()
MANIFEST_PATH=""
if [[ "$DATASET" == "PhysioNet-MI" ]]; then
  if [[ ! -f "$PHYSIO_SPLIT_MANIFEST" ]]; then
    echo "PhysioNet-MI split manifest is unavailable: $PHYSIO_SPLIT_MANIFEST" >&2
    exit 2
  fi
  MANIFEST_PATH="$PHYSIO_SPLIT_MANIFEST"
  SPLIT_ARGS+=(--icassp_split_manifest "$PHYSIO_SPLIT_MANIFEST")
fi

MODEL_ROOT="$(realpath -m "$MODEL_ROOT")"
if [[ "$CONDITION" == "historical_candidate" ]]; then
  MODEL_DIR="$MODEL_ROOT/$DATASET_TAG/${CONDITION}_${HISTORICAL_CANDIDATE_STAGE}/seed_$SEED"
else
  MODEL_DIR="$MODEL_ROOT/$DATASET_TAG/$CONDITION/seed_$SEED"
fi
mkdir -p "$MODEL_DIR"

DATA_CONTRACT_PATH="$MODEL_DIR/data_contract.json"
VERIFY_CONTRACT_ARGS=(
  --dataset "$DATASET"
  --data-dir "$DATASET_DIR"
  --output "$DATA_CONTRACT_PATH"
)
if [[ "$DATASET" == "PhysioNet-MI" ]]; then
  VERIFY_CONTRACT_ARGS+=(--split-manifest "$PHYSIO_SPLIT_MANIFEST")
fi
"$PYTHON_BIN" "$SCRIPT_DIR/verify_data_contract.py" "${VERIFY_CONTRACT_ARGS[@]}"

RUN_ARGS=(
  --seed "$SEED"
  --cuda "$CUDA_ID"
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
  --input_scale_divisor 100.0
  --num_workers "$NUM_WORKERS"
  --label_smoothing "$LABEL_SMOOTHING"
  --ema_decay "$EMA_DECAY"
  --ema_warmup_steps "$EMA_WARMUP_STEPS"
  --use_pretrained_weights
  --foundation_dir "$FOUNDATION_DIR"
  --experiment_profile icassp2027_revision
  --revision_condition "$CONDITION"
  --revision_protocol "$PROTOCOL"
  --revision_run_mode "$RUN_MODE"
  --historical_candidate_recipe "$HISTORICAL_CANDIDATE_RECIPE_PATH"
  --historical_family_id "$HISTORICAL_FAMILY_ID"
  --selection_metric kappa
  --no-tqdm
  "${SPLIT_ARGS[@]}"
)

if [[ "$CONDITION" == "specialist_augmented_full" ]]; then
  RUN_ARGS+=(--paper_method_recipe "$PAPER_METHOD_RECIPE_PATH")
fi

if [[ "$CONDITION" == "historical_candidate" ]]; then
  RUN_ARGS+=(
    --warmup_epochs "$HISTORICAL_CANDIDATE_WARMUP_EPOCHS"
    --warmup_start_factor "$HISTORICAL_CANDIDATE_WARMUP_START_FACTOR"
    --class_weight_mode "$HISTORICAL_CANDIDATE_CLASS_WEIGHT_MODE"
    --class_weight_clip_min "$HISTORICAL_CANDIDATE_CLASS_WEIGHT_CLIP_MIN"
    --class_weight_clip_max "$HISTORICAL_CANDIDATE_CLASS_WEIGHT_CLIP_MAX"
    --effective_num_beta "$HISTORICAL_CANDIDATE_EFFECTIVE_NUM_BETA"
  )
  if [[ "$HISTORICAL_CANDIDATE_PIN_MEMORY" == "true" ]]; then RUN_ARGS+=(--pin_memory); fi
  if [[ "$HISTORICAL_CANDIDATE_PERSISTENT_WORKERS" == "true" ]]; then RUN_ARGS+=(--persistent_workers); fi
  if [[ "$HISTORICAL_CANDIDATE_TRAIN_DROP_LAST" == "true" ]]; then RUN_ARGS+=(--train_drop_last); fi
  if [[ "$HISTORICAL_CANDIDATE_MULTI_LR" == "true" ]]; then RUN_ARGS+=(--multi_lr); fi
fi

COMPONENT_LR_ENABLED=0
if [[ "$CONDITION" == "historical_candidate" ]]; then
  if [[ "$HISTORICAL_CANDIDATE_USE_COMPONENT_LR" == "true" ]]; then
    COMPONENT_LR_ENABLED=1
  fi
elif [[ "$USE_PAPER_PROTOCOL" == "0" || "$PAPER_PROTOCOL_USE_COMPONENT_LR" == "1" ]]; then
  COMPONENT_LR_ENABLED=1
fi
if [[ "$COMPONENT_LR_ENABLED" == "1" ]]; then
  RUN_ARGS+=(
    --use_component_lr
    --lr_backbone_mult "${HISTORICAL_CANDIDATE_LR_BACKBONE_MULT:-0.5}"
    --lr_router_mult "${HISTORICAL_CANDIDATE_LR_ROUTER_MULT:-3.0}"
    --lr_expert_mult "${HISTORICAL_CANDIDATE_LR_EXPERT_MULT:-1.5}"
    --lr_classifier_mult "${HISTORICAL_CANDIDATE_LR_CLASSIFIER_MULT:-1.0}"
    --lr_other_mult "${HISTORICAL_CANDIDATE_LR_OTHER_MULT:-1.0}"
    --lr_depth_mult "${HISTORICAL_CANDIDATE_LR_DEPTH_MULT:-1.0}"
  )
fi

if [[ "$CONDITION" == "selective_fresh" ]]; then
  RUN_ARGS+=(--fresh_selective_recipe_path "$FRESH_SELECTIVE_RECIPE_PATH")
fi
if [[ "$CONDITION" == "historical_candidate" ]]; then
  RUN_ARGS+=(--historical_recipe_path "$HISTORICAL_CANDIDATE_RECIPE_PATH")
fi

if [[ "$DATASET" == "FACED" ]]; then
  RUN_ARGS+=(--faced_meta_csv "$FACED_META_CSV")
fi
if [[ "$USE_EMA" == "1" || "$USE_EMA" == "true" ]]; then
  RUN_ARGS+=(--use_ema --ema_decay "$EMA_DECAY" --ema_warmup_steps "$EMA_WARMUP_STEPS" --ema_eval_only)
elif [[ "$USE_EMA" != "0" && "$USE_EMA" != "false" ]]; then
  echo "USE_EMA must be 0/1 or false/true, got: $USE_EMA" >&2
  exit 2
else
  RUN_ARGS+=(--no-ema_eval_only)
fi

AUDIT_ARGS=("${RUN_ARGS[@]}" --expected-commit "$EXPECTED_COMMIT")
if [[ "$CONDITION" == "selective_fresh" ]]; then
  AUDIT_ARGS+=(--fresh-selective-recipe "$FRESH_SELECTIVE_RECIPE_PATH")
fi
if [[ "$CONDITION" == "historical_candidate" ]]; then
  if [[ "$HISTORICAL_CANDIDATE_SMOKE" == "1" || "$HISTORICAL_CANDIDATE_SMOKE" == "true" ]]; then
    AUDIT_ARGS+=(--historical-candidate-smoke)
  fi
fi
if [[ -n "$PAPER_PROTOCOL_PATH" ]]; then
  AUDIT_ARGS+=(--paper-protocol "$PAPER_PROTOCOL_PATH")
fi
if [[ "$REQUIRE_CLEAN" == "1" || "$REQUIRE_CLEAN" == "true" ]]; then
  AUDIT_ARGS+=(--require-clean)
elif [[ "$REQUIRE_CLEAN" != "0" && "$REQUIRE_CLEAN" != "false" ]]; then
  echo "REQUIRE_CLEAN must be 0/1 or false/true, got: $REQUIRE_CLEAN" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
"$PYTHON_BIN" "$SCRIPT_DIR/audit_revision_config.py" "${AUDIT_ARGS[@]}" \
  | tee "$MODEL_DIR/config_audit.json"

FOUNDATION_SHA256="$(sha256sum "$FOUNDATION_DIR" | awk '{print $1}')"
MANIFEST_SHA256=""
if [[ -n "$MANIFEST_PATH" ]]; then
  MANIFEST_SHA256="$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')"
fi
FRESH_SELECTIVE_RECIPE_SHA256=""
if [[ "$CONDITION" == "selective_fresh" ]]; then
  FRESH_SELECTIVE_RECIPE_SHA256="$(sha256sum "$FRESH_SELECTIVE_RECIPE_PATH" | awk '{print $1}')"
fi
HISTORICAL_CANDIDATE_RECIPE_SHA256=""
if [[ "$CONDITION" == "historical_candidate" ]]; then
  HISTORICAL_CANDIDATE_RECIPE_SHA256="$(sha256sum "$HISTORICAL_CANDIDATE_RECIPE_PATH" | awk '{print $1}')"
fi
DATA_CONTRACT_SHA256="$(sha256sum "$DATA_CONTRACT_PATH" | awk '{print $1}')"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
GIT_DIRTY="$(git -C "$REPO_DIR" status --porcelain --untracked-files=all)"
RUN_MANIFEST="$MODEL_DIR/run_manifest.json"

export DATASET CONDITION PROTOCOL RUN_MODE SEED MODEL_DIR DATASET_DIR FOUNDATION_DIR MANIFEST_PATH PAPER_PROTOCOL_PATH PAPER_PROTOCOL_ID PAPER_PROTOCOL_SHA256 PAPER_PROTOCOL_USE_COMPONENT_LR PAPER_METHOD_RECIPE_PATH PAPER_METHOD_RECIPE_ID PAPER_METHOD_RECIPE_SHA256 PAPER_METHOD_SEMANTICS_SHA256 COMPONENT_LR_ENABLED FRESH_SELECTIVE_RECIPE_PATH FRESH_SELECTIVE_RECIPE_SHA256 HISTORICAL_CANDIDATE_RECIPE_PATH HISTORICAL_CANDIDATE_RECIPE_SHA256 HISTORICAL_CANDIDATE_STAGE HISTORICAL_CANDIDATE_HISTORICAL_FAMILY_ID DATA_CONTRACT_PATH DATA_CONTRACT_SHA256 GIT_COMMIT
"$PYTHON_BIN" - "$RUN_MANIFEST" "$GIT_COMMIT" "$GIT_DIRTY" "$FOUNDATION_SHA256" "$MANIFEST_SHA256" "${RUN_ARGS[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, commit, dirty, foundation_sha256, manifest_sha256, *command = sys.argv[1:]
payload = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'repository_commit': commit,
    'git_dirty': bool(dirty),
    'dataset': os.environ['DATASET'],
    'condition': os.environ['CONDITION'],
    'protocol': os.environ['PROTOCOL'],
    'run_mode': os.environ['RUN_MODE'],
    'paper_eligible': os.environ['RUN_MODE'] == 'paper',
    'seed': int(os.environ['SEED']),
    'split_manifest_field': (
        'seedv_split_manifest'
        if os.environ['PROTOCOL'] == 'seedv_subject_disjoint'
        else 'icassp_split_manifest'
        if os.environ['DATASET'] == 'PhysioNet-MI'
        else 'lmdb_keys'
    ),
    'split_manifest_source': (
        'seedv_subject_disjoint_manifest'
        if os.environ['PROTOCOL'] == 'seedv_subject_disjoint'
        else 'physionet_subject_disjoint_manifest'
        if os.environ['DATASET'] == 'PhysioNet-MI'
        else 'lmdb___keys__'
    ),
    'model_dir': os.environ['MODEL_DIR'],
    'dataset_dir': os.environ['DATASET_DIR'],
    'foundation_dir': os.environ['FOUNDATION_DIR'],
    'foundation_checkpoint_sha256': foundation_sha256,
    'manifest_path': os.environ.get('MANIFEST_PATH', ''),
    'manifest_sha256': manifest_sha256,
    'data_contract_path': os.environ['DATA_CONTRACT_PATH'],
    'data_contract_sha256': os.environ['DATA_CONTRACT_SHA256'],
    'paper_protocol_path': os.environ.get('PAPER_PROTOCOL_PATH', ''),
    'paper_protocol_id': os.environ.get('PAPER_PROTOCOL_ID', ''),
    'paper_protocol_sha256': os.environ.get('PAPER_PROTOCOL_SHA256', ''),
    'paper_method_recipe_path': (
        os.environ.get('PAPER_METHOD_RECIPE_PATH', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'paper_method_recipe_id': (
        os.environ.get('PAPER_METHOD_RECIPE_ID', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'paper_method_recipe_sha256': (
        os.environ.get('PAPER_METHOD_RECIPE_SHA256', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'paper_method_semantics_sha256': (
        os.environ.get('PAPER_METHOD_SEMANTICS_SHA256', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'use_component_lr': os.environ.get('COMPONENT_LR_ENABLED', '0') == '1',
    'fresh_selective_recipe_path': os.environ.get('FRESH_SELECTIVE_RECIPE_PATH', '') if os.environ['CONDITION'] == 'selective_fresh' else '',
    'fresh_selective_recipe_sha256': os.environ.get('FRESH_SELECTIVE_RECIPE_SHA256', '') if os.environ['CONDITION'] == 'selective_fresh' else '',
    'historical_candidate_recipe_path': os.environ.get('HISTORICAL_CANDIDATE_RECIPE_PATH', '') if os.environ['CONDITION'] == 'historical_candidate' else '',
    'historical_candidate_recipe_sha256': os.environ.get('HISTORICAL_CANDIDATE_RECIPE_SHA256', '') if os.environ['CONDITION'] == 'historical_candidate' else '',
    'historical_candidate_stage': os.environ.get('HISTORICAL_CANDIDATE_STAGE', '') if os.environ['CONDITION'] == 'historical_candidate' else '',
    'historical_family_id': os.environ.get('HISTORICAL_CANDIDATE_HISTORICAL_FAMILY_ID', '') if os.environ['CONDITION'] == 'historical_candidate' else '',
    'command': command,
}
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "[icassp-revision] DRY_RUN=1: launch contract passed; skipping model training" >&2
  exit 0
fi

echo "[icassp-revision] dataset=$DATASET condition=$CONDITION seed=$SEED protocol=$PROTOCOL mode=$RUN_MODE" >&2
echo "[icassp-revision] paper_protocol=${PAPER_PROTOCOL_ID:-none} epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR" >&2
echo "[icassp-revision] model_dir=$MODEL_DIR" >&2
printf '[icassp-revision] command:' >&2
printf ' %q' "$PYTHON_BIN" "$REPO_DIR/finetune_main.py" "${RUN_ARGS[@]}" >&2
printf '\n' >&2

set +e
"$PYTHON_BIN" "$REPO_DIR/finetune_main.py" "${RUN_ARGS[@]}"
TRAIN_EXIT=$?
set -e

"$PYTHON_BIN" - "$MODEL_DIR/result_manifest.json" "$TRAIN_EXIT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, exit_code = sys.argv[1:]
summary = os.path.join(os.environ['MODEL_DIR'], 'experiment_summary.csv')
payload = {
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'exit_code': int(exit_code),
    'repository_commit': os.environ.get('GIT_COMMIT', ''),
    'run_mode': os.environ.get('RUN_MODE', ''),
    'paper_eligible': os.environ.get('RUN_MODE') == 'paper',
    'dataset': os.environ.get('DATASET', ''),
    'condition': os.environ.get('CONDITION', ''),
    'protocol': os.environ.get('PROTOCOL', ''),
    'seed': int(os.environ.get('SEED', '0')),
    'model_dir': os.environ.get('MODEL_DIR', ''),
    'experiment_summary': summary if os.path.isfile(summary) else '',
    'summary_present': os.path.isfile(summary),
    'data_contract_path': os.environ.get('DATA_CONTRACT_PATH', ''),
    'data_contract_sha256': os.environ.get('DATA_CONTRACT_SHA256', ''),
    'paper_protocol_path': os.environ.get('PAPER_PROTOCOL_PATH', ''),
    'paper_protocol_id': os.environ.get('PAPER_PROTOCOL_ID', ''),
    'paper_protocol_sha256': os.environ.get('PAPER_PROTOCOL_SHA256', ''),
    'paper_method_recipe_path': (
        os.environ.get('PAPER_METHOD_RECIPE_PATH', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'paper_method_recipe_id': (
        os.environ.get('PAPER_METHOD_RECIPE_ID', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'paper_method_recipe_sha256': (
        os.environ.get('PAPER_METHOD_RECIPE_SHA256', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'paper_method_semantics_sha256': (
        os.environ.get('PAPER_METHOD_SEMANTICS_SHA256', '')
        if os.environ.get('CONDITION') == 'specialist_augmented_full' else ''
    ),
    'use_component_lr': os.environ.get('COMPONENT_LR_ENABLED', '0') == '1',
    'fresh_selective_recipe_path': (
        os.environ.get('FRESH_SELECTIVE_RECIPE_PATH', '')
        if os.environ.get('CONDITION') == 'selective_fresh' else ''
    ),
    'fresh_selective_recipe_sha256': (
        os.environ.get('FRESH_SELECTIVE_RECIPE_SHA256', '')
        if os.environ.get('CONDITION') == 'selective_fresh' else ''
    ),
    'historical_candidate_recipe_path': (
        os.environ.get('HISTORICAL_CANDIDATE_RECIPE_PATH', '')
        if os.environ.get('CONDITION') == 'historical_candidate' else ''
    ),
    'historical_candidate_recipe_sha256': (
        os.environ.get('HISTORICAL_CANDIDATE_RECIPE_SHA256', '')
        if os.environ.get('CONDITION') == 'historical_candidate' else ''
    ),
    'historical_candidate_stage': (
        os.environ.get('HISTORICAL_CANDIDATE_STAGE', '')
        if os.environ.get('CONDITION') == 'historical_candidate' else ''
    ),
}
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY

exit "$TRAIN_EXIT"
