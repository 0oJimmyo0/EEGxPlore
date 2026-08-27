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
EPOCHS="${5:-${EPOCHS:-40}}"
MODEL_ROOT="${6:-${MODEL_ROOT:-$REPO_DIR/output/icassp2027_revision}}"
EXPECTED_COMMIT="${7:-${EXPECTED_COMMIT:-$(git -C "$REPO_DIR" rev-parse HEAD)}}"
HISTORICAL_RECIPE_CONFIRMED="${HISTORICAL_RECIPE_CONFIRMED:-0}"

CUDA_ID="${CUDA_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-3e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-2}"
DROPOUT="${DROPOUT:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
USE_EMA="${USE_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.9995}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-1000}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HISTORICAL_RECIPE_PATH="${HISTORICAL_RECIPE_PATH:-$SCRIPT_DIR/historical_recipe_1785556.json}"
HISTORICAL_FAMILY_ID="${HISTORICAL_FAMILY_ID:-1785556}"

if [[ "$CONDITION" == "historical_selective" ]]; then
  if [[ "$HISTORICAL_RECIPE_CONFIRMED" != "1" && "$HISTORICAL_RECIPE_CONFIRMED" != "true" ]]; then
    echo "historical_selective is locked until the historical recipe audit is complete" >&2
    echo "set HISTORICAL_RECIPE_CONFIRMED=1 only after completing experiments/icassp2027/revision/HISTORICAL_RECIPE_AUDIT.md" >&2
    exit 2
  fi
  if [[ ! -f "$HISTORICAL_RECIPE_PATH" ]]; then
    echo "historical recipe is unavailable: $HISTORICAL_RECIPE_PATH" >&2
    exit 2
  fi
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
    DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED}"
    FACED_META_CSV="${FACED_META_CSV:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv}"
    ;;
  ISRUC)
    DATASET_TAG="isruc"
    NUM_CLASSES=5
    DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/ISRUC/precessed_filter_35}"
    ;;
  PhysioNet-MI)
    DATASET_TAG="physionet_mi"
    NUM_CLASSES=4
    DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/physionet_mi}"
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac

if [[ "$PROTOCOL" != "cbramod_benchmark" && "$PROTOCOL" != "seedv_subject_disjoint" ]]; then
  echo "Unsupported revision protocol: $PROTOCOL" >&2
  exit 2
fi
if [[ "$PROTOCOL" == "seedv_subject_disjoint" && "$DATASET" != "SEED-V" ]]; then
  echo "seedv_subject_disjoint is valid only for SEED-V" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
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
if [[ "$PROTOCOL" == "seedv_subject_disjoint" ]]; then
  SEEDV_SPLIT_MANIFEST="${SEEDV_SPLIT_MANIFEST:-$REPO_DIR/experiments/icassp2027/manifests/seedv/split_manifest.json}"
  if [[ ! -f "$SEEDV_SPLIT_MANIFEST" ]]; then
    echo "Subject-disjoint manifest is unavailable: $SEEDV_SPLIT_MANIFEST" >&2
    exit 2
  fi
  SPLIT_ARGS+=(--seedv_split_manifest "$SEEDV_SPLIT_MANIFEST")
  MANIFEST_PATH="$SEEDV_SPLIT_MANIFEST"
fi

MODEL_ROOT="$(realpath -m "$MODEL_ROOT")"
MODEL_DIR="$MODEL_ROOT/$DATASET_TAG/$CONDITION/seed_$SEED"
mkdir -p "$MODEL_DIR"

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
  --classifier all_patch_reps
  --downstream_dataset "$DATASET"
  --datasets_dir "$DATASET_DIR"
  --num_of_classes "$NUM_CLASSES"
  --model_dir "$MODEL_DIR"
  --input_scale_divisor 100.0
  --num_workers "$NUM_WORKERS"
  --label_smoothing "$LABEL_SMOOTHING"
  --use_pretrained_weights
  --foundation_dir "$FOUNDATION_DIR"
  --experiment_profile icassp2027_revision
  --revision_condition "$CONDITION"
  --revision_protocol "$PROTOCOL"
  --selection_metric kappa
  --use_component_lr
  --lr_backbone_mult 0.5
  --lr_router_mult 3.0
  --lr_expert_mult 1.5
  --lr_classifier_mult 1.0
  --lr_other_mult 1.0
  --lr_depth_mult 1.0
  --no-tqdm
  "${SPLIT_ARGS[@]}"
)

if [[ "$CONDITION" == "historical_selective" ]]; then
  RUN_ARGS+=(--historical_recipe_path "$HISTORICAL_RECIPE_PATH" --historical_family_id "$HISTORICAL_FAMILY_ID")
fi

if [[ "$DATASET" == "FACED" ]]; then
  RUN_ARGS+=(--faced_meta_csv "$FACED_META_CSV")
fi
if [[ "$USE_EMA" == "1" || "$USE_EMA" == "true" ]]; then
  RUN_ARGS+=(--use_ema --ema_decay "$EMA_DECAY" --ema_warmup_steps "$EMA_WARMUP_STEPS" --ema_eval_only)
elif [[ "$USE_EMA" != "0" && "$USE_EMA" != "false" ]]; then
  echo "USE_EMA must be 0/1 or false/true, got: $USE_EMA" >&2
  exit 2
fi

AUDIT_ARGS=("${RUN_ARGS[@]}" --expected-commit "$EXPECTED_COMMIT")
if [[ "$CONDITION" == "historical_selective" ]]; then
  AUDIT_ARGS+=(--historical-recipe "$HISTORICAL_RECIPE_PATH")
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
HISTORICAL_RECIPE_SHA256=""
if [[ "$CONDITION" == "historical_selective" ]]; then
  HISTORICAL_RECIPE_SHA256="$(sha256sum "$HISTORICAL_RECIPE_PATH" | awk '{print $1}')"
fi
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
GIT_DIRTY="$(git -C "$REPO_DIR" status --porcelain --untracked-files=all)"
RUN_MANIFEST="$MODEL_DIR/run_manifest.json"

export DATASET CONDITION PROTOCOL SEED MODEL_DIR DATASET_DIR FOUNDATION_DIR MANIFEST_PATH HISTORICAL_RECIPE_PATH HISTORICAL_FAMILY_ID HISTORICAL_RECIPE_SHA256
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
    'seed': int(os.environ['SEED']),
    'split_manifest_field': (
        'seedv_split_manifest'
        if os.environ['PROTOCOL'] == 'seedv_subject_disjoint'
        else 'lmdb_keys'
    ),
    'split_manifest_source': (
        'seedv_subject_disjoint_manifest'
        if os.environ['PROTOCOL'] == 'seedv_subject_disjoint'
        else 'lmdb___keys__'
    ),
    'model_dir': os.environ['MODEL_DIR'],
    'dataset_dir': os.environ['DATASET_DIR'],
    'foundation_dir': os.environ['FOUNDATION_DIR'],
    'foundation_checkpoint_sha256': foundation_sha256,
    'manifest_path': os.environ.get('MANIFEST_PATH', ''),
    'manifest_sha256': manifest_sha256,
    'historical_family_id': os.environ.get('HISTORICAL_FAMILY_ID', '') if os.environ['CONDITION'] == 'historical_selective' else '',
    'historical_recipe_path': os.environ.get('HISTORICAL_RECIPE_PATH', '') if os.environ['CONDITION'] == 'historical_selective' else '',
    'historical_recipe_sha256': os.environ.get('HISTORICAL_RECIPE_SHA256', '') if os.environ['CONDITION'] == 'historical_selective' else '',
    'command': command,
}
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY

echo "[icassp-revision] dataset=$DATASET condition=$CONDITION seed=$SEED protocol=$PROTOCOL" >&2
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
    'dataset': os.environ.get('DATASET', ''),
    'condition': os.environ.get('CONDITION', ''),
    'protocol': os.environ.get('PROTOCOL', ''),
    'seed': int(os.environ.get('SEED', '0')),
    'model_dir': os.environ.get('MODEL_DIR', ''),
    'experiment_summary': summary if os.path.isfile(summary) else '',
    'summary_present': os.path.isfile(summary),
}
with open(path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY

exit "$TRAIN_EXIT"
