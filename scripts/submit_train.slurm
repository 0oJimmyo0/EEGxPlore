#!/bin/bash
#SBATCH --job-name=FACED_B1A
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/faced_b1a_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/err/faced_b1a_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

# Lean B1a launcher only.
# Default run is 2x2 geometry-isolation (prototype off):
# subject-only, eeg-only, both, neither.
# gate_2x2 remains available for scalar-gate ablations.

set -euo pipefail

# ------------------------------
# RUN IDENTITY
# ------------------------------
STAGE="${STAGE:-B1a}"
RUN_TAG="${RUN_TAG:-${STAGE}}"

if [[ "$STAGE" != "B1a" ]]; then
  echo "[slurm] This script is B1a-only. Use STAGE=B1a." >&2
  exit 2
fi

# ------------------------------
# DATA + PATHS
# ------------------------------
downstream_dataset="FACED"
dataset_dir="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED}"
model_dir="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/${downstream_dataset}_adapter_${RUN_TAG}}"
mkdir -p "$model_dir"

FACED_META_DIR="/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta"
FACED_META_CSV="${FACED_META_CSV:-$FACED_META_DIR/Recording_info.csv}"
CHANNEL_CONTEXT_FILE="${CHANNEL_CONTEXT_FILE:-$FACED_META_DIR/channel_context_rich.json}"

# ------------------------------
# TRAIN HYPERPARAMS
# ------------------------------
lr="${LR:-5e-5}"
weight_decay="${WEIGHT_DECAY:-1e-2}"
clip_value="${CLIP_VALUE:-1.0}"
dropout="${DROPOUT:-0.3}"
label_smoothing="${LABEL_SMOOTHING:-0.05}"
epochs="${EPOCHS:-30}"
batch_size="${BATCH_SIZE:-32}"
seed="${SEED:-3407}"
cuda_id="${CUDA_ID:-0}"
MULTI_LR="${MULTI_LR:-0}" # 0|1 (recommended 0 for B1a isolation)

# ------------------------------
# B1a MECHANISM KNOBS
# ------------------------------
ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
ATTNRES_START_LAYER="${ATTNRES_START_LAYER:-0}"
ATTNRES_GATED="${ATTNRES_GATED:-1}"
ATTNRES_GATE_INIT="${ATTNRES_GATE_INIT:-0.0}"
ATTNRES_SUBJECT_GATES="${ATTNRES_SUBJECT_GATES:-1}"
ATTNRES_EEG_GATE_SCALE="${ATTNRES_EEG_GATE_SCALE:-0.2}"
ATTNRES_EEG_DEPTH_COND_SCALE="${ATTNRES_EEG_DEPTH_COND_SCALE:-1.0}"
ATTNRES_DIAG_INTERVAL="${ATTNRES_DIAG_INTERVAL:-1}"

ADAPTER_MODE="${ADAPTER_MODE:-subject_domain}"
ADAPTER_NUM_LAYERS="${ADAPTER_NUM_LAYERS:-4}"
ADAPTER_RANK="${ADAPTER_RANK:-8}"
ADAPTER_COND_DIM="${ADAPTER_COND_DIM:-32}"
ADAPTER_SCALE="${ADAPTER_SCALE:-0.001}"
ADAPTER_USE_AGE_BUCKET="${ADAPTER_USE_AGE_BUCKET:-1}"
ADAPTER_USE_SEGMENT_BUCKET="${ADAPTER_USE_SEGMENT_BUCKET:-0}"

# B1a is cross-subject oriented. Default: disable direct subject id conditioning.
ADAPTER_USE_SUBJECT_ID="${ADAPTER_USE_SUBJECT_ID:-0}"
SUBJECT_OVERLAP_POLICY="${SUBJECT_OVERLAP_POLICY:-disable}"
CHANNEL_ID_ALIGN_MODE="${CHANNEL_ID_ALIGN_MODE:-auto}"
SUBJECT_SUMMARY_HANDLING="${SUBJECT_SUMMARY_HANDLING:-project}"

PROTOTYPE_MOMENTUM="${PROTOTYPE_MOMENTUM:-0.95}"
PROTOTYPE_PULL_WEIGHT="${PROTOTYPE_PULL_WEIGHT:-0.05}"
PROTOTYPE_PUSH_WEIGHT="${PROTOTYPE_PUSH_WEIGHT:-0.02}"
PROTOTYPE_MARGIN="${PROTOTYPE_MARGIN:-0.10}"
PROTOTYPE_START_EPOCH="${PROTOTYPE_START_EPOCH:-1}"

FORCE_TQDM="${FORCE_TQDM:-0}"
METADATA_DEBUG="${METADATA_DEBUG:-1}"
B1A_MODE="${B1A_MODE:-geometry_2x2}" # gate_2x2 | geometry_2x2

# Targeted optimization knobs
B1A_SUBJECT_LR="${B1A_SUBJECT_LR:-5e-5}"
B1A_EEG_LR="${B1A_EEG_LR:-5e-5}"
B1A_GEOMETRY_LR="${B1A_GEOMETRY_LR:-5e-5}"
ENABLE_SCALE_SWEEP="${ENABLE_SCALE_SWEEP:-0}" # 0|1
GATE_SCALE_SWEEP="${GATE_SCALE_SWEEP:-0.1,0.2,0.5}"
DEPTH_SCALE_SWEEP="${DEPTH_SCALE_SWEEP:-0.5,1.0,1.5}"

if [[ "$B1A_MODE" != "gate_2x2" && "$B1A_MODE" != "geometry_2x2" ]]; then
  echo "[slurm] B1A_MODE must be gate_2x2 or geometry_2x2 (got '$B1A_MODE')." >&2
  exit 2
fi
if [[ "$MULTI_LR" != "0" && "$MULTI_LR" != "1" ]]; then
  echo "[slurm] MULTI_LR must be 0 or 1 (got '$MULTI_LR')." >&2
  exit 2
fi

# ------------------------------
# VALIDATION
# ------------------------------
if [[ ! -f "$FACED_META_CSV" ]]; then
  echo "[slurm] FACED_META_CSV not found: $FACED_META_CSV" >&2
  exit 2
fi

if [[ ! -f "$CHANNEL_CONTEXT_FILE" ]]; then
  echo "[slurm] CHANNEL_CONTEXT_FILE not found: $CHANNEL_CONTEXT_FILE" >&2
  exit 2
fi

# ------------------------------
# ENV
# ------------------------------
module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

# ------------------------------
# SHARED FLAGS
# ------------------------------
COMMON_FLAGS=(
  --seed "$seed"
  --cuda "$cuda_id"
  --epochs "$epochs"
  --batch_size "$batch_size"
  --lr "$lr"
  --weight_decay "$weight_decay"
  --optimizer AdamW
  --clip_value "$clip_value"
  --dropout "$dropout"
  --classifier all_patch_reps
  --downstream_dataset "$downstream_dataset"
  --datasets_dir "$dataset_dir"
  --num_of_classes 9
  --label_smoothing "$label_smoothing"
  --num_workers 4
  --attnres_variant "$ATTNRES_VARIANT"
  --attnres_start_layer "$ATTNRES_START_LAYER"
  --use_pretrained_weights
  --adapter_mode "$ADAPTER_MODE"
  --adapter_num_layers "$ADAPTER_NUM_LAYERS"
  --adapter_rank "$ADAPTER_RANK"
  --adapter_cond_dim "$ADAPTER_COND_DIM"
  --adapter_scale "$ADAPTER_SCALE"
  --subject_overlap_policy "$SUBJECT_OVERLAP_POLICY"
  --channel_id_align_mode "$CHANNEL_ID_ALIGN_MODE"
  --subject_summary_handling "$SUBJECT_SUMMARY_HANDLING"
  --attnres_eeg_gate_scale "$ATTNRES_EEG_GATE_SCALE"
  --attnres_eeg_depth_cond_scale "$ATTNRES_EEG_DEPTH_COND_SCALE"
  --attnres_diag_interval "$ATTNRES_DIAG_INTERVAL"
  --prototype_momentum "$PROTOTYPE_MOMENTUM"
  --prototype_pull_weight "$PROTOTYPE_PULL_WEIGHT"
  --prototype_push_weight "$PROTOTYPE_PUSH_WEIGHT"
  --prototype_margin "$PROTOTYPE_MARGIN"
  --prototype_start_epoch "$PROTOTYPE_START_EPOCH"
  --faced_meta_csv "$FACED_META_CSV"
  --no-moe_use_attnres_depth_router_concat
)

if [[ "$ATTNRES_GATED" == "1" ]]; then
  COMMON_FLAGS+=(--attnres_gated --attnres_gate_init "$ATTNRES_GATE_INIT")
fi
if [[ "$ATTNRES_SUBJECT_GATES" == "1" ]]; then
  COMMON_FLAGS+=(--attnres_subject_gates)
fi
if [[ "$ADAPTER_USE_AGE_BUCKET" == "1" ]]; then
  COMMON_FLAGS+=(--adapter_use_age_bucket)
else
  COMMON_FLAGS+=(--no-adapter_use_age_bucket)
fi
if [[ "$ADAPTER_USE_SEGMENT_BUCKET" == "1" ]]; then
  COMMON_FLAGS+=(--adapter_use_segment_bucket)
else
  COMMON_FLAGS+=(--no-adapter_use_segment_bucket)
fi
if [[ "$ADAPTER_USE_SUBJECT_ID" == "1" ]]; then
  COMMON_FLAGS+=(--adapter_use_subject_id)
else
  COMMON_FLAGS+=(--no-adapter_use_subject_id)
fi
if [[ "$METADATA_DEBUG" == "1" ]]; then
  COMMON_FLAGS+=(--metadata_debug)
else
  COMMON_FLAGS+=(--no-metadata_debug)
fi
if [[ "$MULTI_LR" == "1" ]]; then
  COMMON_FLAGS+=(--multi_lr)
else
  COMMON_FLAGS+=(--no-multi_lr)
fi
if [[ "$FORCE_TQDM" == "1" ]]; then
  COMMON_FLAGS+=(--tqdm)
else
  COMMON_FLAGS+=(--no-tqdm)
fi

scale_to_label() {
  local v="$1"
  v="${v//-/m}"
  v="${v//./p}"
  echo "$v"
}

run_finetune() {
  local local_model_dir="$1"
  shift

  python /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/finetune_main.py \
    --model_dir "$local_model_dir" \
    "${COMMON_FLAGS[@]}" \
    "$@"
}

summarize_log() {
  local label="$1"
  local log_path="$2"

  local best_vals
  best_vals=$(grep -E "^Epoch [0-9]+ :" "$log_path" | \
    awk -F'[,: ]+' 'BEGIN{bestAcc=-1;bestK=-1} {ep=$2; acc=$7; k=$9; if(acc>bestAcc){bestAcc=acc} if(k>bestK){bestK=k} lastAcc=acc; lastK=k; lastEp=ep} END{printf("%.5f %.5f %d %.5f %.5f",bestAcc,bestK,lastEp,lastAcc,lastK)}')

  if [[ -z "$best_vals" ]]; then
    echo "[summary][parse_error] label=$label log=$log_path" >&2
    return 1
  fi

  local best_acc best_kappa last_ep last_acc last_kappa
  read -r best_acc best_kappa last_ep last_acc last_kappa <<< "$best_vals"

  local test_vals test_acc test_kappa test_f1
  test_vals=$(grep -E "^Test Evaluation:" "$log_path" | tail -n1 | \
    awk -F'[,: ]+' '{printf("%.5f %.5f %.5f",$4,$6,$8)}')
  if [[ -z "$test_vals" ]]; then
    test_acc="nan"
    test_kappa="nan"
    test_f1="nan"
  else
    read -r test_acc test_kappa test_f1 <<< "$test_vals"
  fi

  local eeg_batch_std
  eeg_batch_std=$(grep -E "\\[eeg-summary\\]" "$log_path" | tail -n1 | \
    sed -n 's/.*batch_std_mean=\([0-9.eE+-]*\).*/\1/p')
  eeg_batch_std="${eeg_batch_std:-nan}"

  local max_query_delta_std
  max_query_delta_std=$(grep -oE 'pre_attn_query_delta_std=[0-9.eE+-]+' "$log_path" | \
    sed 's/.*=//' | sort -g | tail -n1)
  max_query_delta_std="${max_query_delta_std:-nan}"

  local max_depth_cond_std
  max_depth_cond_std=$(grep -oE 'pre_attn_depth_cond_std=[0-9.eE+-]+' "$log_path" | \
    sed 's/.*=//' | sort -g | tail -n1)
  max_depth_cond_std="${max_depth_cond_std:-nan}"

  local max_eeg_gate_std
  max_eeg_gate_std=$(grep -oE 'pre_attn_eeg_delta_std=[0-9.eE+-]+' "$log_path" | \
    sed 's/.*=//' | sort -g | tail -n1)
  max_eeg_gate_std="${max_eeg_gate_std:-nan}"

  echo "$label $best_acc $best_kappa $last_ep $last_acc $last_kappa $test_acc $test_kappa $test_f1 $eeg_batch_std $max_query_delta_std $max_depth_cond_std $max_eeg_gate_std"
}

calc_delta() {
  awk -v a="$1" -v b="$2" 'BEGIN {printf("%.5f", a - b)}'
}

B1A_LOG_DIR="${B1A_LOG_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/b1a_${SLURM_JOB_ID}}"
mkdir -p "$B1A_LOG_DIR"

declare -a B1A_ROWS=()
declare -A B1A_BEST_ACC=()
declare -A B1A_BEST_KAPPA=()

run_and_collect_b1a() {
  local label="$1"
  shift

  local run_model_dir="$model_dir/$label"
  local run_log="$B1A_LOG_DIR/${label}.log"
  mkdir -p "$run_model_dir"

  echo "[B1a] start label=$label model_dir=$run_model_dir" >&2
  run_finetune "$run_model_dir" "$@" 2>&1 | tee "$run_log"

  local row
  row=$(summarize_log "$label" "$run_log") || exit 3
  B1A_ROWS+=("$row")

  local _label _best_acc _best_kappa _last_ep _last_acc _last_kappa
  read -r _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _rest <<< "$row"
  B1A_BEST_ACC["$_label"]="$_best_acc"
  B1A_BEST_KAPPA["$_label"]="$_best_kappa"
}

if [[ "$B1A_MODE" == "gate_2x2" ]]; then
  # 2x2 scalar-gate isolation with prototype off.
  run_and_collect_b1a "B1a_exp0_subject_only" \
    --lr "$B1A_SUBJECT_LR" \
    --attnres_subject_gates \
    --no-attnres_eeg_cond_gates \
    --no-attnres_eeg_depth_cond \
    --no-prototype_alignment

  run_and_collect_b1a "B1a_exp1_eeg_only" \
    --lr "$B1A_EEG_LR" \
    --no-attnres_subject_gates \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --attnres_eeg_cond_gates \
    --no-attnres_eeg_depth_cond \
    --attnres_eeg_gate_scale "$ATTNRES_EEG_GATE_SCALE" \
    --no-prototype_alignment

  run_and_collect_b1a "B1a_exp2_both" \
    --lr "$B1A_EEG_LR" \
    --attnres_subject_gates \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --attnres_eeg_cond_gates \
    --no-attnres_eeg_depth_cond \
    --attnres_eeg_gate_scale "$ATTNRES_EEG_GATE_SCALE" \
    --no-prototype_alignment

  run_and_collect_b1a "B1a_exp3_neither" \
    --lr "$B1A_SUBJECT_LR" \
    --no-attnres_subject_gates \
    --no-attnres_eeg_cond_gates \
    --no-attnres_eeg_depth_cond \
    --no-prototype_alignment

  if [[ "$ENABLE_SCALE_SWEEP" == "1" ]]; then
    IFS=',' read -r -a gate_scales <<< "$GATE_SCALE_SWEEP"
    for s in "${gate_scales[@]}"; do
      if [[ "$s" == "$ATTNRES_EEG_GATE_SCALE" ]]; then
        continue
      fi
      slabel="$(scale_to_label "$s")"
      run_and_collect_b1a "B1a_exp1_eeg_only_s${slabel}" \
        --lr "$B1A_EEG_LR" \
        --no-attnres_subject_gates \
        --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
        --attnres_eeg_cond_gates \
        --no-attnres_eeg_depth_cond \
        --attnres_eeg_gate_scale "$s" \
        --no-prototype_alignment

      run_and_collect_b1a "B1a_exp2_both_s${slabel}" \
        --lr "$B1A_EEG_LR" \
        --attnres_subject_gates \
        --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
        --attnres_eeg_cond_gates \
        --no-attnres_eeg_depth_cond \
        --attnres_eeg_gate_scale "$s" \
        --no-prototype_alignment
    done
  fi
else
  # 2x2 geometry-conditioned depth selection isolation with prototype off.
  run_and_collect_b1a "B1a_exp0_subject_only" \
    --lr "$B1A_SUBJECT_LR" \
    --attnres_subject_gates \
    --no-attnres_eeg_cond_gates \
    --no-attnres_eeg_depth_cond \
    --no-prototype_alignment

  run_and_collect_b1a "B1a_exp1_eeg_only" \
    --lr "$B1A_GEOMETRY_LR" \
    --no-attnres_subject_gates \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --no-attnres_eeg_cond_gates \
    --attnres_eeg_depth_cond \
    --attnres_eeg_depth_cond_scale "$ATTNRES_EEG_DEPTH_COND_SCALE" \
    --no-prototype_alignment

  run_and_collect_b1a "B1a_exp2_both" \
    --lr "$B1A_GEOMETRY_LR" \
    --attnres_subject_gates \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --no-attnres_eeg_cond_gates \
    --attnres_eeg_depth_cond \
    --attnres_eeg_depth_cond_scale "$ATTNRES_EEG_DEPTH_COND_SCALE" \
    --no-prototype_alignment

  run_and_collect_b1a "B1a_exp3_neither" \
    --lr "$B1A_SUBJECT_LR" \
    --no-attnres_subject_gates \
    --no-attnres_eeg_cond_gates \
    --no-attnres_eeg_depth_cond \
    --no-prototype_alignment

  if [[ "$ENABLE_SCALE_SWEEP" == "1" ]]; then
    IFS=',' read -r -a depth_scales <<< "$DEPTH_SCALE_SWEEP"
    for s in "${depth_scales[@]}"; do
      if [[ "$s" == "$ATTNRES_EEG_DEPTH_COND_SCALE" ]]; then
        continue
      fi
      slabel="$(scale_to_label "$s")"
      run_and_collect_b1a "B1a_exp1_eeg_only_s${slabel}" \
        --lr "$B1A_GEOMETRY_LR" \
        --no-attnres_subject_gates \
        --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
        --no-attnres_eeg_cond_gates \
        --attnres_eeg_depth_cond \
        --attnres_eeg_depth_cond_scale "$s" \
        --no-prototype_alignment

      run_and_collect_b1a "B1a_exp2_both_s${slabel}" \
        --lr "$B1A_GEOMETRY_LR" \
        --attnres_subject_gates \
        --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
        --no-attnres_eeg_cond_gates \
        --attnres_eeg_depth_cond \
        --attnres_eeg_depth_cond_scale "$s" \
        --no-prototype_alignment
    done
  fi
fi

echo "[B1a][summary] log_dir=$B1A_LOG_DIR"
echo "[B1a][summary] label best_acc best_kappa last_ep last_acc last_kappa test_acc test_kappa test_f1 eeg_batch_std max_query_delta_std max_depth_cond_std max_eeg_gate_std"
for row in "${B1A_ROWS[@]}"; do
  echo "[B1a][summary] $row"
done

s_acc="${B1A_BEST_ACC[B1a_exp0_subject_only]}"
s_kappa="${B1A_BEST_KAPPA[B1a_exp0_subject_only]}"
e_acc="${B1A_BEST_ACC[B1a_exp1_eeg_only]}"
e_kappa="${B1A_BEST_KAPPA[B1a_exp1_eeg_only]}"
b_acc="${B1A_BEST_ACC[B1a_exp2_both]}"
b_kappa="${B1A_BEST_KAPPA[B1a_exp2_both]}"
n_acc="${B1A_BEST_ACC[B1a_exp3_neither]}"
n_kappa="${B1A_BEST_KAPPA[B1a_exp3_neither]}"

echo "[B1a][delta] mode=$B1A_MODE vs=subject_only target=eeg_only d_best_acc=$(calc_delta "$e_acc" "$s_acc") d_best_kappa=$(calc_delta "$e_kappa" "$s_kappa")"
echo "[B1a][delta] mode=$B1A_MODE vs=subject_only target=both d_best_acc=$(calc_delta "$b_acc" "$s_acc") d_best_kappa=$(calc_delta "$b_kappa" "$s_kappa")"
echo "[B1a][delta] mode=$B1A_MODE vs=subject_only target=neither d_best_acc=$(calc_delta "$n_acc" "$s_acc") d_best_kappa=$(calc_delta "$n_kappa" "$s_kappa")"
echo "[B1a][delta] mode=$B1A_MODE vs=eeg_only target=both d_best_acc=$(calc_delta "$b_acc" "$e_acc") d_best_kappa=$(calc_delta "$b_kappa" "$e_kappa")"
echo "[B1a][delta] mode=$B1A_MODE vs=neither target=eeg_only d_best_acc=$(calc_delta "$e_acc" "$n_acc") d_best_kappa=$(calc_delta "$e_kappa" "$n_kappa")"
