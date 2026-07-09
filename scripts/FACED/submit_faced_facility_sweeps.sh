#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/neurogroup/mingyangjiang"
EEGX_DIR="$ROOT/EEGxPlore/EEGxPlore"
LABRAM_DIR="$ROOT/EEGxPlore/LaBraM"

LABRAM_SCRIPT="$LABRAM_DIR/scripts/submit_faced_finetune_accre.slurm"
EEGX_SCRIPT="$EEGX_DIR/scripts/FACED/submit_faced_labram_accre.slurm"

if [[ ! -f "$LABRAM_SCRIPT" ]]; then
  echo "Missing LaBraM script: $LABRAM_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$EEGX_SCRIPT" ]]; then
  echo "Missing EEGxPlore script: $EEGX_SCRIPT" >&2
  exit 1
fi

LRS=(1e-5 3e-5 1e-4 3e-4 5e-4)
MAIN_BATCHES=(16)
EXTRA_BATCHES=(32)
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"

submit_labram_baseline() {
  local lr="$1"
  local bs="$2"
  local tag="$3"
  local run_name="labram_faced_${tag}_lr${lr//./p}_bs${bs}_e${EPOCHS}"
  local out_dir="$LABRAM_DIR/checkpoints/${run_name}"
  local log_dir="$LABRAM_DIR/logs/tensorboard/${run_name}"
  sbatch -J "$run_name" \
    --export=ALL,OUTPUT_DIR="$out_dir",LOG_DIR="$log_dir",LR="$lr",BATCH_SIZE="$bs",EPOCHS="$EPOCHS",WARMUP_EPOCHS="$WARMUP_EPOCHS",NUM_WORKERS=4 \
    "$LABRAM_SCRIPT"
}

submit_eegx_dense() {
  local lr="$1"
  local bs="$2"
  local tag="$3"
  local run_name="faced_labram_dense_${tag}_lr${lr//./p}_bs${bs}_e${EPOCHS}_scale100"
  local model_dir="$EEGX_DIR/output/${run_name}"
  sbatch --export=ALL,RUN_NAME="$run_name",MODEL_DIR="$model_dir",SEED=0,EPOCHS="$EPOCHS",BATCH_SIZE="$bs",NUM_WORKERS=0,LR="$lr",WARMUP_EPOCHS="$WARMUP_EPOCHS",WEIGHT_DECAY=0.05,LABRAM_LAYER_DECAY=0.65,LABRAM_INIT_SCALE=0.001,LABRAM_QKV_BIAS=0,LABRAM_USE_ABS_POS_EMB=1,LABRAM_USE_REL_POS_BIAS=0,LABRAM_ADAPTER_LAYERS=0,LABRAM_FORCE_ADAPTER=0,LABRAM_TOKEN_POOL_NO_ADAPTER=0,INPUT_SCALE_DIVISOR=100.0,SELECTION_METRIC=balanced_accuracy \
    "$EEGX_SCRIPT"
}

submit_eegx_adapt() {
  local lr="$1"
  local bs="$2"
  local tag="$3"
  local run_name="faced_labram_attnres_${tag}_lr${lr//./p}_bs${bs}_e${EPOCHS}_scale100"
  local model_dir="$EEGX_DIR/output/${run_name}"
  sbatch --export=ALL,RUN_NAME="$run_name",MODEL_DIR="$model_dir",SEED=0,EPOCHS="$EPOCHS",BATCH_SIZE="$bs",NUM_WORKERS=0,LR="$lr",WARMUP_EPOCHS="$WARMUP_EPOCHS",WEIGHT_DECAY=0.05,LABRAM_LAYER_DECAY=0.65,LABRAM_INIT_SCALE=0.001,LABRAM_QKV_BIAS=0,LABRAM_USE_ABS_POS_EMB=1,LABRAM_USE_REL_POS_BIAS=0,LABRAM_ADAPTER_LAYERS=1,LABRAM_FORCE_ADAPTER=1,ATTNRES_VARIANT=pre_attn,ATTNRES_GATED=1,ATTNRES_START_LAYER=3,LABRAM_RESIDUAL_GAMMA_INIT=0.3,INPUT_SCALE_DIVISOR=100.0,SELECTION_METRIC=balanced_accuracy \
    "$EEGX_SCRIPT"
}

echo "[sweep] submitting LaBraM original baseline sweeps" >&2
for lr in "${LRS[@]}"; do
  for bs in "${MAIN_BATCHES[@]}"; do
    submit_labram_baseline "$lr" "$bs" "orig"
  done
done
for lr in 3e-4 5e-4; do
  for bs in "${EXTRA_BATCHES[@]}"; do
    submit_labram_baseline "$lr" "$bs" "orig"
  done
done

echo "[sweep] submitting EEGxPlore dense sweeps" >&2
for lr in "${LRS[@]}"; do
  for bs in "${MAIN_BATCHES[@]}"; do
    submit_eegx_dense "$lr" "$bs" "dense"
  done
done
for lr in 3e-4 5e-4; do
  for bs in "${EXTRA_BATCHES[@]}"; do
    submit_eegx_dense "$lr" "$bs" "dense"
  done
done

echo "[sweep] submitting EEGxPlore adaptation sweeps" >&2
for lr in "${LRS[@]}"; do
  for bs in "${MAIN_BATCHES[@]}"; do
    submit_eegx_adapt "$lr" "$bs" "adapt"
  done
done
for lr in 3e-4 5e-4; do
  for bs in "${EXTRA_BATCHES[@]}"; do
    submit_eegx_adapt "$lr" "$bs" "adapt"
  done
done

echo "[sweep] all FACED sweep jobs submitted" >&2
