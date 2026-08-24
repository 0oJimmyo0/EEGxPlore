#!/usr/bin/env bash

# Shared ICASSP pilot settings. Override paths/resources in the environment;
# method-specific settings belong only in run_pilot.sh.
ICASSP_REPO_DIR="${ICASSP_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth}"
SEEDV_DATA_DIR="${SEEDV_DATA_DIR:-/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb}"
FACED_DATA_DIR="${FACED_DATA_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/FACED_data}"
FACED_META_CSV="${FACED_META_CSV:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv}"
ISRUC_DATA_DIR="${ISRUC_DATA_DIR:-/data/neurogroup/mingyangjiang/data/ISRUC}"
PHYSIONET_DATA_DIR="${PHYSIONET_DATA_DIR:-}"
MODEL_ROOT="${MODEL_ROOT:-$ICASSP_REPO_DIR/output/icassp2027_pilots}"
CUDA_ID="${CUDA_ID:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
DROPOUT="${DROPOUT:-0.1}"
INPUT_SCALE_DIVISOR="${INPUT_SCALE_DIVISOR:-100.0}"
