#!/bin/bash
#SBATCH --job-name=FACED_ADAPTER
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/faced_adapter_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/err/faced_adapter_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

# =============================================================================
# CBraMod adapter-first finetuning script (new version)
#
# How to use:
# 1) Edit the KNOBS section below.
# 2) Submit: sbatch scripts/submit_train.slurm
# 3) Monitor:
#    - queue:   squeue -u $USER
#    - logs:    tail -f logs/out/faced_adapter_<JOBID>.out
#    - errors:  tail -f logs/err/faced_adapter_<JOBID>.err
#
# What to watch in logs for performance:
# - per-epoch validation line: "Epoch ... acc ... kappa ... f1"
# - final test line:           "Test Evaluation: acc: ..., kappa: ..., f1: ..."
# - best checkpoint path is printed as "model save in ..." and also includes metrics in filename.
# =============================================================================

# ------------------------------
# RUN IDENTITY
# ------------------------------
RUN_TAG="${RUN_TAG:-adapter_base}"

# ------------------------------
# DATA + PATHS
# ------------------------------
downstream_dataset=FACED
dataset_dir=/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED
model_dir=/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/${downstream_dataset}_adapter_${RUN_TAG}
mkdir -p "$model_dir"

FACED_META_CSV=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv

# Optional EEG channel-context metadata file (.json/.pt/.pth)
# Must contain any subset of: channel_ids, coords, montage_mask, region_ids
CHANNEL_CONTEXT_FILE="${CHANNEL_CONTEXT_FILE:-}"

# Optional subject summary map (.json/.pt/.pth): {"sub000": [...], ...}
SUBJECT_SUMMARY_FILE="${SUBJECT_SUMMARY_FILE:-}"

# ------------------------------
# TRAIN HYPERPARAMS
# ------------------------------
lr="${LR:-5e-4}"
weight_decay="${WEIGHT_DECAY:-5e-2}"
clip_value="${CLIP_VALUE:-1.0}"
dropout="${DROPOUT:-0.3}"
label_smoothing="${LABEL_SMOOTHING:-0.05}"
epochs="${EPOCHS:-30}"
batch_size="${BATCH_SIZE:-32}"
seed="${SEED:-3407}"
cuda_id="${CUDA_ID:-0}"

# ------------------------------
# ADAPTER + EEG CONTEXT KNOBS
# ------------------------------
ADAPTER_MODE="${ADAPTER_MODE:-subject_domain}"   # subject_domain | none
ADAPTER_NUM_LAYERS="${ADAPTER_NUM_LAYERS:-2}"
ADAPTER_RANK="${ADAPTER_RANK:-16}"
ADAPTER_COND_DIM="${ADAPTER_COND_DIM:-32}"
ADAPTER_SCALE="${ADAPTER_SCALE:-0.2}"

# 1 => enable channel/electrode context injection
EEG_CONTEXT="${EEG_CONTEXT:-1}"

# ------------------------------
# CONTINUAL LEARNING KNOBS
# ------------------------------
# off | replay | replay_distill
CONTINUAL_MODE="${CONTINUAL_MODE:-off}"
CONTINUAL_MEMORY_SIZE="${CONTINUAL_MEMORY_SIZE:-512}"
CONTINUAL_REPLAY_BATCH_SIZE="${CONTINUAL_REPLAY_BATCH_SIZE:-32}"
CONTINUAL_REPLAY_WEIGHT="${CONTINUAL_REPLAY_WEIGHT:-0.5}"
CONTINUAL_DISTILL_WEIGHT="${CONTINUAL_DISTILL_WEIGHT:-0.2}"
CONTINUAL_DISTILL_TEMP="${CONTINUAL_DISTILL_TEMP:-2.0}"

# 1 => freeze backbone except adapter/context/classifier (for incremental subject update)
ADAPTER_ONLY_UPDATE="${ADAPTER_ONLY_UPDATE:-0}"

# ------------------------------
# LEGACY MoE SWITCH (optional)
# ------------------------------
# Keep 0 for the new version. Set to 1 only if you want old routing baseline comparisons.
USE_LEGACY_MOE="${USE_LEGACY_MOE:-0}"
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-1.0}"
MOE_DOMAIN_BIAS="${MOE_DOMAIN_BIAS:-1}"
MOE_DOMAIN_EMB_DIM="${MOE_DOMAIN_EMB_DIM:-16}"
MOE_LOAD_BALANCE="${MOE_LOAD_BALANCE:-0.005}"
MOE_DOMAIN_BIAS_REG="${MOE_DOMAIN_BIAS_REG:-0.0005}"
MOE_USE_PSD="${MOE_USE_PSD:-0}"

# tqdm in Slurm logs can generate very large files; keep this 0 on cluster
FORCE_TQDM="${FORCE_TQDM:-0}"

# --- modules ---
module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

# Optional flags assembly
EXTRA_FLAGS=()

if [[ "$EEG_CONTEXT" == "1" ]]; then
  EXTRA_FLAGS+=(--eeg_channel_context)
fi
if [[ -n "$CHANNEL_CONTEXT_FILE" ]]; then
  EXTRA_FLAGS+=(--channel_context_file "$CHANNEL_CONTEXT_FILE")
fi
if [[ -n "$SUBJECT_SUMMARY_FILE" ]]; then
  EXTRA_FLAGS+=(--subject_summary_file "$SUBJECT_SUMMARY_FILE")
fi
if [[ "$ADAPTER_ONLY_UPDATE" == "1" ]]; then
  EXTRA_FLAGS+=(--adapter_only_update)
fi
if [[ "$FORCE_TQDM" == "1" ]]; then
  EXTRA_FLAGS+=(--tqdm)
else
  EXTRA_FLAGS+=(--no-tqdm)
fi

# Legacy MoE compatibility path
if [[ "$USE_LEGACY_MOE" == "1" ]]; then
  EXTRA_FLAGS+=(--moe --moe_route_mode typed_capacity_domain --moe_diagnostics)
  EXTRA_FLAGS+=(--moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS")
  EXTRA_FLAGS+=(--moe_capacity_factor "$MOE_CAPACITY_FACTOR")
  EXTRA_FLAGS+=(--moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM")
  EXTRA_FLAGS+=(--moe_load_balance "$MOE_LOAD_BALANCE")
  EXTRA_FLAGS+=(--moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG")
  EXTRA_FLAGS+=(--faced_meta_csv "$FACED_META_CSV")
  if [[ "$MOE_DOMAIN_BIAS" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_domain_bias)
  fi
  if [[ "$MOE_USE_PSD" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_use_psd_router_features)
  fi
fi

python /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/finetune_main.py \
  --seed "$seed" \
  --cuda "$cuda_id" \
  --epochs "$epochs" \
  --batch_size "$batch_size" \
  --lr "$lr" \
  --weight_decay "$weight_decay" \
  --optimizer AdamW \
  --clip_value "$clip_value" \
  --dropout "$dropout" \
  --classifier all_patch_reps \
  --downstream_dataset "$downstream_dataset" \
  --datasets_dir "$dataset_dir" \
  --num_of_classes 9 \
  --model_dir "$model_dir" \
  --label_smoothing "$label_smoothing" \
  --num_workers 4 \
  --attnres_variant pre_attn \
  --attnres_start_layer 0 \
  --use_pretrained_weights \
  --adapter_mode "$ADAPTER_MODE" \
  --adapter_num_layers "$ADAPTER_NUM_LAYERS" \
  --adapter_rank "$ADAPTER_RANK" \
  --adapter_cond_dim "$ADAPTER_COND_DIM" \
  --adapter_scale "$ADAPTER_SCALE" \
  --continual_mode "$CONTINUAL_MODE" \
  --continual_memory_size "$CONTINUAL_MEMORY_SIZE" \
  --continual_replay_batch_size "$CONTINUAL_REPLAY_BATCH_SIZE" \
  --continual_replay_weight "$CONTINUAL_REPLAY_WEIGHT" \
  --continual_distill_weight "$CONTINUAL_DISTILL_WEIGHT" \
  --continual_distill_temp "$CONTINUAL_DISTILL_TEMP" \
  "${EXTRA_FLAGS[@]}"

# =============================================================================
# Suggested sweeps for impact analysis (change one group at a time)
# -----------------------------------------------------------------------------
# A) Adapter capacity:
#    ADAPTER_RANK=8,16,32 ; ADAPTER_NUM_LAYERS=1,2,4 ; ADAPTER_COND_DIM=16,32,64
#
# B) EEG metadata injection:
#    EEG_CONTEXT=0 vs 1
#    with/without CHANNEL_CONTEXT_FILE
#
# C) Continual-learning stability:
#    CONTINUAL_MODE=off vs replay vs replay_distill
#    CONTINUAL_MEMORY_SIZE=128,512,1024
#    CONTINUAL_REPLAY_WEIGHT=0.25,0.5,1.0
#    CONTINUAL_DISTILL_WEIGHT=0.1,0.2,0.5
#
# D) Incremental update mode:
#    ADAPTER_ONLY_UPDATE=0 vs 1
#
# Quick metric extraction from a finished .out file:
#   grep -E "Epoch .*acc:|Test Evaluation:" logs/out/faced_adapter_<JOBID>.out
# =============================================================================
