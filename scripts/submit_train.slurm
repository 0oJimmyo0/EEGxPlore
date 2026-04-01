#!/bin/bash
#SBATCH --job-name=FACED_ADAPTER
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/faced_adapter_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/err/faced_adapter_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

# CBraMod finetune launcher.
# Submit: sbatch scripts/submit_train.slurm
# Monitor: squeue -u $USER ; tail -f logs/out/faced_adapter_<JOBID>.out

# ------------------------------
# RUN IDENTITY
# ------------------------------
STAGE="${STAGE:-B1a}"  # A0|A1|A2|A2b|A3|A3b|A3c|A3d|A3e|A4|A5|B1a|custom
RUN_TAG="${RUN_TAG:-${STAGE}}"

# ------------------------------
# DATA + PATHS
# ------------------------------
downstream_dataset=FACED
dataset_dir=/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED
model_dir=/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/${downstream_dataset}_adapter_${RUN_TAG}
mkdir -p "$model_dir"

FACED_META_DIR=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta
FACED_META_CSV="${FACED_META_CSV:-$FACED_META_DIR/Recording_info.csv}"

# Optional EEG channel-context metadata file (.json/.pt/.pth/.xlsx)
# Must contain any subset of: channel_ids, coords, montage_mask, region_ids.
# For FACED, Electrode_Location.xlsx is supported directly.
CHANNEL_CONTEXT_FILE="${CHANNEL_CONTEXT_FILE:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/channel_context_rich.json}"

# Optional subject summary map (.json/.pt/.pth): {"sub000": [...], ...}
SUBJECT_SUMMARY_FILE="${SUBJECT_SUMMARY_FILE:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/subject_summary_recording_info.json}"
USE_SUBJECT_SUMMARY="${USE_SUBJECT_SUMMARY:-}"   # 0|1, stage preset applies when unset
SUBJECT_SUMMARY_HANDLING="${SUBJECT_SUMMARY_HANDLING:-project}"  # project|error

# ------------------------------
# TRAIN HYPERPARAMS
# ------------------------------
lr="${LR:-1e-4}"
weight_decay="${WEIGHT_DECAY:-1e-2}"
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
# Stage presets (can be overridden by explicit env vars)
ATTNRES_VARIANT_PRESET="none"
EEG_CONTEXT_PRESET="0"
ADAPTER_MODE_PRESET="none"
USE_SUBJECT_SUMMARY_PRESET="0"
ANTI_COLLAPSE_NO_SUBJECT_ID_PRESET="0"
MOE_LINEAR_ROUTER_INPUT_NORM_PRESET="0"
MOE_ROUTER_TEMPERATURE_PRESET="1.0"
MOE_ROUTER_ENTROPY_COEF_PRESET="0.0"
MOE_ROUTER_BALANCE_KL_COEF_PRESET="0.0"
MOE_ROUTER_Z_LOSS_COEF_PRESET="0.0"
MOE_ROUTER_JITTER_STD_PRESET="0.0"
MOE_ROUTER_JITTER_FINAL_STD_PRESET="0.0"
MOE_ROUTER_JITTER_ANNEAL_EPOCHS_PRESET="0"
MOE_ROUTER_SOFT_WARMUP_EPOCHS_PRESET="0"
MOE_ROUTER_WARMUP_MODE_PRESET="off"
MOE_ROUTER_WARMUP_EPOCHS_PRESET="0"
MOE_ROUTER_WARMUP_LR_SCALE_PRESET="0.1"
MOE_CAPACITY_FACTOR_PRESET="1.25"
MOE_ROUTER_DISPATCH_MODE_PRESET="hard_capacity"
MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT_PRESET="1"
MOE_ATTNRES_DEPTH_ROUTER_DIM_PRESET="4"
MOE_SYSTEM_MODE_PRESET="dense_soft"

case "$STAGE" in
  A0)
    ATTNRES_VARIANT_PRESET="none"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="none"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A1)
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="none"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A2)
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="1"
    ADAPTER_MODE_PRESET="none"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A2b)
    # Special diagnostic bundle; per-run knobs are configured later.
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="1"
    ADAPTER_MODE_PRESET="none"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A3)
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A3b)
    # Special A3 comparison bundle; per-run knobs are configured later.
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A3c)
    # Dense soft ablation (no AttnRes-depth router concat).
    # Per-run knobs for A3c bundle are configured later.
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    # Anti-collapse defaults for A3c. Can still be overridden by explicit env vars.
    ANTI_COLLAPSE_NO_SUBJECT_ID_PRESET="1"
    MOE_LINEAR_ROUTER_INPUT_NORM_PRESET="1"
    MOE_ROUTER_TEMPERATURE_PRESET="1.5"
    MOE_ROUTER_ENTROPY_COEF_PRESET="0.01"
    MOE_ROUTER_BALANCE_KL_COEF_PRESET="0.08"
    MOE_ROUTER_Z_LOSS_COEF_PRESET="0.002"
    MOE_ROUTER_JITTER_STD_PRESET="0.15"
    MOE_ROUTER_JITTER_FINAL_STD_PRESET="0.0"
    MOE_ROUTER_JITTER_ANNEAL_EPOCHS_PRESET="5"
    MOE_ROUTER_SOFT_WARMUP_EPOCHS_PRESET="5"
    MOE_ROUTER_WARMUP_MODE_PRESET="low_lr"
    MOE_ROUTER_WARMUP_EPOCHS_PRESET="3"
    MOE_ROUTER_WARMUP_LR_SCALE_PRESET="0.1"
    MOE_CAPACITY_FACTOR_PRESET="1.25"
    MOE_SYSTEM_MODE_PRESET="dense_soft"
    MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT_PRESET="0"
    MOE_ATTNRES_DEPTH_ROUTER_DIM_PRESET="4"
    ;;
  A3d)
    # Dense soft ablation (+ AttnRes-depth router concat).
    # Per-run knobs for A3d bundle are configured later.
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    # Anti-collapse defaults for A3d. Can still be overridden by explicit env vars.
    ANTI_COLLAPSE_NO_SUBJECT_ID_PRESET="1"
    MOE_LINEAR_ROUTER_INPUT_NORM_PRESET="1"
    MOE_ROUTER_TEMPERATURE_PRESET="1.5"
    MOE_ROUTER_ENTROPY_COEF_PRESET="0.01"
    MOE_ROUTER_BALANCE_KL_COEF_PRESET="0.08"
    MOE_ROUTER_Z_LOSS_COEF_PRESET="0.002"
    MOE_ROUTER_JITTER_STD_PRESET="0.15"
    MOE_ROUTER_JITTER_FINAL_STD_PRESET="0.0"
    MOE_ROUTER_JITTER_ANNEAL_EPOCHS_PRESET="5"
    MOE_ROUTER_SOFT_WARMUP_EPOCHS_PRESET="5"
    MOE_ROUTER_WARMUP_MODE_PRESET="low_lr"
    MOE_ROUTER_WARMUP_EPOCHS_PRESET="3"
    MOE_ROUTER_WARMUP_LR_SCALE_PRESET="0.1"
    MOE_CAPACITY_FACTOR_PRESET="1.25"
    MOE_SYSTEM_MODE_PRESET="dense_soft"
    MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT_PRESET="1"
    MOE_ATTNRES_DEPTH_ROUTER_DIM_PRESET="4"
    ;;
  A3e)
    # Decision bundle:
    # 1) Production baseline: adapter + AttnRes (no MoE)
    # 2) Challenger: dense-soft MoE without AttnRes-depth router concat
    # 3) Challenger: dense-soft MoE with AttnRes-depth router concat
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    # Keep anti-collapse defaults aligned with A3c/A3d family.
    ANTI_COLLAPSE_NO_SUBJECT_ID_PRESET="1"
    MOE_LINEAR_ROUTER_INPUT_NORM_PRESET="1"
    MOE_ROUTER_TEMPERATURE_PRESET="1.5"
    MOE_ROUTER_ENTROPY_COEF_PRESET="0.01"
    MOE_ROUTER_BALANCE_KL_COEF_PRESET="0.08"
    MOE_ROUTER_Z_LOSS_COEF_PRESET="0.002"
    MOE_ROUTER_JITTER_STD_PRESET="0.15"
    MOE_ROUTER_JITTER_FINAL_STD_PRESET="0.0"
    MOE_ROUTER_JITTER_ANNEAL_EPOCHS_PRESET="5"
    MOE_ROUTER_SOFT_WARMUP_EPOCHS_PRESET="5"
    MOE_ROUTER_WARMUP_MODE_PRESET="low_lr"
    MOE_ROUTER_WARMUP_EPOCHS_PRESET="3"
    MOE_ROUTER_WARMUP_LR_SCALE_PRESET="0.1"
    MOE_CAPACITY_FACTOR_PRESET="1.25"
    MOE_SYSTEM_MODE_PRESET="dense_soft"
    MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT_PRESET="0"
    MOE_ATTNRES_DEPTH_ROUTER_DIM_PRESET="4"
    ;;
  A4)
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="1"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ;;
  A5)
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="1"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="1"
    ;;
  B1a)
    # New-feature bundle (adapter+AttnRes baseline vs EEG-gated/prototype ablations).
    # Keep global presets neutral; per-run flags in the bundle block set exact mechanisms.
    ATTNRES_VARIANT_PRESET="pre_attn"
    EEG_CONTEXT_PRESET="0"
    ADAPTER_MODE_PRESET="subject_domain"
    USE_SUBJECT_SUMMARY_PRESET="0"
    ANTI_COLLAPSE_NO_SUBJECT_ID_PRESET="1"
    ;;
  custom)
    ;;
  *)
    echo "[slurm] Unknown STAGE='$STAGE'. Use A0|A1|A2|A2b|A3|A3b|A3c|A3d|A3e|A4|A5|B1a|custom." >&2
    exit 2
    ;;
esac

ATTNRES_VARIANT="${ATTNRES_VARIANT:-$ATTNRES_VARIANT_PRESET}"  # none|pre_attn|full|...
ATTNRES_START_LAYER="${ATTNRES_START_LAYER:-0}"
ATTNRES_GATED="${ATTNRES_GATED:-1}"                           # 0|1
ATTNRES_GATE_INIT="${ATTNRES_GATE_INIT:-0.0}"
ATTNRES_SUBJECT_GATES="${ATTNRES_SUBJECT_GATES:-1}"           # 0|1 (requires ATTNRES_GATED=1)
ATTNRES_EEG_COND_GATES="${ATTNRES_EEG_COND_GATES:-0}"         # 0|1
ATTNRES_EEG_GATE_SCALE="${ATTNRES_EEG_GATE_SCALE:-0.1}"
ATTNRES_DIAG_INTERVAL="${ATTNRES_DIAG_INTERVAL:-1}"
ADAPTER_MODE="${ADAPTER_MODE:-$ADAPTER_MODE_PRESET}"   # subject_domain | none
ADAPTER_NUM_LAYERS="${ADAPTER_NUM_LAYERS:-4}"
ADAPTER_RANK="${ADAPTER_RANK:-8}"
ADAPTER_COND_DIM="${ADAPTER_COND_DIM:-32}"
ADAPTER_SCALE="${ADAPTER_SCALE:-0.001}"
ADAPTER_USE_AGE_BUCKET="${ADAPTER_USE_AGE_BUCKET:-1}"          # 0|1
ADAPTER_USE_SEGMENT_BUCKET="${ADAPTER_USE_SEGMENT_BUCKET:-0}"  # 0|1 (keep 0 to avoid leakage-style shortcut)
ADAPTER_USE_SUBJECT_ID="${ADAPTER_USE_SUBJECT_ID:-1}"          # 0|1
SUBJECT_OVERLAP_POLICY="${SUBJECT_OVERLAP_POLICY:-disable}"    # error|disable|allow
CHANNEL_ID_ALIGN_MODE="${CHANNEL_ID_ALIGN_MODE:-auto}"         # auto|strict|off
ANTI_COLLAPSE_NO_SUBJECT_ID="${ANTI_COLLAPSE_NO_SUBJECT_ID:-$ANTI_COLLAPSE_NO_SUBJECT_ID_PRESET}" # 0|1, convenience switch for subject-disjoint sweeps

if [[ "$ANTI_COLLAPSE_NO_SUBJECT_ID" == "1" ]]; then
  ADAPTER_USE_SUBJECT_ID="0"
fi

# 1 => enable channel/electrode context injection
EEG_CONTEXT="${EEG_CONTEXT:-$EEG_CONTEXT_PRESET}"

# Apply stage preset for subject summary unless explicitly set by user
USE_SUBJECT_SUMMARY="${USE_SUBJECT_SUMMARY:-$USE_SUBJECT_SUMMARY_PRESET}"

# B1a controls EEG context per-subrun. Force global EEG context off to avoid accidental contamination.
if [[ "$STAGE" == "B1a" && "$EEG_CONTEXT" != "0" ]]; then
  echo "[slurm] B1a overrides global EEG_CONTEXT to 0; EEG context is enabled only in specific B1a subruns." >&2
  EEG_CONTEXT="0"
fi

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

# ------------------------------
# PROTOTYPE ALIGNMENT KNOBS
# ------------------------------
PROTOTYPE_ALIGNMENT="${PROTOTYPE_ALIGNMENT:-0}"
PROTOTYPE_MOMENTUM="${PROTOTYPE_MOMENTUM:-0.95}"
PROTOTYPE_PULL_WEIGHT="${PROTOTYPE_PULL_WEIGHT:-0.05}"
PROTOTYPE_PUSH_WEIGHT="${PROTOTYPE_PUSH_WEIGHT:-0.02}"
PROTOTYPE_MARGIN="${PROTOTYPE_MARGIN:-0.10}"
PROTOTYPE_START_EPOCH="${PROTOTYPE_START_EPOCH:-1}"

# 1 => freeze backbone except adapter/context/classifier (for incremental subject update)
ADAPTER_ONLY_UPDATE="${ADAPTER_ONLY_UPDATE:-0}"

# ------------------------------
# LEGACY MoE SWITCH (optional)
# ------------------------------
# Keep 0 for the new version. Set to 1 only if you want old routing baseline comparisons.
USE_LEGACY_MOE="${USE_LEGACY_MOE:-0}"
HYBRID_MOE_ADAPTER="${HYBRID_MOE_ADAPTER:-1}"  # 1 keeps adapter_mode in MoE runs; 0 forces pure MoE
MOE_NUM_LAYERS="${MOE_NUM_LAYERS:-1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-4}"
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-$MOE_CAPACITY_FACTOR_PRESET}"
MOE_DOMAIN_BIAS="${MOE_DOMAIN_BIAS:-1}"
MOE_DOMAIN_EMB_DIM="${MOE_DOMAIN_EMB_DIM:-16}"
MOE_LOAD_BALANCE="${MOE_LOAD_BALANCE:-0.005}"
MOE_DOMAIN_BIAS_REG="${MOE_DOMAIN_BIAS_REG:-0.0005}"
MOE_USE_PSD="${MOE_USE_PSD:-0}"
MOE_USE_ADAPTER_COND_BIAS="${MOE_USE_ADAPTER_COND_BIAS:-0}"
MOE_USE_SUBJECT_SUMMARY_ROUTER_CONCAT="${MOE_USE_SUBJECT_SUMMARY_ROUTER_CONCAT:-0}"
MOE_USE_EEG_SUMMARY_ROUTER_CONCAT_SPATIAL="${MOE_USE_EEG_SUMMARY_ROUTER_CONCAT_SPATIAL:-0}"
MOE_USE_EEG_SUMMARY_ROUTER_CONCAT_SPECTRAL="${MOE_USE_EEG_SUMMARY_ROUTER_CONCAT_SPECTRAL:-0}"
MOE_LINEAR_ROUTER_INPUT_NORM="${MOE_LINEAR_ROUTER_INPUT_NORM:-$MOE_LINEAR_ROUTER_INPUT_NORM_PRESET}"
MOE_ROUTER_TEMPERATURE="${MOE_ROUTER_TEMPERATURE:-$MOE_ROUTER_TEMPERATURE_PRESET}"
MOE_ROUTER_ENTROPY_COEF="${MOE_ROUTER_ENTROPY_COEF:-$MOE_ROUTER_ENTROPY_COEF_PRESET}"
MOE_ROUTER_BALANCE_KL_COEF="${MOE_ROUTER_BALANCE_KL_COEF:-$MOE_ROUTER_BALANCE_KL_COEF_PRESET}"
MOE_ROUTER_Z_LOSS_COEF="${MOE_ROUTER_Z_LOSS_COEF:-$MOE_ROUTER_Z_LOSS_COEF_PRESET}"
MOE_ROUTER_JITTER_STD="${MOE_ROUTER_JITTER_STD:-$MOE_ROUTER_JITTER_STD_PRESET}"
MOE_ROUTER_JITTER_FINAL_STD="${MOE_ROUTER_JITTER_FINAL_STD:-$MOE_ROUTER_JITTER_FINAL_STD_PRESET}"
MOE_ROUTER_JITTER_ANNEAL_EPOCHS="${MOE_ROUTER_JITTER_ANNEAL_EPOCHS:-$MOE_ROUTER_JITTER_ANNEAL_EPOCHS_PRESET}"
MOE_ROUTER_SOFT_WARMUP_EPOCHS="${MOE_ROUTER_SOFT_WARMUP_EPOCHS:-$MOE_ROUTER_SOFT_WARMUP_EPOCHS_PRESET}"
MOE_ROUTER_WARMUP_MODE="${MOE_ROUTER_WARMUP_MODE:-$MOE_ROUTER_WARMUP_MODE_PRESET}" # off|freeze|low_lr
MOE_ROUTER_WARMUP_EPOCHS="${MOE_ROUTER_WARMUP_EPOCHS:-$MOE_ROUTER_WARMUP_EPOCHS_PRESET}"
MOE_ROUTER_WARMUP_LR_SCALE="${MOE_ROUTER_WARMUP_LR_SCALE:-$MOE_ROUTER_WARMUP_LR_SCALE_PRESET}"
MOE_ROUTER_DISPATCH_MODE="${MOE_ROUTER_DISPATCH_MODE:-$MOE_ROUTER_DISPATCH_MODE_PRESET}" # hard_capacity|soft
MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT="${MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT:-$MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT_PRESET}"
MOE_ATTNRES_DEPTH_ROUTER_DIM="${MOE_ATTNRES_DEPTH_ROUTER_DIM:-$MOE_ATTNRES_DEPTH_ROUTER_DIM_PRESET}"

# conservative (existing) | dense_soft (all-layer + soft dispatch + AttnRes-depth router summary)
MOE_SYSTEM_MODE="${MOE_SYSTEM_MODE:-$MOE_SYSTEM_MODE_PRESET}"
if [[ "$MOE_SYSTEM_MODE" == "dense_soft" ]]; then
  # Keep dense-soft defaults without clobbering explicit/user stage toggles.
  if [[ "$MOE_NUM_LAYERS" == "1" ]]; then
    MOE_NUM_LAYERS="12"
  fi
  if [[ "$MOE_ROUTER_DISPATCH_MODE" == "hard_capacity" ]]; then
    MOE_ROUTER_DISPATCH_MODE="soft"
  fi
fi

# tqdm in Slurm logs can generate very large files; keep this 0 on cluster
FORCE_TQDM="${FORCE_TQDM:-0}"
METADATA_DEBUG="${METADATA_DEBUG:-1}"  # 0|1

# --- modules ---
module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

# Optional flags assembly
EXTRA_FLAGS=()

echo "[slurm] stage=$STAGE run_tag=$RUN_TAG" >&2
echo "[slurm] core attnres=$ATTNRES_VARIANT gated=$ATTNRES_GATED subj_gates=$ATTNRES_SUBJECT_GATES eeg_gates=$ATTNRES_EEG_COND_GATES eeg_ctx=$EEG_CONTEXT adapter=$ADAPTER_MODE" >&2
echo "[slurm] proto on=$PROTOTYPE_ALIGNMENT mom=$PROTOTYPE_MOMENTUM pull=$PROTOTYPE_PULL_WEIGHT push=$PROTOTYPE_PUSH_WEIGHT margin=$PROTOTYPE_MARGIN" >&2
echo "[slurm] moe legacy=$USE_LEGACY_MOE mode=$MOE_SYSTEM_MODE layers=$MOE_NUM_LAYERS dispatch=$MOE_ROUTER_DISPATCH_MODE attnres_depth_concat=$MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT" >&2
if [[ "$STAGE" == "A3b" || "$STAGE" == "A3c" || "$STAGE" == "A3d" || "$STAGE" == "A3e" || "$STAGE" == "B1a" ]]; then
  echo "[slurm] note: A3b/A3c/A3d/A3e/B1a are bundle stages; per-run flags below intentionally override some global defaults." >&2
fi

if [[ ! -f "$FACED_META_CSV" ]]; then
  echo "[slurm] FACED_META_CSV not found: $FACED_META_CSV" >&2
  echo "[slurm] Set FACED_META_CSV or FACED_META_DIR to a valid FACED metadata path." >&2
  exit 2
fi

if [[ "$STAGE" != "A2b" && "$EEG_CONTEXT" == "1" ]]; then
  if [[ -z "$CHANNEL_CONTEXT_FILE" ]] || [[ "$CHANNEL_CONTEXT_FILE" == "/path/to/channel_context.json" ]] || [[ "$CHANNEL_CONTEXT_FILE" == "/path/to/Electrode_Location.xlsx" ]]; then
    echo "[slurm] EEG_CONTEXT=1 requires a real CHANNEL_CONTEXT_FILE path (not empty/placeholder)." >&2
    exit 2
  elif [[ ! -f "$CHANNEL_CONTEXT_FILE" ]]; then
    echo "[slurm] EEG_CONTEXT=1 but CHANNEL_CONTEXT_FILE not found: $CHANNEL_CONTEXT_FILE" >&2
    exit 2
  else
    EXTRA_FLAGS+=(--eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE")
  fi
fi
if [[ -n "$SUBJECT_SUMMARY_FILE" ]]; then
  EXTRA_FLAGS+=(--subject_summary_file "$SUBJECT_SUMMARY_FILE")
fi
if [[ "$USE_SUBJECT_SUMMARY" == "1" ]]; then
  if [[ -z "$SUBJECT_SUMMARY_FILE" ]]; then
    echo "[slurm] USE_SUBJECT_SUMMARY=1 requires SUBJECT_SUMMARY_FILE to be set." >&2
    exit 2
  fi
  EXTRA_FLAGS+=(--use_subject_summary)
fi
if [[ "$ADAPTER_ONLY_UPDATE" == "1" ]]; then
  EXTRA_FLAGS+=(--adapter_only_update)
fi
if [[ "$ADAPTER_USE_AGE_BUCKET" == "1" ]]; then
  EXTRA_FLAGS+=(--adapter_use_age_bucket)
else
  EXTRA_FLAGS+=(--no-adapter_use_age_bucket)
fi
if [[ "$ADAPTER_USE_SEGMENT_BUCKET" == "1" ]]; then
  EXTRA_FLAGS+=(--adapter_use_segment_bucket)
else
  EXTRA_FLAGS+=(--no-adapter_use_segment_bucket)
fi
if [[ "$ADAPTER_USE_SUBJECT_ID" == "1" ]]; then
  EXTRA_FLAGS+=(--adapter_use_subject_id)
else
  EXTRA_FLAGS+=(--no-adapter_use_subject_id)
fi
EXTRA_FLAGS+=(--subject_overlap_policy "$SUBJECT_OVERLAP_POLICY")
EXTRA_FLAGS+=(--channel_id_align_mode "$CHANNEL_ID_ALIGN_MODE")
EXTRA_FLAGS+=(--subject_summary_handling "$SUBJECT_SUMMARY_HANDLING")
EXTRA_FLAGS+=(--moe_router_temperature "$MOE_ROUTER_TEMPERATURE")
EXTRA_FLAGS+=(--moe_router_entropy_coef "$MOE_ROUTER_ENTROPY_COEF")
EXTRA_FLAGS+=(--moe_router_balance_kl_coef "$MOE_ROUTER_BALANCE_KL_COEF")
EXTRA_FLAGS+=(--moe_router_z_loss_coef "$MOE_ROUTER_Z_LOSS_COEF")
EXTRA_FLAGS+=(--moe_router_jitter_std "$MOE_ROUTER_JITTER_STD")
EXTRA_FLAGS+=(--moe_router_jitter_final_std "$MOE_ROUTER_JITTER_FINAL_STD")
EXTRA_FLAGS+=(--moe_router_jitter_anneal_epochs "$MOE_ROUTER_JITTER_ANNEAL_EPOCHS")
EXTRA_FLAGS+=(--moe_router_soft_warmup_epochs "$MOE_ROUTER_SOFT_WARMUP_EPOCHS")
EXTRA_FLAGS+=(--moe_router_warmup_mode "$MOE_ROUTER_WARMUP_MODE")
EXTRA_FLAGS+=(--moe_router_warmup_epochs "$MOE_ROUTER_WARMUP_EPOCHS")
EXTRA_FLAGS+=(--moe_router_warmup_lr_scale "$MOE_ROUTER_WARMUP_LR_SCALE")
EXTRA_FLAGS+=(--moe_router_dispatch_mode "$MOE_ROUTER_DISPATCH_MODE")
EXTRA_FLAGS+=(--moe_attnres_depth_router_dim "$MOE_ATTNRES_DEPTH_ROUTER_DIM")
if [[ "$MOE_LINEAR_ROUTER_INPUT_NORM" == "1" ]]; then
  EXTRA_FLAGS+=(--moe_linear_router_input_norm)
else
  EXTRA_FLAGS+=(--no-moe_linear_router_input_norm)
fi
if [[ "$MOE_USE_ATTNRES_DEPTH_ROUTER_CONCAT" == "1" ]]; then
  EXTRA_FLAGS+=(--moe_use_attnres_depth_router_concat)
else
  EXTRA_FLAGS+=(--no-moe_use_attnres_depth_router_concat)
fi
if [[ "$METADATA_DEBUG" == "1" ]]; then
  EXTRA_FLAGS+=(--metadata_debug)
else
  EXTRA_FLAGS+=(--no-metadata_debug)
fi
if [[ "$FORCE_TQDM" == "1" ]]; then
  EXTRA_FLAGS+=(--tqdm)
else
  EXTRA_FLAGS+=(--no-tqdm)
fi
if [[ "$ATTNRES_GATED" == "1" ]]; then
  EXTRA_FLAGS+=(--attnres_gated --attnres_gate_init "$ATTNRES_GATE_INIT")
fi
if [[ "$ATTNRES_SUBJECT_GATES" == "1" ]]; then
  EXTRA_FLAGS+=(--attnres_subject_gates)
fi
if [[ "$ATTNRES_EEG_COND_GATES" == "1" ]]; then
  EXTRA_FLAGS+=(--attnres_eeg_cond_gates)
else
  EXTRA_FLAGS+=(--no-attnres_eeg_cond_gates)
fi
EXTRA_FLAGS+=(--attnres_eeg_gate_scale "$ATTNRES_EEG_GATE_SCALE")
EXTRA_FLAGS+=(--attnres_diag_interval "$ATTNRES_DIAG_INTERVAL")

if [[ "$PROTOTYPE_ALIGNMENT" == "1" ]]; then
  EXTRA_FLAGS+=(--prototype_alignment)
else
  EXTRA_FLAGS+=(--no-prototype_alignment)
fi
EXTRA_FLAGS+=(--prototype_momentum "$PROTOTYPE_MOMENTUM")
EXTRA_FLAGS+=(--prototype_pull_weight "$PROTOTYPE_PULL_WEIGHT")
EXTRA_FLAGS+=(--prototype_push_weight "$PROTOTYPE_PUSH_WEIGHT")
EXTRA_FLAGS+=(--prototype_margin "$PROTOTYPE_MARGIN")
EXTRA_FLAGS+=(--prototype_start_epoch "$PROTOTYPE_START_EPOCH")

# Legacy MoE compatibility path
if [[ "$USE_LEGACY_MOE" == "1" && "$STAGE" != "A3b" && "$STAGE" != "A3c" && "$STAGE" != "A3d" && "$STAGE" != "A3e" && "$STAGE" != "B1a" ]]; then
  if [[ "$HYBRID_MOE_ADAPTER" != "1" ]]; then
    ADAPTER_MODE="none"
    USE_SUBJECT_SUMMARY="0"
  fi
  EXTRA_FLAGS+=(--moe --moe_route_mode typed_capacity_domain --moe_diagnostics)
  EXTRA_FLAGS+=(--moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS")
  EXTRA_FLAGS+=(--moe_capacity_factor "$MOE_CAPACITY_FACTOR")
  EXTRA_FLAGS+=(--moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM")
  EXTRA_FLAGS+=(--moe_load_balance "$MOE_LOAD_BALANCE")
  EXTRA_FLAGS+=(--moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG")
  EXTRA_FLAGS+=(--faced_meta_csv "$FACED_META_CSV")
  if [[ "$MOE_USE_ADAPTER_COND_BIAS" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_use_adapter_cond_bias)
  else
    EXTRA_FLAGS+=(--no-moe_use_adapter_cond_bias)
  fi
  if [[ "$MOE_USE_SUBJECT_SUMMARY_ROUTER_CONCAT" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_use_subject_summary_router_concat)
  else
    EXTRA_FLAGS+=(--no-moe_use_subject_summary_router_concat)
  fi
  if [[ "$MOE_USE_EEG_SUMMARY_ROUTER_CONCAT_SPATIAL" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_use_eeg_summary_router_concat_spatial)
  else
    EXTRA_FLAGS+=(--no-moe_use_eeg_summary_router_concat_spatial)
  fi
  if [[ "$MOE_USE_EEG_SUMMARY_ROUTER_CONCAT_SPECTRAL" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_use_eeg_summary_router_concat_spectral)
  else
    EXTRA_FLAGS+=(--no-moe_use_eeg_summary_router_concat_spectral)
  fi
  if [[ "$MOE_DOMAIN_BIAS" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_domain_bias)
  fi
  if [[ "$MOE_USE_PSD" == "1" ]]; then
    EXTRA_FLAGS+=(--moe_use_psd_router_features)
  fi
fi

# Always pass FACED metadata csv so subject/domain id maps are built consistently.
EXTRA_FLAGS+=(--faced_meta_csv "$FACED_META_CSV")

run_finetune() {
  local local_model_dir="$1"
  shift
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
    --model_dir "$local_model_dir" \
    --label_smoothing "$label_smoothing" \
    --num_workers 4 \
    --attnres_variant "$ATTNRES_VARIANT" \
    --attnres_start_layer "$ATTNRES_START_LAYER" \
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
    "$@"
}

summarize_log() {
  local label="$1"
  local log_path="$2"

  local best_vals
  best_vals=$(grep -E "^Epoch [0-9]+ :" "$log_path" | \
    awk -F'[,: ]+' 'BEGIN{bestAcc=-1;bestK=-1} {ep=$2; acc=$7; k=$9; if(acc>bestAcc){bestAcc=acc;bestAccEp=ep} if(k>bestK){bestK=k;bestKEp=ep} lastAcc=acc; lastK=k; lastEp=ep} END{printf("%.5f %.5f %d %.5f %.5f",bestAcc,bestK,lastEp,lastAcc,lastK)}')
  if [[ -z "$best_vals" ]]; then
    echo "[summary][parse_error] label=$label log=$log_path" >&2
    return 1
  fi

  local best_acc best_kappa last_ep last_acc last_kappa
  read -r best_acc best_kappa last_ep last_acc last_kappa <<< "$best_vals"

  local static_fields="unknown"
  static_fields=$(grep -E "\[channel-context\]" "$log_path" | sed -n '1p' | sed -n 's/.*used_static_fields=\([^ ]*\).*/\1/p')
  if [[ -z "$static_fields" ]]; then
    static_fields="-"
  fi

  local runtime_state="unknown"
  runtime_state=$(grep -E "\[channel-context-runtime\]" "$log_path" | sed -n '1p' | sed -n 's/.*channel_context_active=\([^ ]*\) reason=\(.*\)$/\1:\2/p')
  if [[ -z "$runtime_state" ]]; then
    runtime_state="-"
  fi

  local context_file="unknown"
  context_file=$(grep -E "^Namespace\(" "$log_path" | sed -n '1p' | sed -n "s/.*channel_context_file='\([^']*\)'.*/\1/p")
  if [[ -z "$context_file" ]]; then
    context_file="-"
  fi

  echo "$label $best_acc $best_kappa $last_ep $last_acc $last_kappa $static_fields $runtime_state $context_file"
}

if [[ "$STAGE" == "A2b" ]]; then
  if [[ -z "$CHANNEL_CONTEXT_FILE" ]] || [[ "$CHANNEL_CONTEXT_FILE" == "/path/to/channel_context.json" ]] || [[ "$CHANNEL_CONTEXT_FILE" == "/path/to/Electrode_Location.xlsx" ]]; then
    echo "[slurm] A2b requires CHANNEL_CONTEXT_FILE to be set to a real file path." >&2
    exit 2
  fi
  if [[ ! -f "$CHANNEL_CONTEXT_FILE" ]]; then
    echo "[slurm] A2b CHANNEL_CONTEXT_FILE not found: $CHANNEL_CONTEXT_FILE" >&2
    exit 2
  fi

  A2B_RAW_CONTEXT_FILE="${A2B_RAW_CONTEXT_FILE:-$CHANNEL_CONTEXT_FILE}"
  A2B_RICH_CONTEXT_FILE_DEFAULT="$A2B_RAW_CONTEXT_FILE"
  if [[ "$A2B_RAW_CONTEXT_FILE" == *.xlsx ]]; then
    raw_dir="$(dirname "$A2B_RAW_CONTEXT_FILE")"
    if [[ -f "$raw_dir/channel_context_rich.json" ]]; then
      A2B_RICH_CONTEXT_FILE_DEFAULT="$raw_dir/channel_context_rich.json"
    fi
  fi
  A2B_RICH_CONTEXT_FILE="${A2B_RICH_CONTEXT_FILE:-$A2B_RICH_CONTEXT_FILE_DEFAULT}"
  if [[ ! -f "$A2B_RAW_CONTEXT_FILE" ]]; then
    echo "[slurm] A2b raw context file not found: $A2B_RAW_CONTEXT_FILE" >&2
    exit 2
  fi
  if [[ ! -f "$A2B_RICH_CONTEXT_FILE" ]]; then
    echo "[slurm] A2b rich context file not found: $A2B_RICH_CONTEXT_FILE" >&2
    exit 2
  fi

  A2B_LOG_DIR="${A2B_LOG_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/a2b_${SLURM_JOB_ID}}"
  mkdir -p "$A2B_LOG_DIR"

  declare -a A2B_ROWS=()
  declare -A A2B_BEST_ACC=()
  declare -A A2B_BEST_KAPPA=()

  run_and_collect() {
    local label="$1"
    local eeg_flag="$2"
    local align_mode="$3"
    local context_file="$4"
    local run_model_dir="$model_dir/$label"
    local run_log="$A2B_LOG_DIR/${label}.log"

    mkdir -p "$run_model_dir"

    local -a run_flags=("${EXTRA_FLAGS[@]}")
    if [[ "$eeg_flag" == "1" ]]; then
      run_flags+=(--eeg_channel_context --channel_context_file "$context_file" --channel_id_align_mode "$align_mode")
    else
      run_flags+=(--channel_id_align_mode "$align_mode")
    fi

    echo "[A2b] start label=$label eeg_context=$eeg_flag channel_id_align_mode=$align_mode context_file=${context_file:-none} model_dir=$run_model_dir" >&2
    run_finetune "$run_model_dir" "${run_flags[@]}" 2>&1 | tee "$run_log"

    local row
    row=$(summarize_log "$label" "$run_log") || exit 3
    A2B_ROWS+=("$row")

    local _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile
    read -r _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile <<< "$row"
    A2B_BEST_ACC["$_label"]="$_best_acc"
    A2B_BEST_KAPPA["$_label"]="$_best_kappa"
  }

  # Three diagnostics back-to-back:
  # 1) A1_ref: no channel context (overhead/control baseline)
  # 2) A2_raw_auto: channel context with raw file + auto alignment
  # 3) A2_rich_strict: channel context with richer file + strict alignment
  run_and_collect "A1_ref" "0" "auto" ""
  run_and_collect "A2_raw_auto" "1" "auto" "$A2B_RAW_CONTEXT_FILE"
  run_and_collect "A2_rich_strict" "1" "strict" "$A2B_RICH_CONTEXT_FILE"

  calc_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN {printf("%.5f", a - b)}'
  }

  echo "[A2b][summary] log_dir=$A2B_LOG_DIR"
  echo "[A2b][summary] label best_acc best_kappa last_ep last_acc last_kappa used_static_fields runtime_state context_file"
  for row in "${A2B_ROWS[@]}"; do
    echo "[A2b][summary] $row"
  done

  ref_acc="${A2B_BEST_ACC[A1_ref]}"
  ref_kappa="${A2B_BEST_KAPPA[A1_ref]}"
  auto_acc="${A2B_BEST_ACC[A2_raw_auto]}"
  auto_kappa="${A2B_BEST_KAPPA[A2_raw_auto]}"
  strict_acc="${A2B_BEST_ACC[A2_rich_strict]}"
  strict_kappa="${A2B_BEST_KAPPA[A2_rich_strict]}"

  echo "[A2b][delta] vs=A1_ref target=A2_raw_auto d_best_acc=$(calc_delta "$auto_acc" "$ref_acc") d_best_kappa=$(calc_delta "$auto_kappa" "$ref_kappa")"
  echo "[A2b][delta] vs=A1_ref target=A2_rich_strict d_best_acc=$(calc_delta "$strict_acc" "$ref_acc") d_best_kappa=$(calc_delta "$strict_kappa" "$ref_kappa")"
  echo "[A2b][delta] vs=A2_raw_auto target=A2_rich_strict d_best_acc=$(calc_delta "$strict_acc" "$auto_acc") d_best_kappa=$(calc_delta "$strict_kappa" "$auto_kappa")"
elif [[ "$STAGE" == "A3b" ]]; then
  A3B_LOG_DIR="${A3B_LOG_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/a3b_${SLURM_JOB_ID}}"
  mkdir -p "$A3B_LOG_DIR"

  if [[ "$MOE_DOMAIN_BIAS" == "1" ]]; then
    A3B_MOE_BIAS_FLAGS=(--moe_domain_bias)
  else
    A3B_MOE_BIAS_FLAGS=()
  fi
  if [[ "$MOE_USE_PSD" == "1" ]]; then
    A3B_MOE_PSD_FLAGS=(--moe_use_psd_router_features)
  else
    A3B_MOE_PSD_FLAGS=()
  fi

  declare -a A3B_ROWS=()
  declare -A A3B_BEST_ACC=()
  declare -A A3B_BEST_KAPPA=()

  run_and_collect_a3b() {
    local label="$1"
    shift
    local run_model_dir="$model_dir/$label"
    local run_log="$A3B_LOG_DIR/${label}.log"
    mkdir -p "$run_model_dir"

    local -a run_flags=("${EXTRA_FLAGS[@]}")
    run_flags+=("$@")

    echo "[A3b] start label=$label model_dir=$run_model_dir" >&2
    run_finetune "$run_model_dir" "${run_flags[@]}" 2>&1 | tee "$run_log"

    local row
    row=$(summarize_log "$label" "$run_log") || exit 3
    A3B_ROWS+=("$row")

    local _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile
    read -r _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile <<< "$row"
    A3B_BEST_ACC["$_label"]="$_best_acc"
    A3B_BEST_KAPPA["$_label"]="$_best_kappa"
  }

  # Four A3 variants in one job:
  # 1) hybrid + adapter-conditioned router bias
  # 2) hybrid without adapter-conditioned router bias
  # 3) pure MoE (no adapters)
  # 4) adapter/AttnRes only (no MoE)
  run_and_collect_a3b "A3_hybrid_cond" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3B_MOE_BIAS_FLAGS[@]}" "${A3B_MOE_PSD_FLAGS[@]}" \
    --moe_use_adapter_cond_bias

  run_and_collect_a3b "A3_hybrid_nocond" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3B_MOE_BIAS_FLAGS[@]}" "${A3B_MOE_PSD_FLAGS[@]}" \
    --no-moe_use_adapter_cond_bias

  run_and_collect_a3b "A3_pure_moe" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3B_MOE_BIAS_FLAGS[@]}" "${A3B_MOE_PSD_FLAGS[@]}" \
    --adapter_mode none \
    --no-moe_use_adapter_cond_bias

  run_and_collect_a3b "A3_adapter_only" \
    --adapter_mode subject_domain

  calc_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN {printf("%.5f", a - b)}'
  }

  echo "[A3b][summary] log_dir=$A3B_LOG_DIR"
  echo "[A3b][summary] label best_acc best_kappa last_ep last_acc last_kappa used_static_fields runtime_state context_file"
  for row in "${A3B_ROWS[@]}"; do
    echo "[A3b][summary] $row"
  done

  base_acc="${A3B_BEST_ACC[A3_adapter_only]}"
  base_kappa="${A3B_BEST_KAPPA[A3_adapter_only]}"
  hc_acc="${A3B_BEST_ACC[A3_hybrid_cond]}"
  hc_kappa="${A3B_BEST_KAPPA[A3_hybrid_cond]}"
  hn_acc="${A3B_BEST_ACC[A3_hybrid_nocond]}"
  hn_kappa="${A3B_BEST_KAPPA[A3_hybrid_nocond]}"
  pm_acc="${A3B_BEST_ACC[A3_pure_moe]}"
  pm_kappa="${A3B_BEST_KAPPA[A3_pure_moe]}"

  echo "[A3b][delta] vs=A3_adapter_only target=A3_hybrid_cond d_best_acc=$(calc_delta "$hc_acc" "$base_acc") d_best_kappa=$(calc_delta "$hc_kappa" "$base_kappa")"
  echo "[A3b][delta] vs=A3_adapter_only target=A3_hybrid_nocond d_best_acc=$(calc_delta "$hn_acc" "$base_acc") d_best_kappa=$(calc_delta "$hn_kappa" "$base_kappa")"
  echo "[A3b][delta] vs=A3_adapter_only target=A3_pure_moe d_best_acc=$(calc_delta "$pm_acc" "$base_acc") d_best_kappa=$(calc_delta "$pm_kappa" "$base_kappa")"
  echo "[A3b][delta] vs=A3_hybrid_nocond target=A3_hybrid_cond d_best_acc=$(calc_delta "$hc_acc" "$hn_acc") d_best_kappa=$(calc_delta "$hc_kappa" "$hn_kappa")"
elif [[ "$STAGE" == "A3e" ]]; then
  A3E_LOG_DIR="${A3E_LOG_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/a3e_${SLURM_JOB_ID}}"
  mkdir -p "$A3E_LOG_DIR"

  if [[ "$MOE_DOMAIN_BIAS" == "1" ]]; then
    A3E_MOE_BIAS_FLAGS=(--moe_domain_bias)
  else
    A3E_MOE_BIAS_FLAGS=()
  fi
  if [[ "$MOE_USE_PSD" == "1" ]]; then
    A3E_MOE_PSD_FLAGS=(--moe_use_psd_router_features)
  else
    A3E_MOE_PSD_FLAGS=()
  fi

  declare -a A3E_ROWS=()
  declare -A A3E_BEST_ACC=()
  declare -A A3E_BEST_KAPPA=()

  run_and_collect_a3e() {
    local label="$1"
    shift
    local run_model_dir="$model_dir/$label"
    local run_log="$A3E_LOG_DIR/${label}.log"
    mkdir -p "$run_model_dir"

    local -a run_flags=("${EXTRA_FLAGS[@]}")
    run_flags+=("$@")

    echo "[A3e] start label=$label model_dir=$run_model_dir" >&2
    run_finetune "$run_model_dir" "${run_flags[@]}" 2>&1 | tee "$run_log"

    local row
    row=$(summarize_log "$label" "$run_log") || exit 3
    A3E_ROWS+=("$row")

    local _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile
    read -r _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile <<< "$row"
    A3E_BEST_ACC["$_label"]="$_best_acc"
    A3E_BEST_KAPPA["$_label"]="$_best_kappa"
  }

  # 1) Production baseline: adapter + AttnRes (no MoE)
  run_and_collect_a3e "A3e_adapter_attnres_baseline_no_sid" \
    --adapter_mode subject_domain \
    --no-moe_use_attnres_depth_router_concat \
    --no-moe_linear_router_input_norm \
    --moe_router_dispatch_mode hard_capacity \
    --moe_router_entropy_coef 0.0 \
    --moe_router_balance_kl_coef 0.0 \
    --moe_router_z_loss_coef 0.0 \
    --moe_router_jitter_std 0.0 \
    --moe_router_jitter_final_std 0.0 \
    --moe_router_jitter_anneal_epochs 0 \
    --moe_router_soft_warmup_epochs 0 \
    --moe_router_warmup_mode off \
    --moe_router_warmup_epochs 0 \
    --moe_router_warmup_lr_scale 0.1 \
    --no-adapter_use_subject_id

  # 2) Challenger: dense-soft MoE without AttnRes-depth concat
  run_and_collect_a3e "A3e_dense_soft_no_depth_no_sid" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3E_MOE_BIAS_FLAGS[@]}" "${A3E_MOE_PSD_FLAGS[@]}" \
    --moe_use_adapter_cond_bias \
    --moe_router_dispatch_mode soft \
    --no-moe_use_attnres_depth_router_concat \
    --no-moe_use_subject_summary_router_concat \
    --no-moe_use_eeg_summary_router_concat_spatial \
    --no-moe_use_eeg_summary_router_concat_spectral \
    --no-adapter_use_subject_id

  # 3) Challenger: dense-soft MoE with AttnRes-depth concat
  run_and_collect_a3e "A3e_dense_soft_with_depth_no_sid" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3E_MOE_BIAS_FLAGS[@]}" "${A3E_MOE_PSD_FLAGS[@]}" \
    --moe_use_adapter_cond_bias \
    --moe_router_dispatch_mode soft \
    --moe_use_attnres_depth_router_concat \
    --moe_attnres_depth_router_dim "$MOE_ATTNRES_DEPTH_ROUTER_DIM" \
    --no-moe_use_subject_summary_router_concat \
    --no-moe_use_eeg_summary_router_concat_spatial \
    --no-moe_use_eeg_summary_router_concat_spectral \
    --no-adapter_use_subject_id

  calc_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN {printf("%.5f", a - b)}'
  }

  echo "[A3e][summary] log_dir=$A3E_LOG_DIR"
  echo "[A3e][summary] label best_acc best_kappa last_ep last_acc last_kappa used_static_fields runtime_state context_file"
  for row in "${A3E_ROWS[@]}"; do
    echo "[A3e][summary] $row"
  done

  b_acc="${A3E_BEST_ACC[A3e_adapter_attnres_baseline_no_sid]}"
  b_kappa="${A3E_BEST_KAPPA[A3e_adapter_attnres_baseline_no_sid]}"
  c0_acc="${A3E_BEST_ACC[A3e_dense_soft_no_depth_no_sid]}"
  c0_kappa="${A3E_BEST_KAPPA[A3e_dense_soft_no_depth_no_sid]}"
  c1_acc="${A3E_BEST_ACC[A3e_dense_soft_with_depth_no_sid]}"
  c1_kappa="${A3E_BEST_KAPPA[A3e_dense_soft_with_depth_no_sid]}"

  echo "[A3e][delta] vs=baseline target=dense_soft_no_depth d_best_acc=$(calc_delta "$c0_acc" "$b_acc") d_best_kappa=$(calc_delta "$c0_kappa" "$b_kappa")"
  echo "[A3e][delta] vs=baseline target=dense_soft_with_depth d_best_acc=$(calc_delta "$c1_acc" "$b_acc") d_best_kappa=$(calc_delta "$c1_kappa" "$b_kappa")"
  echo "[A3e][delta] vs=dense_soft_no_depth target=dense_soft_with_depth d_best_acc=$(calc_delta "$c1_acc" "$c0_acc") d_best_kappa=$(calc_delta "$c1_kappa" "$c0_kappa")"
elif [[ "$STAGE" == "B1a" ]]; then
  B1A_LOG_DIR="${B1A_LOG_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/b1a_${SLURM_JOB_ID}}"
  mkdir -p "$B1A_LOG_DIR"

  if [[ -z "$CHANNEL_CONTEXT_FILE" ]] || [[ ! -f "$CHANNEL_CONTEXT_FILE" ]]; then
    echo "[slurm] B1a requires a valid CHANNEL_CONTEXT_FILE for EEG-conditioned AttnRes runs." >&2
    exit 2
  fi

  declare -a B1A_ROWS=()
  declare -A B1A_BEST_ACC=()
  declare -A B1A_BEST_KAPPA=()

  run_and_collect_b1a() {
    local label="$1"
    shift
    local run_model_dir="$model_dir/$label"
    local run_log="$B1A_LOG_DIR/${label}.log"
    mkdir -p "$run_model_dir"

    local -a run_flags=("${EXTRA_FLAGS[@]}")
    run_flags+=("$@")

    echo "[B1a] start label=$label model_dir=$run_model_dir" >&2
    run_finetune "$run_model_dir" "${run_flags[@]}" 2>&1 | tee "$run_log"

    local row
    row=$(summarize_log "$label" "$run_log") || exit 3
    B1A_ROWS+=("$row")

    local _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile
    read -r _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile <<< "$row"
    B1A_BEST_ACC["$_label"]="$_best_acc"
    B1A_BEST_KAPPA["$_label"]="$_best_kappa"
  }

  # 0) Baseline: adapter + AttnRes only
  run_and_collect_b1a "B1a_exp0_baseline" \
    --adapter_mode subject_domain \
    --no-adapter_use_subject_id \
    --no-attnres_eeg_cond_gates \
    --no-prototype_alignment

  # 1) Baseline + EEG-conditioned AttnRes
  run_and_collect_b1a "B1a_exp1_eeg_attnres" \
    --adapter_mode subject_domain \
    --no-adapter_use_subject_id \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --attnres_eeg_cond_gates \
    --attnres_eeg_gate_scale "$ATTNRES_EEG_GATE_SCALE" \
    --no-prototype_alignment

  # 2) Baseline + prototype alignment
  run_and_collect_b1a "B1a_exp2_prototype" \
    --adapter_mode subject_domain \
    --no-adapter_use_subject_id \
    --no-attnres_eeg_cond_gates \
    --prototype_alignment

  # 3) Baseline + EEG-conditioned AttnRes + prototype alignment
  run_and_collect_b1a "B1a_exp3_eeg_plus_prototype" \
    --adapter_mode subject_domain \
    --no-adapter_use_subject_id \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --attnres_eeg_cond_gates \
    --attnres_eeg_gate_scale "$ATTNRES_EEG_GATE_SCALE" \
    --prototype_alignment

  calc_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN {printf("%.5f", a - b)}'
  }

  echo "[B1a][summary] log_dir=$B1A_LOG_DIR"
  echo "[B1a][summary] label best_acc best_kappa last_ep last_acc last_kappa used_static_fields runtime_state context_file"
  for row in "${B1A_ROWS[@]}"; do
    echo "[B1a][summary] $row"
  done

  b_acc="${B1A_BEST_ACC[B1a_exp0_baseline]}"
  b_kappa="${B1A_BEST_KAPPA[B1a_exp0_baseline]}"
  e_acc="${B1A_BEST_ACC[B1a_exp1_eeg_attnres]}"
  e_kappa="${B1A_BEST_KAPPA[B1a_exp1_eeg_attnres]}"
  p_acc="${B1A_BEST_ACC[B1a_exp2_prototype]}"
  p_kappa="${B1A_BEST_KAPPA[B1a_exp2_prototype]}"
  ep_acc="${B1A_BEST_ACC[B1a_exp3_eeg_plus_prototype]}"
  ep_kappa="${B1A_BEST_KAPPA[B1a_exp3_eeg_plus_prototype]}"

  echo "[B1a][delta] vs=exp0_baseline target=exp1_eeg_attnres d_best_acc=$(calc_delta "$e_acc" "$b_acc") d_best_kappa=$(calc_delta "$e_kappa" "$b_kappa")"
  echo "[B1a][delta] vs=exp0_baseline target=exp2_prototype d_best_acc=$(calc_delta "$p_acc" "$b_acc") d_best_kappa=$(calc_delta "$p_kappa" "$b_kappa")"
  echo "[B1a][delta] vs=exp0_baseline target=exp3_eeg_plus_prototype d_best_acc=$(calc_delta "$ep_acc" "$b_acc") d_best_kappa=$(calc_delta "$ep_kappa" "$b_kappa")"
  echo "[B1a][delta] vs=exp1_eeg_attnres target=exp3_eeg_plus_prototype d_best_acc=$(calc_delta "$ep_acc" "$e_acc") d_best_kappa=$(calc_delta "$ep_kappa" "$e_kappa")"
  echo "[B1a][delta] vs=exp2_prototype target=exp3_eeg_plus_prototype d_best_acc=$(calc_delta "$ep_acc" "$p_acc") d_best_kappa=$(calc_delta "$ep_kappa" "$p_kappa")"
elif [[ "$STAGE" == "A3c" || "$STAGE" == "A3d" ]]; then
  BUNDLE_TAG="$STAGE"
  BUNDLE_TAG_LOWER="$(printf '%s' "$BUNDLE_TAG" | tr '[:upper:]' '[:lower:]')"
  BASE_LABEL="${BUNDLE_TAG}_adapter_only_no_sid"
  HYBRID_LABEL="${BUNDLE_TAG}_hybrid_bias_no_sid"
  SUBJ_LABEL="${BUNDLE_TAG}_hybrid_subjconcat_no_sid"
  EEG_SPATIAL_LABEL="${BUNDLE_TAG}_hybrid_subj_eeg_spatial_no_sid"
  EEG_BOTH_LABEL="${BUNDLE_TAG}_hybrid_subj_eeg_both_no_sid"

  A3C_RUN_SET="${A3C_RUN_SET:-performance}"  # performance|full
  if [[ "$A3C_RUN_SET" != "performance" && "$A3C_RUN_SET" != "full" ]]; then
    echo "[slurm] A3C_RUN_SET must be performance|full (got '$A3C_RUN_SET')." >&2
    exit 2
  fi
  if [[ "$A3C_RUN_SET" == "full" ]]; then
    if [[ -z "$SUBJECT_SUMMARY_FILE" ]] || [[ ! -f "$SUBJECT_SUMMARY_FILE" ]]; then
      echo "[slurm] ${BUNDLE_TAG} full run requires SUBJECT_SUMMARY_FILE for subject-summary router concat runs." >&2
      exit 2
    fi
    if [[ -z "$CHANNEL_CONTEXT_FILE" ]] || [[ ! -f "$CHANNEL_CONTEXT_FILE" ]]; then
      echo "[slurm] ${BUNDLE_TAG} full run requires CHANNEL_CONTEXT_FILE for EEG-summary router concat runs." >&2
      exit 2
    fi
  fi

  A3C_LOG_DIR="${A3C_LOG_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/${BUNDLE_TAG_LOWER}_${SLURM_JOB_ID}}"
  mkdir -p "$A3C_LOG_DIR"

  if [[ "$MOE_DOMAIN_BIAS" == "1" ]]; then
    A3C_MOE_BIAS_FLAGS=(--moe_domain_bias)
  else
    A3C_MOE_BIAS_FLAGS=()
  fi
  if [[ "$MOE_USE_PSD" == "1" ]]; then
    A3C_MOE_PSD_FLAGS=(--moe_use_psd_router_features)
  else
    A3C_MOE_PSD_FLAGS=()
  fi

  declare -a A3C_ROWS=()
  declare -A A3C_BEST_ACC=()
  declare -A A3C_BEST_KAPPA=()

  run_and_collect_a3c() {
    local label="$1"
    shift
    local run_model_dir="$model_dir/$label"
    local run_log="$A3C_LOG_DIR/${label}.log"
    mkdir -p "$run_model_dir"

    local -a run_flags=("${EXTRA_FLAGS[@]}")
    run_flags+=("$@")

    echo "[${BUNDLE_TAG}] start label=$label model_dir=$run_model_dir" >&2
    run_finetune "$run_model_dir" "${run_flags[@]}" 2>&1 | tee "$run_log"

    local row
    row=$(summarize_log "$label" "$run_log") || exit 3
    A3C_ROWS+=("$row")

    local _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile
    read -r _label _best_acc _best_kappa _last_ep _last_acc _last_kappa _static _runtime _ctxfile <<< "$row"
    A3C_BEST_ACC["$_label"]="$_best_acc"
    A3C_BEST_KAPPA["$_label"]="$_best_kappa"
  }

  # Subject-disjoint-safe default: disable subject_id conditioning in all A3c runs.
  # 1) Adapter baseline (no MoE)
  run_and_collect_a3c "$BASE_LABEL" \
    --adapter_mode subject_domain \
    --no-moe_use_attnres_depth_router_concat \
    --no-moe_linear_router_input_norm \
    --moe_router_dispatch_mode hard_capacity \
    --moe_router_entropy_coef 0.0 \
    --moe_router_balance_kl_coef 0.0 \
    --moe_router_z_loss_coef 0.0 \
    --moe_router_jitter_std 0.0 \
    --moe_router_jitter_final_std 0.0 \
    --moe_router_jitter_anneal_epochs 0 \
    --moe_router_soft_warmup_epochs 0 \
    --moe_router_warmup_mode off \
    --moe_router_warmup_epochs 0 \
    --moe_router_warmup_lr_scale 0.1 \
    --no-adapter_use_subject_id

  # 2) Hybrid base (bias paths only)
  run_and_collect_a3c "$HYBRID_LABEL" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3C_MOE_BIAS_FLAGS[@]}" "${A3C_MOE_PSD_FLAGS[@]}" \
    --moe_use_adapter_cond_bias \
    --no-moe_use_subject_summary_router_concat \
    --no-moe_use_eeg_summary_router_concat_spatial \
    --no-moe_use_eeg_summary_router_concat_spectral \
    --no-adapter_use_subject_id

  if [[ "$A3C_RUN_SET" == "full" ]]; then

  # 3) Add subject-summary concat to both routers.
  run_and_collect_a3c "$SUBJ_LABEL" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3C_MOE_BIAS_FLAGS[@]}" "${A3C_MOE_PSD_FLAGS[@]}" \
    --moe_use_adapter_cond_bias \
    --moe_use_subject_summary_router_concat \
    --no-moe_use_eeg_summary_router_concat_spatial \
    --no-moe_use_eeg_summary_router_concat_spectral \
    --use_subject_summary --subject_summary_file "$SUBJECT_SUMMARY_FILE" \
    --no-adapter_use_subject_id

  # 4) Add EEG compact summary concat to spatial router.
  run_and_collect_a3c "$EEG_SPATIAL_LABEL" \
    --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
    --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
    --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
    --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
    --moe_load_balance "$MOE_LOAD_BALANCE" \
    --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
    "${A3C_MOE_BIAS_FLAGS[@]}" "${A3C_MOE_PSD_FLAGS[@]}" \
    --moe_use_adapter_cond_bias \
    --moe_use_subject_summary_router_concat \
    --moe_use_eeg_summary_router_concat_spatial \
    --no-moe_use_eeg_summary_router_concat_spectral \
    --use_subject_summary --subject_summary_file "$SUBJECT_SUMMARY_FILE" \
    --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
    --no-adapter_use_subject_id

  fi

  if [[ "$A3C_RUN_SET" == "full" ]]; then
    # 5) Optional symmetry ablation: add EEG compact summary concat to spectral router too.
    run_and_collect_a3c "$EEG_BOTH_LABEL" \
      --moe --moe_route_mode typed_capacity_domain --moe_diagnostics \
      --moe_num_layers "$MOE_NUM_LAYERS" --moe_num_experts "$MOE_NUM_EXPERTS" \
      --moe_capacity_factor "$MOE_CAPACITY_FACTOR" \
      --moe_domain_emb_dim "$MOE_DOMAIN_EMB_DIM" \
      --moe_load_balance "$MOE_LOAD_BALANCE" \
      --moe_domain_bias_reg "$MOE_DOMAIN_BIAS_REG" \
      "${A3C_MOE_BIAS_FLAGS[@]}" "${A3C_MOE_PSD_FLAGS[@]}" \
      --moe_use_adapter_cond_bias \
      --moe_use_subject_summary_router_concat \
      --moe_use_eeg_summary_router_concat_spatial \
      --moe_use_eeg_summary_router_concat_spectral \
      --use_subject_summary --subject_summary_file "$SUBJECT_SUMMARY_FILE" \
      --eeg_channel_context --channel_context_file "$CHANNEL_CONTEXT_FILE" \
      --no-adapter_use_subject_id
  fi

  calc_delta() {
    awk -v a="$1" -v b="$2" 'BEGIN {printf("%.5f", a - b)}'
  }

  echo "[${BUNDLE_TAG}][summary] log_dir=$A3C_LOG_DIR"
  echo "[${BUNDLE_TAG}][summary] label best_acc best_kappa last_ep last_acc last_kappa used_static_fields runtime_state context_file"
  for row in "${A3C_ROWS[@]}"; do
    echo "[${BUNDLE_TAG}][summary] $row"
  done

  b_acc="${A3C_BEST_ACC[$BASE_LABEL]}"
  b_kappa="${A3C_BEST_KAPPA[$BASE_LABEL]}"
  hb_acc="${A3C_BEST_ACC[$HYBRID_LABEL]}"
  hb_kappa="${A3C_BEST_KAPPA[$HYBRID_LABEL]}"

  echo "[${BUNDLE_TAG}][delta] vs=$BASE_LABEL target=$HYBRID_LABEL d_best_acc=$(calc_delta "$hb_acc" "$b_acc") d_best_kappa=$(calc_delta "$hb_kappa" "$b_kappa")"
  if [[ "$A3C_RUN_SET" == "full" ]]; then
    hs_acc="${A3C_BEST_ACC[$SUBJ_LABEL]}"
    hs_kappa="${A3C_BEST_KAPPA[$SUBJ_LABEL]}"
    he_acc="${A3C_BEST_ACC[$EEG_SPATIAL_LABEL]}"
    he_kappa="${A3C_BEST_KAPPA[$EEG_SPATIAL_LABEL]}"
    h2_acc="${A3C_BEST_ACC[$EEG_BOTH_LABEL]}"
    h2_kappa="${A3C_BEST_KAPPA[$EEG_BOTH_LABEL]}"
    echo "[${BUNDLE_TAG}][delta] vs=$HYBRID_LABEL target=$SUBJ_LABEL d_best_acc=$(calc_delta "$hs_acc" "$hb_acc") d_best_kappa=$(calc_delta "$hs_kappa" "$hb_kappa")"
    echo "[${BUNDLE_TAG}][delta] vs=$SUBJ_LABEL target=$EEG_SPATIAL_LABEL d_best_acc=$(calc_delta "$he_acc" "$hs_acc") d_best_kappa=$(calc_delta "$he_kappa" "$hs_kappa")"
    echo "[${BUNDLE_TAG}][delta] vs=$EEG_SPATIAL_LABEL target=$EEG_BOTH_LABEL d_best_acc=$(calc_delta "$h2_acc" "$he_acc") d_best_kappa=$(calc_delta "$h2_kappa" "$he_kappa")"
  fi
else
  run_finetune "$model_dir" "${EXTRA_FLAGS[@]}"
fi

