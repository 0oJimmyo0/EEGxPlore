#!/bin/bash
#SBATCH --job-name=FACED_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/faced_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/err/faced_hier_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

# =============================================================================
# COMPARISON KNOBS — change these to compare runs (keep everything else equal)
# =============================================================================
# 1) USE_PSD: 0 = typed MoE, NO PSD on spectral router (main baseline per doc/next_step.md)
#             1 = add --moe_use_psd_router_features (spectral router gets 5-band PSD)
# 2) RUN_TAG: short label folded into model_dir so checkpoints never overwrite between A/B
# 3) ROUTING_EXPORT_DIR: leave empty to skip CSV export; or set to a path ON SCRATCH.
#    Use ${SLURM_JOB_ID} UNESCAPED so sbatch expands it on the compute node (not \$...).
#    CSVs are written ONLY after training finishes (test + checkpoint save).
# 4) ROUTING_SPLITS: test only, or val,test
# 5) ROUTING_RUN_NAME: stored in per-sample CSV (default: SLURM_JOB_ID when set; else empty)
# 6) TQDM: default --no-tqdm on cluster (tqdm → huge .out files / quota); use FORCE_TQDM=1 to debug
# =============================================================================
USE_PSD=1
RUN_TAG=psd
ROUTING_EXPORT_DIR="/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_${RUN_TAG}_${SLURM_JOB_ID}"
# Personal scratch (saves group quota on shared): uncomment and comment line above
# ROUTING_EXPORT_DIR="/gpfs/radev/scratch/xu_hua/mj756/routing_exports/run_${RUN_TAG}_${SLURM_JOB_ID}"
ROUTING_SPLITS=test
# Override if not under Slurm (e.g. interactive): ROUTING_RUN_NAME=my_manual_run
ROUTING_RUN_NAME="${SLURM_JOB_ID:-}"
FORCE_TQDM=0

# --- modules: compiler + CUDA that match your build ---
module load GCCcore/12.2.0
module load CUDA/12.6.3

module load miniconda/24.3.0
conda activate cbramod

# --- choose dataset ---
downstream_dataset=FACED
dataset_dir=/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED

# Checkpoints: separate subdir per comparison (USE_PSD + RUN_TAG)
model_dir=/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/${downstream_dataset}_typed_moe_${RUN_TAG}
mkdir -p "$model_dir"

FACED_META_CSV=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv

# -------------------------
# training hyperparameters (FACED MoE sweet spot from logs)
# -------------------------
lr=5e-4
weight_decay=5e-2
clip_value=1.0
dropout=0.3
label_smoothing=0.05
epochs=30
batch_size=32
seed=3407
cuda_id=0

# Optional flags from USE_PSD / ROUTING_EXPORT_DIR / TQDM
PSD_FLAGS=()
if [[ "$USE_PSD" == "1" ]]; then
  PSD_FLAGS=(--moe_use_psd_router_features)
fi

ROUTING_FLAGS=()
if [[ -n "$ROUTING_EXPORT_DIR" ]]; then
  mkdir -p "$ROUTING_EXPORT_DIR"
  ROUTING_FLAGS=(
    --routing_export_dir "$ROUTING_EXPORT_DIR"
    --routing_export_splits "$ROUTING_SPLITS"
    --faced_meta_csv "$FACED_META_CSV"
  )
  if [[ -n "$ROUTING_RUN_NAME" ]]; then
    ROUTING_FLAGS+=(--routing_run_name "$ROUTING_RUN_NAME")
  fi
  echo "[slurm] routing export dir: $ROUTING_EXPORT_DIR" >&2
  echo "[slurm] routing_run_name: ${ROUTING_RUN_NAME:-<empty>}" >&2
fi

TQDM_FLAGS=(--no-tqdm)
if [[ "$FORCE_TQDM" == "1" ]]; then
  TQDM_FLAGS=(--tqdm)
fi

# -------------------------
# This job: typed dual-bank MoE (shared dense + spatial + spectral specialists)
#   --moe_shared_specialist   pretrained dense always on; specialists start with zero output
#   --moe_diagnostics         after each val epoch, log expert usage / entropy (one val batch)
#
# Other comparisons (comment this block, adjust vars above):
#   Replacement MoE (no shared path): omit --moe_shared_specialist; use a different model_dir suffix
#   Optional: --moe_specialist_rand_linear1  (Kaiming linear1 on specialists)
#   Optional: --moe_top_k 2
# -------------------------
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
  --moe \
  --moe_shared_specialist \
  --moe_diagnostics \
  --moe_specialist_rand_linear1 \
  --moe_num_layers 1 \
  --moe_num_experts 4 \
  --moe_top_k 2 \
  --moe_router_mode sample_attnres \
  --moe_router_arch mlp \
  --moe_router_mlp_hidden 128 \
  --moe_expert_type typed \
  "${PSD_FLAGS[@]}" \
  "${ROUTING_FLAGS[@]}" \
  "${TQDM_FLAGS[@]}"

# -------------------------
# Quick A/B cheat sheet (edit COMPARISON KNOBS at top)
# -------------------------
# | Goal                         | USE_PSD | RUN_TAG  | Extra                          |
# |-------------------------------|---------|----------|--------------------------------|
# | Baseline: typed, no PSD       | 0       | no_psd   | (default)                      |
# | Compare: typed + PSD router | 1       | psd      | same hyperparams               |
# | Routing CSV + metadata join | any     | any      | set ROUTING_EXPORT_DIR scratch |
# | Tag rows with job id         | any     | any      | ROUTING_RUN_NAME (default JOBID)|
# | Interactive tqdm            | any     | any      | FORCE_TQDM=1                   |
#
# =============================================================================
# NEXT STEPS — post-hoc analysis (run on login or devel node; not inside this job)
# =============================================================================
# Project root:
#   CBRA=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod
#
# 1) Rebuild grouped CSVs from one existing per-sample export (adds routing_summary_by_group, etc.):
#   cd "$CBRA"
#   python3 scripts/analyze_facced_routing.py \
#     --csv /gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_no_psd_1405305/faced_routing_test_ebest_ep21_epoch21_acc_0.56441_kappa_0.50606_f1_0.56680_per_sample.csv \
#     --outdir /gpfs/radev/scratch/xu_hua/mj756/routing_exports/analysis_no_psd_1405305 \
#     --prefix faced_rerun_1405305
#
# 2) Paired no-PSD vs PSD compare (FIRST csv = no-PSD, SECOND = PSD) → four compare_*.csv files:
#   cd "$CBRA"
#   python3 scripts/analyze_facced_routing.py \
#     --compare \
#     /gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_no_psd_1405305/faced_routing_test_ebest_ep21_epoch21_acc_0.56441_kappa_0.50606_f1_0.56680_per_sample.csv \
#     /gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_psd_1405304/faced_routing_test_ebest_epNN_epochNN_acc_*_per_sample.csv \
#     --outdir /gpfs/radev/scratch/xu_hua/mj756/routing_exports/compare_no_psd_vs_psd
#   (Replace the PSD path with your real filename: ls run_psd_*/faced_routing_test_*_per_sample.csv)
#
# 3) Submit training (after editing USE_PSD / RUN_TAG at top):
#   cd /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod
#   sbatch scripts/submit_train.slurm
#
# Optional experiments (older notes):
# Replacement MoE only: omit --moe_shared_specialist; point model_dir to ..._moe_opt
# MoE + pre_attn, warmer gate, top-2 (~0.58 test in older logs):
#   --attnres_gated --attnres_gate_init -1.0 --attnres_start_layer 6 --moe_top_k 2 --moe_load_balance 0.05 --moe_router_noise 0.01
# Disable aux: --moe_load_balance 0 --moe_router_noise 0
# sample_hidden router: --moe_router_mode sample_hidden
