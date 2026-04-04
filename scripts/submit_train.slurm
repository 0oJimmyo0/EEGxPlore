#!/bin/bash
#SBATCH --job-name=FACED_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/out/faced_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/err/faced_hier_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

DATASET=FACED
DATASET_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED
FACED_META_CSV=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv

EPOCHS="${EPOCHS:-50}"
ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
DEPTH_ROUTER="${DEPTH_ROUTER:-on}"
if [[ -z "${DEPTH_ROUTER_DIM:-}" ]]; then
  if [[ "$ATTNRES_VARIANT" == "pre_attn" ]]; then
    DEPTH_ROUTER_DIM=13
  else
    DEPTH_ROUTER_DIM=26
  fi
fi
ROUTER_ARCH="${ROUTER_ARCH:-linear}"
ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-128}"
ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.8}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.01}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.01}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.001}"
ROUTER_JITTER_STD="${ROUTER_JITTER_STD:-0.02}"
ROUTER_JITTER_FINAL_STD="${ROUTER_JITTER_FINAL_STD:-0.005}"
ROUTER_JITTER_ANNEAL_EPOCHS="${ROUTER_JITTER_ANNEAL_EPOCHS:-20}"
ROUTER_SOFT_WARMUP_EPOCHS="${ROUTER_SOFT_WARMUP_EPOCHS:-8}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_A1_stable_${ATTNRES_VARIANT}_${DEPTH_ROUTER}}"
MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/FACED_typed_moe_A1_A1_base}"
ROUTING_EXPORT_DIR="${ROUTING_EXPORT_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_A1_A1_base_${SLURM_JOB_ID:-manual}}"

DEPTH_ROUTER_ARGS=()
if [[ "$DEPTH_ROUTER" == "on" ]]; then
  DEPTH_ROUTER_ARGS+=(--moe_use_attnres_depth_router_features --moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM")
fi

mkdir -p "$MODEL_DIR" "$ROUTING_EXPORT_DIR"

echo "[slurm] running A1_base only | epochs=$EPOCHS" >&2
echo "[slurm] attnres_variant=$ATTNRES_VARIANT depth_router=$DEPTH_ROUTER dispatch=$ROUTER_DISPATCH_MODE" >&2
echo "[slurm] model_dir=$MODEL_DIR" >&2
echo "[slurm] routing_export_dir=$ROUTING_EXPORT_DIR | run_name=$RUN_NAME" >&2

python /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/finetune_main.py \
  --seed 3407 \
  --cuda 0 \
  --epochs "$EPOCHS" \
  --batch_size 32 \
  --lr 5e-4 \
  --weight_decay 5e-2 \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout 0.3 \
  --classifier all_patch_reps \
  --downstream_dataset "$DATASET" \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes 9 \
  --model_dir "$MODEL_DIR" \
  --label_smoothing 0.05 \
  --num_workers 4 \
  --attnres_variant "$ATTNRES_VARIANT" \
  --attnres_start_layer 0 \
  --use_pretrained_weights \
  --moe \
  --moe_diagnostics \
  --moe_num_layers 1 \
  --moe_num_experts 4 \
  --moe_router_arch "$ROUTER_ARCH" \
  --moe_router_mlp_hidden "$ROUTER_MLP_HIDDEN" \
  "${DEPTH_ROUTER_ARGS[@]}" \
  --moe_router_dispatch_mode "$ROUTER_DISPATCH_MODE" \
  --moe_router_temperature "$ROUTER_TEMPERATURE" \
  --moe_router_entropy_coef "$ROUTER_ENTROPY_COEF" \
  --moe_router_balance_kl_coef "$ROUTER_BALANCE_KL_COEF" \
  --moe_router_z_loss_coef "$ROUTER_Z_LOSS_COEF" \
  --moe_router_jitter_std "$ROUTER_JITTER_STD" \
  --moe_router_jitter_final_std "$ROUTER_JITTER_FINAL_STD" \
  --moe_router_jitter_anneal_epochs "$ROUTER_JITTER_ANNEAL_EPOCHS" \
  --moe_router_soft_warmup_epochs "$ROUTER_SOFT_WARMUP_EPOCHS" \
  --routing_export_dir "$ROUTING_EXPORT_DIR" \
  --routing_export_splits val,test \
  --faced_meta_csv "$FACED_META_CSV" \
  --routing_run_name "$RUN_NAME" \
  --multi_lr \
  --no-tqdm
