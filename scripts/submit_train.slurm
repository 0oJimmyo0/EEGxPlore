#!/bin/bash
#SBATCH --job-name=FACED_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/out/faced_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/err/faced_hier_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
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
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_A1_base}"
MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/FACED_typed_moe_A1_A1_base}"
ROUTING_EXPORT_DIR="${ROUTING_EXPORT_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_A1_A1_base_${SLURM_JOB_ID:-manual}}"

mkdir -p "$MODEL_DIR" "$ROUTING_EXPORT_DIR"

echo "[slurm] running A1_base only | epochs=$EPOCHS" >&2
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
  --attnres_variant pre_attn \
  --attnres_start_layer 0 \
  --use_pretrained_weights \
  --moe \
  --moe_diagnostics \
  --moe_num_layers 1 \
  --moe_num_experts 4 \
  --moe_router_arch mlp \
  --moe_router_mlp_hidden 128 \
  --moe_router_dispatch_mode hard_capacity \
  --moe_router_temperature 1.2 \
  --moe_router_entropy_coef 0.003 \
  --moe_router_balance_kl_coef 0.003 \
  --moe_router_z_loss_coef 0.0001 \
  --moe_router_jitter_std 0.01 \
  --moe_router_jitter_final_std 0.002 \
  --moe_router_jitter_anneal_epochs 12 \
  --moe_router_soft_warmup_epochs 2 \
  --routing_export_dir "$ROUTING_EXPORT_DIR" \
  --routing_export_splits val,test \
  --faced_meta_csv "$FACED_META_CSV" \
  --routing_run_name "$RUN_NAME" \
  --multi_lr \
  --no-tqdm
