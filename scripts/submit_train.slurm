#!/bin/bash
#SBATCH --job-name=FACED_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/faced_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/err/faced_hier_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

# --- modules: compiler + CUDA that match your build ---
module load GCCcore/12.2.0
module load CUDA/12.6.3

module load miniconda/24.3.0
conda activate cbramod

# --- choose dataset ---
downstream_dataset=FACED
dataset_dir=/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED

# -------------------------
# output
# -------------------------
model_dir=/gpfs/radev/scratch/xu_hua/shared/data/localLLM/checkpoints/${downstream_dataset}_attnres_scratch
mkdir -p "$model_dir"

# -------------------------
# training hyperparameters
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

# -------------------------
# run
# Baseline: --attnres_variant none (matches original CBraMod finetune loop).
# AttnRes:   e.g. --attnres_variant final (same optimizer/scheduler as baseline).
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
  --attnres_variant pre_attn \

#  --use_pretrained_weights