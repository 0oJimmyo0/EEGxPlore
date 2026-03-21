#!/bin/bash
#SBATCH --job-name=FACED_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/out/faced_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CBraMod/logs/err/faced_hier_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
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
# output (use a unique path per experiment family)
# -------------------------
model_dir=/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/${downstream_dataset}_moe_opt
mkdir -p "$model_dir"

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

# -------------------------
# Recommended MoE baseline (before you add new MoE modules):
#   --attnres_variant none     strongest FACED test acc in recent sweeps (~0.585 κ~0.53)
#   --moe_top_k 1              simpler router, slightly faster than k=2
#   moe_load_balance / moe_router_noise: defaults in finetune_main.py are now 0.05 / 0.01
#
# Recent reference (same hparams): MoE-only > AttnRes+MoE on test; pre_attn+k2 ~0.58 after LB fix;
# final/full ~0.578–0.579. Re-enable AttnRes only when comparing, not as default MoE shell.
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
  --num_workers 8 \
  --attnres_variant none \
  --use_pretrained_weights \
  --moe \
  --moe_num_layers 2 \
  --moe_num_experts 4 \
  --moe_top_k 1

# Explicit tuning (optional; matches new argparse defaults — omit to use defaults):
#   --moe_load_balance 0.05 --moe_router_noise 0.01

# -------------------------
# Optional experiments (comment main block above, uncomment one)
# -------------------------
# MoE + pre_attn, warmer gate, top-2 experts (~0.58 test in 1400054):
#   --attnres_variant pre_attn --attnres_start_layer 6 \
#   --attnres_gated --attnres_gate_init -1.0 \
#   --moe --moe_top_k 2 --moe_load_balance 0.05 --moe_router_noise 0.01
#
# Disable aux noise ablation:
#   --moe_load_balance 0 --moe_router_noise 0
