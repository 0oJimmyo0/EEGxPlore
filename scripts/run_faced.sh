#!/usr/bin/env bash
set -euo pipefail

# Minimal FACED run script for the refactored FACED+SEED-V branch.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_ID="${CUDA_ID:-0}"
DATASET_DIR="${DATASET_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/FACED_data}"
FOUNDATION_DIR="${FOUNDATION_DIR:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth}"
FACED_META_CSV="${FACED_META_CSV:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/output/faced_refactor_anchor}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "$MODEL_DIR"

python "$REPO_DIR/finetune_main.py" \
  --seed 42 \
  --cuda "$CUDA_ID" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr 2e-4 \
  --weight_decay 2e-2 \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout 0.3 \
  --classifier all_patch_reps \
  --downstream_dataset FACED \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes 9 \
  --model_dir "$MODEL_DIR" \
  --label_smoothing 0.05 \
  --num_workers "$NUM_WORKERS" \
  --use_pretrained_weights \
  --foundation_dir "$FOUNDATION_DIR" \
  --faced_meta_csv "$FACED_META_CSV" \
  --attnres_variant pre_attn \
  --attnres_start_layer 0 \
  --moe \
  --moe_diagnostics \
  --moe_num_layers 1 \
  --moe_num_experts 4 \
  --moe_router_arch mlp \
  --moe_router_mlp_hidden 128 \
  --moe_use_attnres_depth_router_features \
  --moe_attnres_depth_router_dim 15 \
  --moe_attnres_depth_summary_mode attn_delta4 \
  --moe_attnres_depth_probe_mlp_for_router \
  --moe_attnres_depth_summary_grad_mode detached \
  --moe_attnres_depth_summary_unfreeze_epoch 16 \
  --moe_router_dispatch_mode soft \
  --moe_router_temperature 1.5 \
  --moe_router_entropy_coef 0.005 \
  --moe_router_balance_kl_coef 0.01 \
  --moe_router_z_loss_coef 0.001 \
  --moe_router_jitter_std 0.02 \
  --moe_router_jitter_final_std 0.005 \
  --moe_router_jitter_anneal_epochs 30 \
  --moe_router_soft_warmup_epochs 15 \
  --no-tqdm
