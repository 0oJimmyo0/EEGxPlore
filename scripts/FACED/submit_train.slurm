#!/bin/bash
#SBATCH --job-name=FACED_TUNE
#SBATCH --output=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/FACED/out/faced_hier_%j.out
#SBATCH --error=/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/FACED/err/faced_hier_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00

set -euo pipefail

module load GCCcore/12.2.0
module load CUDA/12.6.3
module load miniconda/24.3.0
conda activate cbramod

DATASET="${DATASET:-FACED}"
DATASET_DIR="${DATASET_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/downstream_preped/FACED}"
NUM_CLASSES="${NUM_CLASSES:-9}"
FACED_META_CSV="${FACED_META_CSV:-/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv}"

EPOCHS="${EPOCHS:-50}"
ATTNRES_VARIANT="${ATTNRES_VARIANT:-pre_attn}"
DEPTH_ROUTER="${DEPTH_ROUTER:-on}"
if [[ -z "${DEPTH_ROUTER_DIM:-}" ]]; then
  if [[ "$ATTNRES_VARIANT" == "pre_attn" ]]; then
    DEPTH_ROUTER_DIM=15
  else
    DEPTH_ROUTER_DIM=26
  fi
fi
ROUTER_ARCH="${ROUTER_ARCH:-mlp}"
ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-128}"
DEPTH_SUMMARY_MODE="${DEPTH_SUMMARY_MODE:-attn_delta4}"
DEPTH_PROBE_MLP_FOR_ROUTER="${DEPTH_PROBE_MLP_FOR_ROUTER:-on}"
DEPTH_SUMMARY_GRAD_MODE="${DEPTH_SUMMARY_GRAD_MODE:-none}"
DEPTH_SUMMARY_UNFREEZE_EPOCH="${DEPTH_SUMMARY_UNFREEZE_EPOCH:-16}"
ABLATION_GROUP="${ABLATION_GROUP:-grad}"
ABLATION_VALUE="${ABLATION_VALUE:-trainable}"
ROUTER_DISPATCH_MODE="${ROUTER_DISPATCH_MODE:-soft}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.8}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.01}"
ROUTER_BALANCE_KL_COEF="${ROUTER_BALANCE_KL_COEF:-0.01}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.001}"
ROUTER_JITTER_STD="${ROUTER_JITTER_STD:-0.02}"
ROUTER_JITTER_FINAL_STD="${ROUTER_JITTER_FINAL_STD:-0.005}"
ROUTER_JITTER_ANNEAL_EPOCHS="${ROUTER_JITTER_ANNEAL_EPOCHS:-30}"
ROUTER_SOFT_WARMUP_EPOCHS="${ROUTER_SOFT_WARMUP_EPOCHS:-15}"
MODEL_DIR="${MODEL_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/FACED_typed_moe_A1_A1_base}"
ROUTING_EXPORT_DIR="${ROUTING_EXPORT_DIR:-/gpfs/radev/scratch/xu_hua/shared/models/checkpoints/routing_record_dir/run_A1_A1_base_${SLURM_JOB_ID:-manual}}"

ABLATION_TAG="none"
case "$ABLATION_GROUP" in
  none)
    ;;
  grad)
    if [[ -z "$ABLATION_VALUE" ]]; then
      echo "[slurm] ABLATION_GROUP=grad requires ABLATION_VALUE in {detached,delayed_unfreeze,trainable}" >&2
      exit 2
    fi
    case "$ABLATION_VALUE" in
      detached|delayed_unfreeze|trainable)
        DEPTH_SUMMARY_MODE="attn_delta4"
        DEPTH_SUMMARY_GRAD_MODE="$ABLATION_VALUE"
        ABLATION_TAG="grad_${ABLATION_VALUE}"
        ;;
      *)
        echo "[slurm] invalid ABLATION_VALUE for grad: $ABLATION_VALUE" >&2
        exit 2
        ;;
    esac
    ;;
  content)
    if [[ -z "$ABLATION_VALUE" ]]; then
      echo "[slurm] ABLATION_GROUP=content requires ABLATION_VALUE in {attn_delta4,attn_mlp_balanced,attn_mlp_latemix}" >&2
      exit 2
    fi
    case "$ABLATION_VALUE" in
      attn_delta4|attn_mlp_balanced|attn_mlp_latemix)
        DEPTH_SUMMARY_MODE="$ABLATION_VALUE"
        DEPTH_SUMMARY_GRAD_MODE="detached"
        ABLATION_TAG="content_${ABLATION_VALUE}"
        ;;
      *)
        echo "[slurm] invalid ABLATION_VALUE for content: $ABLATION_VALUE" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "[slurm] invalid ABLATION_GROUP: $ABLATION_GROUP (expected: none|grad|content)" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-manual}_A1_${ABLATION_TAG}_${ATTNRES_VARIANT}_${DEPTH_ROUTER}}"

DEPTH_ROUTER_ARGS=()
if [[ "$DEPTH_ROUTER" == "on" ]]; then
  DEPTH_ROUTER_ARGS+=(--moe_use_attnres_depth_router_features --moe_attnres_depth_router_dim "$DEPTH_ROUTER_DIM")
  DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_summary_mode "$DEPTH_SUMMARY_MODE")
  if [[ "$DEPTH_PROBE_MLP_FOR_ROUTER" == "on" ]]; then
    DEPTH_ROUTER_ARGS+=(--moe_attnres_depth_probe_mlp_for_router)
  fi
fi

mkdir -p "$MODEL_DIR" "$ROUTING_EXPORT_DIR"

echo "[slurm] running A1_base only | epochs=$EPOCHS" >&2
echo "[slurm] dataset=$DATASET num_classes=$NUM_CLASSES" >&2
echo "[slurm] ablation_group=$ABLATION_GROUP ablation_value=${ABLATION_VALUE:-NA}" >&2
echo "[slurm] attnres_variant=$ATTNRES_VARIANT depth_router=$DEPTH_ROUTER dispatch=$ROUTER_DISPATCH_MODE" >&2
echo "[slurm] depth_summary_mode=$DEPTH_SUMMARY_MODE probe_mlp_for_router=$DEPTH_PROBE_MLP_FOR_ROUTER" >&2
echo "[slurm] depth_summary_grad_mode=$DEPTH_SUMMARY_GRAD_MODE depth_summary_unfreeze_epoch=$DEPTH_SUMMARY_UNFREEZE_EPOCH" >&2
echo "[slurm] router_temp=$ROUTER_TEMPERATURE entropy=$ROUTER_ENTROPY_COEF bal_kl=$ROUTER_BALANCE_KL_COEF" >&2
echo "[slurm] model_dir=$MODEL_DIR" >&2
echo "[slurm] routing_export_dir=$ROUTING_EXPORT_DIR | run_name=$RUN_NAME" >&2

python /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/finetune_main.py \
  --seed 42 \
  --cuda 0 \
  --epochs "$EPOCHS" \
  --batch_size 32 \
  --lr 2e-4 \
  --weight_decay 2e-2 \
  --optimizer AdamW \
  --clip_value 1.0 \
  --dropout 0.3 \
  --classifier all_patch_reps \
  --downstream_dataset "$DATASET" \
  --datasets_dir "$DATASET_DIR" \
  --num_of_classes "$NUM_CLASSES" \
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
  --moe_attnres_depth_summary_grad_mode "$DEPTH_SUMMARY_GRAD_MODE" \
  --moe_attnres_depth_summary_unfreeze_epoch "$DEPTH_SUMMARY_UNFREEZE_EPOCH" \
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
  --no-tqdm 

