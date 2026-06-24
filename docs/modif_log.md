mlp 1e-5, lr_router_mul 1.5 ->0.75 / router temp 1.5, router warmup epoch 8, warmup blend + / MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL="0.025" MOE_ROUTER_ENTROPY_COEF_SPECTRAL="0.02" / ROUTER_MLP_HIDDEN="${ROUTER_MLP_HIDDEN:-64}" / 

3e-5, mlp 128 baseline/ routr_mul = 1.5 + label smooth 0.02 / jitter那边

change moe_router_entropy_coef_spatial = 0.02
moe_router_entropy_coef_spectral = 0.01
moe_router_balance_kl_coef_spatial = 0.03
moe_router_balance_kl_coef_spectral = 0.015 / lr 2e-5 / lr 1e-5

mlp, lr 5e-5/ 3e-5 +dual query / 3e-5 + router mult 1.25

router mul 1.25 + EMA / router mul 1.5 + unfreeze 16 / unfreeze 8(baseline) + 
increase moe_shared_blend_warmup_epochs a little, like 5 → 7
increase moe_router_soft_warmup_epochs a little, like 8 → 12



router temp 1.5-1.75/ 
moe_router_entropy_coef_spatial=0.02
moe_router_entropy_coef_spectral=0.015
moe_router_balance_kl_coef_spatial=0.03
moe_router_balance_kl_coef_spectral=0.02
/
SP_ENT=0.02, SP_KL=0.03 (current best family) /
SP_ENT=0.015, SP_KL=0.03 /
SP_ENT=0.02, SP_KL=0.025 /
SP_ENT=0.015, SP_KL=0.025
Keep spectral fixed at:

SC_ENT=0.01
SC_KL=0.015

/
moe_router_z_loss_coef=5e-4
moe_router_jitter_std=0.015
moe_router_jitter_final_std=0.005
moe_router_jitter_anneal_epochs=30

/
SP_ENT=0.02, SP_KL=0.03 fixed
Try SC_ENT=0.0075, SC_KL=0.0125 /
Try SC_ENT=0.005, SC_KL=0.010


/ lr router mult 1.5-1.25 / 1.5-1 


/moe_router_compact_feature_mode=eeg_summary, moe_router_compact_feature_dim=8
moe_router_compact_feature_mode=psd_summary, moe_router_compact_feature_dim=8



/
eeg_summary plain:
COMPACT_FEATURE_MODE=eeg_summary \
COMPACT_FEATURE_DIM=8 \
COMPACT_ROUTER_WARMUP_EPOCHS=0 \
COMPACT_ROUTER_GATE_INIT=1.0 \
sbatch EEGxPlore/scripts/PHYSIO-MI/train_physio_compact_shared.slurm

eeg_summary with warmup only:
COMPACT_FEATURE_MODE=eeg_summary \
COMPACT_FEATURE_DIM=8 \
COMPACT_ROUTER_WARMUP_EPOCHS=8 \
COMPACT_ROUTER_GATE_INIT=1.0 \
sbatch EEGxPlore/scripts/PHYSIO-MI/train_physio_compact_shared.slurm

eeg_summary with warmup plus weaker initial gate:
COMPACT_FEATURE_MODE=eeg_summary \
COMPACT_FEATURE_DIM=8 \
COMPACT_ROUTER_WARMUP_EPOCHS=8 \
COMPACT_ROUTER_GATE_INIT=0.25 \
sbatch EEGxPlore/scripts/PHYSIO-MI/train_physio_compa

psd_summ with warmup 8 / 
psd summ wth warmup wth init gate 0.25


----------
/ baseline: apply_bank_reg_first_try() {
  if [[ "$MOE_ROUTER_ENTROPY_COEF_SPATIAL" == "-1.0" ]]; then
    MOE_ROUTER_ENTROPY_COEF_SPATIAL="0.02"
  fi
  if [[ "$MOE_ROUTER_ENTROPY_COEF_SPECTRAL" == "-1.0" ]]; then
    MOE_ROUTER_ENTROPY_COEF_SPECTRAL="0.01"
  fi
  if [[ "$MOE_ROUTER_BALANCE_KL_COEF_SPATIAL" == "-1.0" ]]; then
    MOE_ROUTER_BALANCE_KL_COEF_SPATIAL="0.03"
  fi
  if [[ "$MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL" == "-1.0" ]]; then
    MOE_ROUTER_BALANCE_KL_COEF_SPECTRAL="0.015"
  fi
}
----------




change lr: 5e-5, 1e-4, 2e-5 / 5e-5+delay unfreexe / same + dual query / same + block shared typed 



/ 2e-4 + unfreexe


lr 4e-5, 6e-5, 2e-5 / 5e-5+ema / 5e-5+component lr /5e-5+effective num+num beta 0.995 / 5e-5+dual query / 


moe_router_compact_feature_mode=none + 5e-5 + ema / moe_router_compact_feature_mode=none + 5e-5 + compo lr + ema + class weight: effective num / same + inv freq clip / 5e-5+ema+component lr / 5e-5+ dual query

**检测component lr, eeg summary*

5e-5 + compo lr onlly / linear proj not mlp / 5e-5+ no compo lr + inv freq / same with compo lr /
5e-5 + unfreexe 1

5e-5+unfreexe 6/ unfreexe 4 / unfreexe 8 (curent besr) with ema step 1000 / 

** CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.75}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.5}" **

change to:
CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.5}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.75}" /
 CLASS_WEIGHT_CLIP_MIN="${CLASS_WEIGHT_CLIP_MIN:-0.9}"
CLASS_WEIGHT_CLIP_MAX="${CLASS_WEIGHT_CLIP_MAX:-1.2}"

/ current best + lr 4e-5/ 6e-5/ \
0.95 / 1.15
1.0 / 1.1
0.85 / 1.25
\

**classifier all patch twolayer/all patch onelayer / *** (tbd) --moe_shared_blend_warmup_epochs 5 -- 3 / 

ROUTER_ENTROPY_COEF_SPATIAL=-1.0
ROUTER_ENTROPY_COEF_SPECTRAL=0.0075
ROUTER_BALANCE_KL_COEF_SPATIAL=-1.0
ROUTER_BALANCE_KL_COEF_SPECTRAL=-1.0 / 

ROUTER_ENTROPY_COEF_SPATIAL=-1.0
ROUTER_ENTROPY_COEF_SPECTRAL=-1.0
ROUTER_BALANCE_KL_COEF_SPATIAL=-1.0
ROUTER_BALANCE_KL_COEF_SPECTRAL=0.015 /

ROUTER_ENTROPY_COEF_SPATIAL=-1.0
ROUTER_ENTROPY_COEF_SPECTRAL=0.0075
ROUTER_BALANCE_KL_COEF_SPATIAL=-1.0
ROUTER_BALANCE_KL_COEF_SPECTRAL=0.015 / 

ROUTER_ENTROPY_COEF_SPATIAL=0.003
ROUTER_ENTROPY_COEF_SPECTRAL=0.0075
ROUTER_BALANCE_KL_COEF_SPATIAL=0.005
ROUTER_BALANCE_KL_COEF_SPECTRAL=0.015

/ multi lr 
/moe_router_compact_gate_init = 0.5
moe_router_compact_warmup_epochs = 3 or 5
/moe_expert_init_noise_std = 0.01


/moe_router_compact_gate_init = 0.5
moe_router_compact_warmup_epochs = 5
/both 上下
/moe_expert_init_noise_std = 0.01

/Step-1 (short schedule / early-stop proxy) x2
/Step-3 (component LR ablation) x2
/avg polling
/anti-collapse depth separation x2



----------
seed-v ablation
sbatch --export=ALL,RUN_MODE=dense,SEEDV_PROTOCOL=cbramod_benchmark,SEED_LIST="42 3407 2024",RUN_NAME=seedv_dense submit_seedv_train.slurm

# AttnRes-only
sbatch --export=ALL,RUN_MODE=custom,USE_MOE=0,ATTNRES_VARIANT=pre_attn,SEEDV_PROTOCOL=cbramod_benchmark,SEED_LIST="42 3407 2024",RUN_NAME=seedv_attnres submit_seedv_train.slurm

# full model
sbatch --export=ALL,RUN_MODE=custom,USE_MOE=1,ATTNRES_VARIANT=pre_attn,A1_STRICT_BLOCK_ABLATION=1,SEEDV_PROTOCOL=cbramod_benchmark,SEED_LIST="42 3407 2024",RUN_NAME=seedv_full submit_seedv_train.slurm

# context: compact_shared
sbatch --export=ALL,RUN_MODE=custom,USE_MOE=1,ATTNRES_VARIANT=pre_attn,A1_STRICT_BLOCK_ABLATION=0,DEPTH_CONTEXT_MODE=compact_shared,DEPTH_SUMMARY_MODE=attn_delta4,DEPTH_PROBE_MLP_FOR_ROUTER=on,SEED_LIST="42 3407 2024",RUN_NAME=seedv_ctx_compact submit_seedv_train.slurm

# context: block_shared
sbatch --export=ALL,RUN_MODE=custom,USE_MOE=1,ATTNRES_VARIANT=pre_attn,A1_STRICT_BLOCK_ABLATION=0,DEPTH_CONTEXT_MODE=block_shared_typed_proj,DEPTH_QUERY_MODE=shared,SEED_LIST="42 3407 2024",RUN_NAME=seedv_ctx_block submit_seedv_train.slurm

# context: dual_query
sbatch --export=ALL,RUN_MODE=custom,USE_MOE=1,ATTNRES_VARIANT=pre_attn,A1_STRICT_BLOCK_ABLATION=0,DEPTH_CONTEXT_MODE=dual_query_block_typed_proj,DEPTH_QUERY_MODE=dual,SEED_LIST="42 3407 2024",RUN_NAME=seedv_ctx_dual submit_seedv_train.slurm


--------- TUEV
cd /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/scripts/TUEV

# dense
sbatch --export=ALL,USE_MOE=0,ATTNRES_VARIANT=none,SEED=42,RUN_NAME=tuev_dense_s42 submit_train.slurm

# AttnRes-only
sbatch --export=ALL,USE_MOE=0,ATTNRES_VARIANT=pre_attn,SEED=42,RUN_NAME=tuev_attnres_s42 submit_train.slurm

# full
sbatch --export=ALL,USE_MOE=1,ATTNRES_VARIANT=pre_attn,SEED=42,RUN_NAME=tuev_full_s42 submit_train.slurm

# component LR
sbatch --export=ALL,USE_MOE=1,ATTNRES_VARIANT=pre_attn,USE_COMPONENT_LR=1,SEED=42,RUN_NAME=tuev_componentlr_s42 submit_train.slurm

# EMA
sbatch --export=ALL,USE_MOE=1,ATTNRES_VARIANT=pre_attn,USE_EMA=1,EMA_WARMUP_STEPS=300,SEED=42,RUN_NAME=tuev_ema_s42 submit_train.slurm

---------FACED
cd /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/scripts/FACED

sbatch --export=ALL,ABLATION_GROUP=grad,ABLATION_VALUE=detached,RUN_NAME=faced_grad_detached submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=grad,ABLATION_VALUE=delayed_unfreeze,RUN_NAME=faced_grad_delayed submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=grad,ABLATION_VALUE=trainable,RUN_NAME=faced_grad_trainable submit_train.slurm

sbatch --export=ALL,ABLATION_GROUP=content,ABLATION_VALUE=attn_delta4,RUN_NAME=faced_content_delta4 submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=content,ABLATION_VALUE=attn_mlp_balanced,RUN_NAME=faced_content_balanced submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=content,ABLATION_VALUE=attn_mlp_latemix,RUN_NAME=faced_content_latemix submit_train.slurm

sbatch --export=ALL,USE_EMA=1,RUN_NAME=faced_ema_on submit_train.slurm
sbatch --export=ALL,USE_EMA=0,RUN_NAME=faced_ema_off submit_train.slurm

sbatch --export=ALL,DEPTH_CONTEXT_MODE=compact_shared,RUN_NAME=faced_ctx_compact submit_train.slurm
sbatch --export=ALL,DEPTH_CONTEXT_MODE=dual_query_block_typed_proj,RUN_NAME=faced_ctx_dual submit_train.slurm
sbatch --export=ALL,DEPTH_CONTEXT_MODE=block_shared_typed_proj,RUN_NAME=faced_ctx_block submit_train.slurm


--------isruc 
cd /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/scripts/ISRUC

sbatch --export=ALL,ABLATION_GROUP=context,ABLATION_VALUE=compact_shared,RUN_NAME=isruc_ctx_compact submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=context,ABLATION_VALUE=dual_query_block_typed_proj,RUN_NAME=isruc_ctx_dual submit_train.slurm

sbatch --export=ALL,ABLATION_GROUP=grad,ABLATION_VALUE=detached,RUN_NAME=isruc_grad_detached submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=grad,ABLATION_VALUE=delayed_unfreeze,RUN_NAME=isruc_grad_delayed submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=grad,ABLATION_VALUE=trainable,RUN_NAME=isruc_grad_trainable submit_train.slurm

sbatch --export=ALL,ABLATION_GROUP=ema,ABLATION_VALUE=on,RUN_NAME=isruc_ema_on submit_train.slurm
sbatch --export=ALL,ABLATION_GROUP=ema,ABLATION_VALUE=off,RUN_NAME=isruc_ema_off submit_train.slurm

-------physio-mi
cd /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/scripts/PHYSIO-MI

# context
sbatch --export=ALL,ABLATION_GROUP=context,ABLATION_VALUE=compact_shared,RUN_NAME=physio_ctx_compact train_physio_compact_shared.slurm
sbatch --export=ALL,ABLATION_GROUP=context,ABLATION_VALUE=dual_query_block_typed_proj,RUN_NAME=physio_ctx_dual train_physio_compact_shared.slurm

# EMA
sbatch --export=ALL,ABLATION_GROUP=ema,ABLATION_VALUE=on,RUN_NAME=physio_ema_on train_physio_compact_shared.slurm
sbatch --export=ALL,ABLATION_GROUP=ema,ABLATION_VALUE=off,RUN_NAME=physio_ema_off train_physio_compact_shared.slurm

# reg / component LR
sbatch --export=ALL,REG_LR_ABLATION=none,RUN_NAME=physio_reg_none train_physio_compact_shared.slurm
sbatch --export=ALL,REG_LR_ABLATION=bank_reg,RUN_NAME=physio_bank_reg train_physio_compact_shared.slurm
sbatch --export=ALL,REG_LR_ABLATION=mild_component_lr,RUN_NAME=physio_component_lr train_physio_compact_shared.slurm
sbatch --export=ALL,REG_LR_ABLATION=bank_reg_plus_mild_component_lr,RUN_NAME=physio_bankreg_componentlr train_physio_compact_shared.slurm

# compact feature source
sbatch --export=ALL,COMPACT_FEATURE_MODE=none,RUN_NAME=physio_compact_none train_physio_compact_shared.slurm
sbatch --export=ALL,COMPACT_FEATURE_MODE=eeg_summary,RUN_NAME=physio_compact_eeg train_physio_compact_shared.slurm
sbatch --export=ALL,COMPACT_FEATURE_MODE=psd_summary,RUN_NAME=physio_compact_psd train_physio_compact_shared.slurm
