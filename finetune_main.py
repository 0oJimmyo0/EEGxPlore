import argparse
import random

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='MentalArithmetic',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/mental-arithmetic/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/data/wjq/models_weights/Big/BigFaced', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument(
        '--multi_lr',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='AdamW/SGD: backbone lr=--lr, head lr scaled like CBraMod (off: --no-multi_lr)',
    )
    parser.add_argument('--frozen', action='store_true',
                        help='frozen only if you are using pretrained weights')
    parser.add_argument('--use_pretrained_weights', action='store_true',
                    help='load pretrained CBraMod weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth',
                        help='foundation_dir')
    parser.add_argument('--attnres_final_output', type=str, default='attnres',
                    choices=['attnres', 'last_source'],
                    help='how to form the encoder output in AttnRes mode')
    
    # AttnRes ablation
    parser.add_argument(
        '--attnres_variant',
        type=str,
        default='none',
        choices=['none', 'final', 'pre_attn', 'pre_mlp', 'full'],
        help='Encoder path: none=original CBraMod stack; other values add FullAttnRes ablations. '
             'This is the only switch for AttnRes in finetuning (do not use a separate use_attnres flag). '
             'Finetuning uses the same optimizer/scheduler as none (see finetune_trainer).',
    )
    parser.add_argument(
        '--attnres_gated',
        action='store_true',
        help='use gated interpolation between baseline input and AttnRes input'
    )

    parser.add_argument(
        '--attnres_gate_init',
        type=float,
        default=0.0,
        help='Logit for pre-attn (and pre-mlp) gate before sigmoid. Very negative (e.g. -5) ≈ '
             'pure baseline stream early; with MoE, try -1..0 so AttnRes can actually contribute.'
    )

    parser.add_argument(
        '--attnres_subject_gates',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use subject/domain adapter conditioning to modulate AttnRes gates per sample. '
             'Only effective when --attnres_gated is enabled and adapter_mode is subject_domain.',
    )

    parser.add_argument(
    '--attnres_start_layer',
    type=int,
    default=0,
    help='first layer index that uses AttnRes; for 12 layers, 8 means top-4 only'
)

    parser.add_argument(
        '--adapter_mode',
        type=str,
        default='none',
        choices=['none', 'subject_domain'],
        help='Subject/domain adaptation mode. subject_domain enables lightweight conditioned adapters.',
    )
    parser.add_argument(
        '--adapter_num_layers',
        type=int,
        default=2,
        help='Number of top encoder layers with adapters.',
    )
    parser.add_argument(
        '--adapter_rank',
        type=int,
        default=16,
        help='Low-rank adapter bottleneck rank.',
    )
    parser.add_argument(
        '--adapter_cond_dim',
        type=int,
        default=32,
        help='Condition embedding dim for subject/domain adapters.',
    )
    parser.add_argument(
        '--adapter_scale',
        type=float,
        default=0.2,
        help='Residual scale for adapter outputs.',
    )
    parser.add_argument(
        '--adapter_only_update',
        action='store_true',
        help='Freeze backbone except adapter/context modules; useful for incremental subject updates.',
    )
    parser.add_argument(
        '--eeg_channel_context',
        action='store_true',
        help='Enable EEG channel/electrode context encoding in early backbone blocks.',
    )
    parser.add_argument(
        '--channel_context_file',
        type=str,
        default='',
        help='Optional .json/.pt/.pth/.xlsx with channel_ids/coords/montage_mask/region_ids.',
    )
    parser.add_argument(
        '--channel_id_align_mode',
        type=str,
        default='auto',
        choices=['auto', 'strict', 'off'],
        help='When loading channel_context_file: auto=align common ID conventions/reorder, '
             'strict=exact expected IDs/order, off=skip ID alignment checks.',
    )
    parser.add_argument(
        '--subject_summary_file',
        type=str,
        default='',
        help='Optional .json/.pt/.pth mapping subject id to compact summary vectors.',
    )
    parser.add_argument(
        '--use_subject_summary',
        action='store_true',
        help='Enable subject_summary conditioning in subject/domain adapters.',
    )
    parser.add_argument(
        '--subject_summary_handling',
        type=str,
        default='project',
        choices=['project', 'error'],
        help='How to handle subject_summary dim mismatch vs adapter_cond_dim.',
    )
    parser.add_argument(
        '--metadata_debug',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Log one-time metadata produced/consumed summaries at runtime.',
    )
    parser.add_argument(
        '--adapter_use_age_bucket',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If adapters are enabled, include age_bucket_id in adapter conditioning.',
    )
    parser.add_argument(
        '--adapter_use_subject_id',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If adapters are enabled, include subject_id in adapter conditioning.',
    )
    parser.add_argument(
        '--adapter_use_segment_bucket',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If adapters are enabled, include segment_bucket_id in adapter conditioning. '
             'Default is off to avoid segment-position shortcut leakage.',
    )
    parser.add_argument(
        '--subject_overlap_policy',
        type=str,
        default='disable',
        choices=['error', 'disable', 'allow'],
        help='If subject overlap is detected across train/val/test and adapter_use_subject_id=True: '
             'error=fail, disable=auto-disable subject_id conditioning, allow=keep it enabled.',
    )
    parser.add_argument(
        '--continual_mode',
        type=str,
        default='off',
        choices=['off', 'replay', 'replay_distill'],
        help='Continual-learning mode: off, replay memory, or replay + distillation regularization.',
    )
    parser.add_argument(
        '--continual_memory_size',
        type=int,
        default=512,
        help='Replay memory budget in number of samples.',
    )
    parser.add_argument(
        '--continual_replay_batch_size',
        type=int,
        default=32,
        help='Replay samples per optimization step.',
    )
    parser.add_argument(
        '--continual_replay_weight',
        type=float,
        default=0.5,
        help='Weight for replay supervised loss.',
    )
    parser.add_argument(
        '--continual_distill_weight',
        type=float,
        default=0.2,
        help='Weight for replay distillation loss (replay_distill mode).',
    )
    parser.add_argument(
        '--continual_distill_temp',
        type=float,
        default=2.0,
        help='Temperature for multiclass replay distillation.',
    )

    # Deprecated/legacy MoE path retained for backward compatibility.
    parser.add_argument(
        '--moe',
        action='store_true',
        help='[Deprecated] Enable typed routed MoE in top encoder layers.',
    )
    parser.add_argument(
        '--moe_num_layers',
        type=int,
        default=2,
        help='Number of top Transformer layers that use MoE (1–12; default 2)',
    )
    parser.add_argument(
        '--moe_num_experts',
        type=int,
        default=4,
        help='Experts per MoE layer for spatial/spectral specialist banks.',
    )
    parser.add_argument(
        '--moe_route_mode',
        type=str,
        default='typed_capacity_domain',
        choices=['typed_capacity_domain'],
        help='Routed MoE mode (single supported mode in this refactor).',
    )
    parser.add_argument(
        '--moe_capacity_factor',
        type=float,
        default=1.0,
        help='Per-expert capacity factor for top-1 routing (capacity=ceil(factor*B/E)).',
    )
    parser.add_argument(
        '--moe_load_balance',
        type=float,
        default=0.005,
        help='Load-balance auxiliary coefficient (0=off).',
    )
    parser.add_argument(
        '--moe_domain_bias',
        action='store_true',
        help='Enable additive domain-aware logit bias (zero-init).',
    )
    parser.add_argument(
        '--moe_domain_emb_dim',
        type=int,
        default=16,
        help='Domain metadata embedding dim for domain-aware routing bias.',
    )
    parser.add_argument(
        '--moe_domain_bias_reg',
        type=float,
        default=0.0,
        help='Domain-bias regularization coefficient (0=off).',
    )
    parser.add_argument(
        '--moe_diagnostics',
        action='store_true',
        help='After each val epoch, log MoE routing/capacity diagnostics (one val batch, eval mode).',
    )
    parser.add_argument(
        '--moe_use_psd_router_features',
        action='store_true',
        help='Append PSD features to the spectral bank router only.',
    )
    parser.add_argument(
        '--moe_use_adapter_cond_bias',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='When MoE and subject adapters are both enabled, use adapter conditioning as a zero-init router bias.',
    )
    parser.add_argument(
        '--moe_use_subject_summary_router_concat',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Concat projected subject_summary features into both spatial/spectral router inputs.',
    )
    parser.add_argument(
        '--moe_use_eeg_summary_router_concat_spatial',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Concat compact EEG-context summary features into the spatial router input.',
    )
    parser.add_argument(
        '--moe_use_eeg_summary_router_concat_spectral',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Concat compact EEG-context summary features into the spectral router input.',
    )
    parser.add_argument(
        '--moe_linear_router_input_norm',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Apply LayerNorm on router inputs when router_arch=linear.',
    )
    parser.add_argument(
        '--moe_router_temperature',
        type=float,
        default=1.0,
        help='Temperature for router logits (logits / temperature). >1.0 softens routing.',
    )
    parser.add_argument(
        '--moe_router_entropy_coef',
        type=float,
        default=0.0,
        help='Entropy regularization coefficient for router probs. Positive encourages higher entropy.',
    )
    parser.add_argument(
        '--moe_router_balance_kl_coef',
        type=float,
        default=0.0,
        help='Batch-level KL(mean_router_probs || uniform) coefficient. Positive discourages one-expert collapse.',
    )
    parser.add_argument(
        '--moe_router_z_loss_coef',
        type=float,
        default=0.0,
        help='Router z-loss coefficient to suppress logit explosion (Switch-style logsumexp penalty).',
    )
    parser.add_argument(
        '--moe_router_jitter_std',
        type=float,
        default=0.0,
        help='Std of Gaussian jitter added to router logits during training only (0 disables).',
    )
    parser.add_argument(
        '--moe_router_jitter_final_std',
        type=float,
        default=0.0,
        help='Final router jitter std after annealing (used with --moe_router_jitter_anneal_epochs).',
    )
    parser.add_argument(
        '--moe_router_jitter_anneal_epochs',
        type=int,
        default=0,
        help='Linear jitter anneal epochs from start std to final std (0 keeps fixed jitter).',
    )
    parser.add_argument(
        '--moe_router_soft_warmup_epochs',
        type=int,
        default=0,
        help='Use soft-routing warmup for first N training epochs (0 disables).',
    )
    parser.add_argument(
        '--moe_router_warmup_mode',
        type=str,
        default='off',
        choices=['off', 'freeze', 'low_lr'],
        help='Router parameter warmup policy for first N epochs.',
    )
    parser.add_argument(
        '--moe_router_warmup_epochs',
        type=int,
        default=0,
        help='Number of epochs for router freeze/low_lr warmup.',
    )
    parser.add_argument(
        '--moe_router_warmup_lr_scale',
        type=float,
        default=0.1,
        help='Gradient scale for router params during low_lr warmup mode.',
    )

    parser.add_argument(
        '--tqdm',
        dest='use_tqdm',
        action='store_true',
        help='Force tqdm progress bars even when stderr is not a TTY (e.g. batch logs).',
    )
    parser.add_argument(
        '--no-tqdm',
        dest='use_tqdm',
        action='store_false',
        help='Disable tqdm (recommended on cluster to avoid quota issues from progress bar writes).',
    )
    parser.set_defaults(use_tqdm=None)

    parser.add_argument(
        '--routing_export_dir',
        type=str,
        default='',
        help='FACED typed MoE: after training, write per-sample routing CSVs under this directory (empty=off).',
    )
    parser.add_argument(
        '--routing_export_splits',
        type=str,
        default='test',
        help='Comma-separated: val, test (default: test only).',
    )
    parser.add_argument(
        '--faced_meta_csv',
        type=str,
        default='/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv',
        help='Recording_info.csv used for FACED domain ids (routing) and export joins.',
    )
    parser.add_argument(
        '--routing_run_name',
        type=str,
        default='',
        help='Optional label stored in per-sample routing CSV (e.g. Slurm job id); analysis only.',
    )

    params = parser.parse_args()
    params.subject_adapter = (params.adapter_mode != 'none')
    if params.moe:
        print(
            "[warning] --moe is deprecated; prefer --adapter_mode subject_domain unless running controlled MoE ablations.",
            flush=True,
        )
    if params.moe and params.subject_adapter:
        print(
            "[warning] Running hybrid MoE+adapter mode. "
            "Adapters are now applied after MoE in selected top layers.",
            flush=True,
        )
    if params.moe_use_adapter_cond_bias and not (params.moe and params.subject_adapter):
        raise ValueError("--moe_use_adapter_cond_bias requires both --moe and adapter_mode subject_domain.")
    if params.moe_router_temperature <= 0:
        raise ValueError("--moe_router_temperature must be > 0.")
    if params.moe_router_balance_kl_coef < 0:
        raise ValueError("--moe_router_balance_kl_coef must be >= 0.")
    if params.moe_router_z_loss_coef < 0:
        raise ValueError("--moe_router_z_loss_coef must be >= 0.")
    if params.moe_router_jitter_std < 0:
        raise ValueError("--moe_router_jitter_std must be >= 0.")
    if params.moe_router_jitter_final_std < 0:
        raise ValueError("--moe_router_jitter_final_std must be >= 0.")
    if params.moe_router_jitter_anneal_epochs < 0:
        raise ValueError("--moe_router_jitter_anneal_epochs must be >= 0.")
    if params.moe_router_soft_warmup_epochs < 0:
        raise ValueError("--moe_router_soft_warmup_epochs must be >= 0.")
    if params.moe_router_warmup_epochs < 0:
        raise ValueError("--moe_router_warmup_epochs must be >= 0.")
    if params.moe_router_warmup_lr_scale <= 0:
        raise ValueError("--moe_router_warmup_lr_scale must be > 0.")
    if (
        params.moe
        and params.moe_router_warmup_mode == 'freeze'
        and params.moe_router_soft_warmup_epochs > 0
    ):
        print(
            "[warning] moe_router_warmup_mode=freeze with moe_router_soft_warmup_epochs>0 can lock"
            " router probs near initialization during soft warmup. Prefer warmup_mode=low_lr for"
            " first anti-collapse sweeps.",
            flush=True,
        )
    if params.moe_use_subject_summary_router_concat and not params.moe:
        raise ValueError("--moe_use_subject_summary_router_concat requires --moe.")
    if params.moe_use_subject_summary_router_concat and not params.use_subject_summary:
        raise ValueError(
            "--moe_use_subject_summary_router_concat requires --use_subject_summary and --subject_summary_file."
        )
    if (
        (params.moe_use_eeg_summary_router_concat_spatial or params.moe_use_eeg_summary_router_concat_spectral)
        and not params.moe
    ):
        raise ValueError("EEG summary router concat flags require --moe.")
    if (
        params.moe_use_eeg_summary_router_concat_spatial or params.moe_use_eeg_summary_router_concat_spectral
    ) and not params.eeg_channel_context:
        raise ValueError(
            "EEG summary router concat flags require --eeg_channel_context and a valid --channel_context_file."
        )
    if params.use_subject_summary and not params.subject_adapter:
        raise ValueError("--use_subject_summary requires adapter_mode subject_domain.")
    if params.use_subject_summary and not params.subject_summary_file:
        raise ValueError("--use_subject_summary requires --subject_summary_file.")
    if params.adapter_only_update and not params.subject_adapter:
        raise ValueError("--adapter_only_update requires adapter_mode subject_domain.")
    if params.adapter_use_segment_bucket:
        print(
            "[warning] adapter_use_segment_bucket=True can introduce segment-position shortcut leakage on FACED. "
            "Use only for controlled ablation.",
            flush=True,
        )
    if params.adapter_use_subject_id and params.subject_overlap_policy == 'allow':
        print(
            "[warning] subject_id conditioning is enabled with subject_overlap_policy=allow. "
            "If splits overlap by subject, identity shortcut risk remains.",
            flush=True,
        )
    if params.attnres_subject_gates and not params.attnres_gated:
        print(
            "[warning] attnres_subject_gates=True but attnres_gated=False. "
            "Subject-conditioned gates are disabled unless --attnres_gated is enabled.",
            flush=True,
        )
    print(params)
    print(
        "[ablation-config] "
        f"attnres_variant={params.attnres_variant} "
        f"attnres_gated={params.attnres_gated} "
        f"attnres_subject_gates={params.attnres_subject_gates} "
        f"eeg_channel_context={params.eeg_channel_context} "
        f"subject_adapters={params.subject_adapter} "
        f"use_subject_summary={params.use_subject_summary} "
        f"adapter_use_subject_id={params.adapter_use_subject_id} "
        f"adapter_use_age_bucket={params.adapter_use_age_bucket} "
        f"adapter_use_segment_bucket={params.adapter_use_segment_bucket} "
        f"subject_overlap_policy={params.subject_overlap_policy} "
        f"channel_id_align_mode={params.channel_id_align_mode} "
        f"continual_mode={params.continual_mode} "
        f"adapter_only_update={params.adapter_only_update} "
        f"moe={params.moe} "
        f"moe_use_adapter_cond_bias={params.moe_use_adapter_cond_bias} "
        f"moe_use_subject_summary_router_concat={params.moe_use_subject_summary_router_concat} "
        f"moe_use_eeg_summary_router_concat_spatial={params.moe_use_eeg_summary_router_concat_spatial} "
        f"moe_use_eeg_summary_router_concat_spectral={params.moe_use_eeg_summary_router_concat_spectral} "
        f"moe_linear_router_input_norm={params.moe_linear_router_input_norm} "
        f"moe_router_temperature={params.moe_router_temperature} "
        f"moe_router_entropy_coef={params.moe_router_entropy_coef} "
        f"moe_router_balance_kl_coef={params.moe_router_balance_kl_coef} "
        f"moe_router_z_loss_coef={params.moe_router_z_loss_coef} "
        f"moe_router_jitter_std={params.moe_router_jitter_std} "
        f"moe_router_jitter_final_std={params.moe_router_jitter_final_std} "
        f"moe_router_jitter_anneal_epochs={params.moe_router_jitter_anneal_epochs} "
        f"moe_router_soft_warmup_epochs={params.moe_router_soft_warmup_epochs} "
        f"moe_router_warmup_mode={params.moe_router_warmup_mode} "
        f"moe_router_warmup_epochs={params.moe_router_warmup_epochs} "
        f"moe_router_warmup_lr_scale={params.moe_router_warmup_lr_scale}",
        flush=True,
    )

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        params.return_sample_keys = bool(getattr(params, 'routing_export_dir', None))
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_physio.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mumtaz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedvig.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        load_dataset = bciciv2a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_bciciv2a.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
