import argparse
import os
import random

import numpy as np
import torch

from datasets import faced_dataset, isruc_dataset, mumtaz_dataset, physio_dataset, seedv_dataset, tuev_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_isruc, model_for_mumtaz, model_for_physio, model_for_seedv, model_for_tuev


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument(
        '--use_ema',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable exponential moving average tracking of trainable model parameters.',
    )
    parser.add_argument(
        '--ema_decay',
        type=float,
        default=0.999,
        help='EMA decay factor in [0, 1).',
    )
    parser.add_argument(
        '--ema_warmup_steps',
        type=int,
        default=300,
        help='Number of optimizer steps before EMA updates start.',
    )
    parser.add_argument(
        '--ema_eval_only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true, select best epoch using EMA validation metrics when EMA is available.',
    )
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument(
        '--classifier',
        type=str,
        default='all_patch_reps',
        choices=[
            'all_patch_reps',
            'all_patch_reps_twolayer',
            'all_patch_reps_onelayer',
            'avgpooling_patch_reps',
        ],
    )

    parser.add_argument(
        '--downstream_dataset',
        type=str,
        default='FACED',
        choices=['FACED', 'SEED-V', 'ISRUC', 'PhysioNet-MI', 'Mumtaz2016', 'TUEV'],
    )
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--num_of_classes', type=int, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument(
        '--class_weight_mode',
        type=str,
        default='none',
        choices=['none', 'inv_freq_clip', 'effective_num'],
        help='Optional class-weighted CE mode. Weights are computed from train split only.',
    )
    parser.add_argument(
        '--class_weight_clip_min',
        type=float,
        default=0.75,
        help='Lower bound for clipped inverse-frequency class weights.',
    )
    parser.add_argument(
        '--class_weight_clip_max',
        type=float,
        default=1.5,
        help='Upper bound for clipped inverse-frequency class weights.',
    )
    parser.add_argument(
        '--effective_num_beta',
        type=float,
        default=0.999,
        help='Beta for effective-number class weights: (1-beta)/(1-beta^n).',
    )
    parser.add_argument(
        '--multi_lr',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Backbone/head dual-LR mode used by prior FACED runs.',
    )
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--use_pretrained_weights', action='store_true')
    parser.add_argument(
        '--foundation_dir',
        type=str,
        default='/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/others/pretrained_weights.pth',
    )

    parser.add_argument('--attnres_final_output', type=str, default='attnres', choices=['attnres', 'last_source'])
    parser.add_argument(
        '--attnres_variant',
        type=str,
        default='none',
        choices=['none', 'final', 'pre_attn', 'pre_mlp', 'full'],
    )
    parser.add_argument('--attnres_gated', action='store_true')
    parser.add_argument('--attnres_gate_init', type=float, default=0.0)
    parser.add_argument('--attnres_start_layer', type=int, default=0)

    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--moe_num_layers', type=int, default=2)
    parser.add_argument('--moe_num_experts', type=int, default=4)
    parser.add_argument('--moe_route_mode', type=str, default='typed_capacity_domain', choices=['typed_capacity_domain'])
    parser.add_argument('--moe_capacity_factor', type=float, default=1.0)
    parser.add_argument('--moe_load_balance', type=float, default=0.005)
    parser.add_argument('--moe_domain_bias', action='store_true')
    parser.add_argument('--moe_domain_emb_dim', type=int, default=16)
    parser.add_argument('--moe_domain_bias_reg', type=float, default=0.0)
    parser.add_argument('--moe_diagnostics', action='store_true')
    parser.add_argument('--moe_use_psd_router_features', action='store_true')
    parser.add_argument('--moe_use_attnres_depth_router_features', action='store_true')
    parser.add_argument('--moe_attnres_depth_router_dim', type=int, default=26)
    parser.add_argument(
        '--moe_attnres_depth_context_mode',
        type=str,
        default='compact_shared',
        choices=['compact_shared', 'block_shared_typed_proj', 'dual_query_block_typed_proj'],
    )
    parser.add_argument('--moe_attnres_depth_block_count', type=int, default=4)
    parser.add_argument(
        '--moe_attnres_depth_summary_mode',
        type=str,
        default='auto',
        choices=['auto', 'attn_delta4', 'attn_mlp_balanced', 'attn_mlp_latemix'],
        help='Compact depth-summary composition mode (ignored by block_shared_typed_proj and dual_query_block_typed_proj).',
    )
    parser.add_argument(
        '--moe_attnres_depth_probe_mlp_for_router',
        action='store_true',
        help='Use pre-MLP AttnRes alpha in compact summary mode (ignored by block_shared_typed_proj and dual_query_block_typed_proj).',
    )
    parser.add_argument(
        '--moe_attnres_depth_router_init',
        type=str,
        default='xavier',
        choices=['xavier', 'zero'],
        help='Initialization for depth-summary router projections (spatial/spectral).',
    )
    parser.add_argument(
        '--moe_attnres_depth_router_norm_gate',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply RMSNorm + small learned gates to dual-query depth summaries before router projection.',
    )
    parser.add_argument(
        '--moe_attnres_depth_router_gate_init',
        type=float,
        default=0.075,
        help='Initial gate value for dual-query depth summary stabilization (recommended 0.05-0.1).',
    )
    parser.add_argument(
        '--moe_attnres_depth_router_norm_eps',
        type=float,
        default=1e-6,
        help='Epsilon for RMS normalization applied before depth summary router projection.',
    )
    parser.add_argument(
        '--moe_attnres_depth_block_separation_coef',
        type=float,
        default=0.0,
        help='Mild anti-collapse regularizer coefficient for hinge penalty on depth-block JS separation.',
    )
    parser.add_argument(
        '--moe_attnres_depth_block_separation_target_js',
        type=float,
        default=0.03,
        help='Target JS floor for hinge-style depth block separation regularizer.',
    )
    parser.add_argument(
        '--moe_attnres_depth_summary_grad_mode',
        type=str,
        default='delayed_unfreeze',
        choices=['detached', 'delayed_unfreeze', 'trainable'],
    )
    parser.add_argument('--moe_attnres_depth_summary_unfreeze_epoch', type=int, default=8)
    parser.add_argument('--moe_router_arch', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--moe_router_mlp_hidden', type=int, default=128)
    parser.add_argument('--moe_router_dispatch_mode', type=str, default='hard_capacity', choices=['hard_capacity', 'soft'])
    parser.add_argument('--moe_router_temperature', type=float, default=1.0)
    parser.add_argument('--moe_router_entropy_coef', type=float, default=0.0)
    parser.add_argument('--moe_router_balance_kl_coef', type=float, default=0.0)
    parser.add_argument('--moe_router_z_loss_coef', type=float, default=0.0)
    parser.add_argument('--moe_router_jitter_std', type=float, default=0.0)
    parser.add_argument('--moe_router_jitter_final_std', type=float, default=0.0)
    parser.add_argument('--moe_router_jitter_anneal_epochs', type=int, default=0)
    parser.add_argument('--moe_router_soft_warmup_epochs', type=int, default=0)

    parser.add_argument('--moe_uniform_dispatch_warmup_epochs', type=int, default=0)
    parser.add_argument('--moe_shared_blend_warmup_epochs', type=int, default=0)
    parser.add_argument('--moe_shared_blend_start', type=float, default=1.0)
    parser.add_argument('--moe_shared_blend_end', type=float, default=0.0)
    parser.add_argument('--moe_router_entropy_coef_spatial', type=float, default=-1.0)
    parser.add_argument('--moe_router_entropy_coef_spectral', type=float, default=-1.0)
    parser.add_argument('--moe_router_balance_kl_coef_spatial', type=float, default=-1.0)
    parser.add_argument('--moe_router_balance_kl_coef_spectral', type=float, default=-1.0)
    parser.add_argument('--moe_specialist_branch_mode', type=str, default='both', choices=['both', 'spatial_only', 'spectral_only'])
    parser.add_argument('--moe_router_compact_feature_mode', type=str, default='eeg_summary', choices=['none', 'eeg_summary', 'psd_summary'])
    parser.add_argument('--moe_router_compact_feature_dim', type=int, default=8)
    parser.add_argument('--moe_router_compact_warmup_epochs', type=int, default=0)
    parser.add_argument('--moe_router_compact_gate_init', type=float, default=1.0)
    parser.add_argument('--moe_expert_init_noise_std', type=float, default=0.0)

    parser.add_argument('--use_component_lr', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--lr_backbone_mult', type=float, default=0.5)
    parser.add_argument('--lr_router_mult', type=float, default=2.0)
    parser.add_argument('--lr_expert_mult', type=float, default=1.5)
    parser.add_argument('--lr_classifier_mult', type=float, default=1.0)
    parser.add_argument('--lr_other_mult', type=float, default=1.0)
    parser.add_argument(
        '--selection_metric',
        type=str,
        default='kappa',
        choices=['kappa', 'balanced_accuracy', 'weighted_f1'],
        help='Validation metric used to select the primary multiclass checkpoint.',
    )

    parser.add_argument('--tqdm', dest='use_tqdm', action='store_true')
    parser.add_argument('--no-tqdm', dest='use_tqdm', action='store_false')
    parser.set_defaults(use_tqdm=None)


def add_faced_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--routing_export_dir', type=str, default='')
    parser.add_argument('--routing_export_splits', type=str, default='test')
    parser.add_argument(
        '--faced_meta_csv',
        type=str,
        default='/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/CLEEG/data/faced_data_info/FACED_meta/Recording_info.csv',
    )
    parser.add_argument('--routing_run_name', type=str, default='')


def add_seedv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--seedv_split_manifest',
        type=str,
        default='',
        help='Optional JSON/PKL split manifest override for SEED-V keys. Empty means use LMDB __keys__ (CBraMod cohort).',
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.min_lr <= 0:
        raise ValueError('--min_lr must be > 0.')
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError('--ema_decay must satisfy 0 <= ema_decay < 1.')
    if args.ema_warmup_steps < 0:
        raise ValueError('--ema_warmup_steps must be >= 0.')
    if args.min_lr > args.lr:
        print(
            f"[warn] --min_lr ({args.min_lr}) is higher than --lr ({args.lr}); "
            'cosine LR schedule will increase toward eta_min floor behavior.'
        )
    if args.moe_router_temperature <= 0:
        raise ValueError('--moe_router_temperature must be > 0.')
    if args.moe_router_mlp_hidden <= 0:
        raise ValueError('--moe_router_mlp_hidden must be > 0.')
    if args.moe_router_balance_kl_coef < 0:
        raise ValueError('--moe_router_balance_kl_coef must be >= 0.')
    if args.moe_router_z_loss_coef < 0:
        raise ValueError('--moe_router_z_loss_coef must be >= 0.')
    if args.moe_router_jitter_std < 0:
        raise ValueError('--moe_router_jitter_std must be >= 0.')
    if args.moe_router_jitter_final_std < 0:
        raise ValueError('--moe_router_jitter_final_std must be >= 0.')
    if args.moe_router_jitter_anneal_epochs < 0:
        raise ValueError('--moe_router_jitter_anneal_epochs must be >= 0.')
    if args.moe_router_soft_warmup_epochs < 0:
        raise ValueError('--moe_router_soft_warmup_epochs must be >= 0.')
    if args.moe_uniform_dispatch_warmup_epochs < 0:
        raise ValueError('--moe_uniform_dispatch_warmup_epochs must be >= 0.')
    if args.moe_shared_blend_warmup_epochs < 0:
        raise ValueError('--moe_shared_blend_warmup_epochs must be >= 0.')
    if not (0.0 <= args.moe_shared_blend_start <= 1.0):
        raise ValueError('--moe_shared_blend_start must be in [0, 1].')
    if not (0.0 <= args.moe_shared_blend_end <= 1.0):
        raise ValueError('--moe_shared_blend_end must be in [0, 1].')
    if args.moe_attnres_depth_summary_unfreeze_epoch < 1:
        raise ValueError('--moe_attnres_depth_summary_unfreeze_epoch must be >= 1.')
    if (
        args.moe_attnres_depth_summary_grad_mode == 'delayed_unfreeze'
        and args.moe_attnres_depth_summary_unfreeze_epoch > args.epochs
    ):
        print(
            '[warn] delayed_unfreeze is enabled but '
            f'--moe_attnres_depth_summary_unfreeze_epoch={args.moe_attnres_depth_summary_unfreeze_epoch} '
            f'exceeds --epochs={args.epochs}; depth summary will remain detached for this run.'
        )
    if args.moe_attnres_depth_router_dim <= 0:
        raise ValueError('--moe_attnres_depth_router_dim must be > 0.')
    if args.moe_attnres_depth_block_count < 1:
        raise ValueError('--moe_attnres_depth_block_count must be >= 1.')
    if args.moe_attnres_depth_router_gate_init <= 0:
        raise ValueError('--moe_attnres_depth_router_gate_init must be > 0.')
    if args.moe_attnres_depth_router_norm_eps <= 0:
        raise ValueError('--moe_attnres_depth_router_norm_eps must be > 0.')
    if args.moe_attnres_depth_block_separation_coef < 0:
        raise ValueError('--moe_attnres_depth_block_separation_coef must be >= 0.')
    if args.moe_attnres_depth_block_separation_target_js < 0:
        raise ValueError('--moe_attnres_depth_block_separation_target_js must be >= 0.')
    typed_block_modes = {'block_shared_typed_proj', 'dual_query_block_typed_proj'}
    if (
        args.moe_attnres_depth_context_mode in typed_block_modes
        and not args.moe_use_attnres_depth_router_features
    ):
        print(
            f"[warn] {args.moe_attnres_depth_context_mode} selected but --moe_use_attnres_depth_router_features is off; "
            'depth block context will not be used by routers in this run.'
        )
    if args.moe_attnres_depth_context_mode in typed_block_modes:
        if args.moe_attnres_depth_summary_mode != 'auto':
            print(
                '[warn] --moe_attnres_depth_summary_mode is ignored for '
                f'{args.moe_attnres_depth_context_mode}; forcing auto.'
            )
            args.moe_attnres_depth_summary_mode = 'auto'
        if args.moe_attnres_depth_probe_mlp_for_router:
            print(
                '[warn] --moe_attnres_depth_probe_mlp_for_router is compact-summary-only; '
                f'disabling it for {args.moe_attnres_depth_context_mode}.'
            )
            args.moe_attnres_depth_probe_mlp_for_router = False
    if args.moe_router_compact_feature_dim <= 0:
        raise ValueError('--moe_router_compact_feature_dim must be > 0.')
    if args.moe_router_compact_warmup_epochs < 0:
        raise ValueError('--moe_router_compact_warmup_epochs must be >= 0.')
    if args.moe_router_compact_gate_init <= 0:
        raise ValueError('--moe_router_compact_gate_init must be > 0.')
    if args.moe_expert_init_noise_std < 0:
        raise ValueError('--moe_expert_init_noise_std must be >= 0.')
    if args.moe_router_entropy_coef_spatial < 0 and args.moe_router_entropy_coef_spatial != -1.0:
        raise ValueError('--moe_router_entropy_coef_spatial must be >= 0 or -1 for fallback.')
    if args.moe_router_entropy_coef_spectral < 0 and args.moe_router_entropy_coef_spectral != -1.0:
        raise ValueError('--moe_router_entropy_coef_spectral must be >= 0 or -1 for fallback.')
    if args.moe_router_balance_kl_coef_spatial < 0 and args.moe_router_balance_kl_coef_spatial != -1.0:
        raise ValueError('--moe_router_balance_kl_coef_spatial must be >= 0 or -1 for fallback.')
    if args.moe_router_balance_kl_coef_spectral < 0 and args.moe_router_balance_kl_coef_spectral != -1.0:
        raise ValueError('--moe_router_balance_kl_coef_spectral must be >= 0 or -1 for fallback.')
    if args.class_weight_clip_min <= 0:
        raise ValueError('--class_weight_clip_min must be > 0.')
    if args.class_weight_clip_max < args.class_weight_clip_min:
        raise ValueError('--class_weight_clip_max must be >= --class_weight_clip_min.')
    if args.effective_num_beta < 0.0 or args.effective_num_beta >= 1.0:
        raise ValueError('--effective_num_beta must satisfy 0 <= effective_num_beta < 1.')

    for key in ['lr_backbone_mult', 'lr_router_mult', 'lr_expert_mult', 'lr_classifier_mult', 'lr_other_mult']:
        if getattr(args, key) <= 0:
            raise ValueError(f'--{key} must be > 0.')

    if args.downstream_dataset == 'FACED' and args.num_of_classes != 9:
        print(f"[FACED] warning: expected num_of_classes=9, got {args.num_of_classes}")
    if args.downstream_dataset == 'SEED-V' and args.num_of_classes != 5:
        print(f"[SEED-V] warning: expected num_of_classes=5, got {args.num_of_classes}")
    if args.downstream_dataset == 'ISRUC' and args.num_of_classes != 5:
        print(f"[ISRUC] warning: expected num_of_classes=5 for standard sleep staging, got {args.num_of_classes}")
    if args.downstream_dataset == 'Mumtaz2016' and args.num_of_classes != 2:
        print(f"[Mumtaz2016] warning: expected num_of_classes=2 for MDD-vs-control, got {args.num_of_classes}")
    if args.downstream_dataset == 'TUEV' and args.num_of_classes != 6:
        print(f"[TUEV] warning: expected num_of_classes=6 for event classification, got {args.num_of_classes}")


def build_dataset(args: argparse.Namespace):
    args.return_sample_keys = False
    args.return_domain_ids = False

    if args.downstream_dataset == 'FACED':
        args.return_sample_keys = bool(getattr(args, 'routing_export_dir', None))
        args.return_domain_ids = bool(getattr(args, 'moe', False) and args.moe_route_mode == 'typed_capacity_domain')
        return faced_dataset.LoadDataset(args).get_data_loader()

    if args.downstream_dataset == 'SEED-V':
        args.return_sample_keys = bool(getattr(args, 'routing_export_dir', None))
        if args.moe_domain_bias:
            print('[SEED-V] warning: --moe_domain_bias enabled without FACED metadata; zero/unknown ids will be used.')
        if args.return_sample_keys:
            print('[SEED-V] routing_export_dir is set; per-sample export is FACED-only and will be skipped.')

        if args.seedv_split_manifest:
            print(
                f"[SEED-V] using split manifest override: {args.seedv_split_manifest} "
                '(this differs from CBraMod LMDB __keys__ cohort)'
            )
        else:
            print('[SEED-V] using LMDB __keys__ train/val/test split (CBraMod cohort).')

        return seedv_dataset.LoadDataset(args).get_data_loader()

    if args.downstream_dataset == 'ISRUC':
        args.return_sample_keys = False
        args.return_domain_ids = False
        if args.moe_domain_bias:
            print('[ISRUC] warning: --moe_domain_bias is ignored; ISRUC loader does not provide domain metadata.')
        if getattr(args, 'routing_export_dir', ''):
            print('[ISRUC] warning: routing_export_dir is FACED-only; ISRUC run will skip per-sample routing export.')
        return isruc_dataset.LoadDataset(args).get_data_loader()

    if args.downstream_dataset == 'PhysioNet-MI':
        args.return_sample_keys = False
        args.return_domain_ids = False
        print('[PhysioNet-MI] using physio_dataset.LoadDataset in selective-adaptation finetune pipeline.')
        return physio_dataset.LoadDataset(args).get_data_loader()

    if args.downstream_dataset == 'Mumtaz2016':
        args.return_sample_keys = False
        args.return_domain_ids = False
        if args.moe_domain_bias:
            print('[Mumtaz2016] warning: --moe_domain_bias is ignored; Mumtaz loader does not provide domain metadata.')
        if getattr(args, 'routing_export_dir', ''):
            print('[Mumtaz2016] warning: routing_export_dir is FACED-only; Mumtaz run will skip per-sample routing export.')
        return mumtaz_dataset.LoadDataset(args).get_data_loader()

    if args.downstream_dataset == 'TUEV':
        args.return_sample_keys = False
        args.return_domain_ids = False
        if args.moe_domain_bias:
            print('[TUEV] warning: --moe_domain_bias is ignored; TUEV loader does not provide domain metadata.')
        if getattr(args, 'routing_export_dir', ''):
            print('[TUEV] warning: routing_export_dir is FACED-only; TUEV run will skip per-sample routing export.')
        return tuev_dataset.LoadDataset(args).get_data_loader()

    raise ValueError(f'Unsupported downstream_dataset: {args.downstream_dataset}')


def build_model(args: argparse.Namespace):
    if args.downstream_dataset == 'FACED':
        return model_for_faced.Model(args)
    if args.downstream_dataset == 'SEED-V':
        return model_for_seedv.Model(args)
    if args.downstream_dataset == 'ISRUC':
        return model_for_isruc.Model(args)
    if args.downstream_dataset == 'PhysioNet-MI':
        print('[PhysioNet-MI] building model_for_physio.Model')
        return model_for_physio.Model(args)
    if args.downstream_dataset == 'Mumtaz2016':
        print('[Mumtaz2016] building model_for_mumtaz.Model')
        return model_for_mumtaz.Model(args)
    if args.downstream_dataset == 'TUEV':
        print('[TUEV] building model_for_tuev.Model')
        return model_for_tuev.Model(args)
    raise ValueError(f'Unsupported downstream_dataset: {args.downstream_dataset}')


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description='FACED + SEED-V + ISRUC + PhysioNet-MI + Mumtaz2016 + TUEV finetuning')
    add_shared_args(parser)
    add_faced_args(parser)
    add_seedv_args(parser)
    args = parser.parse_args()

    validate_args(args)
    print(args)

    setup_seed(args.seed)
    torch.cuda.set_device(args.cuda)
    print(f'The downstream dataset is {args.downstream_dataset}')

    data_loader = build_dataset(args)
    model = build_model(args)
    trainer = Trainer(args, data_loader, model)
    if args.downstream_dataset == 'Mumtaz2016' and args.num_of_classes == 2:
        print('[Mumtaz2016] using binary training loop (AUROC-oriented selection).')
        trainer.train_for_binaryclass()
    else:
        trainer.train_for_multiclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
