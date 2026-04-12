import argparse
import os
import random

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    parser.add_argument('--clip_value', type=float, default=1.0)
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

    parser.add_argument('--downstream_dataset', type=str, default='FACED', choices=['FACED', 'SEED-V'])
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--num_of_classes', type=int, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
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
        choices=['compact_shared', 'block_shared_typed_proj'],
    )
    parser.add_argument('--moe_attnres_depth_block_count', type=int, default=4)
    parser.add_argument(
        '--moe_attnres_depth_summary_mode',
        type=str,
        default='auto',
        choices=['auto', 'attn_delta4', 'attn_mlp_balanced', 'attn_mlp_latemix'],
        help='Compact depth-summary composition mode (ignored by block_shared_typed_proj).',
    )
    parser.add_argument(
        '--moe_attnres_depth_probe_mlp_for_router',
        action='store_true',
        help='Use pre-MLP AttnRes alpha in compact summary mode (ignored by block_shared_typed_proj).',
    )
    parser.add_argument(
        '--moe_attnres_depth_router_init',
        type=str,
        default='xavier',
        choices=['xavier', 'zero'],
        help='Initialization for depth-summary router projections (spatial/spectral).',
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
    parser.add_argument('--moe_router_compact_feature_mode', type=str, default='none', choices=['none', 'eeg_summary', 'psd_summary'])
    parser.add_argument('--moe_router_compact_feature_dim', type=int, default=8)
    parser.add_argument('--moe_expert_init_noise_std', type=float, default=0.0)

    parser.add_argument('--use_component_lr', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--lr_backbone_mult', type=float, default=1.0)
    parser.add_argument('--lr_router_mult', type=float, default=1.0)
    parser.add_argument('--lr_expert_mult', type=float, default=1.0)
    parser.add_argument('--lr_classifier_mult', type=float, default=1.0)
    parser.add_argument('--lr_other_mult', type=float, default=1.0)

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
    if (
        args.moe_attnres_depth_context_mode == 'block_shared_typed_proj'
        and not args.moe_use_attnres_depth_router_features
    ):
        print(
            '[warn] block_shared_typed_proj selected but --moe_use_attnres_depth_router_features is off; '
            'depth block context will not be used by routers in this run.'
        )
    if args.moe_attnres_depth_context_mode == 'block_shared_typed_proj':
        if args.moe_attnres_depth_summary_mode != 'auto':
            print(
                '[warn] --moe_attnres_depth_summary_mode is ignored for '
                'block_shared_typed_proj; forcing auto.'
            )
            args.moe_attnres_depth_summary_mode = 'auto'
        if args.moe_attnres_depth_probe_mlp_for_router:
            print(
                '[warn] --moe_attnres_depth_probe_mlp_for_router is compact-summary-only; '
                'disabling it for block_shared_typed_proj.'
            )
            args.moe_attnres_depth_probe_mlp_for_router = False
    if args.moe_router_compact_feature_dim <= 0:
        raise ValueError('--moe_router_compact_feature_dim must be > 0.')
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

    for key in ['lr_backbone_mult', 'lr_router_mult', 'lr_expert_mult', 'lr_classifier_mult', 'lr_other_mult']:
        if getattr(args, key) <= 0:
            raise ValueError(f'--{key} must be > 0.')

    if args.downstream_dataset == 'FACED' and args.num_of_classes != 9:
        print(f"[FACED] warning: expected num_of_classes=9, got {args.num_of_classes}")
    if args.downstream_dataset == 'SEED-V' and args.num_of_classes != 5:
        print(f"[SEED-V] warning: expected num_of_classes=5, got {args.num_of_classes}")


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

    raise ValueError(f'Unsupported downstream_dataset: {args.downstream_dataset}')


def build_model(args: argparse.Namespace):
    if args.downstream_dataset == 'FACED':
        return model_for_faced.Model(args)
    if args.downstream_dataset == 'SEED-V':
        return model_for_seedv.Model(args)
    raise ValueError(f'Unsupported downstream_dataset: {args.downstream_dataset}')


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description='FACED + SEED-V finetuning')
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
    trainer.train_for_multiclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
