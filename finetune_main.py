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
    parser.add_argument('--seedv_split_manifest', type=str, default='',
                        help='SEED-V only: optional JSON/PKL split manifest with train/val/test key lists. '
                             'If empty, use LMDB __keys__.')
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
    '--attnres_start_layer',
    type=int,
    default=0,
    help='first layer index that uses AttnRes; for 12 layers, 8 means top-4 only'
)

    parser.add_argument(
        '--moe',
        action='store_true',
        help='Enable typed routed MoE in the top encoder layers.',
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
        '--moe_use_attnres_depth_router_features',
        action='store_true',
        help='Append AttnRes depth-selection summary features to both bank routers.',
    )
    parser.add_argument(
        '--moe_attnres_depth_router_dim',
        type=int,
        default=26,
        help='Projected dim for AttnRes depth-summary router features when enabled (26 keeps full feature set).',
    )
    parser.add_argument(
        '--moe_attnres_depth_summary_mode',
        type=str,
        default='auto',
        choices=['auto', 'attn_delta4', 'attn_mlp_balanced', 'attn_mlp_latemix'],
        help='How to compose AttnRes depth summary features before projection. '
             'For dim=15: attn_delta4 (current best-compatible), attn_mlp_balanced, attn_mlp_latemix.',
    )
    parser.add_argument(
        '--moe_attnres_depth_probe_mlp_for_router',
        action='store_true',
        help='In pre_attn mode, compute a lightweight pre-MLP AttnRes alpha for routing summary only '
             '(does not change main forward path).',
    )
    parser.add_argument(
        '--moe_attnres_depth_summary_grad_mode',
        type=str,
        default='detached',
        choices=['detached', 'delayed_unfreeze', 'trainable'],
        help='Gradient path for router depth summary: detached, delayed_unfreeze, or trainable.',
    )
    parser.add_argument(
        '--moe_attnres_depth_summary_unfreeze_epoch',
        type=int,
        default=16,
        help='For delayed_unfreeze mode, first epoch index (1-based) where depth-summary gradients are enabled.',
    )
    parser.add_argument(
        '--moe_router_arch',
        type=str,
        default='linear',
        choices=['linear', 'mlp'],
        help='Router head architecture for typed MoE.',
    )
    parser.add_argument(
        '--moe_router_mlp_hidden',
        type=int,
        default=128,
        help='Hidden size when --moe_router_arch mlp is used.',
    )
    parser.add_argument(
        '--moe_router_dispatch_mode',
        type=str,
        default='hard_capacity',
        choices=['hard_capacity', 'soft'],
        help='Router dispatch mode: hard_capacity (top-1 + capacity correction) or soft (weighted experts).',
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
        '--moe_uniform_dispatch_warmup_epochs',
        type=int,
        default=0,
        help='For first N epochs, dispatch uniformly across experts to avoid early lock-in.',
    )
    parser.add_argument(
        '--moe_shared_blend_warmup_epochs',
        type=int,
        default=0,
        help='Warmup epochs for shared-vs-expert blending ramp (0 disables).',
    )
    parser.add_argument(
        '--moe_shared_blend_start',
        type=float,
        default=1.0,
        help='Shared blend value at warmup start (1=shared-only, 0=full expert residual).',
    )
    parser.add_argument(
        '--moe_shared_blend_end',
        type=float,
        default=0.0,
        help='Shared blend value after warmup (typically 0 for full expert residual).',
    )
    parser.add_argument(
        '--moe_router_entropy_coef_spatial',
        type=float,
        default=-1.0,
        help='Spatial branch entropy coef; negative means fallback to --moe_router_entropy_coef.',
    )
    parser.add_argument(
        '--moe_router_entropy_coef_spectral',
        type=float,
        default=-1.0,
        help='Spectral branch entropy coef; negative means fallback to --moe_router_entropy_coef.',
    )
    parser.add_argument(
        '--moe_router_balance_kl_coef_spatial',
        type=float,
        default=-1.0,
        help='Spatial branch KL balance coef; negative means fallback to --moe_router_balance_kl_coef.',
    )
    parser.add_argument(
        '--moe_router_balance_kl_coef_spectral',
        type=float,
        default=-1.0,
        help='Spectral branch KL balance coef; negative means fallback to --moe_router_balance_kl_coef.',
    )
    parser.add_argument(
        '--moe_specialist_branch_mode',
        type=str,
        default='both',
        choices=['both', 'spatial_only', 'spectral_only'],
        help='Enable both specialist banks, or only spatial/spectral specialists with the other branch shared-only.',
    )
    parser.add_argument(
        '--moe_router_compact_feature_mode',
        type=str,
        default='none',
        choices=['none', 'eeg_summary', 'psd_summary'],
        help='Optional compact extra router signal added to both banks (small pooled summary only).',
    )
    parser.add_argument(
        '--moe_router_compact_feature_dim',
        type=int,
        default=8,
        help='Dimension of compact extra router signal (recommended 8-16).',
    )
    parser.add_argument(
        '--moe_expert_init_noise_std',
        type=float,
        default=0.0,
        help='Tiny Gaussian noise std applied to expert linear1 after dense warm-start for symmetry breaking.',
    )

    parser.add_argument(
        '--use_component_lr',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use component-wise LR multipliers (backbone/router/experts/classifier/other).',
    )
    parser.add_argument('--lr_backbone_mult', type=float, default=1.0,
                        help='LR multiplier for backbone parameters.')
    parser.add_argument('--lr_router_mult', type=float, default=1.0,
                        help='LR multiplier for MoE router parameters.')
    parser.add_argument('--lr_expert_mult', type=float, default=1.0,
                        help='LR multiplier for MoE expert/shared FFN parameters.')
    parser.add_argument('--lr_classifier_mult', type=float, default=1.0,
                        help='LR multiplier for classifier head parameters.')
    parser.add_argument('--lr_other_mult', type=float, default=1.0,
                        help='LR multiplier for non-classifier/non-backbone parameters.')

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
        help='Optional per-sample routing export directory (empty=off). '
             'Currently implemented for FACED; other datasets are skipped gracefully.',
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
        help='FACED-only Recording_info.csv for domain ids and routing export joins.',
    )
    parser.add_argument(
        '--routing_run_name',
        type=str,
        default='',
        help='Optional label stored in per-sample routing CSV (e.g. Slurm job id); analysis only.',
    )

    params = parser.parse_args()
    if params.moe_router_temperature <= 0:
        raise ValueError('--moe_router_temperature must be > 0.')
    if params.moe_router_mlp_hidden <= 0:
        raise ValueError('--moe_router_mlp_hidden must be > 0.')
    if params.moe_router_balance_kl_coef < 0:
        raise ValueError('--moe_router_balance_kl_coef must be >= 0.')
    if params.moe_router_z_loss_coef < 0:
        raise ValueError('--moe_router_z_loss_coef must be >= 0.')
    if params.moe_router_jitter_std < 0:
        raise ValueError('--moe_router_jitter_std must be >= 0.')
    if params.moe_router_jitter_final_std < 0:
        raise ValueError('--moe_router_jitter_final_std must be >= 0.')
    if params.moe_router_jitter_anneal_epochs < 0:
        raise ValueError('--moe_router_jitter_anneal_epochs must be >= 0.')
    if params.moe_router_soft_warmup_epochs < 0:
        raise ValueError('--moe_router_soft_warmup_epochs must be >= 0.')
    if params.moe_uniform_dispatch_warmup_epochs < 0:
        raise ValueError('--moe_uniform_dispatch_warmup_epochs must be >= 0.')
    if params.moe_shared_blend_warmup_epochs < 0:
        raise ValueError('--moe_shared_blend_warmup_epochs must be >= 0.')
    if not (0.0 <= params.moe_shared_blend_start <= 1.0):
        raise ValueError('--moe_shared_blend_start must be in [0, 1].')
    if not (0.0 <= params.moe_shared_blend_end <= 1.0):
        raise ValueError('--moe_shared_blend_end must be in [0, 1].')
    if params.moe_attnres_depth_summary_unfreeze_epoch < 1:
        raise ValueError('--moe_attnres_depth_summary_unfreeze_epoch must be >= 1.')
    if params.moe_attnres_depth_router_dim <= 0:
        raise ValueError('--moe_attnres_depth_router_dim must be > 0.')
    if params.moe_router_compact_feature_dim <= 0:
        raise ValueError('--moe_router_compact_feature_dim must be > 0.')
    if params.moe_expert_init_noise_std < 0:
        raise ValueError('--moe_expert_init_noise_std must be >= 0.')
    if params.moe_router_entropy_coef_spatial < 0 and params.moe_router_entropy_coef_spatial != -1.0:
        raise ValueError('--moe_router_entropy_coef_spatial must be >= 0 or -1 for fallback.')
    if params.moe_router_entropy_coef_spectral < 0 and params.moe_router_entropy_coef_spectral != -1.0:
        raise ValueError('--moe_router_entropy_coef_spectral must be >= 0 or -1 for fallback.')
    if params.moe_router_balance_kl_coef_spatial < 0 and params.moe_router_balance_kl_coef_spatial != -1.0:
        raise ValueError('--moe_router_balance_kl_coef_spatial must be >= 0 or -1 for fallback.')
    if params.moe_router_balance_kl_coef_spectral < 0 and params.moe_router_balance_kl_coef_spectral != -1.0:
        raise ValueError('--moe_router_balance_kl_coef_spectral must be >= 0 or -1 for fallback.')
    for k in ['lr_backbone_mult', 'lr_router_mult', 'lr_expert_mult', 'lr_classifier_mult', 'lr_other_mult']:
        if getattr(params, k) <= 0:
            raise ValueError(f'--{k} must be > 0.')
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    params.return_sample_keys = False
    params.return_domain_ids = False

    if params.downstream_dataset == 'FACED':
        params.return_sample_keys = bool(getattr(params, 'routing_export_dir', None))
        params.return_domain_ids = bool(getattr(params, 'moe', False) and params.moe_route_mode == 'typed_capacity_domain')
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        params.return_sample_keys = bool(getattr(params, 'routing_export_dir', None))
        if params.num_of_classes != 5:
            print(f"[SEED-V] warning: expected num_of_classes=5, got {params.num_of_classes}")
        if params.moe_domain_bias:
            print("[SEED-V] warning: --moe_domain_bias enabled without FACED metadata; routing uses zero/unknown ids.")
        if params.return_sample_keys:
            print('[SEED-V] routing_export_dir is set; per-sample export is FACED-only and will be skipped.')
        if params.seedv_split_manifest:
            print(f"[SEED-V] using external split manifest: {params.seedv_split_manifest}")
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