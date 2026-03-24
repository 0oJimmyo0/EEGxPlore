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
    '--attnres_start_layer',
    type=int,
    default=0,
    help='first layer index that uses AttnRes; for 12 layers, 8 means top-4 only'
)

    parser.add_argument(
        '--moe',
        action='store_true',
        help='Replace dense FFN with sparse MoE in the top encoder layers (see --moe_num_layers, ...)',
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
        help='Experts per MoE layer',
    )
    parser.add_argument(
        '--moe_top_k',
        type=int,
        default=1,
        help='Top-k experts per token (load-balancing aux is normalized by k so scale matches k=1)',
    )
    parser.add_argument(
        '--moe_expert_zero_only',
        action='store_true',
        help='MoE warm-start: copy pretrained dense FFN only into expert 0 (default: copy into all experts)',
    )
    parser.add_argument(
        '--moe_load_balance',
        type=float,
        default=0.005,
        help='Switch-style MoE load-balancing aux weight (0=off). Default 0.05 matches stable FACED MoE runs.',
    )
    parser.add_argument(
        '--moe_router_noise',
        type=float,
        default=0.005,
        help='Gaussian noise std on MoE router logits in training (0=off). Default 0.01; set 0 to ablate.',
    )
    parser.add_argument(
        '--moe_shared_specialist',
        action='store_true',
        help='MoE: keep pretrained dense FFN as shared path; routed specialists add residual (vs full replacement)',
    )
    parser.add_argument(
        '--moe_specialist_rand_linear1',
        action='store_true',
        help='With --moe_shared_specialist: Kaiming specialist linear1 instead of dense copy (+symmetry noise)',
    )
    parser.add_argument(
        '--moe_diagnostics',
        action='store_true',
        help='After each val epoch, log MoE expert usage / entropy / load-balance (one val batch, eval mode)',
    )
    parser.add_argument(
        '--moe_router_mode',
        type=str,
        default='token',
        choices=['token', 'sample_hidden', 'sample_attnres'],
        help='MoE routing: token (default), sample_hidden (mean-pool FFN input per sample), '
             'sample_attnres (pre-attn AttnRes: concat pooled baseline, attnres, diff → 3D router)',
    )
    parser.add_argument(
        '--moe_router_arch',
        type=str,
        default='linear',
        choices=['linear', 'mlp'],
        help='MoE router head: linear or LayerNorm→Linear→GELU→Linear (hidden=moe_router_mlp_hidden)',
    )
    parser.add_argument(
        '--moe_router_mlp_hidden',
        type=int,
        default=128,
        help='Hidden size when --moe_router_arch mlp',
    )
    parser.add_argument(
        '--moe_use_psd_router_features',
        action='store_true',
        help='sample_attnres only: concat 5-band log1p PSD from raw EEG (set with sample_attnres)',
    )
    parser.add_argument(
        '--moe_expert_type',
        type=str,
        default='generic',
        choices=['generic', 'typed'],
        help='With --moe_shared_specialist: generic=one specialist pool; typed=spatial+spectral banks '
             '(requires --moe_router_mode sample_attnres --moe_top_k 1). PSD attaches only to spectral router.',
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
        help='Recording_info.csv for cohort / sample rate / age join (analysis export only).',
    )
    parser.add_argument(
        '--routing_run_name',
        type=str,
        default='',
        help='Optional label stored in per-sample routing CSV (e.g. Slurm job id); analysis only.',
    )

    params = parser.parse_args()
    print(params)

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