"""Contract tests for the focused ICASSP revision argument firewall."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_main import add_faced_args, add_seedv_args, add_shared_args, validate_args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_faced_args(parser)
    add_seedv_args(parser)
    return parser


def _args(condition: str):
    argv = [
        '--datasets_dir', '/tmp/icassp_revision_dataset',
        '--num_of_classes', '5',
        '--model_dir', str(REPO_ROOT / 'output/icassp2027_revision/contract_test/seed_42'),
        '--downstream_dataset', 'SEED-V',
        '--experiment_profile', 'icassp2027_revision',
        '--revision_condition', condition,
        '--revision_protocol', 'cbramod_benchmark',
    ]
    if condition == 'specialist_augmented_full':
        argv.extend([
            '--downstream_dataset', 'FACED',
            '--num_of_classes', '9',
            '--paper_method_recipe', str(
                Path(__file__).with_name('paper_method_specialist_augmented_full_v1.json')
            ),
        ])
    return _parser().parse_args(argv)


def main() -> None:
    expected = {
        'frozen': {'trainability_mode': 'frozen', 'attnres_variant': 'none', 'moe': False},
        'upper1': {'trainability_mode': 'upper1', 'attnres_variant': 'none', 'moe': False},
        'full': {'trainability_mode': 'full', 'attnres_variant': 'none', 'moe': False},
        'attnres_only': {'trainability_mode': 'attnres_only', 'attnres_variant': 'pre_attn', 'moe': False},
        'specialist_only': {
            'trainability_mode': 'specialist_only',
            'attnres_variant': 'none',
            'moe': True,
            'moe_router_base_feature_mode': 'baseline_only',
        },
        'combined': {
            'trainability_mode': 'combined',
            'attnres_variant': 'pre_attn',
            'moe': True,
            'moe_router_base_feature_mode': 'full',
        },
        'selective_fresh': {
            'trainability_mode': 'combined',
            'attnres_variant': 'pre_attn',
            'moe': True,
            'moe_router_base_feature_mode': 'full',
        },
        'selective_paper': {
            'trainability_mode': 'combined',
            'attnres_variant': 'pre_attn',
            'moe': True,
            'moe_router_base_feature_mode': 'full',
        },
        'specialist_augmented_full': {
            'trainability_mode': 'full',
            'attnres_variant': 'pre_attn',
            'moe': True,
            'moe_router_base_feature_mode': 'full',
            'moe_use_attnres_depth_router_features': False,
            'moe_attnres_depth_context_mode': 'compact_shared',
        },
    }
    for condition, fields in expected.items():
        args = _args(condition)
        validate_args(args)
        for field, value in fields.items():
            assert getattr(args, field) == value, (condition, field, getattr(args, field), value)
        assert args.backbone == 'cbramod'
        assert args.selection_metric == 'kappa'
        assert args.input_scale_divisor == 100.0

    historical = _args('historical_selective')
    try:
        validate_args(historical)
    except ValueError as exc:
        assert 'permanently locked' in str(exc)
    else:
        raise AssertionError('historical_selective was not permanently locked')

    archived_protocol = _args('upper1')
    archived_protocol.revision_protocol = 'seedv_subject_disjoint'
    try:
        validate_args(archived_protocol)
    except ValueError as exc:
        assert 'archived' in str(exc)
    else:
        raise AssertionError('archived subject-disjoint protocol was accepted')

    physio_missing_manifest = _parser().parse_args([
        '--datasets_dir', '/tmp/icassp_revision_dataset',
        '--num_of_classes', '4',
        '--model_dir', str(REPO_ROOT / 'output/icassp2027_revision/contract_test/seed_42'),
        '--downstream_dataset', 'PhysioNet-MI',
        '--experiment_profile', 'icassp2027_revision',
        '--revision_condition', 'full',
        '--revision_protocol', 'cbramod_benchmark',
    ])
    try:
        validate_args(physio_missing_manifest)
    except ValueError as exc:
        assert 'split manifest' in str(exc)
    else:
        raise AssertionError('PhysioNet-MI without its split manifest was accepted')

    physio = _parser().parse_args([
        '--datasets_dir', '/tmp/icassp_revision_dataset',
        '--num_of_classes', '4',
        '--model_dir', str(REPO_ROOT / 'output/icassp2027_revision/contract_test/seed_42'),
        '--downstream_dataset', 'PhysioNet-MI',
        '--experiment_profile', 'icassp2027_revision',
        '--revision_condition', 'full',
        '--revision_protocol', 'cbramod_benchmark',
        '--icassp_split_manifest', str(
            REPO_ROOT / 'experiments/icassp2027/manifests/physionet_mi/split_manifest.json'
        ),
    ])
    validate_args(physio)

    smoke = _args('upper1')
    smoke.revision_run_mode = 'smoke'
    smoke.epochs = 1
    smoke.model_dir = str(REPO_ROOT / 'output/icassp2027_smoke/contract_test/seed_42')
    validate_args(smoke)

    rejected = _args('combined')
    rejected.model_dir = '/tmp/not-an-icasp-output'
    try:
        validate_args(rejected)
    except ValueError as exc:
        assert 'model_dir' in str(exc)
    else:
        raise AssertionError('revision profile accepted an output directory outside its root')

    print('revision firewall contract: PASS')


if __name__ == '__main__':
    main()
