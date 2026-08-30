"""CPU contract tests for the final paper-facing method recipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_main import add_faced_args, add_seedv_args, add_shared_args, validate_args
from paper_method_schema import load_method_recipe, verify_args_against_method


ROOT = Path(__file__).parent
RECIPE = ROOT / 'paper_method_specialist_augmented_full_v1.json'


def _args(dataset: str = 'FACED'):
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_faced_args(parser)
    add_seedv_args(parser)
    return parser.parse_args([
        '--datasets_dir', '/tmp/icassp_paper_method_dataset',
        '--num_of_classes', '9' if dataset == 'FACED' else '5',
        '--model_dir', str(REPO_ROOT / 'output/icassp2027_revision/method_contract/seed_3407'),
        '--downstream_dataset', dataset,
        '--experiment_profile', 'icassp2027_revision',
        '--revision_condition', 'specialist_augmented_full',
        '--revision_protocol', 'cbramod_benchmark',
        '--revision_run_mode', 'paper',
        '--paper_method_recipe', str(RECIPE),
    ])


def main() -> None:
    recipe = load_method_recipe(RECIPE)
    assert recipe['method_id'] == 'icaspp_specialist_augmented_full_v1'
    for dataset in ('FACED', 'ISRUC', 'SEED-V'):
        args = _args(dataset)
        validate_args(args)
        info = verify_args_against_method(args, recipe)
        assert info['paper_method_id'] == recipe['method_id']
        assert args.trainability_mode == 'full'
        assert args.attnres_variant == 'pre_attn'
        assert args.moe is True
        assert args.moe_use_attnres_depth_router_features is False
        assert args.moe_attnres_depth_context_mode == 'compact_shared'
        assert args.use_component_lr is False

    missing = _args()
    missing.paper_method_recipe = ''
    try:
        validate_args(missing)
    except ValueError as exc:
        assert 'requires' in str(exc) and 'paper_method_recipe' in str(exc)
    else:
        raise AssertionError('missing paper method recipe was accepted')

    print('paper method contract: PASS')


if __name__ == '__main__':
    main()
