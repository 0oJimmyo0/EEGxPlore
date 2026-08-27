"""Validate and print the resolved focused ICASSP revision configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_main import add_faced_args, add_seedv_args, add_shared_args, validate_args
from verify_historical_recipe import verify_recipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_args(parser)
    add_faced_args(parser)
    add_seedv_args(parser)
    parser.add_argument(
        '--expected-commit',
        default='',
        help='Expected repository commit. If set, the current HEAD must match it.',
    )
    parser.add_argument(
        '--require-clean',
        action='store_true',
        help='Reject tracked or untracked working-tree changes.',
    )
    parser.add_argument(
        '--historical-recipe',
        default=str(Path(__file__).with_name('historical_recipe_1785556.json')),
        help='Machine-readable recipe required for historical_selective.',
    )
    return parser


def _git_state() -> tuple[str, str]:
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    commit, status = _git_state()
    expected = str(args.expected_commit).strip()
    if expected and expected != 'HEAD' and commit != expected:
        raise SystemExit(f'expected commit {expected}, found {commit}')
    if args.require_clean and status:
        raise SystemExit('working tree is not clean; commit the revision before an evidence run')

    historical_recipe_info = {}
    if args.revision_condition == 'historical_selective':
        historical_recipe_info = verify_recipe(Path(args.historical_recipe), args)

    resolved = {
        'repository_root': str(REPO_ROOT),
        'git_commit': commit,
        'git_dirty': bool(status),
        'dataset': args.downstream_dataset,
        'condition': args.revision_condition,
        'protocol': args.revision_protocol,
        'model_dir': os.path.realpath(os.path.abspath(args.model_dir)),
        'backbone': args.backbone,
        'trainability_mode': args.trainability_mode,
        'attnres_variant': args.attnres_variant,
        'attnres_start_layer': args.attnres_start_layer,
        'moe': args.moe,
        'moe_num_layers': args.moe_num_layers,
        'moe_route_mode': args.moe_route_mode,
        'moe_router_base_feature_mode': args.moe_router_base_feature_mode,
        'moe_use_attnres_depth_router_features': args.moe_use_attnres_depth_router_features,
        'moe_router_compact_feature_mode': args.moe_router_compact_feature_mode,
        'selection_metric': args.selection_metric,
        'input_scale_divisor': args.input_scale_divisor,
        'foundation_dir': os.path.realpath(os.path.abspath(args.foundation_dir)),
        'datasets_dir': os.path.realpath(os.path.abspath(args.datasets_dir)),
        'historical_recipe_path': os.path.realpath(os.path.abspath(args.historical_recipe))
        if args.revision_condition == 'historical_selective' else '',
        **historical_recipe_info,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
