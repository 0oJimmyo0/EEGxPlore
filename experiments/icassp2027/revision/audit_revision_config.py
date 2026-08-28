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
from verify_fresh_selective_recipe import verify_recipe as verify_fresh_recipe
from verify_paper_protocol import (
    validate_args_against_protocol,
    verify_protocol as verify_paper_protocol,
)


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
        '--fresh-selective-recipe',
        default=str(Path(__file__).with_name('fresh_selective_recipe.json')),
        help='Machine-readable recipe required for selective_fresh.',
    )
    parser.add_argument(
        '--paper-protocol',
        default='',
        help='Locked dataset-specific protocol required for paper-facing new rows.',
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

    if args.revision_run_mode in {'paper', 'smoke'} and not args.paper_protocol:
        raise SystemExit(
            'paper/smoke revision audits require --paper-protocol; '
            'use the dataset-specific locked protocol'
        )

    commit, status = _git_state()
    expected = str(args.expected_commit).strip()
    if expected and expected != 'HEAD' and commit != expected:
        raise SystemExit(f'expected commit {expected}, found {commit}')
    if args.require_clean and status:
        raise SystemExit('working tree is not clean; commit the revision before an evidence run')

    fresh_selective_recipe_info = {}
    if args.revision_condition == 'selective_fresh':
        fresh_selective_recipe_info = verify_fresh_recipe(
            Path(args.fresh_selective_recipe), args
        )

    paper_protocol_info = {}
    if args.paper_protocol:
        paper_protocol_info = verify_paper_protocol(
            Path(args.paper_protocol), args.downstream_dataset
        )
        if args.revision_run_mode in {'paper', 'smoke'}:
            validate_args_against_protocol(args, paper_protocol_info)

    resolved = {
        'repository_root': str(REPO_ROOT),
        'git_commit': commit,
        'git_dirty': bool(status),
        'dataset': args.downstream_dataset,
        'condition': args.revision_condition,
        'protocol': args.revision_protocol,
        'revision_run_mode': args.revision_run_mode,
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
        'fresh_selective_recipe_path': os.path.realpath(os.path.abspath(args.fresh_selective_recipe))
        if args.revision_condition == 'selective_fresh' else '',
        'paper_protocol_path': os.path.realpath(os.path.abspath(args.paper_protocol))
        if args.paper_protocol else '',
        **fresh_selective_recipe_info,
        **{
            'paper_protocol_id': paper_protocol_info.get('protocol_id', ''),
            'paper_protocol_sha256': paper_protocol_info.get('sha256', ''),
            'paper_protocol_status': 'locked' if paper_protocol_info else '',
        },
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
