"""Validate that two ICASSP Static/Routed summaries form a causal pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _config(summary: dict) -> dict:
    return summary.get('config', {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--static_summary', type=Path, required=True)
    parser.add_argument('--routed_summary', type=Path, required=True)
    args = parser.parse_args()

    static = _load(args.static_summary)
    routed = _load(args.routed_summary)
    static_config = _config(static)
    routed_config = _config(routed)
    static_prov = static.get('provenance', {})
    routed_prov = routed.get('provenance', {})

    static_policy = static_config.get('moe_router_policy')
    routed_policy = routed_config.get('moe_router_policy')
    if {static_policy, routed_policy} != {'static', 'sample'}:
        raise AssertionError(
            f"Expected one static and one sample policy, got {static_policy!r}, {routed_policy!r}"
        )

    for key in ('dataset', 'seed', 'selection_metric'):
        if static.get(key, static_config.get(key)) != routed.get(key, routed_config.get(key)):
            raise AssertionError(f"Pair field differs: {key}")
    if static_config.get('selection_metric') != 'kappa':
        raise AssertionError('Pair selection metric is not kappa')

    for key in ('manifest_sha256', 'git_commit', 'pair_contract_sha256'):
        if static_prov.get(key) != routed_prov.get(key):
            raise AssertionError(f"Pair provenance differs: {key}")
    if static_prov.get('git_dirty') is not False or routed_prov.get('git_dirty') is not False:
        raise AssertionError('Pair must come from a clean git worktree')

    static_counts = static_prov.get('trainable_parameter_counts', {})
    routed_counts = routed_prov.get('trainable_parameter_counts', {})
    if static_counts != routed_counts:
        raise AssertionError('Static/Routed trainable parameter counts differ')

    print('Static/Routed pair: PASS')
    print(json.dumps({
        'dataset': static.get('dataset', static_config.get('downstream_dataset')),
        'seed': static.get('seed', static_config.get('seed')),
        'manifest_sha256': static_prov.get('manifest_sha256'),
        'pair_contract_sha256': static_prov.get('pair_contract_sha256'),
        'trainable_parameter_counts': static_counts,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
