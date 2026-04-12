#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple


REQUIRED_FILES = (
    'block_summary_stats.json',
    'router_context_stats.json',
    'routing_diagnostics.json',
)


def _load_json_rows(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError(f'expected JSON list in {path}')


def _numeric_series(rows: List[Dict], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)):
            out.append(float(val))
    return out


def _span(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return max(values) - min(values)


def _check_run(run_dir: str, min_records: int) -> Tuple[bool, Dict[str, float]]:
    ok = True
    stats: Dict[str, float] = {}

    print(f'\n[verify] run_dir={run_dir}')
    paths = {name: os.path.join(run_dir, name) for name in REQUIRED_FILES}
    for name, path in paths.items():
        if not os.path.isfile(path):
            print(f'[verify][fail] missing {name}')
            ok = False

    if not ok:
        return False, stats

    block_rows = _load_json_rows(paths['block_summary_stats.json'])
    router_rows = _load_json_rows(paths['router_context_stats.json'])
    routing_rows = _load_json_rows(paths['routing_diagnostics.json'])

    print(
        f'[verify] rows: block={len(block_rows)} router={len(router_rows)} routing={len(routing_rows)} '
        f'(min_records={min_records})'
    )
    if len(block_rows) < min_records or len(router_rows) < min_records:
        print('[verify][fail] not enough diagnostic records')
        ok = False

    block_mode_rows = [r for r in block_rows if r.get('depth_context_mode') == 'block_shared_typed_proj']
    if not block_mode_rows:
        print('[verify][fail] no rows with depth_context_mode=block_shared_typed_proj')
        ok = False

    block_counts = sorted({int(r.get('block_count', 0) or 0) for r in block_mode_rows})
    print(f'[verify] block_count values={block_counts}')
    if not block_counts or block_counts == [0]:
        print('[verify][fail] block_count is missing/zero')
        ok = False

    layer_counts_present = any(isinstance(r.get('block_layer_counts'), list) for r in block_mode_rows)
    if not layer_counts_present:
        print('[verify][fail] block_layer_counts missing in block rows')
        ok = False

    shared_norm = _numeric_series(router_rows, 'shared_context_norm')
    proj_sp = _numeric_series(router_rows, 'spatial_projected_context_norm')
    proj_sc = _numeric_series(router_rows, 'spectral_projected_context_norm')

    shared_span = _span(shared_norm)
    proj_sp_span = _span(proj_sp)
    proj_sc_span = _span(proj_sc)

    print(
        '[verify] context spans: '
        f'shared={shared_span if shared_span is not None else "NA"} '
        f'spatial_proj={proj_sp_span if proj_sp_span is not None else "NA"} '
        f'spectral_proj={proj_sc_span if proj_sc_span is not None else "NA"}'
    )

    if shared_span is None and proj_sp_span is None and proj_sc_span is None:
        print('[verify][fail] no numeric context norm fields found')
        ok = False

    stats['shared_mean'] = sum(shared_norm) / len(shared_norm) if shared_norm else float('nan')
    stats['proj_sp_mean'] = sum(proj_sp) / len(proj_sp) if proj_sp else float('nan')
    stats['proj_sc_mean'] = sum(proj_sc) / len(proj_sc) if proj_sc else float('nan')

    collapsed_sp = _numeric_series(routing_rows, 'spatial_collapsed_experts')
    collapsed_sc = _numeric_series(routing_rows, 'spectral_collapsed_experts')
    if collapsed_sp or collapsed_sc:
        print(
            f'[verify] collapsed experts mean: '
            f'spatial={sum(collapsed_sp)/len(collapsed_sp) if collapsed_sp else "NA"} '
            f'spectral={sum(collapsed_sc)/len(collapsed_sc) if collapsed_sc else "NA"}'
        )

    return ok, stats


def main() -> None:
    ap = argparse.ArgumentParser(description='Verify block-shared depth diagnostics are present and non-trivial.')
    ap.add_argument('--run_dir', action='append', required=True, help='Model output directory containing diagnostics JSON files.')
    ap.add_argument('--min_records', type=int, default=1)
    args = ap.parse_args()

    all_ok = True
    per_run_stats: List[Tuple[str, Dict[str, float]]] = []

    for run_dir in args.run_dir:
        ok, stats = _check_run(run_dir, max(1, int(args.min_records)))
        all_ok = all_ok and ok
        per_run_stats.append((run_dir, stats))

    if len(per_run_stats) > 1:
        shared_means = [s.get('shared_mean') for _, s in per_run_stats if isinstance(s.get('shared_mean'), float)]
        proj_sp_means = [s.get('proj_sp_mean') for _, s in per_run_stats if isinstance(s.get('proj_sp_mean'), float)]
        proj_sc_means = [s.get('proj_sc_mean') for _, s in per_run_stats if isinstance(s.get('proj_sc_mean'), float)]

        def _distinct_count(vals: List[float]) -> int:
            clean = [v for v in vals if v == v]
            return len({round(v, 8) for v in clean})

        print('\n[verify] cross-run variation summary')
        print(f'[verify] unique shared_mean={_distinct_count(shared_means)}')
        print(f'[verify] unique proj_sp_mean={_distinct_count(proj_sp_means)}')
        print(f'[verify] unique proj_sc_mean={_distinct_count(proj_sc_means)}')

    if not all_ok:
        raise SystemExit(2)

    print('\n[verify] PASS')


if __name__ == '__main__':
    main()
