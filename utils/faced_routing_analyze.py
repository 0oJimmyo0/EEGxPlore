"""
Post-hoc FACED typed-MoE routing analysis from per-sample CSVs (no training / no model).
Writes CSVs only: grouped summaries, entropy stats, cross-tabs, optional two-run comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _get(r: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in r and r[k] != "":
            return r[k]
    return default


def _as_int_correct(x: Dict[str, Any]) -> int:
    v = _get(x, "correct", default=0)
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _spatial_top1(r: Dict[str, Any]) -> int:
    v = _get(r, "spatial_top1_expert", "spatial_expert", default=None)
    if v is None or v == "":
        return -1
    return int(v)


def _spectral_top1(r: Dict[str, Any]) -> int:
    v = _get(r, "spectral_top1_expert", "spectral_expert", default=None)
    if v is None or v == "":
        return -1
    return int(v)


def _infer_num_experts(rows: List[Dict[str, Any]]) -> int:
    n = 0
    pat = re.compile(r"^spatial_p(\d+)$")
    for r in rows:
        for k in r:
            m = pat.match(k)
            if m:
                n = max(n, int(m.group(1)) + 1)
    if n > 0:
        return n
    for r in rows:
        n = max(n, _spatial_top1(r) + 1, _spectral_top1(r) + 1)
    return max(n, 4)


def load_per_sample_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_all_analyses(
    rows: List[Dict[str, Any]],
    out_dir: str,
    file_prefix: str,
    num_experts: Optional[int] = None,
) -> Dict[str, str]:
    """
    Write analysis CSVs under out_dir. Returns map name -> path.
    """
    os.makedirs(out_dir, exist_ok=True)
    ne = num_experts or _infer_num_experts(rows)
    out: Dict[str, str] = {}

    out["entropy_by_group"] = _entropy_by_group(rows, out_dir, file_prefix)
    out["expert_usage_by_group"] = _expert_usage_by_group(rows, out_dir, file_prefix, ne)
    out["performance_by_group"] = _performance_by_group(rows, out_dir, file_prefix)
    out["crosstabs"] = _write_crosstabs(rows, out_dir, file_prefix, ne)
    return out


def _entropy_by_group(rows: List[Dict[str, Any]], out_dir: str, prefix: str) -> str:
    """Mean / std / count for spatial and spectral entropy by metadata group."""
    dims = ("cohort", "sample_rate_group", "segment_index", "age_bucket")
    summaries: List[Dict[str, Any]] = []

    for gname in dims:
        bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if gname == "cohort":
                gv = str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
            else:
                gv = str(_get(r, gname, default="NA") or "NA")
            bucket[gv].append(r)
        for gval, rs in sorted(bucket.items()):
            se = [float(_get(x, "spatial_entropy", default=0)) for x in rs]
            sc = [float(_get(x, "spectral_entropy", default=0)) for x in rs]
            n = len(rs)
            summaries.append(
                {
                    "group_dimension": gname,
                    "group_value": gval,
                    "n": n,
                    "spatial_entropy_mean": statistics.mean(se) if se else "",
                    "spatial_entropy_std": statistics.pstdev(se) if len(se) > 1 else 0.0,
                    "spectral_entropy_mean": statistics.mean(sc) if sc else "",
                    "spectral_entropy_std": statistics.pstdev(sc) if len(sc) > 1 else 0.0,
                }
            )

    path = os.path.join(out_dir, f"{prefix}_entropy_by_group.csv")
    if summaries:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
    return path


def _expert_usage_by_group(
    rows: List[Dict[str, Any]], out_dir: str, prefix: str, num_e: int
) -> str:
    dims = ("cohort", "sample_rate_group", "segment_index", "age_bucket")
    summaries: List[Dict[str, Any]] = []

    for gname in dims:
        bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if gname == "cohort":
                gv = str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
            else:
                gv = str(_get(r, gname, default="NA") or "NA")
            bucket[gv].append(r)
        for gval, rs in sorted(bucket.items()):
            n = len(rs)
            hsp = [0] * num_e
            hsc = [0] * num_e
            for x in rs:
                a, b = _spatial_top1(x), _spectral_top1(x)
                if 0 <= a < num_e:
                    hsp[a] += 1
                if 0 <= b < num_e:
                    hsc[b] += 1
            summaries.append(
                {
                    "group_dimension": gname,
                    "group_value": gval,
                    "n": n,
                    "spatial_expert_counts": str(hsp),
                    "spectral_expert_counts": str(hsc),
                    "spatial_expert_frac": str([round(c / max(n, 1), 6) for c in hsp]),
                    "spectral_expert_frac": str([round(c / max(n, 1), 6) for c in hsc]),
                }
            )

    path = os.path.join(out_dir, f"{prefix}_expert_usage_by_group.csv")
    if summaries:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
    return path


def _performance_by_group(rows: List[Dict[str, Any]], out_dir: str, prefix: str) -> str:
    dims = ("cohort", "sample_rate_group", "segment_index", "age_bucket")
    summaries: List[Dict[str, Any]] = []
    n_cls = 9

    for gname in dims:
        bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if gname == "cohort":
                gv = str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
            else:
                gv = str(_get(r, gname, default="NA") or "NA")
            bucket[gv].append(r)
        for gval, rs in sorted(bucket.items()):
            n = len(rs)
            acc = sum(_as_int_correct(x) for x in rs) / max(n, 1)
            confs = [float(_get(x, "max_softmax_confidence", "confidence", default=0)) for x in rs]
            mean_conf = statistics.mean(confs) if confs else 0.0
            # confusion counts [true][pred]
            cm = [[0] * n_cls for _ in range(n_cls)]
            for x in rs:
                try:
                    ti = int(_get(x, "true_label", default=-1))
                    pj = int(_get(x, "pred_label", default=-1))
                    if 0 <= ti < n_cls and 0 <= pj < n_cls:
                        cm[ti][pj] += 1
                except (TypeError, ValueError):
                    continue
            summaries.append(
                {
                    "group_dimension": gname,
                    "group_value": gval,
                    "n": n,
                    "accuracy": round(acc, 6),
                    "mean_confidence": round(mean_conf, 6),
                    "confusion_matrix_json": str(cm),
                }
            )

    path = os.path.join(out_dir, f"{prefix}_performance_by_group.csv")
    if summaries:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
    return path


def _write_crosstabs(rows: List[Dict[str, Any]], out_dir: str, prefix: str, num_e: int) -> str:
    """Write multiple crosstab CSVs; return manifest path listing files."""
    n_cls = 9
    paths: List[str] = []

    def mat_expert_label(bank: str) -> List[List[int]]:
        m = [[0] * n_cls for _ in range(num_e)]
        for r in rows:
            try:
                ti = int(_get(r, "true_label", default=-1))
                if bank == "spatial":
                    ei = _spatial_top1(r)
                else:
                    ei = _spectral_top1(r)
                if 0 <= ei < num_e and 0 <= ti < n_cls:
                    m[ei][ti] += 1
            except (TypeError, ValueError):
                continue
        return m

    for bank in ("spatial", "spectral"):
        m = mat_expert_label(bank)
        p = os.path.join(out_dir, f"{prefix}_crosstab_{bank}_expert_x_true_label.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["expert_id"] + [f"true_label_{c}" for c in range(n_cls)])
            for e in range(num_e):
                w.writerow([e] + m[e])
        paths.append(p)

    def mat_expert_dim(dim: str, bank: str) -> Tuple[List[str], List[List[int]]]:
        seen = set()
        for r in rows:
            if dim == "cohort":
                v = str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
            else:
                v = str(_get(r, dim, default="NA") or "NA")
            seen.add(v)
        col_vals = sorted(seen, key=lambda x: (x == "NA", x))
        mat = [[0] * len(col_vals) for _ in range(num_e)]
        idx = {v: i for i, v in enumerate(col_vals)}
        for r in rows:
            if dim == "cohort":
                v = str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
            else:
                v = str(_get(r, dim, default="NA") or "NA")
            j = idx[v]
            ei = _spatial_top1(r) if bank == "spatial" else _spectral_top1(r)
            if 0 <= ei < num_e:
                mat[ei][j] += 1
        return col_vals, mat

    for dim, short in (("cohort", "cohort"), ("sample_rate_group", "sr"), ("segment_index", "seg")):
        for bank in ("spatial", "spectral"):
            cols, m = mat_expert_dim(dim, bank)
            p = os.path.join(out_dir, f"{prefix}_crosstab_{bank}_expert_x_{short}.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["expert_id"] + cols)
                for e in range(num_e):
                    w.writerow([e] + m[e])
            paths.append(p)

    manifest = os.path.join(out_dir, f"{prefix}_crosstab_manifest.txt")
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write("\n".join(paths))
    return manifest


def compare_two_runs(path_a: str, path_b: str, out_dir: str, label_a: str, label_b: str) -> str:
    """Aggregate comparison: mean metrics by cohort for nuisance-alignment inspection."""
    os.makedirs(out_dir, exist_ok=True)
    ra = load_per_sample_csv(path_a)
    rb = load_per_sample_csv(path_b)

    def by_cohort(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            c = str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
            d[c].append(r)
        return d

    ca, cb = by_cohort(ra), by_cohort(rb)
    keys = sorted(set(ca.keys()) | set(cb.keys()))
    rows_out: List[Dict[str, Any]] = []
    for k in keys:
        def stats(rows: List[Dict[str, Any]], tag: str) -> Dict[str, float]:
            if not rows:
                return {f"{tag}_n": 0, f"{tag}_acc": 0, f"{tag}_mean_sp_H": 0, f"{tag}_mean_spc_H": 0}
            n = len(rows)
            acc = sum(_as_int_correct(x) for x in rows) / max(n, 1)
            sp = [float(_get(x, "spatial_entropy", 0)) for x in rows]
            sc = [float(_get(x, "spectral_entropy", 0)) for x in rows]
            return {
                f"{tag}_n": n,
                f"{tag}_acc": round(acc, 6),
                f"{tag}_mean_spatial_entropy": round(statistics.mean(sp), 6) if sp else 0,
                f"{tag}_mean_spectral_entropy": round(statistics.mean(sc), 6) if sc else 0,
            }

        row: Dict[str, Any] = {"cohort": k}
        row.update(stats(ca.get(k, []), label_a))
        row.update(stats(cb.get(k, []), label_b))
        rows_out.append(row)

    cmp_path = os.path.join(out_dir, f"compare_{label_a}_vs_{label_b}_by_cohort.csv")
    if rows_out:
        with open(cmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    return cmp_path


def main() -> None:
    ap = argparse.ArgumentParser(description="FACED routing CSV analysis (post-hoc, no model).")
    ap.add_argument("--csv", type=str, help="Per-sample routing CSV from export_facced_routing_split")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for analysis CSVs")
    ap.add_argument("--prefix", type=str, default="analysis", help="Filename prefix")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("CSV_A", "CSV_B"),
        help="Two per-sample CSVs (e.g. no-PSD vs PSD); writes comparison summary",
    )
    ap.add_argument("--label_a", type=str, default="no_psd", help="Tag for first CSV in comparison")
    ap.add_argument("--label_b", type=str, default="psd", help="Tag for second CSV in comparison")
    args = ap.parse_args()

    if args.compare:
        p = compare_two_runs(args.compare[0], args.compare[1], args.outdir, args.label_a, args.label_b)
        print(f"[analyze] wrote {p}", flush=True)
        return

    if not args.csv:
        ap.error("provide --csv or --compare")
    rows = load_per_sample_csv(args.csv)
    if not rows:
        raise SystemExit("empty CSV")
    ne = _infer_num_experts(rows)
    outs = write_all_analyses(rows, args.outdir, args.prefix, ne)
    for k, v in outs.items():
        print(f"[analyze] {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
