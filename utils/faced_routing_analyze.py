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


def _layer_indices_in_row(r: Dict[str, Any]) -> List[int]:
    out = set()
    pat = re.compile(r"^layer(\d+)_")
    for k in r:
        m = pat.match(k)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def _pick_analysis_layer(rows: List[Dict[str, Any]], layer_idx: Optional[int] = None) -> Optional[int]:
    if layer_idx is not None:
        return layer_idx
    seen = set()
    for r in rows:
        seen.update(_layer_indices_in_row(r))
    if not seen:
        return None
    return min(seen)


def _layer_key(r: Dict[str, Any], layer_idx: Optional[int], suffix: str) -> Optional[str]:
    if layer_idx is not None:
        key = f"layer{layer_idx}_{suffix}"
        return key if key in r else None
    for li in _layer_indices_in_row(r):
        key = f"layer{li}_{suffix}"
        if key in r:
            return key
    return None


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def project_rows_for_analysis(
    rows: List[Dict[str, Any]],
    expert_mode: str = "assigned",
    layer_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Normalize per-sample rows into canonical analysis columns.

    expert_mode:
      - assigned: use post-capacity assigned experts
      - raw: use raw pre-capacity top-1 experts
    """
    if expert_mode not in {"assigned", "raw"}:
        raise ValueError(f"expert_mode must be assigned or raw, got {expert_mode!r}")

    picked_layer = _pick_analysis_layer(rows, layer_idx)
    projected: List[Dict[str, Any]] = []
    for r in rows:
        rc = dict(r)
        if expert_mode == "assigned":
            sp_key = _layer_key(r, picked_layer, "assigned_spatial_expert")
            sc_key = _layer_key(r, picked_layer, "assigned_spectral_expert")
        else:
            sp_key = _layer_key(r, picked_layer, "raw_top1_spatial")
            sc_key = _layer_key(r, picked_layer, "raw_top1_spectral")

        ent_sp_key = _layer_key(r, picked_layer, "pre_entropy_spatial")
        ent_sc_key = _layer_key(r, picked_layer, "pre_entropy_spectral")

        rc["spatial_top1_expert"] = _safe_int(r.get(sp_key, _get(r, "spatial_top1_expert", default=-1)))
        rc["spectral_top1_expert"] = _safe_int(r.get(sc_key, _get(r, "spectral_top1_expert", default=-1)))
        rc["spatial_entropy"] = _safe_float(r.get(ent_sp_key, _get(r, "spatial_entropy", default=0.0)))
        rc["spectral_entropy"] = _safe_float(r.get(ent_sc_key, _get(r, "spectral_entropy", default=0.0)))
        rc["analysis_layer_idx"] = "" if picked_layer is None else picked_layer
        rc["expert_source"] = "assigned" if expert_mode == "assigned" else "raw_top1"
        projected.append(rc)
    return projected


def _ensure_analysis_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    sample = rows[0]
    has_generic_experts = "spatial_top1_expert" in sample and "spectral_top1_expert" in sample
    has_generic_entropy = "spatial_entropy" in sample and "spectral_entropy" in sample
    if has_generic_experts and has_generic_entropy:
        return rows
    return project_rows_for_analysis(rows, expert_mode="assigned")


def _spatial_top1(r: Dict[str, Any]) -> int:
    v = _get(r, "spatial_top1_expert", "assigned_spatial_expert", "spatial_expert", default=None)
    if v is not None and v != "":
        return _safe_int(v)
    layer_key = _layer_key(r, None, "assigned_spatial_expert") or _layer_key(r, None, "raw_top1_spatial")
    return _safe_int(r.get(layer_key, -1))


def _spectral_top1(r: Dict[str, Any]) -> int:
    v = _get(r, "spectral_top1_expert", "assigned_spectral_expert", "spectral_expert", default=None)
    if v is not None and v != "":
        return _safe_int(v)
    layer_key = _layer_key(r, None, "assigned_spectral_expert") or _layer_key(r, None, "raw_top1_spectral")
    return _safe_int(r.get(layer_key, -1))


def _entropy_value(r: Dict[str, Any], bank: str) -> float:
    generic = _get(r, f"{bank}_entropy", default=None)
    if generic is not None and generic != "":
        return _safe_float(generic)
    layer_key = _layer_key(r, None, f"pre_entropy_{bank}")
    if layer_key is not None:
        return _safe_float(r.get(layer_key, 0.0))
    fallback = _get(r, f"pre_entropy_{bank}", default=0.0)
    return _safe_float(fallback)


def _group_value_for_dim(r: Dict[str, Any], gname: str) -> str:
    """Normalize metadata group key for cohort / sample_rate / segment / age."""
    if gname == "cohort":
        return str(_get(r, "recording_cohort", "cohort", default="NA") or "NA")
    return str(_get(r, gname, default="NA") or "NA")


def _sort_group_values(vals: List[str], dim: str) -> List[str]:
    def key(v: str) -> Tuple:
        if v == "NA":
            return (2, 0, v)
        if dim == "segment_index":
            try:
                return (0, int(float(v)), "")
            except (TypeError, ValueError):
                return (1, 0, v)
        try:
            return (0, float(v), "")
        except (TypeError, ValueError):
            return (1, 0, v)

    return sorted(vals, key=key)


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
    analysis_rows = _ensure_analysis_rows(rows)
    ne = num_experts or _infer_num_experts(analysis_rows)
    out: Dict[str, str] = {}

    out["entropy_by_group"] = _entropy_by_group(analysis_rows, out_dir, file_prefix)
    out["expert_usage_by_group"] = _expert_usage_by_group(analysis_rows, out_dir, file_prefix, ne)
    out["performance_by_group"] = _performance_by_group(analysis_rows, out_dir, file_prefix)
    out["routing_summary_by_group"] = _routing_summary_by_group(analysis_rows, out_dir, file_prefix, ne)
    out["crosstabs"] = _write_crosstabs(analysis_rows, out_dir, file_prefix, ne)
    return out


def _entropy_by_group(rows: List[Dict[str, Any]], out_dir: str, prefix: str) -> str:
    """Mean / std / count for spatial and spectral entropy by metadata group."""
    dims = ("cohort", "sample_rate_group", "segment_index", "age_bucket")
    summaries: List[Dict[str, Any]] = []

    for gname in dims:
        bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            gv = _group_value_for_dim(r, gname)
            bucket[gv].append(r)
        for gval in _sort_group_values(list(bucket.keys()), gname):
            rs = bucket[gval]
            se = [_entropy_value(x, "spatial") for x in rs]
            sc = [_entropy_value(x, "spectral") for x in rs]
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
            gv = _group_value_for_dim(r, gname)
            bucket[gv].append(r)
        for gval in _sort_group_values(list(bucket.keys()), gname):
            rs = bucket[gval]
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
            gv = _group_value_for_dim(r, gname)
            bucket[gv].append(r)
        for gval in _sort_group_values(list(bucket.keys()), gname):
            rs = bucket[gval]
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


def _routing_summary_by_group(
    rows: List[Dict[str, Any]], out_dir: str, prefix: str, num_e: int
) -> str:
    """
    One table: per (group_dimension, group_value) — n, accuracy, confidence,
    mean entropies, expert top-1 counts and fractions (collapse diagnostics).
    """
    dims = ("cohort", "sample_rate_group", "segment_index", "age_bucket")
    summaries: List[Dict[str, Any]] = []
    n_cls = 9

    for gname in dims:
        bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            bucket[_group_value_for_dim(r, gname)].append(r)
        for gval in _sort_group_values(list(bucket.keys()), gname):
            rs = bucket[gval]
            n = len(rs)
            acc = sum(_as_int_correct(x) for x in rs) / max(n, 1)
            confs = [float(_get(x, "max_softmax_confidence", "confidence", default=0)) for x in rs]
            mean_conf = statistics.mean(confs) if confs else 0.0
            se = [_entropy_value(x, "spatial") for x in rs]
            sc = [_entropy_value(x, "spectral") for x in rs]
            hsp = [0] * num_e
            hsc = [0] * num_e
            for x in rs:
                a, b = _spatial_top1(x), _spectral_top1(x)
                if 0 <= a < num_e:
                    hsp[a] += 1
                if 0 <= b < num_e:
                    hsc[b] += 1
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
                    "mean_spatial_entropy": round(statistics.mean(se), 6) if se else 0.0,
                    "mean_spectral_entropy": round(statistics.mean(sc), 6) if sc else 0.0,
                    "spatial_expert_counts": str(hsp),
                    "spectral_expert_counts": str(hsc),
                    "spatial_expert_frac": str([round(c / max(n, 1), 6) for c in hsp]),
                    "spectral_expert_frac": str([round(c / max(n, 1), 6) for c in hsc]),
                    "confusion_matrix_json": str(cm),
                }
            )

    path = os.path.join(out_dir, f"{prefix}_routing_summary_by_group.csv")
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


def _aggregate_run_group(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        return {
            "n": 0,
            "acc": 0.0,
            "mean_spatial_entropy": 0.0,
            "mean_spectral_entropy": 0.0,
        }
    acc = sum(_as_int_correct(x) for x in rows) / n
    sp = [_entropy_value(x, "spatial") for x in rows]
    sc = [_entropy_value(x, "spectral") for x in rows]
    return {
        "n": n,
        "acc": round(acc, 6),
        "mean_spatial_entropy": round(statistics.mean(sp), 6),
        "mean_spectral_entropy": round(statistics.mean(sc), 6),
    }


def _bucket_by_dim(rows: List[Dict[str, Any]], dim: str) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[_group_value_for_dim(r, dim)].append(r)
    return d


def compare_no_psd_vs_psd_pair(
    path_no_psd: str,
    path_psd: str,
    out_dir: str,
    out_prefix: str = "compare_no_psd_vs_psd",
) -> Dict[str, str]:
    """
    Paired no-PSD vs PSD per-sample exports: write four comparison CSVs
    (cohort, sample_rate_group, segment_index, age_bucket) with acc_delta = acc_psd - acc_no_psd.
    """
    os.makedirs(out_dir, exist_ok=True)
    ra = load_per_sample_csv(path_no_psd)
    rb = load_per_sample_csv(path_psd)
    if not ra or not rb:
        raise ValueError("compare: empty per-sample CSV")

    dims = (
        ("cohort", "cohort", f"{out_prefix}_by_cohort.csv"),
        ("sample_rate_group", "sample_rate_group", f"{out_prefix}_by_sample_rate_group.csv"),
        ("segment_index", "segment_index", f"{out_prefix}_by_segment_index.csv"),
        ("age_bucket", "age_bucket", f"{out_prefix}_by_age_bucket.csv"),
    )
    out_paths: Dict[str, str] = {}
    fieldnames = [
        "n_no_psd",
        "n_psd",
        "acc_no_psd",
        "acc_psd",
        "acc_delta",
        "mean_spatial_entropy_no_psd",
        "mean_spatial_entropy_psd",
        "mean_spectral_entropy_no_psd",
        "mean_spectral_entropy_psd",
    ]

    for dim, colname, fname in dims:
        ba, bb = _bucket_by_dim(ra, dim), _bucket_by_dim(rb, dim)
        keys = _sort_group_values(list(set(ba.keys()) | set(bb.keys())), dim)
        rows_out: List[Dict[str, Any]] = []
        for k in keys:
            sa = _aggregate_run_group(ba.get(k, []))
            sb = _aggregate_run_group(bb.get(k, []))
            row: Dict[str, Any] = {colname: k}
            row["n_no_psd"] = sa["n"]
            row["n_psd"] = sb["n"]
            row["acc_no_psd"] = sa["acc"]
            row["acc_psd"] = sb["acc"]
            row["acc_delta"] = round(float(sb["acc"]) - float(sa["acc"]), 6)
            row["mean_spatial_entropy_no_psd"] = sa["mean_spatial_entropy"]
            row["mean_spatial_entropy_psd"] = sb["mean_spatial_entropy"]
            row["mean_spectral_entropy_no_psd"] = sa["mean_spectral_entropy"]
            row["mean_spectral_entropy_psd"] = sb["mean_spectral_entropy"]
            rows_out.append(row)
        path = os.path.join(out_dir, fname)
        if rows_out:
            fn = [colname] + fieldnames
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fn)
                w.writeheader()
                w.writerows(rows_out)
        out_paths[dim] = path
    return out_paths


def compare_two_runs(path_a: str, path_b: str, out_dir: str, label_a: str, label_b: str) -> str:
    """Backward-compatible: cohort-only table with custom column prefixes."""
    os.makedirs(out_dir, exist_ok=True)
    ra = load_per_sample_csv(path_a)
    rb = load_per_sample_csv(path_b)
    ca, cb = _bucket_by_dim(ra, "cohort"), _bucket_by_dim(rb, "cohort")
    keys = _sort_group_values(list(set(ca.keys()) | set(cb.keys())), "cohort")
    rows_out: List[Dict[str, Any]] = []
    for k in keys:
        sa = _aggregate_run_group(ca.get(k, []))
        sb = _aggregate_run_group(cb.get(k, []))

        row: Dict[str, Any] = {"cohort": k}
        row[f"{label_a}_n"] = sa["n"]
        row[f"{label_a}_acc"] = sa["acc"]
        row[f"{label_a}_mean_spatial_entropy"] = sa["mean_spatial_entropy"]
        row[f"{label_a}_mean_spectral_entropy"] = sa["mean_spectral_entropy"]
        row[f"{label_b}_n"] = sb["n"]
        row[f"{label_b}_acc"] = sb["acc"]
        row[f"{label_b}_mean_spatial_entropy"] = sb["mean_spatial_entropy"]
        row[f"{label_b}_mean_spectral_entropy"] = sb["mean_spectral_entropy"]
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
        "--expert_mode",
        type=str,
        default="assigned",
        choices=["assigned", "raw"],
        help="Which expert ids to summarize when reading a per-sample CSV (default: assigned).",
    )
    ap.add_argument(
        "--layer_idx",
        type=int,
        default=None,
        help="Optional MoE layer index to analyze when the CSV contains multiple MoE layers.",
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("CSV_NO_PSD", "CSV_PSD"),
        help="Two per-sample CSVs: first=no-PSD, second=PSD; writes compare_no_psd_vs_psd_by_*.csv (four files)",
    )
    ap.add_argument(
        "--compare_legacy_cohort",
        nargs=2,
        metavar=("CSV_A", "CSV_B"),
        help="Legacy: single cohort comparison with --label_a / --label_b column prefixes",
    )
    ap.add_argument("--label_a", type=str, default="no_psd", help="Tag for first CSV in --compare_legacy_cohort")
    ap.add_argument("--label_b", type=str, default="psd", help="Tag for second CSV in --compare_legacy_cohort")
    ap.add_argument(
        "--compare_prefix",
        type=str,
        default="compare_no_psd_vs_psd",
        help="Filename prefix for --compare outputs (default: compare_no_psd_vs_psd)",
    )
    args = ap.parse_args()

    if args.compare:
        paths = compare_no_psd_vs_psd_pair(
            args.compare[0], args.compare[1], args.outdir, out_prefix=args.compare_prefix
        )
        for dim, p in paths.items():
            print(f"[analyze] compare {dim}: {p}", flush=True)
        return

    if args.compare_legacy_cohort:
        p = compare_two_runs(
            args.compare_legacy_cohort[0],
            args.compare_legacy_cohort[1],
            args.outdir,
            args.label_a,
            args.label_b,
        )
        print(f"[analyze] wrote {p}", flush=True)
        return

    if not args.csv:
        ap.error("provide --csv or --compare")
    rows = load_per_sample_csv(args.csv)
    if not rows:
        raise SystemExit("empty CSV")
    analysis_rows = project_rows_for_analysis(rows, expert_mode=args.expert_mode, layer_idx=args.layer_idx)
    ne = _infer_num_experts(analysis_rows)
    outs = write_all_analyses(analysis_rows, args.outdir, args.prefix, ne)
    for k, v in outs.items():
        print(f"[analyze] {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
