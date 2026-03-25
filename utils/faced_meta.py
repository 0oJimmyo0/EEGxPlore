"""FACED metadata helpers for domain-aware MoE routing."""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict

UNKNOWN_ID = 0


def parse_faced_lmdb_key(key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lmdb_key": key,
        "source_file": "",
        "segment_index": -1,
        "chunk_index": -1,
        "sub_id": "",
    }
    parts = key.rsplit("-", 2)
    if len(parts) != 3:
        return out
    file, si, sj = parts
    out["source_file"] = file
    try:
        out["segment_index"] = int(si)
        out["chunk_index"] = int(sj)
    except ValueError:
        pass
    m = re.search(r"(sub\d+)", file, re.IGNORECASE)
    if m:
        out["sub_id"] = m.group(1).lower()
    return out


def _age_bucket(age: Any) -> str:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return ""
    if a < 22:
        return "<22"
    if a < 30:
        return "22-29"
    if a < 40:
        return "30-39"
    return "40+"


def _segment_bucket(segment_index: int) -> str:
    if segment_index < 0:
        return ""
    return f"seg_{min(7, segment_index // 20)}"


def load_recording_info_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Map sub_id (e.g. sub000) -> safe metadata fields."""
    if not path or not os.path.isfile(path):
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = (row.get("sub") or row.get("sub ") or "").strip().lower()
            if not sub:
                continue
            rows[sub] = {
                "cohort": (row.get("Cohort ") or row.get("Cohort") or "").strip(),
                "sample_rate_group": (row.get("Sample_rate") or "").strip(),
                "age_bucket": _age_bucket(row.get("Age")),
            }
    return rows


def _value_id_map(values) -> Dict[str, int]:
    vocab = sorted({str(v).strip() for v in values if str(v).strip()})
    return {v: i + 1 for i, v in enumerate(vocab)}


def build_faced_domain_maps(meta_csv_path: str) -> Dict[str, Any]:
    rec_map = load_recording_info_csv(meta_csv_path)
    return {
        "recordings": rec_map,
        "cohort_ids": _value_id_map(v.get("cohort", "") for v in rec_map.values()),
        "sample_rate_group_ids": _value_id_map(v.get("sample_rate_group", "") for v in rec_map.values()),
        "age_bucket_ids": _value_id_map(v.get("age_bucket", "") for v in rec_map.values()),
        "segment_bucket_ids": {f"seg_{i}": i + 1 for i in range(8)},
    }


def lmdb_key_to_domain_ids(key: str, domain_maps: Dict[str, Any]) -> Dict[str, int]:
    parsed = parse_faced_lmdb_key(key)
    sid = parsed.get("sub_id", "")
    rec = domain_maps.get("recordings", {}).get(sid, {})

    cohort = str(rec.get("cohort", "")).strip()
    sample_rate = str(rec.get("sample_rate_group", "")).strip()
    age_bucket = str(rec.get("age_bucket", "")).strip()
    segment_bucket = _segment_bucket(int(parsed.get("segment_index", -1)))

    cohort_id = domain_maps.get("cohort_ids", {}).get(cohort, UNKNOWN_ID)
    sample_rate_group_id = domain_maps.get("sample_rate_group_ids", {}).get(sample_rate, UNKNOWN_ID)
    age_bucket_id = domain_maps.get("age_bucket_ids", {}).get(age_bucket, UNKNOWN_ID)
    segment_bucket_id = domain_maps.get("segment_bucket_ids", {}).get(segment_bucket, UNKNOWN_ID)

    return {
        "cohort_id": int(cohort_id),
        "sample_rate_group_id": int(sample_rate_group_id),
        "age_bucket_id": int(age_bucket_id),
        "segment_bucket_id": int(segment_bucket_id),
    }


def join_meta_for_key(key: str, rec_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    parsed = parse_faced_lmdb_key(key)
    sid = parsed.get("sub_id") or ""
    base = dict(parsed)
    if sid and sid in rec_map:
        base.update(rec_map[sid])
    else:
        base.setdefault("cohort", "")
        base.setdefault("sample_rate_group", "")
        base.setdefault("age_bucket", "")
    base["segment_bucket"] = _segment_bucket(int(base.get("segment_index", -1)))
    return base
