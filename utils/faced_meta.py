"""FACED Recording_info.csv join helpers (analysis only; not for model input)."""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict, List, Optional


def parse_faced_lmdb_key(key: str) -> Dict[str, Any]:
    """
    Keys from preprocessing_faced: '{file}-{segment_i}-{chunk_j}'.
    file = original pickle filename (often contains subXXX).
    """
    out: Dict[str, Any] = {
        "lmdb_key": key,
        "source_file": "",
        "segment_index": "",
        "chunk_index": "",
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


def load_recording_info_csv(path: str) -> Dict[str, Dict[str, Any]]:
    """Map sub_id (e.g. sub000) -> safe metadata fields."""
    if not path or not os.path.isfile(path):
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = (row.get("sub") or row.get("sub ") or "").strip()
            if not sub:
                continue
            sub_l = sub.lower()
            cohort = (row.get("Cohort ") or row.get("Cohort") or "").strip()
            sr = (row.get("Sample_rate") or "").strip()
            age = row.get("Age")
            rows[sub_l] = {
                "sub": sub_l,
                "cohort": cohort,
                "sample_rate_group": sr,
                "age": age,
                "age_bucket": _age_bucket(age),
                "gender": (row.get("Gender") or "").strip(),
            }
    return rows


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
    return base
