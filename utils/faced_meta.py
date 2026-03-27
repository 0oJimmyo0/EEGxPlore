"""FACED metadata helpers for domain/adaptation-aware training."""

from __future__ import annotations

import csv
import json
import os
import re
import pickle
from typing import Any, Dict, Iterable

import lmdb

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


def _subject_id_map_from_sub_ids(sub_ids: Iterable[str]) -> Dict[str, int]:
    vocab = sorted({str(v).strip().lower() for v in sub_ids if str(v).strip()})
    return {v: i + 1 for i, v in enumerate(vocab)}


def build_subject_id_map_from_lmdb(data_dir: str) -> Dict[str, int]:
    if not data_dir or not os.path.isdir(data_dir):
        return {}
    db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with db.begin(write=False) as txn:
            blob = txn.get("__keys__".encode())
            if blob is None:
                return {}
            by_split = pickle.loads(blob)
    except Exception:
        return {}
    finally:
        db.close()

    subs = []
    for split_keys in by_split.values():
        for key in split_keys:
            kstr = key.decode() if isinstance(key, bytes) else str(key)
            subs.append(parse_faced_lmdb_key(kstr).get("sub_id", ""))
    return _subject_id_map_from_sub_ids(subs)


def load_subject_summary_map(path: str) -> Dict[str, Any]:
    """Optional JSON/PT map: sub_id -> compact numeric summary vector."""
    if not path or not os.path.isfile(path):
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    elif ext in {".pt", ".pth"}:
        import torch
        blob = torch.load(path, map_location="cpu")
    else:
        raise ValueError("subject_summary_file must be .json/.pt/.pth")
    if not isinstance(blob, dict):
        raise ValueError("subject_summary_file must contain a dict mapping sub_id -> vector")
    out: Dict[str, Any] = {}
    for k, v in blob.items():
        kk = str(k).strip().lower()
        if kk:
            out[kk] = v
    return out


def build_faced_domain_maps(meta_csv_path: str) -> Dict[str, Any]:
    rec_map = load_recording_info_csv(meta_csv_path)
    return {
        "recordings": rec_map,
        "cohort_ids": _value_id_map(v.get("cohort", "") for v in rec_map.values()),
        "sample_rate_group_ids": _value_id_map(v.get("sample_rate_group", "") for v in rec_map.values()),
        "age_bucket_ids": _value_id_map(v.get("age_bucket", "") for v in rec_map.values()),
        "segment_bucket_ids": {f"seg_{i}": i + 1 for i in range(8)},
    }


def build_faced_meta_maps(
    data_dir: str,
    meta_csv_path: str,
    subject_summary_file: str = "",
    use_subject_summary: bool = False,
) -> Dict[str, Any]:
    domain = build_faced_domain_maps(meta_csv_path)
    domain["subject_ids"] = build_subject_id_map_from_lmdb(data_dir)
    domain["subject_summaries"] = load_subject_summary_map(subject_summary_file) if use_subject_summary else {}
    domain["dataset_id"] = 1  # FACED
    return domain


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


def lmdb_key_to_subject_meta(key: str, meta_maps: Dict[str, Any]) -> Dict[str, Any]:
    parsed = parse_faced_lmdb_key(key)
    sid = parsed.get("sub_id", "")
    domain_ids = lmdb_key_to_domain_ids(key, meta_maps)
    subject_id = int(meta_maps.get("subject_ids", {}).get(sid, UNKNOWN_ID))
    dataset_id = int(meta_maps.get("dataset_id", 1))
    summary = meta_maps.get("subject_summaries", {}).get(sid)
    return {
        "subject_id": subject_id,
        "dataset_id": dataset_id,
        "subject_summary": summary,
        **domain_ids,
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
