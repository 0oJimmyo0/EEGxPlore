#!/usr/bin/env python3
"""Extract auditable per-sample metadata for the ICASSP routing study.

This script is intentionally read-only with respect to the source datasets. It
writes CSV/JSON audit artifacts only under the requested experiment output
directory. ISRUC is represented at epoch granularity while retaining the
sequence-file ``container_key`` used by its loader.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import pickletools
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FIELDS = [
    "sample_key",
    "container_key",
    "subject_id",
    "session_id",
    "recording_id",
    "label",
    "existing_split",
    "key_exists",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _key_text(key: Any) -> str:
    return _text(key)


def _parse_seedv_key(key: str) -> Dict[str, str]:
    match = re.match(r"^([^_]+)_([^_]+)_t(\d+)_g(\d+)$", key)
    if match:
        subject, session, trial, segment = match.groups()
        return {
            "subject_id": subject,
            "session_id": session,
            "recording_id": f"{subject}_{session}_t{trial}",
        }

    parts = key.rsplit("-", 2)
    prefix = parts[0] if len(parts) == 3 else key
    fields = prefix.split("_")
    subject = fields[0] if fields else ""
    session = fields[1] if len(fields) > 1 else ""
    return {
        "subject_id": subject,
        "session_id": session,
        "recording_id": prefix,
    }


def _parse_faced_key(key: str) -> Dict[str, str]:
    parts = key.rsplit("-", 2)
    source_file = parts[0] if len(parts) == 3 else key
    match = re.search(r"(sub\d+)", source_file, re.IGNORECASE)
    return {
        "subject_id": match.group(1).lower() if match else "",
        "session_id": "",
        "recording_id": source_file,
    }


def _parse_physio_key(key: str) -> Dict[str, str]:
    match = re.match(r"^([A-Za-z]+\d+)R(\d+)-(.+)$", key)
    if not match:
        return {"subject_id": "", "session_id": "", "recording_id": ""}
    subject, run, _ = match.groups()
    return {
        "subject_id": subject,
        "session_id": f"R{run}",
        "recording_id": f"{subject}R{run}",
    }


def _scalar_label(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


_INTEGER_OPCODES = {
    "BININT",
    "BININT1",
    "BININT2",
    "BININT4",
    "LONG1",
    "LONG4",
}


def _pickle_label(raw: bytes) -> int:
    """Read the dict's scalar ``label`` without materializing the EEG array.

    The LMDB values pickle a large NumPy array before the label.  The parser
    therefore locates the final ``label`` key and disassembles only the small
    reconstruction tail, never the array payload.  The current writers encode
    labels either as a plain integer (SEED-V) or as a NumPy scalar
    (FACED/PhysioNet-MI).
    """

    # All current writers use SHORT_BINUNICODE for this key.  Keep a
    # BINUNICODE fallback for future preprocessing revisions.
    short_marker = b"\x8c\x05label"
    long_marker = b"X\x05\x00\x00\x00label"
    start = raw.rfind(short_marker)
    if start >= 0:
        tail = raw[start:]
    else:
        start = raw.rfind(long_marker)
        if start < 0:
            return -1
        tail = raw[start:]

    # genops only needs a valid protocol prefix to disassemble this isolated
    # tail; memo references are intentionally not dereferenced.
    operations = list(pickletools.genops(b"\x80\x04" + tail))
    label_positions = [
        index for index, (_, argument, _) in enumerate(operations) if argument == "label"
    ]
    if not label_positions:
        return -1
    label_index = label_positions[-1]
    following = operations[label_index + 1 : label_index + 48]
    # NumPy scalar pickles carry the value in the final short byte string of
    # the scalar reconstruction rather than in an integer opcode.
    scalar_bytes = [
        argument
        for opcode, argument, _ in following
        if opcode.name in {"BINBYTES", "SHORT_BINBYTES", "BINBYTES8"}
        and isinstance(argument, bytes)
        and len(argument) <= 16
    ]
    if scalar_bytes:
        return int.from_bytes(scalar_bytes[-1], byteorder="little", signed=True)
    for opcode, argument, _ in following:
        if opcode.name in _INTEGER_OPCODES:
            return _scalar_label(argument)
    return -1


def _sample_shape(raw: bytes) -> str:
    """Materialize only a bounded number of payloads to record shape metadata."""

    payload = pickle.loads(raw)
    sample = payload.get("sample") if isinstance(payload, dict) else None
    return str(tuple(sample.shape)) if hasattr(sample, "shape") else ""


def _load_lmdb_rows(dataset: str, data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import lmdb

    db = lmdb.open(str(data_dir), readonly=True, lock=False, readahead=False, meminit=False)
    rows: List[Dict[str, Any]] = []
    missing_keys = 0
    shapes: Dict[str, str] = {}

    with db.begin(write=False) as txn:
        raw_index = txn.get(b"__keys__")
        if raw_index is None:
            raise RuntimeError(f"{dataset}: LMDB is missing __keys__: {data_dir}")
        split_index = pickle.loads(raw_index)

        for split in ("train", "val", "test"):
            for raw_key in split_index.get(split, []):
                key = _key_text(raw_key)
                encoded = raw_key if isinstance(raw_key, bytes) else key.encode("utf-8")
                raw = txn.get(encoded)
                if raw is None:
                    missing_keys += 1
                    rows.append(
                        {
                            "sample_key": key,
                            "container_key": key,
                            "subject_id": "",
                            "session_id": "",
                            "recording_id": "",
                            "label": -1,
                            "existing_split": split,
                            "key_exists": False,
                        }
                    )
                    continue

                if dataset == "SEED-V":
                    parsed = _parse_seedv_key(key)
                elif dataset == "FACED":
                    parsed = _parse_faced_key(key)
                elif dataset == "PhysioNet-MI":
                    parsed = _parse_physio_key(key)
                else:
                    raise ValueError(f"Unsupported LMDB dataset: {dataset}")
                rows.append(
                    {
                        "sample_key": key,
                        "container_key": key,
                        **parsed,
                        "label": _pickle_label(raw),
                        "existing_split": split,
                        "key_exists": True,
                    }
                )
                if split not in shapes:
                    shapes[split] = _sample_shape(raw)

    return rows, {"missing_keys": missing_keys, "example_shapes": shapes, "split_index_counts": {
        split: len(split_index.get(split, [])) for split in ("train", "val", "test")
    }}


def _load_isruc_rows(data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import numpy as np

    seq_root = data_dir / "seq"
    label_root = data_dir / "labels"
    if not seq_root.is_dir() or not label_root.is_dir():
        raise RuntimeError(f"ISRUC requires seq/ and labels/ under {data_dir}")

    rows: List[Dict[str, Any]] = []
    missing_pairs = 0
    shape_counts: Counter[str] = Counter()
    sequence_count = 0

    subject_dirs = sorted(
        (p for p in seq_root.iterdir() if p.is_dir() and re.match(r"ISRUC-group1-\d+$", p.name)),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name.rsplit("-", 1)[1]
        subject_num = int(subject_id)
        existing_split = "train" if subject_num <= 80 else "val" if subject_num <= 90 else "test"
        label_dir = label_root / subject_dir.name
        for seq_path in sorted(subject_dir.glob("*.npy")):
            label_path = label_dir / seq_path.name
            if not label_path.is_file():
                missing_pairs += 1
                continue
            sequence_count += 1
            seq = np.load(seq_path, mmap_mode="r")
            labels = np.asarray(np.load(label_path, mmap_mode="r")).reshape(-1)
            shape_counts[str(tuple(seq.shape))] += 1
            if len(seq) != len(labels):
                raise RuntimeError(
                    f"ISRUC sequence/label length mismatch: {seq_path} has {len(seq)} samples and {len(labels)} labels"
                )
            container_key = f"{subject_dir.name}/{seq_path.name}"
            for epoch_idx, label in enumerate(labels):
                rows.append(
                    {
                        "sample_key": f"{container_key}::epoch_{epoch_idx:04d}",
                        "container_key": container_key,
                        "subject_id": subject_id,
                        "session_id": "",
                        "recording_id": container_key,
                        "label": _scalar_label(label),
                        "existing_split": existing_split,
                        "key_exists": True,
                    }
                )

    return rows, {
        "missing_sequence_label_pairs": missing_pairs,
        "sequence_count": sequence_count,
        "sequence_shapes": dict(shape_counts),
    }


def _write_outputs(rows: Iterable[Dict[str, Any]], audit: Dict[str, Any], output_dir: Path, dataset: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    csv_path = output_dir / "all_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)

    subjects = sorted({str(row["subject_id"]) for row in rows if row.get("subject_id")})
    labels = Counter(str(row["label"]) for row in rows if int(row.get("label", -1)) >= 0)
    split_counts = Counter(str(row["existing_split"]) for row in rows)
    audit = {
        "dataset": dataset,
        "rows": len(rows),
        "subjects": len(subjects),
        "subject_ids": subjects,
        "missing_subject_rows": sum(not row.get("subject_id") for row in rows),
        "missing_or_invalid_labels": sum(int(row.get("label", -1)) < 0 for row in rows),
        "missing_key_rows": sum(not row.get("key_exists") for row in rows),
        "label_counts": dict(sorted(labels.items())),
        "existing_split_counts": dict(sorted(split_counts.items())),
        **audit,
    }
    with (output_dir / "metadata_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    if audit["missing_subject_rows"] or audit["missing_or_invalid_labels"] or audit["missing_key_rows"]:
        raise SystemExit(
            f"{dataset}: metadata audit failed; see {output_dir / 'metadata_audit.json'}"
        )

    print(json.dumps({"dataset": dataset, "output_dir": str(output_dir), **audit}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["SEED-V", "FACED", "ISRUC", "PhysioNet-MI"], required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if args.dataset == "ISRUC":
        rows, audit = _load_isruc_rows(args.data_dir)
    else:
        rows, audit = _load_lmdb_rows(args.dataset, args.data_dir)
    _write_outputs(rows, audit, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
