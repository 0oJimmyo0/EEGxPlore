#!/usr/bin/env python3
"""Audit ICASSP metadata and frozen manifests as one reproducible bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


DATASETS = {
    "SEED-V": ("seedv", "datasets/seedv", "manifests/seedv"),
    "FACED": ("faced", "datasets/faced", "manifests/faced"),
    "ISRUC": ("isruc", "datasets/isruc", "manifests/isruc"),
    "PhysioNet-MI": ("physionet_mi", "datasets/physionet_mi", "manifests/physionet_mi"),
}
SPLITS = ("train", "val", "test")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_metadata(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({
                "sample_key": raw["sample_key"],
                "container_key": raw["container_key"],
                "subject_id": raw["subject_id"],
                "label": int(raw["label"]),
                "existing_split": raw.get("existing_split", ""),
                "key_exists": raw.get("key_exists", "").lower() in {"true", "1"},
            })
    return rows


def _overlaps(groups: Mapping[str, Set[str]]) -> Dict[str, List[str]]:
    return {
        f"{left}_{right}": sorted(groups[left] & groups[right])
        for left, right in combinations(SPLITS, 2)
    }


def _audit_dataset(name: str, root: Path, metadata_rel: str, manifest_rel: str) -> Dict[str, Any]:
    metadata_path = root / metadata_rel / "all_samples.csv"
    manifest_dir = root / manifest_rel
    manifest_path = manifest_dir / "split_manifest.json"
    rows = _read_metadata(metadata_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    container_subject: Dict[str, str] = {}
    container_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["key_exists"] or not row["subject_id"] or row["label"] < 0:
            raise RuntimeError(f"{name}: invalid metadata row {row}")
        key = row["container_key"]
        if key in container_subject and container_subject[key] != row["subject_id"]:
            raise RuntimeError(f"{name}: container has multiple subjects: {key}")
        container_subject[key] = row["subject_id"]
        container_rows[key].append(row)

    fresh_subjects = {
        split: {container_subject[key] for key in manifest.get(split, [])}
        for split in SPLITS
    }
    fresh_keys = [key for split in SPLITS for key in manifest.get(split, [])]
    fresh_key_set = set(fresh_keys)
    source_key_set = set(container_subject)
    fresh_class_counts = {
        split: dict(sorted(Counter(
            row["label"]
            for key in manifest.get(split, [])
            for row in container_rows.get(key, [])
        ).items()))
        for split in SPLITS
    }

    existing_subjects: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        if row["existing_split"]:
            existing_subjects[row["existing_split"]].add(row["subject_id"])
    existing_overlap = {
        f"{left}_{right}": sorted(existing_subjects[left] & existing_subjects[right])
        for left, right in combinations(sorted(existing_subjects), 2)
    }

    manifest_bytes = manifest_path.read_bytes()
    stored_hash = (manifest_dir / "split_manifest.sha256").read_text(encoding="utf-8").split()[0]
    computed_hash = _sha256(manifest_bytes)
    unknown_keys = sorted(fresh_key_set - source_key_set)
    missing_keys = sorted(source_key_set - fresh_key_set)
    duplicate_keys = sorted(key for key, count in Counter(fresh_keys).items() if count > 1)
    fresh_overlap = _overlaps(fresh_subjects)
    missing_classes = {
        split: sorted(set(label for row in rows for label in [row["label"]]) - set(fresh_class_counts[split]))
        for split in SPLITS
    }

    passed = not any([
        unknown_keys,
        missing_keys,
        duplicate_keys,
        any(fresh_overlap.values()),
        any(missing_classes.values()),
        stored_hash != computed_hash,
    ])
    return {
        "dataset": name,
        "metadata_rows": len(rows),
        "source_container_keys": len(source_key_set),
        "source_subjects": len(set(container_subject.values())),
        "fresh_manifest_sha256": computed_hash,
        "stored_manifest_sha256": stored_hash,
        "manifest_hash_match": stored_hash == computed_hash,
        "fresh_split_subject_counts": {split: len(fresh_subjects[split]) for split in SPLITS},
        "fresh_split_container_counts": {split: len(manifest.get(split, [])) for split in SPLITS},
        "fresh_split_sample_counts": {
            split: sum(len(container_rows[key]) for key in manifest.get(split, []))
            for split in SPLITS
        },
        "fresh_class_counts": fresh_class_counts,
        "fresh_subject_overlap": fresh_overlap,
        "existing_split_subject_counts": {
            split: len(subjects) for split, subjects in sorted(existing_subjects.items())
        },
        "existing_split_subject_overlap": existing_overlap,
        "unknown_manifest_keys": unknown_keys,
        "missing_manifest_keys": missing_keys,
        "duplicate_manifest_keys": duplicate_keys,
        "missing_classes_by_fresh_split": missing_classes,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/icassp2027"))
    args = parser.parse_args()
    results = {
        name: _audit_dataset(name, args.root, metadata_rel, manifest_rel)
        for name, (_, metadata_rel, manifest_rel) in DATASETS.items()
    }
    summary = {
        "datasets": results,
        "all_fresh_manifest_gates_passed": all(result["passed"] for result in results.values()),
        "note": "Existing split overlap is diagnostic only; fresh manifests are the ICASSP protocol.",
    }
    output_dir = args.root / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "phase0_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "passed", "metadata_rows", "source_subjects", "train_subjects", "val_subjects", "test_subjects", "existing_overlap_pairs"],
        )
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({
                "dataset": name,
                "passed": result["passed"],
                "metadata_rows": result["metadata_rows"],
                "source_subjects": result["source_subjects"],
                "train_subjects": result["fresh_split_subject_counts"]["train"],
                "val_subjects": result["fresh_split_subject_counts"]["val"],
                "test_subjects": result["fresh_split_subject_counts"]["test"],
                "existing_overlap_pairs": sum(bool(value) for value in result["existing_split_subject_overlap"].values()),
            })
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_fresh_manifest_gates_passed"]:
        raise SystemExit("Phase 0 manifest audit failed")


if __name__ == "__main__":
    main()
