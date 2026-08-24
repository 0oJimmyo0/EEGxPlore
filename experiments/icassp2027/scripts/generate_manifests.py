#!/usr/bin/env python3
"""Generate deterministic subject-disjoint ICASSP manifests.

The input is the audited ``all_samples.csv`` produced by
``extract_metadata.py``.  Subject assignment is performed on loader-level
``container_key`` groups, so ISRUC sequence containers are never split across
train/validation/test even though its metadata table has one row per epoch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


SPLITS = ("train", "val", "test")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("key_exists", "").lower() not in {"true", "1"}:
                raise ValueError(f"Metadata contains a missing key: {raw.get('sample_key')}")
            if not raw.get("subject_id"):
                raise ValueError(f"Metadata contains an empty subject_id: {raw.get('sample_key')}")
            try:
                label = int(raw["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label in metadata row: {raw}") from exc
            if label < 0:
                raise ValueError(f"Invalid negative label in metadata row: {raw}")
            rows.append(
                {
                    "sample_key": raw["sample_key"],
                    "container_key": raw["container_key"],
                    "subject_id": raw["subject_id"],
                    "session_id": raw.get("session_id", ""),
                    "recording_id": raw.get("recording_id", ""),
                    "label": label,
                    "existing_split": raw.get("existing_split", ""),
                }
            )
    if not rows:
        raise ValueError(f"Metadata table is empty: {path}")
    return rows


def _group_containers(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    containers: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row["container_key"])
        subject = str(row["subject_id"])
        if key in containers and containers[key]["subject_id"] != subject:
            raise ValueError(f"Container maps to multiple subjects: {key}")
        if key not in containers:
            containers[key] = {
                "container_key": key,
                "subject_id": subject,
                "rows": 0,
                "labels": Counter(),
                "existing_splits": set(),
            }
        containers[key]["rows"] += 1
        containers[key]["labels"][int(row["label"])] += 1
        if row.get("existing_split"):
            containers[key]["existing_splits"].add(str(row["existing_split"]))
    return containers


def _subject_stats(containers: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for container in containers.values():
        subject = str(container["subject_id"])
        if subject not in stats:
            stats[subject] = {"rows": 0, "labels": Counter(), "containers": []}
        stats[subject]["rows"] += int(container["rows"])
        stats[subject]["labels"].update(container["labels"])
        stats[subject]["containers"].append(str(container["container_key"]))
    return stats


def _target_subject_counts(n_subjects: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    train = max(1, int(round(n_subjects * ratios["train"])))
    val = max(1, int(round(n_subjects * ratios["val"])))
    test = n_subjects - train - val
    if test < 1:
        test = 1
        train = max(1, n_subjects - val - test)
    return {"train": train, "val": val, "test": test}


def _assignment_objective(
    assignment: Mapping[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    labels: Sequence[int],
    ratios: Mapping[str, float],
    target_subjects: Mapping[str, int],
) -> float:
    total_rows = sum(int(value["rows"]) for value in stats.values())
    total_labels = Counter()
    for value in stats.values():
        total_labels.update(value["labels"])
    split_subjects = Counter(assignment.values())
    split_rows = Counter()
    split_labels = {split: Counter() for split in SPLITS}
    for subject, split in assignment.items():
        split_rows[split] += int(stats[subject]["rows"])
        split_labels[split].update(stats[subject]["labels"])

    objective = 0.0
    for split in SPLITS:
        target_rows = max(1.0, ratios[split] * total_rows)
        objective += ((split_rows[split] - target_rows) / target_rows) ** 2
        objective += 0.5 * (
            (split_subjects[split] - target_subjects[split])
            / max(1, target_subjects[split])
        ) ** 2
        for label in labels:
            target_label = max(1.0, ratios[split] * total_labels[label])
            objective += 0.5 * (
                (split_labels[split][label] - target_label) / target_label
            ) ** 2
            if split_labels[split][label] == 0:
                objective += 1000.0
    return objective


def _assign_subjects(
    stats: Mapping[str, Mapping[str, Any]],
    seed: int,
    ratios: Mapping[str, float],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    subjects = sorted(stats)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for disjoint splits")
    labels = sorted({int(label) for value in stats.values() for label in value["labels"]})
    target_subjects = _target_subject_counts(len(subjects), ratios)

    best: Tuple[float, Dict[str, str], int] | None = None
    # A small deterministic restart set makes the greedy assignment robust to
    # subject-size and class-composition outliers without turning splitting
    # into a tunable model-selection procedure.
    for attempt in range(256):
        rng = random.Random(seed + attempt * 1009)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        order = sorted(
            shuffled,
            key=lambda subject: (
                -int(stats[subject]["rows"]),
                -len(stats[subject]["labels"]),
                shuffled.index(subject),
            ),
        )
        assignment: Dict[str, str] = {}
        split_rows = Counter()
        split_labels = {split: Counter() for split in SPLITS}
        split_counts = Counter()

        # Seed each split with a broad-support subject. This makes class
        # support explicit, especially for ISRUC's two four-class subjects.
        remaining = order[:]
        for split in SPLITS:
            candidates = [
                subject
                for subject in remaining
                if len(stats[subject]["labels"]) == len(labels)
            ] or remaining
            subject = min(
                candidates,
                key=lambda item: (
                    abs(int(stats[item]["rows"]) - ratios[split] * sum(
                        int(value["rows"]) for value in stats.values()
                    )),
                    item,
                ),
            )
            assignment[subject] = split
            remaining.remove(subject)
            split_counts[split] += 1
            split_rows[split] += int(stats[subject]["rows"])
            split_labels[split].update(stats[subject]["labels"])

        for subject in remaining:
            eligible = [
                split for split in SPLITS if split_counts[split] < target_subjects[split]
            ]
            if not eligible:
                raise RuntimeError("Subject target accounting exhausted early")

            def score(split: str) -> float:
                after_rows = split_rows[split] + int(stats[subject]["rows"])
                target_rows = ratios[split] * sum(
                    int(value["rows"]) for value in stats.values()
                )
                row_term = ((after_rows - target_rows) / max(1.0, target_rows)) ** 2
                after_subjects = split_counts[split] + 1
                subject_term = (
                    (after_subjects - target_subjects[split])
                    / max(1, target_subjects[split])
                ) ** 2
                label_term = 0.0
                for label in labels:
                    target_label = ratios[split] * sum(
                        int(value["labels"].get(label, 0)) for value in stats.values()
                    )
                    after_label = split_labels[split][label] + stats[subject]["labels"].get(label, 0)
                    label_term += ((after_label - target_label) / max(1.0, target_label)) ** 2
                return row_term + 0.5 * subject_term + 0.5 * label_term

            selected = min(eligible, key=lambda split: (score(split), split))
            assignment[subject] = selected
            split_counts[selected] += 1
            split_rows[selected] += int(stats[subject]["rows"])
            split_labels[selected].update(stats[subject]["labels"])

        objective = _assignment_objective(assignment, stats, labels, ratios, target_subjects)
        if best is None or objective < best[0]:
            best = (objective, assignment, attempt)

    if best is None:
        raise RuntimeError("Could not construct a subject assignment")
    objective, assignment, attempt = best
    missing = {
        split: [label for label in labels if not any(
            label in stats[subject]["labels"]
            for subject, assigned_split in assignment.items()
            if assigned_split == split
        )]
        for split in SPLITS
    }
    if any(missing.values()):
        raise RuntimeError(f"Class support failed after deterministic assignment: {missing}")
    return assignment, {
        "seed": seed,
        "restart_count": 256,
        "selected_restart": attempt,
        "objective": objective,
        "target_subject_counts": target_subjects,
        "target_sample_ratios": dict(ratios),
    }


def _canonical_manifest(manifest: Mapping[str, Sequence[str]]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate(dataset: str, metadata_path: Path, output_dir: Path, seed: int) -> None:
    rows = _read_rows(metadata_path)
    containers = _group_containers(rows)
    stats = _subject_stats(containers)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    assignment, generation = _assign_subjects(stats, seed, ratios)

    manifest = {
        split: sorted(
            key for key, container in containers.items()
            if assignment[str(container["subject_id"])] == split
        )
        for split in SPLITS
    }
    all_manifest_keys = [key for split in SPLITS for key in manifest[split]]
    duplicate_keys = sorted(key for key, count in Counter(all_manifest_keys).items() if count > 1)
    source_keys = set(containers)
    manifest_keys = set(all_manifest_keys)
    unknown_keys = sorted(manifest_keys - source_keys)
    missing_keys = sorted(source_keys - manifest_keys)

    split_subjects = {
        split: sorted({str(containers[key]["subject_id"]) for key in manifest[split]})
        for split in SPLITS
    }
    split_sample_counts = {
        split: sum(int(containers[key]["rows"]) for key in manifest[split])
        for split in SPLITS
    }
    split_class_counts = {
        split: dict(sorted(
            Counter(
                label
                for key in manifest[split]
                for label, count in containers[key]["labels"].items()
                for _ in range(int(count))
            ).items()
        ))
        for split in SPLITS
    }
    overlap = {
        f"{left}_{right}": sorted(set(split_subjects[left]) & set(split_subjects[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    source_split_counts = Counter(
        split
        for container in containers.values()
        for split in container["existing_splits"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _canonical_manifest(manifest)
    (output_dir / "split_manifest.json").write_bytes(manifest_bytes)
    (output_dir / "split_manifest.sha256").write_text(
        _sha256_bytes(manifest_bytes) + "  split_manifest.json\n", encoding="utf-8"
    )

    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split", "container_key", "subject_id", "rows", "labels", "existing_splits"],
        )
        writer.writeheader()
        for split in SPLITS:
            for key in manifest[split]:
                container = containers[key]
                writer.writerow({
                    "dataset": dataset,
                    "split": split,
                    "container_key": key,
                    "subject_id": container["subject_id"],
                    "rows": container["rows"],
                    "labels": json.dumps(dict(sorted(container["labels"].items())), sort_keys=True),
                    "existing_splits": json.dumps(sorted(container["existing_splits"])),
                })

    _write_json(output_dir / "subject_counts.json", {
        split: {"count": len(split_subjects[split]), "subjects": split_subjects[split]}
        for split in SPLITS
    })
    _write_json(output_dir / "sample_counts.json", split_sample_counts)
    _write_json(output_dir / "class_counts.json", split_class_counts)
    _write_json(output_dir / "overlap_audit.json", {
        "dataset": dataset,
        "subject_overlap": overlap,
        "passed": not any(overlap.values()),
    })
    _write_json(output_dir / "key_existence_audit.json", {
        "dataset": dataset,
        "metadata_rows": len(rows),
        "source_container_keys": len(source_keys),
        "manifest_container_keys": len(manifest_keys),
        "duplicate_manifest_keys": duplicate_keys,
        "unknown_manifest_keys": unknown_keys,
        "missing_manifest_keys": missing_keys,
        "metadata_key_exists_false": 0,
        "passed": not (duplicate_keys or unknown_keys or missing_keys),
    })
    _write_json(output_dir / "split_generation.json", {
        "dataset": dataset,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_bytes(metadata_path.read_bytes()),
        "source_rows": len(rows),
        "source_container_keys": len(source_keys),
        "source_existing_split_container_counts": dict(sorted(source_split_counts.items())),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "generation": generation,
        "class_labels": sorted({int(row["label"]) for row in rows}),
        "container_key_is_split_unit": True,
    })

    if any(overlap.values()) or duplicate_keys or unknown_keys or missing_keys:
        raise SystemExit(f"{dataset}: manifest audit failed; see {output_dir}")
    if any(not split_class_counts[split] for split in SPLITS):
        raise SystemExit(f"{dataset}: empty class-count split; see {output_dir}")
    print(json.dumps({
        "dataset": dataset,
        "output_dir": str(output_dir),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "subjects": {split: len(split_subjects[split]) for split in SPLITS},
        "containers": {split: len(manifest[split]) for split in SPLITS},
        "samples": split_sample_counts,
        "classes": split_class_counts,
        "existing_split_container_counts": dict(sorted(source_split_counts.items())),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(args.metadata_csv)
    _generate(args.dataset, args.metadata_csv, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
