"""Summarize replayed class- and subject-level ICASSP diagnostics.

The input files are produced by replay_selected_diagnostics.py. Class deltas
are paired within seed. PhysioNet-MI subject deltas first average each
subject's BA across the three matched seeds, then bootstrap subjects (not EEG
windows) for confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev


CONDITIONS = ("full", "full_attnres_only", "specialist_augmented_full")
CLASS_LABELS = {
    "ISRUC": ["W", "N1", "N2", "N3", "REM"],
    "PhysioNet-MI": ["Class 0", "Class 1", "Class 2", "Class 3"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_sd(values):
    values = [float(value) for value in values]
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def _bootstrap(values, statistic, seed=2027, draws=10000):
    import random

    values = [float(value) for value in values]
    rng = random.Random(seed)
    samples = [statistic([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws)]
    samples.sort()
    return {
        "estimate": statistic(values),
        "ci95": [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]],
        "n_subjects": len(values),
        "draws": draws,
        "seed": seed,
    }


def _records(paths):
    all_records = []
    sources = []
    for raw_path in paths:
        path = raw_path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"path": str(path), "sha256": _sha256(path)})
        all_records.extend(payload.get("records", []))
    return all_records, sources


def _by_dataset_condition_seed(records):
    grouped = {}
    for record in records:
        dataset = str(record["dataset"])
        condition = str(record["condition"])
        seed = int(record["seed"])
        if condition not in CONDITIONS:
            raise ValueError(f"unexpected condition {condition!r}")
        paper_eligible = record.get("paper_eligible")
        if paper_eligible is None:
            manifest_path = Path(str(record["run_root"])) / "result_manifest.json"
            if manifest_path.is_file():
                paper_eligible = json.loads(manifest_path.read_text(encoding="utf-8")).get("paper_eligible")
        if paper_eligible is not True:
            raise ValueError(f"diagnostic row is not paper eligible: {dataset}/{condition}/{seed}")
        grouped.setdefault(dataset, {}).setdefault(condition, {})[seed] = record
    return grouped


def _classwise(dataset, grouped):
    required = [set(grouped.get(condition, {})) for condition in CONDITIONS]
    seeds = sorted(set.intersection(*required))
    if not seeds:
        raise ValueError(f"no matched seeds for {dataset}")
    labels = CLASS_LABELS.get(dataset, [f"Class {idx}" for idx in range(len(grouped["full"][seeds[0]]["test_metrics"]["classwise_recall"]))])
    result = {"seeds": seeds, "class_labels": labels, "per_seed": {}, "mean_sd": {}}
    for seed in seeds:
        metrics = {
            condition: grouped[condition][seed]["test_metrics"] for condition in CONDITIONS
        }
        recalls = {condition: metrics[condition]["classwise_recall"] for condition in CONDITIONS}
        if not all(len(recalls[condition]) == len(labels) for condition in CONDITIONS):
            raise ValueError(f"class count mismatch in {dataset} seed {seed}")
        result["per_seed"][str(seed)] = {
            "recall": recalls,
            "delta_attnres": [a - f for a, f in zip(recalls["full_attnres_only"], recalls["full"])],
            "delta_specialist_given_attnres": [s - a for s, a in zip(recalls["specialist_augmented_full"], recalls["full_attnres_only"])],
        }

    for key, left, right in (
        ("delta_attnres", "full_attnres_only", "full"),
        ("delta_specialist_given_attnres", "specialist_augmented_full", "full_attnres_only"),
    ):
        result["mean_sd"][key] = []
        for index in range(len(labels)):
            values = [result["per_seed"][str(seed)][key][index] for seed in seeds]
            result["mean_sd"][key].append(_mean_sd(values))
    return result


def _subject_level(grouped):
    conditions = {condition: grouped[condition] for condition in CONDITIONS}
    seeds = sorted(set.intersection(*(set(value) for value in conditions.values())))
    subject_sets = []
    for condition in CONDITIONS:
        for seed in seeds:
            subject_sets.append(set(conditions[condition][seed]["test_metrics"].get("subject_metrics", {})))
    subjects = sorted(set.intersection(*subject_sets))
    if not subjects:
        raise ValueError("no matched subject metrics")

    by_subject = {}
    for subject in subjects:
        subject_ba = {}
        for condition in CONDITIONS:
            subject_ba[condition] = mean(
                float(conditions[condition][seed]["test_metrics"]["subject_metrics"][subject]["balanced_accuracy"])
                for seed in seeds
            )
        by_subject[subject] = {
            "balanced_accuracy": subject_ba,
            "delta_attnres": subject_ba["full_attnres_only"] - subject_ba["full"],
            "delta_specialist_given_attnres": subject_ba["specialist_augmented_full"] - subject_ba["full_attnres_only"],
        }

    summary = {"seeds": seeds, "n_subjects": len(subjects), "by_subject": by_subject, "effects": {}}
    for key in ("delta_attnres", "delta_specialist_given_attnres"):
        values = [row[key] for row in by_subject.values()]
        summary["effects"][key] = {
            "positive_subjects": sum(value > 0 for value in values),
            "negative_subjects": sum(value < 0 for value in values),
            "zero_subjects": sum(value == 0 for value in values),
            "mean_sd": _mean_sd(values),
            "median_bootstrap": _bootstrap(values, median, seed=2027),
            "mean_bootstrap": _bootstrap(values, mean, seed=2028),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records, sources = _records(args.input)
    grouped = _by_dataset_condition_seed(records)
    datasets = sorted(grouped)
    payload = {
        "schema": "icassp2027_stage_diagnostics_summary_v1",
        "sources": sources,
        "datasets": {},
    }
    for dataset in datasets:
        payload["datasets"][dataset] = {"classwise": _classwise(dataset, grouped[dataset])}
        if dataset == "PhysioNet-MI":
            payload["datasets"][dataset]["subject_level"] = _subject_level(grouped[dataset])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
