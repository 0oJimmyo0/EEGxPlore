"""Aggregate the audited ICASSP confirmatory rows and paired deltas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

from audit_confirmatory_matrix import audit_matrix
from common import DEFAULT_OUTPUT_ROOT, DEFAULT_PAPER_MANIFEST


METRICS = (
    "test_balanced_accuracy",
    "test_weighted_f1",
    "test_kappa",
    "test_macro_f1",
)
EFFICIENCY = (
    "runtime_seconds",
    "peak_cuda_memory_mb",
    "trainable_parameters",
    "total_parameters",
)


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"n": 0, "mean": None, "sd": None}
    return {
        "n": len(numbers),
        "mean": mean(numbers),
        "sd": stdev(numbers) if len(numbers) > 1 else 0.0,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["dataset"]), int(row["seed"])), {})[str(row["executable_condition"])] = row

    output: List[Dict[str, Any]] = []
    for (dataset, seed), methods in sorted(by_key.items()):
        full = methods.get("full")
        specialist = methods.get("specialist_augmented_full")
        if full is None or specialist is None:
            raise ValueError(f"missing paired method for {dataset} seed {seed}")
        row: Dict[str, Any] = {"dataset": dataset, "seed": seed}
        for metric in METRICS:
            full_value = float(full[metric])
            specialist_value = float(specialist[metric])
            row[f"full_{metric}"] = full_value
            row[f"specialist_{metric}"] = specialist_value
            row[f"delta_{metric}"] = specialist_value - full_value
        row["full_runtime_seconds"] = full["runtime_seconds"]
        row["specialist_runtime_seconds"] = specialist["runtime_seconds"]
        row["runtime_ratio_specialist_over_full"] = specialist["runtime_seconds"] / full["runtime_seconds"]
        row["full_peak_cuda_memory_mb"] = full["peak_cuda_memory_mb"]
        row["specialist_peak_cuda_memory_mb"] = specialist["peak_cuda_memory_mb"]
        row["peak_memory_ratio_specialist_over_full"] = specialist["peak_cuda_memory_mb"] / full["peak_cuda_memory_mb"]
        row["full_trainable_parameters"] = full["trainable_parameters"]
        row["specialist_trainable_parameters"] = specialist["trainable_parameters"]
        row["trainable_parameter_delta"] = specialist["trainable_parameters"] - full["trainable_parameters"]
        row["trainable_parameter_ratio_specialist_over_full"] = specialist["trainable_parameters"] / full["trainable_parameters"]
        output.append(row)
    return output


def _aggregate(rows: List[Dict[str, Any]], pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_dataset_method": {}, "paired_by_dataset": {}}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        summary["by_dataset_method"][dataset] = {}
        for method in ("full", "specialist_augmented_full"):
            selected = [row for row in rows if row["dataset"] == dataset and row["executable_condition"] == method]
            summary["by_dataset_method"][dataset][method] = {
                metric: _summary(float(row[metric]) for row in selected) for metric in METRICS
            }
            for field in EFFICIENCY:
                summary["by_dataset_method"][dataset][method][field] = _summary(
                    float(row[field]) for row in selected
                )

        selected_pairs = [row for row in pairs if row["dataset"] == dataset]
        summary["paired_by_dataset"][dataset] = {
            metric: _summary(float(row[f"delta_{metric}"]) for row in selected_pairs)
            for metric in METRICS
        }
        summary["paired_by_dataset"][dataset]["positive_delta_count"] = {
            metric: sum(float(row[f"delta_{metric}"]) > 0 for row in selected_pairs)
            for metric in METRICS
        }
        summary["paired_by_dataset"][dataset]["pair_count"] = len(selected_pairs)
    return summary


def aggregate(audit: Dict[str, Any]) -> Dict[str, Any]:
    if not audit.get("passed"):
        raise ValueError("confirmatory audit did not pass; refusing to aggregate incomplete or mismatched rows")
    rows = list(audit.get("rows", []))
    expected_count = int(audit.get("expected_cell_count", 0) or 0)
    if len(rows) != expected_count or any(row.get("status") != "pass" for row in rows):
        raise ValueError(f"expected exactly {expected_count} passing confirmatory rows")
    pairs = _pair_rows(rows)
    return {
        "schema_version": 1,
        "training_commit": audit.get("training_commit"),
        "paper_manifest": audit.get("paper_manifest"),
        "source_audit_passed": True,
        "seed_rows": rows,
        "paired_rows": pairs,
        "aggregate": _aggregate(rows, pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_OUTPUT_ROOT / "confirmatory_audit.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.audit.is_file():
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
    else:
        audit = audit_matrix(args.output_root, args.paper_manifest)
    result = aggregate(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "confirmatory_aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "confirmatory_seed_results.csv", result["seed_rows"])
    _write_csv(args.output_dir / "confirmatory_paired_deltas.csv", result["paired_rows"])
    print(json.dumps({"seed_rows": len(result["seed_rows"]), "paired_rows": len(result["paired_rows"]), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
