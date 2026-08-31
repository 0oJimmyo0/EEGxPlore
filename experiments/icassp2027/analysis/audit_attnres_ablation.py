"""Audit the prespecified Full FT + AttnRes component-control runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "revision" / "paper_attnres_ablation_manifest_v1.csv"
DEFAULT_OUTPUT = ROOT.parent.parent / "output" / "icassp2027_revision"
DATASET_TAG = {
    "FACED": "faced",
    "ISRUC": "isruc",
    "SEED-V": "seedv",
    "PhysioNet-MI": "physionet_mi",
}
SEEDS = {"3407", "2024", "2027"}
EXPECTED_CONDITION = "full_attnres_only"
EXPECTED_RECIPE = "icassp2027_full_attnres_only_v1"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_last(path: Path) -> dict[str, str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return {}
    return dict(rows[-1]) if rows else {}


def audit(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        planned = list(csv.DictReader(handle))
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    expected = {
        (row.get("dataset", ""), row.get("executable_condition", ""), row.get("seed", ""))
        for row in planned
    }
    if len(planned) != 12 or len(expected) != 12:
        errors.append(f"component manifest must contain 12 unique rows, found {len(expected)}")

    for row in planned:
        dataset = str(row.get("dataset", ""))
        condition = str(row.get("executable_condition", ""))
        seed = str(row.get("seed", ""))
        identity = f"{dataset}/{condition}/{seed}"
        if condition != EXPECTED_CONDITION:
            errors.append(f"{identity}: unexpected condition")
        if seed not in SEEDS:
            errors.append(f"{identity}: unexpected seed")
        if row.get("source_reference") != "paper_method_attnres_only_v1":
            errors.append(f"{identity}: incorrect source reference")
        run_dir = output_root / DATASET_TAG.get(dataset, dataset) / condition / f"seed_{seed}"
        run = read_json(run_dir / "run_manifest.json")
        result = read_json(run_dir / "result_manifest.json")
        summary = read_last(run_dir / "experiment_summary.csv")
        missing = [str(path) for path in (
            run_dir / "run_manifest.json",
            run_dir / "result_manifest.json",
            run_dir / "experiment_summary.csv",
        ) if not path.is_file()]
        if missing:
            errors.extend(f"{identity}: missing {path}" for path in missing)
            continue
        if int(result.get("exit_code", 1)) != 0 or result.get("paper_eligible") is not True:
            errors.append(f"{identity}: result manifest is not a successful paper-eligible run")
        if run.get("condition") != condition or str(run.get("seed")) != seed:
            errors.append(f"{identity}: run identity mismatch")
        if run.get("paper_component_recipe_id") != EXPECTED_RECIPE:
            errors.append(f"{identity}: component recipe provenance missing or incorrect")
        expected_fields = {
            "trainability_mode": "full",
            "attnres_variant": "pre_attn",
            "moe": "False",
        }
        config = read_json(run_dir / "config_audit.json")
        for field, expected_value in expected_fields.items():
            actual = config.get(field)
            if str(actual) != expected_value:
                errors.append(f"{identity}: config {field}={actual!r}, expected {expected_value!r}")
        for field in ("test_balanced_accuracy", "test_weighted_f1", "test_kappa"):
            try:
                value = float(summary[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{identity}: missing metric {field}")
                continue
            if not math.isfinite(value):
                errors.append(f"{identity}: non-finite metric {field}")
        rows.append({
            "dataset": dataset,
            "condition": condition,
            "seed": int(seed),
            "run_dir": str(run_dir),
            "test_balanced_accuracy": float(summary.get("test_balanced_accuracy", "nan")),
            "test_weighted_f1": float(summary.get("test_weighted_f1", "nan")),
            "test_kappa": float(summary.get("test_kappa", "nan")),
            "runtime_seconds": float(summary.get("total_wall_seconds", "nan")),
            "peak_cuda_memory_mb": float(summary.get("peak_cuda_mb", "nan")),
            "trainable_parameters": int(float(summary.get("trainable_parameter_count", 0))),
        })
    return {
        "passed": not errors,
        "errors": errors,
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root.resolve()),
        "expected_rows": len(expected),
        "passing_rows": len(rows) if not errors else 0,
        "rows": rows,
        "condition": EXPECTED_CONDITION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.manifest, args.output_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and not result["passed"]:
        raise SystemExit("AttnRes ablation audit failed")


if __name__ == "__main__":
    main()
