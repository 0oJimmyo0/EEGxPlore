"""Audit the 18-cell ICASSP scope, including the locked SEED-V extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = ROOT / "paper_table_manifest_v3.csv"
EXPECTED_SEEDS = {"3407", "2024", "2027"}
EXPECTED_DATASETS = {"FACED", "ISRUC", "SEED-V"}
EXPECTED_CONDITIONS = {"full", "specialist_augmented_full"}


def audit(table_path: Path) -> dict:
    errors = []
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 18:
        errors.append(f"v3 paper table must contain exactly 18 rows, found {len(rows)}")
    observed = {(row.get("dataset", ""), row.get("executable_condition", "")) for row in rows}
    expected = {(dataset, condition) for dataset in EXPECTED_DATASETS for condition in EXPECTED_CONDITIONS}
    if observed != expected:
        errors.append(f"final cell set mismatch: {sorted(observed)}")

    for row in rows:
        identity = f"{row.get('dataset')}/{row.get('executable_condition')}/{row.get('seed')}"
        dataset = row.get("dataset", "")
        condition = row.get("executable_condition", "")
        if row.get("paper_eligibility") != "primary_new_evidence":
            errors.append(f"{identity}: paper_eligibility is not primary_new_evidence")
        if dataset not in EXPECTED_DATASETS or condition not in EXPECTED_CONDITIONS:
            errors.append(f"{identity}: dataset/condition is outside v3 scope")
        if row.get("seed") not in EXPECTED_SEEDS:
            errors.append(f"{identity}: seed is not in {sorted(EXPECTED_SEEDS)}")
        expected_new_run = "yes" if dataset == "SEED-V" else "no"
        if row.get("required_new_run") != expected_new_run:
            errors.append(f"{identity}: required_new_run must be {expected_new_run}")
        if condition == "specialist_augmented_full":
            if row.get("source_reference") != "paper_method_specialist_augmented_full_v1":
                errors.append(f"{identity}: specialist method reference is incorrect")
        else:
            expected_protocol = f"paper_protocol_{dataset.lower()}_v1".replace("seed-v", "seedv")
            if row.get("source_reference") != expected_protocol:
                errors.append(f"{identity}: full baseline protocol reference is incorrect")

    for dataset, condition in sorted(expected):
        seeds = {
            row.get("seed", "")
            for row in rows
            if row.get("dataset") == dataset and row.get("executable_condition") == condition
        }
        if seeds != EXPECTED_SEEDS:
            errors.append(f"{dataset}/{condition}: seed set is {sorted(seeds)}")

    return {
        "passed": not errors,
        "errors": errors,
        "table": str(table_path.resolve()),
        "confirmatory_rows": len(rows),
        "datasets": sorted(EXPECTED_DATASETS),
        "methods": sorted(EXPECTED_CONDITIONS),
        "seeds": sorted(EXPECTED_SEEDS),
        "seedv_scope": "prespecified_extension_v3",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit(args.table)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and not result["passed"]:
        raise SystemExit("v3 paper scope audit failed")


if __name__ == "__main__":
    main()
