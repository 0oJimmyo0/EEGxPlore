"""Audit the final 12-cell confirmatory ICASSP experiment scope."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = ROOT / "paper_table_manifest_v2.csv"
EXPECTED_SEEDS = {"3407", "2024", "2027"}
EXPECTED_CELLS = {
    (dataset, condition)
    for dataset in ("FACED", "ISRUC")
    for condition in ("full", "specialist_augmented_full")
}


def audit(table_path: Path) -> dict:
    errors = []
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 12:
        errors.append(f"final paper table must contain exactly 12 rows, found {len(rows)}")

    observed_cells = {(row.get("dataset", ""), row.get("executable_condition", "")) for row in rows}
    if observed_cells != EXPECTED_CELLS:
        errors.append(f"final cell set mismatch: {sorted(observed_cells)}")

    for row in rows:
        identity = f"{row.get('dataset')}/{row.get('executable_condition')}/{row.get('seed')}"
        if row.get("source_class") != "new_confirmatory":
            errors.append(f"{identity}: source_class must be new_confirmatory")
        if row.get("paper_eligibility") != "primary_new_evidence":
            errors.append(f"{identity}: paper_eligibility is not primary_new_evidence")
        if row.get("required_new_run") != "yes":
            errors.append(f"{identity}: required_new_run must be yes")
        if row.get("status") != "planned":
            errors.append(f"{identity}: status must remain planned before execution")
        if row.get("seed") not in EXPECTED_SEEDS:
            errors.append(f"{identity}: seed is not in {sorted(EXPECTED_SEEDS)}")
        if row.get("dataset") not in {"FACED", "ISRUC"}:
            errors.append(f"{identity}: dataset is outside the final scope")
        if row.get("executable_condition") == "specialist_augmented_full":
            if row.get("source_reference") != "paper_method_specialist_augmented_full_v1":
                errors.append(f"{identity}: specialist method reference is incorrect")
        elif row.get("executable_condition") == "full":
            expected_protocol = f"paper_protocol_{row.get('dataset', '').lower()}_v1"
            if row.get("source_reference") != expected_protocol:
                errors.append(f"{identity}: full baseline protocol reference is incorrect")

    for dataset, condition in sorted(EXPECTED_CELLS):
        seeds = {
            row.get("seed", "")
            for row in rows
            if row.get("dataset") == dataset and row.get("executable_condition") == condition
        }
        if seeds != EXPECTED_SEEDS:
            errors.append(f"{dataset}/{condition}: seed set is {sorted(seeds)}, expected {sorted(EXPECTED_SEEDS)}")

    return {
        "passed": not errors,
        "errors": errors,
        "table": str(table_path.resolve()),
        "confirmatory_rows": len(rows),
        "datasets": ["FACED", "ISRUC"],
        "methods": ["full", "specialist_augmented_full"],
        "seeds": ["3407", "2024", "2027"],
        "development_seed_excluded": "42",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit(args.table)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and not result["passed"]:
        raise SystemExit("final paper scope audit failed")


if __name__ == "__main__":
    main()
