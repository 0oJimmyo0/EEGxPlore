"""Audit the frozen ICASSP paper scope and planned evidence rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORICAL = ROOT / "historical_candidates.csv"
DEFAULT_TABLE = ROOT / "paper_table_manifest.csv"
SEEDS = {"42", "3407", "2024"}
EXPECTED_LEGACY = {
    "historical_dense",
    "historical_attnres_only",
    "historical_selective",
}
EXPECTED_NEW = {
    ("SEED-V", "upper1"),
    ("FACED", "full"),
    ("FACED", "selective_paper"),
    ("ISRUC", "full"),
    ("ISRUC", "selective_paper"),
}


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(historical_path: Path, table_path: Path) -> Dict[str, Any]:
    historical = _rows(historical_path)
    table = _rows(table_path)
    errors: List[str] = []

    legacy_groups: Dict[str, set[str]] = {}
    for row in historical:
        condition = row.get("condition", "")
        legacy_groups.setdefault(condition, set()).add(row.get("seed", ""))
        if row.get("provenance_class") != "legacy_reported_paper":
            errors.append(f"historical row is not legacy_reported_paper: {condition}/{row.get('seed')}")
        if row.get("verification_level") != "manuscript_reported_not_reproduced":
            errors.append(f"historical row has wrong verification level: {condition}/{row.get('seed')}")
        if row.get("run_status") != "reported_not_reproduced":
            errors.append(f"historical row has wrong run status: {condition}/{row.get('seed')}")
        if row.get("test_macro_f1"):
            errors.append(f"historical weighted-F1 was placed in macro-F1: {condition}/{row.get('seed')}")
        if not row.get("reported_weighted_f1"):
            errors.append(f"historical weighted-F1 is missing: {condition}/{row.get('seed')}")
        for field in ("code_commit", "checkpoint_path", "checkpoint_sha256"):
            if row.get(field):
                errors.append(f"historical row contains fabricated {field}: {condition}/{row.get('seed')}")

    if set(legacy_groups) != EXPECTED_LEGACY:
        errors.append(f"historical condition set mismatch: {sorted(legacy_groups)}")
    for condition in EXPECTED_LEGACY:
        if legacy_groups.get(condition) != SEEDS:
            errors.append(f"historical seed set mismatch for {condition}: {sorted(legacy_groups.get(condition, set()))}")

    planned_new = {
        (row.get("dataset", ""), row.get("executable_condition", ""))
        for row in table
        if row.get("source_class") == "new_multiseed"
    }
    missing = sorted(EXPECTED_NEW - planned_new)
    if missing:
        errors.append(f"required new condition rows missing: {missing}")
    if len([row for row in table if row.get("source_class") == "new_multiseed"]) != 15:
        errors.append("paper table must contain exactly 15 planned new rows")
    if len([row for row in table if row.get("source_class") == "legacy_reported_paper"]) != 9:
        errors.append("paper table must contain exactly 9 legacy SEED-V rows")

    for dataset, condition in EXPECTED_NEW:
        count = sum(
            1
            for row in table
            if row.get("source_class") == "new_multiseed"
            and row.get("dataset") == dataset
            and row.get("executable_condition") == condition
        )
        if count != 3:
            errors.append(f"new row count for {dataset}/{condition} is {count}, expected 3")

    for row in table:
        if row.get("source_class") == "new_multiseed" and row.get("seed") not in SEEDS:
            errors.append(f"new row uses undeclared seed: {row.get('dataset')}/{row.get('executable_condition')}/{row.get('seed')}")

    return {
        "passed": not errors,
        "errors": errors,
        "paper_scope": {
            "backbone": "CBraMod",
            "datasets": ["SEED-V", "FACED", "ISRUC"],
            "seeds": [42, 3407, 2024],
        },
        "legacy_rows": {condition: len(legacy_groups.get(condition, set())) for condition in sorted(EXPECTED_LEGACY)},
        "required_new_rows": {
            f"{dataset}/{condition}": sum(
                1
                for row in table
                if row.get("source_class") == "new_multiseed"
                and row.get("dataset") == dataset
                and row.get("executable_condition") == condition
            )
            for dataset, condition in sorted(EXPECTED_NEW)
        },
        "forbidden": [
            "TUEV",
            "LaBraM",
            "TMLR results",
            "subject-disjoint primary",
            "PhysioNet primary",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(args.historical, args.table)
    except (OSError, csv.Error) as exc:
        raise SystemExit(f"paper scope audit could not read inputs: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and not result["passed"]:
        raise SystemExit("paper scope audit failed")


if __name__ == "__main__":
    main()
