"""Audit whether the three historical SEED-V reference families are matched."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = REPO_ROOT / "experiments" / "icassp2027" / "revision" / "historical_candidates.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "icassp2027_revision" / "historical_bundle_audit.json"
TARGET_FAMILIES = {
    "1769781": "historical_dense",
    "1769783": "historical_attnres_only",
    "1785556": "historical_selective",
}
REQUIRED_FIELDS = (
    "dataset",
    "split",
    "preprocessing",
    "epoch_budget",
    "selection_rule",
    "code_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "seed",
)
MATCH_FIELDS = ("dataset", "split", "preprocessing", "epoch_budget", "selection_rule")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_bundle(index_path: Path) -> Dict[str, Any]:
    rows = _read_rows(index_path)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        family_id = str(row.get("historical_family_id", "")).strip()
        if family_id in TARGET_FAMILIES:
            grouped[family_id].append(row)

    families: Dict[str, Any] = {}
    all_passed = True
    for family_id, expected_condition in TARGET_FAMILIES.items():
        family_rows = grouped.get(family_id, [])
        seeds = sorted({row.get("seed", "").strip() for row in family_rows if row.get("seed", "").strip()})
        missing_fields = sorted({field for row in family_rows for field in REQUIRED_FIELDS if not row.get(field, "").strip()})
        condition_values = sorted({row.get("condition", "").strip() for row in family_rows})
        field_values = {
            field: sorted({row.get(field, "").strip() for row in family_rows if row.get(field, "").strip()})
            for field in MATCH_FIELDS
        }
        mismatched_fields = sorted(field for field, values in field_values.items() if len(values) > 1)
        passed = bool(family_rows) and not missing_fields and len(seeds) >= 3 and not mismatched_fields
        all_passed = all_passed and passed
        families[family_id] = {
            "expected_condition": expected_condition,
            "rows": len(family_rows),
            "seeds": seeds,
            "condition_values": condition_values,
            "missing_required_fields": missing_fields,
            "matched_field_values": field_values,
            "mismatched_fields": mismatched_fields,
            "passed": passed,
        }

    return {
        "index_path": str(index_path),
        "target_families": TARGET_FAMILIES,
        "required_seed_count": 3,
        "all_families_matched": all_passed,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless all three families pass.")
    args = parser.parse_args()
    report = audit_bundle(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not report["all_families_matched"]:
        raise SystemExit("historical bundle audit is pending")


if __name__ == "__main__":
    main()
