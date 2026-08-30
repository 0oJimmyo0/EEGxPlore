"""Shared constants and readers for the frozen ICASSP confirmatory analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "icassp2027_revision"
DEFAULT_PAPER_MANIFEST = (
    REPO_ROOT / "experiments" / "icassp2027" / "revision" / "paper_table_manifest_v2.csv"
)

TRAINING_COMMIT = "fd425cdfd0ff08d57ac30ee9b8737b895e9d46ad"
FOUNDATION_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
METHOD_ID = "icaspp_specialist_augmented_full_v1"
METHOD_SHA256 = "32dbc9266225fefce3eaa4fb8f4faf2cc727ca0611712e9b7520157c13eba10a"

DATASET_CONTRACT_SHA256 = {
    "FACED": "3269f80c4f89e346362e363e3ef328331f853ea8f6917d960b6ee6ff99e564d5",
    "ISRUC": "6db187940c432334346aa3a7a4aac7bfa5bdc493f637678188d2380048e6ea4a",
}
DATASET_PROTOCOL = {
    "FACED": {
        "id": "icaspp_paper_derived_faced_v1",
        "sha256": "dce264b69304759613a1fddd9028fc2d380e23a488372746d58d0886be9d0836",
    },
    "ISRUC": {
        "id": "icaspp_paper_derived_isruc_v1",
        "sha256": "995270018fbf0162e3a88b395fb5006c2ade9f441a6217694e18d5aa637cedd8",
    },
}
DATASETS = ("FACED", "ISRUC")
CONDITIONS = ("full", "specialist_augmented_full")
SEEDS = ("3407", "2024", "2027")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_last_csv_row(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return {}
    return dict(rows[-1]) if rows else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def run_directory(output_root: Path, dataset: str, condition: str, seed: str) -> Path:
    return output_root / dataset.lower() / condition / f"seed_{seed}"


def load_manifest_cells(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows


def expected_primary_cells(path: Path = DEFAULT_PAPER_MANIFEST) -> List[Tuple[str, str, str]]:
    rows = load_manifest_cells(path)
    cells = [
        (
            str(row.get("dataset", "")),
            str(row.get("executable_condition", "")),
            str(row.get("seed", "")),
        )
        for row in rows
        if row.get("paper_eligibility") == "primary_new_evidence"
        and row.get("required_new_run") == "yes"
    ]
    return sorted(cells)


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted({str(value) for value in values})
