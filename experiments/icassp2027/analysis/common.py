"""Shared constants and readers for the frozen ICASSP confirmatory analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "icassp2027_revision"
DEFAULT_PAPER_MANIFEST = (
    REPO_ROOT / "experiments" / "icassp2027" / "revision" / "paper_table_manifest_v4.csv"
)

TRAINING_COMMIT = "fd425cdfd0ff08d57ac30ee9b8737b895e9d46ad"
TRAINING_SEMANTICS_ID = "icassp2027_specialist_full_v1"
EXECUTION_COMMIT_CONTRACT = (
    REPO_ROOT
    / "experiments"
    / "icassp2027"
    / "revision"
    / "accepted_execution_commits.json"
)
FOUNDATION_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
METHOD_ID = "icaspp_specialist_augmented_full_v1"
METHOD_SHA256 = "08ded624f8d12da6c98e42436b8e4dcc87f20d5188b5541f851ca8e429384ace"

DATASET_CONTRACT_SHA256 = {
    "SEED-V": "db8fb3219e2acf74e1427e50a84a96d1c31fada78deac86bbde82f2a9c2a02ea",
    "FACED": "3269f80c4f89e346362e363e3ef328331f853ea8f6917d960b6ee6ff99e564d5",
    "ISRUC": "6db187940c432334346aa3a7a4aac7bfa5bdc493f637678188d2380048e6ea4a",
    "PhysioNet-MI": "c28f125d6ebd54ca306a697eb3b7a3d1fdbef04d0a7b4da7c50a3a2c7c67cac6",
}
DATASET_PROTOCOL = {
    "SEED-V": {
        "id": "icaspp_paper_derived_seedv_v1",
        "sha256": "ace6b6283ba014cf37a943b93e58d0ef1e018a93216e3f6b83fb86a4296c3296",
    },
    "FACED": {
        "id": "icaspp_paper_derived_faced_v1",
        "sha256": "dce264b69304759613a1fddd9028fc2d380e23a488372746d58d0886be9d0836",
    },
    "ISRUC": {
        "id": "icaspp_paper_derived_isruc_v1",
        "sha256": "995270018fbf0162e3a88b395fb5006c2ade9f441a6217694e18d5aa637cedd8",
    },
    "PhysioNet-MI": {
        "id": "icaspp_paper_derived_physionet_mi_v1",
        "sha256": "2eb1bfde072e8e83af5510b4bf4e148b3f422a7c9fba72913cb54a1fd4ed7c3c",
    },
}
DATASETS = ("FACED", "ISRUC", "SEED-V", "PhysioNet-MI")
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
    dataset_tag = {
        "SEED-V": "seedv",
        "FACED": "faced",
        "ISRUC": "isruc",
        "PhysioNet-MI": "physionet_mi",
    }[dataset]
    return output_root / dataset_tag / condition / f"seed_{seed}"


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
    ]
    return sorted(cells)


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted({str(value) for value in values})


def load_execution_contract() -> Dict[str, Any]:
    payload = read_json(EXECUTION_COMMIT_CONTRACT)
    if payload.get("training_source_commit") != TRAINING_COMMIT:
        raise ValueError("execution commit contract has an unexpected training source commit")
    if payload.get("training_semantics_id") != TRAINING_SEMANTICS_ID:
        raise ValueError("execution commit contract has an unexpected training semantics ID")
    return payload


def execution_commit_info(commit: str) -> Dict[str, Any] | None:
    contract = load_execution_contract()
    entries = contract.get("accepted_execution_commits", {})
    entry = entries.get(str(commit))
    if not isinstance(entry, dict):
        return None
    return {"execution_commit": str(commit), **entry}


def verify_execution_commit_diff(commit: str, info: Dict[str, Any]) -> List[str]:
    if commit == TRAINING_COMMIT:
        return []
    errors: List[str] = []
    try:
        names = subprocess.run(
            ["git", "diff", "--name-only", TRAINING_COMMIT, commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        diff_bytes = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", TRAINING_COMMIT, commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"cannot verify execution commit diff {TRAINING_COMMIT}..{commit}: {exc}"]
    allowed = sorted(str(path) for path in info.get("allowed_diff_paths", []))
    if sorted(names) != allowed:
        errors.append(f"execution diff paths {sorted(names)!r} do not match allowlist {allowed!r}")
    expected_sha = str(info.get("diff_sha256", ""))
    actual_sha = hashlib.sha256(diff_bytes).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        errors.append(f"execution diff sha256={actual_sha!r}, expected {expected_sha!r}")
    return errors
