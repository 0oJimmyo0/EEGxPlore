"""Strictly audit the frozen 12-cell ICASSP confirmatory matrix.

This audit is intentionally independent of the training path.  It reads only
run manifests, result manifests, summaries, and locked hash contracts.  It
does not promote smoke, seed-42, historical-candidate, or other diagnostic
artifacts into confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import (
    CONDITIONS,
    DATASET_CONTRACT_SHA256,
    DATASET_PROTOCOL,
    DATASETS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PAPER_MANIFEST,
    FOUNDATION_SHA256,
    METHOD_ID,
    METHOD_SHA256,
    REPO_ROOT,
    SEEDS,
    TRAINING_COMMIT,
    as_bool,
    as_float,
    expected_primary_cells,
    read_json,
    read_last_csv_row,
    run_directory,
)


REQUIRED_METRICS = (
    "test_balanced_accuracy",
    "test_weighted_f1",
    "test_kappa",
    "test_macro_f1",
)
REQUIRED_SUMMARY_FIELDS = REQUIRED_METRICS + (
    "best_epoch",
    "trainable_parameter_count",
    "total_wall_seconds",
    "peak_cuda_mb",
    "model_path",
    "checkpoint_sha256",
)


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _canonical_row(
    dataset: str,
    condition: str,
    seed: str,
    run_dir: Path,
    run_manifest: Dict[str, Any],
    result_manifest: Dict[str, Any],
    summary: Dict[str, str],
) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    protocol = DATASET_PROTOCOL.get(dataset, {})
    expected_contract = DATASET_CONTRACT_SHA256.get(dataset, "")
    expected_method_sha = METHOD_SHA256 if condition == "specialist_augmented_full" else ""
    expected_method_id = METHOD_ID if condition == "specialist_augmented_full" else ""

    actual_dataset = str(_first(summary.get("dataset"), run_manifest.get("dataset"), result_manifest.get("dataset")))
    actual_condition = str(_first(summary.get("revision_condition"), run_manifest.get("condition"), result_manifest.get("condition")))
    actual_seed = str(_first(summary.get("seed"), run_manifest.get("seed"), result_manifest.get("seed")))
    run_mode = str(_first(run_manifest.get("run_mode"), result_manifest.get("run_mode")))
    paper_eligible = as_bool(_first(run_manifest.get("paper_eligible"), result_manifest.get("paper_eligible")))
    exit_code = result_manifest.get("exit_code")

    if actual_dataset != dataset:
        errors.append(f"dataset={actual_dataset!r}, expected {dataset!r}")
    if actual_condition != condition:
        errors.append(f"condition={actual_condition!r}, expected {condition!r}")
    if actual_seed != seed:
        errors.append(f"seed={actual_seed!r}, expected {seed!r}")
    if run_mode != "paper":
        errors.append(f"run_mode={run_mode!r}, expected 'paper'")
    if not paper_eligible:
        errors.append("paper_eligible is not true")
    if exit_code is None or int(exit_code) != 0:
        errors.append(f"exit_code={exit_code!r}, expected 0")

    code_commit = str(_first(summary.get("git_commit"), run_manifest.get("repository_commit"), result_manifest.get("repository_commit")))
    if code_commit != TRAINING_COMMIT:
        errors.append(f"code_commit={code_commit!r}, expected frozen training commit")
    if as_bool(_first(summary.get("git_dirty"), run_manifest.get("git_dirty"))):
        errors.append("git_dirty is true")

    protocol_id = str(_first(summary.get("paper_protocol_id"), run_manifest.get("paper_protocol_id"), result_manifest.get("paper_protocol_id")))
    protocol_sha = str(_first(summary.get("paper_protocol_sha256"), run_manifest.get("paper_protocol_sha256"), result_manifest.get("paper_protocol_sha256")))
    if protocol_id != protocol.get("id"):
        errors.append(f"paper_protocol_id={protocol_id!r}, expected {protocol.get('id')!r}")
    if protocol_sha != protocol.get("sha256"):
        errors.append("paper_protocol_sha256 does not match the locked protocol")

    data_contract_sha = str(_first(run_manifest.get("data_contract_sha256"), result_manifest.get("data_contract_sha256")))
    if data_contract_sha != expected_contract:
        errors.append("data_contract_sha256 does not match the locked rejected-paper data contract")

    foundation_sha = str(_first(run_manifest.get("foundation_checkpoint_sha256"), result_manifest.get("foundation_checkpoint_sha256")))
    if foundation_sha != FOUNDATION_SHA256:
        errors.append("foundation_checkpoint_sha256 does not match the locked foundation checkpoint")

    method_id = str(_first(run_manifest.get("paper_method_recipe_id"), result_manifest.get("paper_method_recipe_id")))
    method_sha = str(_first(run_manifest.get("paper_method_recipe_sha256"), result_manifest.get("paper_method_recipe_sha256")))
    if method_id != expected_method_id:
        errors.append(f"paper_method_recipe_id={method_id!r}, expected {expected_method_id!r}")
    if method_sha != expected_method_sha:
        errors.append("paper_method_recipe_sha256 does not match the locked method contract")

    for field in REQUIRED_SUMMARY_FIELDS:
        if not str(summary.get(field, "") or "").strip():
            errors.append(f"summary field missing: {field}")
    for field in REQUIRED_METRICS + ("total_wall_seconds", "peak_cuda_mb"):
        if as_float(summary.get(field)) is None:
            errors.append(f"summary field is not finite: {field}")

    checkpoint_path = Path(str(summary.get("model_path", "")))
    if not checkpoint_path.is_file():
        errors.append(f"checkpoint missing: {checkpoint_path}")
    checkpoint_sha = str(summary.get("checkpoint_sha256", "") or "")
    if len(checkpoint_sha) != 64:
        errors.append("checkpoint_sha256 is missing or malformed")

    trainable_parameters = as_float(summary.get("trainable_parameter_count"))
    if trainable_parameters is None or trainable_parameters <= 0:
        errors.append("trainable_parameter_count must be positive")

    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": "Full fine-tuning" if condition == "full" else "AttnRes + Typed Specialists",
        "executable_condition": condition,
        "test_balanced_accuracy": as_float(summary.get("test_balanced_accuracy")),
        "test_weighted_f1": as_float(summary.get("test_weighted_f1")),
        "test_kappa": as_float(summary.get("test_kappa")),
        "test_macro_f1": as_float(summary.get("test_macro_f1")),
        "best_epoch": int(float(summary.get("best_epoch", 0) or 0)),
        "runtime_seconds": as_float(summary.get("total_wall_seconds")),
        "peak_cuda_memory_mb": as_float(summary.get("peak_cuda_mb")),
        "trainable_parameters": int(trainable_parameters or 0),
        "total_parameters": int(trainable_parameters or 0),
        "total_parameter_note": "equal_to_trainable_for_locked_full_trainability",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha,
        "method_id": method_id,
        "method_sha256": method_sha,
        "data_contract_sha256": data_contract_sha,
        "foundation_sha256": foundation_sha,
        "code_commit": code_commit,
        "paper_eligible": paper_eligible,
        "run_mode": run_mode,
        "run_directory": str(run_dir),
        "status": "pass" if not errors else "fail",
        "errors": errors,
    }
    return row, errors


def _excluded_artifacts(output_root: Path, primary_cells: set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    excluded: List[Dict[str, Any]] = []
    if not output_root.is_dir():
        return excluded
    for manifest_path in sorted(output_root.rglob("run_manifest.json")):
        manifest = read_json(manifest_path)
        cell = (
            str(manifest.get("dataset", "")),
            str(manifest.get("condition", "")),
            str(manifest.get("seed", "")),
        )
        if cell in primary_cells:
            continue
        reasons: List[str] = []
        mode = str(manifest.get("run_mode", ""))
        condition = str(manifest.get("condition", ""))
        seed = str(manifest.get("seed", ""))
        if mode == "smoke":
            reasons.append("smoke")
        if seed == "42":
            reasons.append("seed_42_development")
        if condition in {"historical_candidate", "historical_selective"}:
            reasons.append("historical_candidate")
        if condition in {"selective_fresh", "selective_paper", "combined", "specialist_only", "attnres_only"}:
            reasons.append("non_confirmatory_condition")
        if not reasons:
            reasons.append("outside_frozen_matrix")
        excluded.append({"run_manifest": str(manifest_path), "cell": cell, "reasons": reasons})
    return excluded


def audit_matrix(output_root: Path, paper_manifest: Path) -> Dict[str, Any]:
    expected = expected_primary_cells(paper_manifest)
    expected_set = set(expected)
    required_set = {
        (dataset, condition, seed)
        for dataset in DATASETS
        for condition in CONDITIONS
        for seed in SEEDS
    }
    failures: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    if expected_set != required_set:
        failures.append({"cell": "paper_manifest", "errors": ["manifest does not exactly equal the frozen 12-cell matrix"]})

    for dataset, condition, seed in expected:
        run_dir = run_directory(output_root, dataset, condition, seed)
        run_manifest_path = run_dir / "run_manifest.json"
        result_manifest_path = run_dir / "result_manifest.json"
        summary_path = run_dir / "experiment_summary.csv"
        missing = [str(path) for path in (run_manifest_path, result_manifest_path, summary_path) if not path.is_file()]
        if missing:
            failures.append({"cell": [dataset, condition, seed], "errors": [f"required artifact missing: {path}" for path in missing]})
            rows.append({"dataset": dataset, "executable_condition": condition, "seed": int(seed), "status": "fail", "errors": [f"required artifact missing: {path}" for path in missing]})
            continue
        row, errors = _canonical_row(
            dataset,
            condition,
            seed,
            run_dir,
            read_json(run_manifest_path),
            read_json(result_manifest_path),
            read_last_csv_row(summary_path),
        )
        rows.append(row)
        if errors:
            failures.append({"cell": [dataset, condition, seed], "errors": errors})

    passed = not failures and len(rows) == len(required_set)
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_commit": TRAINING_COMMIT,
        "paper_manifest": str(paper_manifest),
        "output_root": str(output_root),
        "expected_cell_count": len(required_set),
        "complete_cell_count": sum(row.get("status") == "pass" for row in rows),
        "passed": passed,
        "failures": failures,
        "rows": rows,
        "excluded_artifacts": _excluded_artifacts(output_root, expected_set),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "confirmatory_audit.json",
    )
    parser.add_argument("--strict", action="store_true", help="return failure status unless all 12 cells pass")
    args = parser.parse_args()
    audit = audit_matrix(args.output_root, args.paper_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("passed", "expected_cell_count", "complete_cell_count", "failures")}, indent=2))
    return 0 if audit["passed"] or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
