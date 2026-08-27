"""Build the row-level ICASSP evidence registry from revision run artifacts.

The registry is intentionally generated below ``output/`` (which is ignored by
Git).  A row is never promoted to a paper result automatically: overlap status
and reuse decision remain explicit audit fields and default to pending review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "icassp2027_revision"
DEFAULT_REGISTRY = DEFAULT_OUTPUT_ROOT / "evidence_registry.csv"

FIELDNAMES = [
    "source_kind",
    "run_status",
    "dataset",
    "condition",
    "historical_family_id",
    "historical_recipe_sha256",
    "seed",
    "split",
    "preprocessing",
    "epoch_budget",
    "selection_rule",
    "code_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "metric_files",
    "trainable_parameter_count",
    "runtime_seconds",
    "gpu",
    "peak_memory_mb",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_kappa",
    "tmlr_overlap_status",
    "reuse_decision",
    "notes",
]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _last_csv_row(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return {}
    return dict(rows[-1]) if rows else {}


def _command_value(command: Any, flag: str) -> str:
    if not isinstance(command, list):
        return ""
    values = [str(value) for value in command]
    try:
        index = values.index(flag)
    except ValueError:
        return ""
    return values[index + 1] if index + 1 < len(values) else ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_files(run_dir: Path) -> str:
    names = {
        path.name
        for path in run_dir.iterdir()
        if path.is_file()
        and (
            path.name == "experiment_summary.csv"
            or path.name in {
                "adaptation_diagnosis.json",
                "config_audit.json",
                "ema_comparison_summary.json",
                "result_manifest.json",
                "selected_checkpoint_diagnostics.json",
            }
            or path.name.startswith("run_summary_")
        )
    }
    return ";".join(sorted(names))


def _row_for_run(run_manifest_path: Path, hash_checkpoints: bool) -> Dict[str, str]:
    run_dir = run_manifest_path.parent
    manifest = _read_json(run_manifest_path)
    summary_path = run_dir / "experiment_summary.csv"
    summary = _last_csv_row(summary_path)
    result = _read_json(run_dir / "result_manifest.json")

    protocol = str(summary.get("revision_protocol") or manifest.get("protocol") or "")
    split_source = str(
        summary.get("split_manifest_source")
        or manifest.get("split_manifest_source")
        or ("seedv_subject_disjoint_manifest" if protocol == "seedv_subject_disjoint" else "lmdb___keys__")
    )
    checkpoint_path = str(summary.get("model_path") or "")
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    checkpoint_sha256 = str(summary.get("checkpoint_sha256") or "")
    if hash_checkpoints and checkpoint is not None and checkpoint.is_file() and not checkpoint_sha256:
        checkpoint_sha256 = _sha256_file(checkpoint)

    exit_code = result.get("exit_code")
    if exit_code is None:
        run_status = "complete" if summary else "incomplete"
    elif int(exit_code) == 0:
        run_status = "complete"
    else:
        run_status = "failed"

    code_commit = str(summary.get("git_commit") or manifest.get("repository_commit") or "")
    gpu = _command_value(manifest.get("command"), "--cuda")
    epoch_budget = str(summary.get("epochs") or _command_value(manifest.get("command"), "--epochs") or "")
    selection_rule = str(summary.get("selection_metric") or "kappa")
    notes: List[str] = []
    if not checkpoint_sha256:
        notes.append("checkpoint_hash_not_computed")
    if not summary:
        notes.append("experiment_summary_missing")
    if not code_commit:
        notes.append("code_commit_missing")
    if run_status != "complete":
        notes.append("not_eligible_for_main_table")

    return {
        "source_kind": "new_revision_run",
        "run_status": run_status,
        "dataset": str(summary.get("dataset") or manifest.get("dataset") or ""),
        "condition": str(summary.get("revision_condition") or manifest.get("condition") or ""),
        "historical_family_id": str(summary.get("historical_family_id") or manifest.get("historical_family_id") or ""),
        "historical_recipe_sha256": str(summary.get("historical_recipe_sha256") or manifest.get("historical_recipe_sha256") or ""),
        "seed": str(summary.get("seed") or manifest.get("seed") or ""),
        "split": protocol,
        "preprocessing": f"{split_source};dataset_dir={manifest.get('dataset_dir', '')}",
        "epoch_budget": epoch_budget,
        "selection_rule": selection_rule,
        "code_commit": code_commit,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "metric_files": _metric_files(run_dir),
        "trainable_parameter_count": str(summary.get("trainable_parameter_count") or ""),
        "runtime_seconds": str(summary.get("total_wall_seconds") or ""),
        "gpu": gpu,
        "peak_memory_mb": str(summary.get("peak_cuda_mb") or ""),
        "test_balanced_accuracy": str(summary.get("test_balanced_accuracy") or ""),
        "test_macro_f1": str(summary.get("test_macro_f1") or ""),
        "test_kappa": str(summary.get("test_kappa") or ""),
        "tmlr_overlap_status": "unreviewed_pending_row_audit",
        "reuse_decision": "candidate_pending_audit" if run_status == "complete" else "invalid_failed_or_incomplete",
        "notes": ";".join(notes),
    }


def _historical_rows(index_path: Optional[Path]) -> List[Dict[str, str]]:
    if index_path is None or not index_path.is_file():
        return []
    try:
        with index_path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    rows: List[Dict[str, str]] = []
    for source_row in source_rows:
        row = {field: str(source_row.get(field, "") or "") for field in FIELDNAMES}
        row["source_kind"] = "rejected_paper_historical"
        row["run_status"] = row["run_status"] or "candidate"
        row["tmlr_overlap_status"] = row["tmlr_overlap_status"] or "unreviewed_pending_row_audit"
        row["reuse_decision"] = row["reuse_decision"] or "candidate_pending_audit"
        rows.append(row)
    return rows


def build_registry(
    output_root: Path,
    registry_path: Path,
    hash_checkpoints: bool = False,
    historical_index: Optional[Path] = None,
) -> int:
    manifests = sorted(output_root.rglob("run_manifest.json")) if output_root.is_dir() else []
    rows = [_row_for_run(path, hash_checkpoints=hash_checkpoints) for path in manifests]
    rows.extend(_historical_rows(historical_index))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--historical-index",
        type=Path,
        default=REPO_ROOT / "experiments" / "icassp2027" / "revision" / "historical_candidates.csv",
        help="CSV index of rejected-paper candidate rows to append to the generated registry.",
    )
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Compute missing checkpoint SHA-256 hashes; this can be expensive for large runs.",
    )
    args = parser.parse_args()
    count = build_registry(
        args.output_root,
        args.registry,
        hash_checkpoints=args.hash_checkpoints,
        historical_index=args.historical_index,
    )
    print(f"evidence registry: wrote {count} rows to {args.registry}")


if __name__ == "__main__":
    main()
