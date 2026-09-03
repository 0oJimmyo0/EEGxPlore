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
from typing import Any, Dict, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "icassp2027_revision"
DEFAULT_REGISTRY = DEFAULT_OUTPUT_ROOT / "evidence_registry.csv"
DEFAULT_PAPER_MANIFEST = REPO_ROOT / "experiments" / "icassp2027" / "revision" / "paper_table_manifest_v4.csv"
EXECUTION_COMMIT_CONTRACT = REPO_ROOT / "experiments" / "icassp2027" / "revision" / "accepted_execution_commits.json"
TRAINING_SEMANTICS_ID = "icassp2027_specialist_full_v1"

FIELDNAMES = [
    "source_kind",
    "provenance_class",
    "verification_level",
    "evidence_role",
    "paper_eligibility",
    "source_location",
    "run_mode",
    "run_status",
    "dataset",
    "condition",
    "historical_family_id",
    "historical_recipe_sha256",
    "fresh_selective_recipe_sha256",
    "paper_method_recipe_id",
    "paper_method_recipe_sha256",
    "paper_method_semantics_sha256",
    "paper_method_recipe_path",
    "paper_component_recipe_id",
    "paper_component_recipe_sha256",
    "paper_component_semantics_sha256",
    "paper_component_recipe_path",
    "paper_protocol_id",
    "paper_protocol_sha256",
    "paper_protocol_path",
    "use_component_lr",
    "seed",
    "split",
    "preprocessing",
    "data_contract_sha256",
    "epoch_budget",
    "selection_rule",
    "code_commit",
    "execution_commit",
    "execution_commit_classification",
    "training_source_commit",
    "training_semantics_id",
    "semantic_config_sha256",
    "pair_contract_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "metric_files",
    "trainable_parameter_count",
    "runtime_seconds",
    "gpu",
    "peak_memory_mb",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "reported_weighted_f1",
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


def _execution_contract() -> Dict[str, Any]:
    return _read_json(EXECUTION_COMMIT_CONTRACT)


def _provenance_fields(code_commit: str) -> Dict[str, str]:
    contract = _execution_contract()
    entry = contract.get("accepted_execution_commits", {}).get(code_commit, {})
    if not isinstance(entry, dict):
        return {
            "execution_commit": code_commit,
            "execution_commit_classification": "unaccepted" if code_commit else "",
            "training_source_commit": "",
            "training_semantics_id": "",
        }
    return {
        "execution_commit": code_commit,
        "execution_commit_classification": str(entry.get("classification", "")),
        "training_source_commit": str(contract.get("training_source_commit", "")),
        "training_semantics_id": str(contract.get("training_semantics_id", TRAINING_SEMANTICS_ID)),
    }


def _semantic_config_sha256(run_dir: Path) -> str:
    summaries = sorted(run_dir.glob("run_summary_*.json"))
    if not summaries:
        return ""
    payload = _read_json(summaries[-1]).get("config")
    if not isinstance(payload, dict):
        return ""
    normalized = {
        str(key): value
        for key, value in payload.items()
        if key not in {
            "seed",
            "model_dir",
            "paper_method_recipe",
            "paper_method_semantics_sha256",
        }
    }
    encoded = json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _primary_cells(manifest_path: Optional[Path]) -> Optional[Set[Tuple[str, str, str]]]:
    """Load exact paper-facing cells; None preserves legacy function behavior."""
    if manifest_path is None:
        return None
    if not manifest_path.is_file():
        raise FileNotFoundError(f"paper manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (
                str(row.get("dataset", "") or ""),
                str(row.get("executable_condition", "") or ""),
                str(row.get("seed", "") or ""),
            )
            for row in rows
            if row.get("paper_eligibility") == "primary_new_evidence"
        }


def _row_for_run(
    run_manifest_path: Path,
    hash_checkpoints: bool,
    primary_cells: Optional[Set[Tuple[str, str, str]]] = None,
) -> Dict[str, str]:
    run_dir = run_manifest_path.parent
    manifest = _read_json(run_manifest_path)
    summary_path = run_dir / "experiment_summary.csv"
    summary = _last_csv_row(summary_path)
    result = _read_json(run_dir / "result_manifest.json")
    run_mode = str(manifest.get("run_mode") or result.get("run_mode") or "paper")

    protocol = str(summary.get("revision_protocol") or manifest.get("protocol") or result.get("protocol") or "")
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

    code_commit = str(
        summary.get("git_commit")
        or manifest.get("repository_commit")
        or result.get("repository_commit")
        or ""
    )
    provenance = _provenance_fields(code_commit)
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
    if run_mode == "smoke":
        notes.append("smoke_run_not_paper_evidence")
    elif run_mode == "internal":
        notes.append("internal_run_not_paper_evidence")

    dataset = str(summary.get("dataset") or manifest.get("dataset") or result.get("dataset") or "")
    condition = str(
        summary.get("revision_condition")
        or manifest.get("condition")
        or result.get("condition")
        or ""
    )
    seed = str(summary.get("seed") or manifest.get("seed") or "")
    is_primary_cell = primary_cells is not None and (dataset, condition, seed) in primary_cells

    if run_mode == "smoke":
        evidence_role = "smoke"
        paper_eligibility = "not_eligible_smoke"
        reuse_decision = "not_paper_smoke"
    elif run_mode == "internal":
        evidence_role = "internal_development"
        paper_eligibility = "not_eligible_internal"
        reuse_decision = "internal_not_paper"
    elif primary_cells is not None and is_primary_cell:
        if run_status == "complete":
            evidence_role = "confirmatory_candidate"
            paper_eligibility = "primary_new_evidence_pending_audit"
            reuse_decision = "candidate_pending_audit"
        else:
            evidence_role = "confirmatory_incomplete"
            paper_eligibility = "not_eligible_incomplete"
            reuse_decision = "invalid_failed_or_incomplete"
    else:
        evidence_role = (
            "development_reference"
            if condition == "full" and seed == "42"
            else "development_diagnostic"
            if seed == "42"
            else "out_of_scope_paper_run"
        )
        paper_eligibility = "development_only_not_primary"
        reuse_decision = "development_context_only" if run_status == "complete" else "invalid_failed_or_incomplete"

    use_component_lr = manifest.get("use_component_lr")
    if use_component_lr is None:
        use_component_lr = result.get("use_component_lr")
    if use_component_lr is None:
        use_component_lr = "--use_component_lr" in (manifest.get("command") or [])

    return {
        "source_kind": "new_revision_run",
        "provenance_class": "new_multiseed",
        "verification_level": "run_artifacts_present" if run_status == "complete" else "run_artifacts_incomplete",
        "evidence_role": evidence_role,
        "paper_eligibility": paper_eligibility,
        "source_location": str(run_manifest_path),
        "run_mode": run_mode,
        "run_status": run_status,
        "dataset": dataset,
        "condition": condition,
        "historical_family_id": str(summary.get("historical_family_id") or manifest.get("historical_family_id") or ""),
        "historical_recipe_sha256": str(summary.get("historical_recipe_sha256") or manifest.get("historical_recipe_sha256") or ""),
        "fresh_selective_recipe_sha256": str(
            summary.get("fresh_selective_recipe_sha256")
            or manifest.get("fresh_selective_recipe_sha256")
            or result.get("fresh_selective_recipe_sha256")
            or ""
        ),
        "paper_method_recipe_id": str(
            manifest.get("paper_method_recipe_id")
            or result.get("paper_method_recipe_id")
            or ""
        ),
        "paper_method_recipe_sha256": str(
            manifest.get("paper_method_recipe_sha256")
            or result.get("paper_method_recipe_sha256")
            or summary.get("paper_method_recipe_sha256")
            or ""
        ),
        "paper_method_semantics_sha256": str(
            manifest.get("paper_method_semantics_sha256")
            or result.get("paper_method_semantics_sha256")
            or summary.get("paper_method_semantics_sha256")
            or ""
        ),
        "paper_method_recipe_path": str(
            manifest.get("paper_method_recipe_path")
            or result.get("paper_method_recipe_path")
            or ""
        ),
        "paper_component_recipe_id": str(
            manifest.get("paper_component_recipe_id")
            or result.get("paper_component_recipe_id")
            or ""
        ),
        "paper_component_recipe_sha256": str(
            manifest.get("paper_component_recipe_sha256")
            or result.get("paper_component_recipe_sha256")
            or ""
        ),
        "paper_component_semantics_sha256": str(
            manifest.get("paper_component_semantics_sha256")
            or result.get("paper_component_semantics_sha256")
            or ""
        ),
        "paper_component_recipe_path": str(
            manifest.get("paper_component_recipe_path")
            or result.get("paper_component_recipe_path")
            or ""
        ),
        "paper_protocol_id": str(
            summary.get("paper_protocol_id")
            or manifest.get("paper_protocol_id")
            or result.get("paper_protocol_id")
            or ""
        ),
        "paper_protocol_sha256": str(
            summary.get("paper_protocol_sha256")
            or manifest.get("paper_protocol_sha256")
            or result.get("paper_protocol_sha256")
            or ""
        ),
        "paper_protocol_path": str(
            summary.get("paper_protocol_path")
            or manifest.get("paper_protocol_path")
            or result.get("paper_protocol_path")
            or ""
        ),
        "use_component_lr": str(use_component_lr),
        "seed": seed,
        "split": protocol,
        "preprocessing": f"{split_source};dataset_dir={manifest.get('dataset_dir', '')}",
        "data_contract_sha256": str(
            manifest.get("data_contract_sha256")
            or result.get("data_contract_sha256")
            or ""
        ),
        "epoch_budget": epoch_budget,
        "selection_rule": selection_rule,
        "code_commit": code_commit,
        **provenance,
        "semantic_config_sha256": _semantic_config_sha256(run_dir),
        "pair_contract_sha256": str(summary.get("pair_contract_sha256") or ""),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "metric_files": _metric_files(run_dir),
        "trainable_parameter_count": str(summary.get("trainable_parameter_count") or ""),
        "runtime_seconds": str(summary.get("total_wall_seconds") or ""),
        "gpu": gpu,
        "peak_memory_mb": str(summary.get("peak_cuda_mb") or ""),
        "test_balanced_accuracy": str(summary.get("test_balanced_accuracy") or ""),
        "test_macro_f1": str(summary.get("test_macro_f1") or ""),
        "test_weighted_f1": str(summary.get("test_weighted_f1") or ""),
        "reported_weighted_f1": "",
        "test_kappa": str(summary.get("test_kappa") or ""),
        "tmlr_overlap_status": "unreviewed_pending_row_audit",
        "reuse_decision": reuse_decision,
        "notes": ";".join(notes),
    }


def _frozen_rows(
    audit_path: Optional[Path],
    primary_cells: Optional[Set[Tuple[str, str, str]]],
) -> List[Dict[str, str]]:
    """Import rows from a passed frozen audit snapshot.

    The frozen ICASSP package intentionally keeps the audited row-level
    metadata and aggregate tables, but not every original run directory.  A
    snapshot row is therefore evidence with complete audit provenance, not a
    locally replayable run artifact.  Primary cells represented by this
    snapshot replace matching rows discovered below the live output root.
    """
    if audit_path is None:
        return []
    payload = _read_json(audit_path)
    source_rows = payload.get("rows")
    if not payload.get("passed") or not isinstance(source_rows, list):
        raise ValueError(f"frozen audit is not a passed row snapshot: {audit_path}")

    rows: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    source_location = str(audit_path.resolve())
    for source in source_rows:
        if not isinstance(source, dict):
            raise ValueError(f"frozen audit contains a non-object row: {audit_path}")
        dataset = str(source.get("dataset", "") or "")
        condition = str(source.get("executable_condition", "") or "")
        seed = str(source.get("seed", "") or "")
        cell = (dataset, condition, seed)
        if cell in seen:
            raise ValueError(f"duplicate frozen audit cell: {cell}")
        seen.add(cell)
        if primary_cells is not None and cell not in primary_cells:
            raise ValueError(f"frozen audit row is outside the active paper manifest: {cell}")
        if source.get("status") != "pass" or not source.get("paper_eligible"):
            raise ValueError(f"frozen audit row is not paper-eligible: {cell}")

        row = {field: "" for field in FIELDNAMES}
        row.update({
            "source_kind": "frozen_confirmatory_snapshot",
            "provenance_class": "frozen_confirmatory",
            "verification_level": "frozen_audit_passed",
            "evidence_role": "confirmatory_frozen",
            "paper_eligibility": "primary_new_evidence_audited",
            "source_location": f"{source_location}::{dataset}/{condition}/seed_{seed}",
            "run_mode": str(source.get("run_mode", "paper")),
            "run_status": "complete",
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "split": str(source.get("protocol_id", "") or ""),
            "data_contract_sha256": str(source.get("data_contract_sha256", "") or ""),
            "selection_rule": "validation kappa (frozen audit)",
            "code_commit": str(source.get("code_commit", "") or ""),
            "execution_commit": str(source.get("execution_commit", "") or ""),
            "execution_commit_classification": str(source.get("execution_commit_classification", "") or ""),
            "training_source_commit": str(source.get("training_source_commit", "") or ""),
            "training_semantics_id": str(source.get("training_semantics_id", "") or ""),
            "semantic_config_sha256": str(source.get("semantic_config_sha256", "") or ""),
            "pair_contract_sha256": str(source.get("pair_contract_sha256", "") or ""),
            "checkpoint_path": str(source.get("checkpoint_path", "") or ""),
            "checkpoint_sha256": str(source.get("checkpoint_sha256", "") or ""),
            "metric_files": "confirmatory_audit.json;confirmatory_seed_results.csv",
            "trainable_parameter_count": str(source.get("trainable_parameters", "") or ""),
            "runtime_seconds": str(source.get("runtime_seconds", "") or ""),
            "gpu": "",
            "peak_memory_mb": str(source.get("peak_cuda_memory_mb", "") or ""),
            "test_balanced_accuracy": str(source.get("test_balanced_accuracy", "") or ""),
            "test_macro_f1": str(source.get("test_macro_f1", "") or ""),
            "test_weighted_f1": str(source.get("test_weighted_f1", "") or ""),
            "test_kappa": str(source.get("test_kappa", "") or ""),
            "tmlr_overlap_status": "unreviewed_pending_row_audit",
            "reuse_decision": "eligible_frozen_snapshot",
            "notes": "audited frozen snapshot; original run directory is external to this checkout",
        })
        if condition == "full":
            row["paper_method_semantics_sha256"] = str(source.get("method_semantics_sha256", "") or "")
        else:
            row["paper_method_recipe_id"] = str(source.get("method_id", "") or "")
            row["paper_method_recipe_sha256"] = str(source.get("method_sha256", "") or "")
            row["paper_method_semantics_sha256"] = str(source.get("method_semantics_sha256", "") or "")
        rows.append(row)

    if primary_cells is not None and seen != primary_cells:
        missing = sorted(primary_cells - seen)
        extra = sorted(seen - primary_cells)
        raise ValueError(f"frozen audit cells do not match active manifest; missing={missing}, extra={extra}")
    return rows


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
        row["source_kind"] = row["source_kind"] or "rejected_paper_report"
        row["provenance_class"] = row["provenance_class"] or "legacy_context_only"
        row["verification_level"] = row["verification_level"] or "unverified_historical"
        row["paper_eligibility"] = row["paper_eligibility"] or "not_eligible_pending_audit"
        row["source_location"] = row["source_location"] or "historical_candidates.csv"
        row["run_mode"] = row["run_mode"] or "legacy_report"
        row["run_status"] = row["run_status"] or "reported_not_reproduced"
        row["evidence_role"] = row["evidence_role"] or "legacy_reported_context"
        row["tmlr_overlap_status"] = row["tmlr_overlap_status"] or "unreviewed_pending_row_audit"
        row["reuse_decision"] = row["reuse_decision"] or "candidate_pending_audit"
        rows.append(row)
    return rows


def build_registry(
    output_root: Path,
    registry_path: Path,
    hash_checkpoints: bool = False,
    historical_index: Optional[Path] = None,
    paper_manifest: Optional[Path] = None,
    frozen_audit: Optional[Path] = None,
) -> int:
    manifests = sorted(output_root.rglob("run_manifest.json")) if output_root.is_dir() else []
    primary_cells = _primary_cells(paper_manifest)
    frozen_rows = _frozen_rows(frozen_audit, primary_cells)
    frozen_cells = {
        (row["dataset"], row["condition"], row["seed"])
        for row in frozen_rows
    }
    run_primary_cells = (
        None
        if primary_cells is None
        else primary_cells - frozen_cells
    )
    rows = [
        _row_for_run(path, hash_checkpoints=hash_checkpoints, primary_cells=run_primary_cells)
        for path in manifests
    ]
    rows.extend(frozen_rows)
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
    parser.add_argument(
        "--paper-manifest",
        type=Path,
        default=DEFAULT_PAPER_MANIFEST,
        help="Active paper manifest used to distinguish confirmatory rows from development artifacts.",
    )
    parser.add_argument(
        "--frozen-audit",
        type=Path,
        default=None,
        help="Passed confirmatory_audit.json snapshot. Matching primary rows replace live run discovery.",
    )
    args = parser.parse_args()
    count = build_registry(
        args.output_root,
        args.registry,
        hash_checkpoints=args.hash_checkpoints,
        historical_index=args.historical_index,
        paper_manifest=args.paper_manifest,
        frozen_audit=args.frozen_audit,
    )
    print(f"evidence registry: wrote {count} rows to {args.registry}")


if __name__ == "__main__":
    main()
