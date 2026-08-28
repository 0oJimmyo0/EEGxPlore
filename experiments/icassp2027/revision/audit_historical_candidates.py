"""Audit the recoverable evidence for the rejected-paper FACED/ISRUC runs.

This audit is intentionally conservative.  Missing logs, checkpoints, or
historical source details remain ``unknown``; they are never filled from the
current revision defaults and never promoted to paper evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]

TARGETS = {
    "1464851": {
        "dataset": "FACED",
        "headline_role": "historical_full_selective",
        "headline_metrics": {"balanced_accuracy": 0.60548, "kappa": 0.55276, "f1": 0.60721},
        "context": "block_shared_typed_proj",
        "epoch_evidence": {"legacy_launcher": 40, "docs_note": 50},
        "launcher": "scripts/FACED/submit_train.slurm",
        "log_names": ["faced_hier_1464851.out"],
    },
    "1521786": {
        "dataset": "ISRUC",
        "headline_role": "historical_full_selective",
        "headline_metrics": {
            "balanced_accuracy": 0.80681,
            "kappa": 0.78019,
            "f1": 0.83062,
            "ema_balanced_accuracy": 0.79972,
            "ema_kappa": 0.76805,
            "ema_f1": 0.82202,
        },
        "context": "dual_query_block_typed_proj",
        "epoch_evidence": {"legacy_launcher": 30},
        "launcher": "scripts/ISRUC/submit_train.slurm",
        "log_names": ["isruc_hier_1521786.out"],
    },
}


def _candidate_paths(repo_root: Path, target: Dict[str, Any]) -> List[Path]:
    names = set(target["log_names"])
    paths = [
        repo_root / "logs" / target["dataset"] / "out" / name
        for name in names
    ]
    paths.extend(
        Path(raw)
        for raw in (
            "/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/logs/"
            f"{target['dataset']}/out/{name}"
            for name in names
        )
    )
    return paths


def _find_namespace(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["namespace_line"] = ""
        return result
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["read_error"] = str(exc)
        return result
    match = re.search(r"Namespace\([^\n]*\)", text)
    result["namespace_line"] = match.group(0) if match else ""
    result["namespace_recovered"] = bool(match)
    return result


def _git_trainability_evidence(repo_root: Path) -> Dict[str, Any]:
    command = [
        "git",
        "log",
        "--format=%h %ad %s",
        "--date=short",
        "-Strainability_mode",
        "--",
        "finetune_main.py",
        "finetune_trainer.py",
    ]
    try:
        output = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"search_status": "error", "error": str(exc), "commits": []}
    return {
        "search_status": "completed",
        "commits": output[:20],
        "current_legacy_fallback": "full",
        "current_legacy_fallback_basis": (
            "resolve_trainability_mode maps auto to full when no freeze/profile or "
            "typed_conditional route is active"
        ),
        "historical_exact_mode": "unknown",
    }


def build_audit(repo_root: Path) -> Dict[str, Any]:
    families: Dict[str, Any] = {}
    for family_id, target in TARGETS.items():
        log_candidates = [_find_namespace(path) for path in _candidate_paths(repo_root, target)]
        recovered = [entry for entry in log_candidates if entry.get("namespace_recovered")]
        launcher_path = repo_root / target["launcher"]
        families[family_id] = {
            "dataset": target["dataset"],
            "headline_role": target["headline_role"],
            "headline_metrics": target["headline_metrics"],
            "historical_context": target["context"],
            "epoch_evidence": target["epoch_evidence"],
            "legacy_launcher": {
                "path": str(launcher_path),
                "exists": launcher_path.is_file(),
                "passes_explicit_trainability_mode": (
                    "--trainability_mode" in launcher_path.read_text(encoding="utf-8", errors="replace")
                    if launcher_path.is_file() else False
                ),
            },
            "exact_log_candidates": log_candidates,
            "namespace_recovered": bool(recovered),
            "historical_trainability": {
                "status": "unknown",
                "current_resolver_inference": "full",
                "historical_code_behavior_confirmed": False,
            },
        }

    unresolved = []
    for family_id, family in families.items():
        if not family["namespace_recovered"]:
            unresolved.append(f"{family_id}.namespace")
        if not family["legacy_launcher"]["passes_explicit_trainability_mode"]:
            unresolved.append(f"{family_id}.trainability_mode")
        if len(family["epoch_evidence"]) > 1:
            unresolved.append(f"{family_id}.epoch_budget")

    return {
        "schema_version": 1,
        "audit_type": "historical_candidate_forensics",
        "repository_root": str(repo_root),
        "families": families,
        "trainability_source_audit": _git_trainability_evidence(repo_root),
        "all_historical_fields_resolved": not unresolved,
        "unresolved_fields": unresolved,
        "candidate_policy": {
            "paper_eligible": False,
            "do_not_use_test_metrics_to_choose_recipe": True,
            "default_trainability_inference": "full",
            "default_trainability_basis": (
                "legacy launchers omit --trainability_mode and current auto resolver falls back to full; "
                "historical exact source remains unavailable"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    payload = build_audit(args.repo_root.resolve())
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if args.strict and not payload["all_historical_fields_resolved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
