"""Contract test for the row-level ICASSP evidence registry."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from build_evidence_registry import (
    DEFAULT_PAPER_MANIFEST,
    FIELDNAMES,
    _row_for_run,
    build_registry,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "revision"
        run_dir = root / "seedv" / "combined" / "seed_42"
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "repository_commit": "abc123",
                    "dataset": "SEED-V",
                    "condition": "combined",
                    "protocol": "cbramod_benchmark",
                    "seed": 42,
                    "dataset_dir": "/data/seedv",
                    "data_contract_sha256": "contract123",
                    "fresh_selective_recipe_sha256": "recipe123",
                    "paper_method_recipe_id": "method123",
                    "paper_method_recipe_sha256": "methodsha123",
                    "paper_method_recipe_path": "/repo/paper_method.json",
                    "use_component_lr": True,
                    "command": ["--cuda", "1", "--epochs", "40"],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "result_manifest.json").write_text(
            json.dumps({"exit_code": 0}), encoding="utf-8"
        )
        with (run_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset", "revision_condition", "revision_protocol", "seed",
                    "epochs", "selection_metric", "git_commit", "model_path",
                    "test_balanced_accuracy", "test_macro_f1", "test_weighted_f1", "test_kappa",
                    "trainable_parameter_count", "total_wall_seconds", "peak_cuda_mb",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "dataset": "SEED-V",
                "revision_condition": "combined",
                "revision_protocol": "cbramod_benchmark",
                "seed": "42",
                "epochs": "40",
                "selection_metric": "kappa",
                "git_commit": "abc123",
                "model_path": "",
                    "test_balanced_accuracy": "0.41",
                    "test_macro_f1": "0.40",
                    "test_weighted_f1": "0.41",
                    "test_kappa": "0.27",
                "trainable_parameter_count": "123",
                "total_wall_seconds": "12.5",
                "peak_cuda_mb": "2048",
            })

        forced_primary = _row_for_run(
            run_dir / "run_manifest.json",
            hash_checkpoints=False,
            primary_cells={("SEED-V", "combined", "42")},
        )
        assert forced_primary["evidence_role"] == "confirmatory_candidate"
        assert forced_primary["paper_eligibility"] == "primary_new_evidence_pending_audit"
        assert forced_primary["reuse_decision"] == "candidate_pending_audit"

        registry = Path(tmp) / "evidence_registry.csv"
        historical_index = Path(tmp) / "historical_candidates.csv"
        with historical_index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "condition", "historical_family_id", "seed", "run_status"],
            )
            writer.writeheader()
            writer.writerow({
                "dataset": "SEED-V",
                "condition": "historical_selective",
                "historical_family_id": "1785556",
                "seed": "42",
                "run_status": "candidate",
            })
        assert build_registry(
            root,
            registry,
            historical_index=historical_index,
            paper_manifest=DEFAULT_PAPER_MANIFEST,
        ) == 2
        with registry.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert all(set(row) == set(FIELDNAMES) for row in rows)
        assert rows[0]["condition"] == "combined"
        assert rows[0]["provenance_class"] == "new_multiseed"
        assert rows[0]["evidence_role"] == "development_diagnostic"
        assert rows[0]["paper_eligibility"] == "development_only_not_primary"
        assert rows[0]["reuse_decision"] == "development_context_only"
        assert rows[0]["run_mode"] == "paper"
        assert rows[0]["tmlr_overlap_status"] == "unreviewed_pending_row_audit"
        assert rows[0]["gpu"] == "1"
        assert rows[0]["split"] == "cbramod_benchmark"
        assert rows[0]["data_contract_sha256"] == "contract123"
        assert rows[0]["fresh_selective_recipe_sha256"] == "recipe123"
        assert rows[0]["paper_method_recipe_id"] == "method123"
        assert rows[0]["paper_method_recipe_sha256"] == "methodsha123"
        assert rows[0]["paper_method_recipe_path"] == "/repo/paper_method.json"
        assert rows[0]["use_component_lr"] == "True"
        assert rows[0]["test_weighted_f1"] == "0.41"
        assert rows[1]["source_kind"] == "rejected_paper_report"
        assert rows[1]["provenance_class"] == "legacy_context_only"
        assert rows[1]["run_mode"] == "legacy_report"
        assert rows[1]["historical_family_id"] == "1785556"
        assert rows[1]["reuse_decision"] == "candidate_pending_audit"

    print("evidence registry contract: PASS")


if __name__ == "__main__":
    main()
