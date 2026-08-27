"""Contract test for the row-level ICASSP evidence registry."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from build_evidence_registry import FIELDNAMES, build_registry


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
                    "test_balanced_accuracy", "test_macro_f1", "test_kappa",
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
                "test_kappa": "0.27",
                "trainable_parameter_count": "123",
                "total_wall_seconds": "12.5",
                "peak_cuda_mb": "2048",
            })

        registry = Path(tmp) / "evidence_registry.csv"
        assert build_registry(root, registry) == 1
        with registry.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert set(rows[0]) == set(FIELDNAMES)
        assert rows[0]["condition"] == "combined"
        assert rows[0]["tmlr_overlap_status"] == "unreviewed_pending_row_audit"
        assert rows[0]["reuse_decision"] == "candidate_pending_audit"
        assert rows[0]["gpu"] == "1"
        assert rows[0]["split"] == "cbramod_benchmark"

    print("evidence registry contract: PASS")


if __name__ == "__main__":
    main()
