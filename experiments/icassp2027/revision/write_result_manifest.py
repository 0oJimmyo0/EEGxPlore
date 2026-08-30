"""Write the terminal result manifest for one ICASSP revision attempt."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: write_result_manifest.py PATH FINAL_EXIT TRAINING_EXIT", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    final_exit = int(sys.argv[2])
    training_exit = int(sys.argv[3])
    summary = Path(_env("MODEL_DIR")) / "experiment_summary.csv"
    condition = _env("CONDITION")
    payload = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": final_exit,
        "training_exit_code": training_exit,
        "repository_commit": _env("ICASSP_EXECUTION_COMMIT", _env("GIT_COMMIT")),
        "expected_commit": _env("EXPECTED_COMMIT"),
        "execution_commit_start": _env("ICASSP_EXECUTION_COMMIT", _env("GIT_COMMIT")),
        "execution_commit_end": _env("ICASSP_EXECUTION_COMMIT_END"),
        "git_dirty": _env("ICASSP_EXECUTION_DIRTY", "0") == "1",
        "provenance_consistent": _env("ICASSP_PROVENANCE_CONSISTENT", "0") == "1",
        "run_mode": _env("RUN_MODE"),
        "paper_eligible": _env("RUN_MODE") == "paper" and final_exit == 0,
        "dataset": _env("DATASET"),
        "condition": condition,
        "protocol": _env("PROTOCOL"),
        "seed": int(_env("SEED", "0")),
        "model_dir": _env("MODEL_DIR"),
        "experiment_summary": str(summary) if summary.is_file() else "",
        "summary_present": summary.is_file(),
        "data_contract_path": _env("DATA_CONTRACT_PATH"),
        "data_contract_sha256": _env("DATA_CONTRACT_SHA256"),
        "paper_protocol_path": _env("PAPER_PROTOCOL_PATH"),
        "paper_protocol_id": _env("PAPER_PROTOCOL_ID"),
        "paper_protocol_sha256": _env("PAPER_PROTOCOL_SHA256"),
        "paper_method_recipe_path": _env("PAPER_METHOD_RECIPE_PATH") if condition == "specialist_augmented_full" else "",
        "paper_method_recipe_id": _env("PAPER_METHOD_RECIPE_ID") if condition == "specialist_augmented_full" else "",
        "paper_method_recipe_sha256": _env("PAPER_METHOD_RECIPE_SHA256") if condition == "specialist_augmented_full" else "",
        "paper_method_semantics_sha256": _env("PAPER_METHOD_SEMANTICS_SHA256") if condition == "specialist_augmented_full" else "",
        "use_component_lr": _env("COMPONENT_LR_ENABLED", "0") == "1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
