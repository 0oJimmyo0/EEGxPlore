"""Replay selected ICASSP checkpoints for class- and subject-level diagnostics.

This utility performs inference only. It reconstructs each run from the saved
run summary configuration, loads the checkpoint already selected by validation
kappa, and writes detailed test metrics. It does not train, tune, or replace
any paper-facing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune_evaluator import Evaluator  # noqa: E402
from finetune_main import build_dataset, build_model  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_summary(run_root: Path) -> Path:
    summaries = sorted(run_root.glob("run_summary*.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"expected exactly one run_summary*.json in {run_root}, found {len(summaries)}")
    return summaries[0]


def _find_checkpoint(run_root: Path, summary: dict) -> Path:
    configured = Path(str(summary.get("model_path", ""))).name
    if configured:
        candidate = run_root / configured
        if candidate.is_file():
            return candidate
    checkpoints = sorted(run_root.glob("*.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"could not resolve selected checkpoint in {run_root}; configured={configured!r}, "
            f"local checkpoints={len(checkpoints)}"
        )
    return checkpoints[0]


def _load_state_dict(path: Path):
    state = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    return state


def replay(run_root: Path, cuda: int) -> dict:
    summary_path = _find_summary(run_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = dict(summary.get("config") or {})
    if not config:
        raise RuntimeError(f"run summary has no config: {summary_path}")

    checkpoint = _find_checkpoint(run_root, summary)
    dataset = str(summary.get("dataset") or config.get("downstream_dataset") or "unknown")
    config["cuda"] = int(cuda)
    config["selected_checkpoint_diagnostics"] = True
    config["return_sample_keys"] = dataset == "PhysioNet-MI"
    config["return_domain_ids"] = False
    params = argparse.Namespace(**config)

    torch.cuda.set_device(int(cuda))
    data_loader = build_dataset(params)
    model = build_model(params).cuda()
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    model.eval()
    with torch.no_grad():
        detailed = Evaluator(params, data_loader["test"]).get_detailed_metrics_for_multiclass(model)

    result = {
        "dataset": dataset,
        "condition": summary.get("revision_condition") or summary.get("condition"),
        "seed": config.get("seed"),
        "paper_eligible": summary.get("paper_eligible"),
        "run_root": str(run_root.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "selected_checkpoint": {
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": config.get("selection_metric"),
            "eval_source": summary.get("primary_eval_source"),
        },
        "test_metrics": detailed,
    }
    del model, data_loader
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    records = []
    for run_root in args.run_root:
        print(f"[replay] {run_root}", flush=True)
        records.append(replay(run_root.resolve(), args.cuda))

    payload = {
        "schema": "icassp2027_selected_diagnostics_v1",
        "inference_only": True,
        "selection_rule": "checkpoint selected by validation kappa in original run",
        "diagnostic_code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[replay] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
