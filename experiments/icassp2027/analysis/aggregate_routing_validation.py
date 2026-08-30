"""Summarize stored Specialist validation routing snapshots.

The training code currently stores the final diagnostic snapshot from each
validation epoch, rather than a complete validation-set aggregate.  This tool
therefore labels its output explicitly as ``last_validation_batch_per_epoch``
and must not be used to claim globally balanced routing.  A future full-set
inference tool can replace this analysis without changing the training path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from common import DATASETS, DEFAULT_OUTPUT_ROOT, SEEDS, run_directory


def _effective_experts(counts: List[int]) -> float | None:
    total = sum(counts)
    if total <= 0:
        return None
    probs = [count / total for count in counts if count > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return math.exp(entropy)


def summarize(output_root: Path, datasets: List[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset in datasets:
        for seed in SEEDS:
            run_dir = run_directory(output_root, dataset, "specialist_augmented_full", seed)
            path = run_dir / "routing_diagnostics.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                diagnostics = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing.append(str(path))
                continue
            for entry in diagnostics:
                spatial = list(entry.get("spatial_assigned_count_per_expert") or [])
                spectral = list(entry.get("spectral_assigned_count_per_expert") or [])
                rows.append({
                    "dataset": dataset,
                    "condition": "specialist_augmented_full",
                    "seed": int(seed),
                    "epoch": int(entry.get("epoch", 0)),
                    "spatial_top_share": max(spatial) / sum(spatial) if sum(spatial) else None,
                    "spectral_top_share": max(spectral) / sum(spectral) if sum(spectral) else None,
                    "spatial_entropy": entry.get("spatial_routing_entropy_pre_capacity"),
                    "spectral_entropy": entry.get("spectral_routing_entropy_pre_capacity"),
                    "spatial_effective_experts": _effective_experts(spatial),
                    "spectral_effective_experts": _effective_experts(spectral),
                    "spatial_counts": spatial,
                    "spectral_counts": spectral,
                })
    return {
        "schema_version": 1,
        "scope": "last_validation_batch_per_epoch",
        "full_validation_set": False,
        "rows": rows,
        "missing": missing,
        "summary": {
            "snapshot_count": len(rows),
            "mean_spatial_effective_experts": mean([r["spatial_effective_experts"] for r in rows if r["spatial_effective_experts"] is not None]) if rows else None,
            "mean_spectral_effective_experts": mean([r["spectral_effective_experts"] for r in rows if r["spectral_effective_experts"] is not None]) if rows else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "routing_validation_summary.json")
    args = parser.parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    result = summarize(args.output_root, datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"] | {"scope": result["scope"], "full_validation_set": result["full_validation_set"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
