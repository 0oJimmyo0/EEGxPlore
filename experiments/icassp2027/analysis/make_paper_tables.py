"""Generate manuscript-ready Markdown tables from audited aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import DEFAULT_OUTPUT_ROOT


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _mean_sd(payload: Dict[str, Any]) -> str:
    return f"{_fmt(payload.get('mean'))} ± {_fmt(payload.get('sd'))}"


def _markdown(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def make_tables(payload: Dict[str, Any]) -> Dict[str, str]:
    aggregate = payload["aggregate"]
    main_rows: List[List[Any]] = []
    efficiency_rows: List[List[Any]] = []
    delta_rows: List[List[Any]] = []
    labels = {
        "full": "Full CBraMod FT",
        "specialist_augmented_full": "AttnRes + Typed Specialists",
    }
    for dataset in sorted(aggregate["by_dataset_method"]):
        methods = aggregate["by_dataset_method"][dataset]
        for method in ("full", "specialist_augmented_full"):
            stats = methods[method]
            main_rows.append([
                dataset,
                labels[method],
                _mean_sd(stats["test_balanced_accuracy"]),
                _mean_sd(stats["test_weighted_f1"]),
                _mean_sd(stats["test_kappa"]),
                _mean_sd(stats["test_macro_f1"]),
            ])
            efficiency_rows.append([
                dataset,
                labels[method],
                _fmt(stats["trainable_parameters"]["mean"] / 1e6, 2) + "M",
                _fmt(stats["runtime_seconds"]["mean"] / 60.0, 1) + " min",
                _fmt(stats["peak_cuda_memory_mb"]["mean"] / 1024.0, 2) + " GiB",
            ])
        paired = aggregate["paired_by_dataset"][dataset]
        delta_rows.append([
            dataset,
            _mean_sd(paired["test_balanced_accuracy"]),
            _mean_sd(paired["test_weighted_f1"]),
            _mean_sd(paired["test_kappa"]),
            f"{paired['positive_delta_count']['test_balanced_accuracy']}/{paired['pair_count']}",
        ])

    return {
        "paper_main_table.md": _markdown(
            ["Dataset", "Method", "BA", "Weighted-F1", "κ", "Macro-F1"], main_rows
        ),
        "paper_paired_delta_table.md": _markdown(
            ["Dataset", "Δ BA", "Δ Weighted-F1", "Δ κ", "Positive Δ BA"], delta_rows
        ),
        "paper_efficiency_table.md": _markdown(
            ["Dataset", "Method", "Trainable Params", "Runtime", "Peak CUDA Memory"], efficiency_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_OUTPUT_ROOT / "confirmatory_aggregate.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if not args.aggregate.is_file():
        raise SystemExit(f"aggregate not found: {args.aggregate}")
    payload = json.loads(args.aggregate.read_text(encoding="utf-8"))
    tables = make_tables(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")
    print(json.dumps({"tables": sorted(tables), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
