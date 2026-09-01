"""Plot paired class-wise stage deltas for the ICASSP manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    datasets = [name for name in ("ISRUC", "PhysioNet-MI") if name in payload["datasets"]]
    if len(datasets) != 2:
        raise ValueError(f"expected ISRUC and PhysioNet-MI diagnostics, found {datasets}")

    arrays = []
    for dataset in datasets:
        classwise = payload["datasets"][dataset]["classwise"]
        labels = classwise["class_labels"]
        means = classwise["mean_sd"]
        arrays.append((dataset, labels, [
            [100.0 * cell["mean"] for cell in means["delta_attnres"]],
            [100.0 * cell["mean"] for cell in means["delta_specialist_given_attnres"]],
        ]))

    max_abs = max(abs(value) for _, _, rows in arrays for row in rows for value in row)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), constrained_layout=False)
    cmap = plt.get_cmap("RdBu_r")
    image = None
    for panel_idx, (ax, (dataset, labels, rows)) in enumerate(zip(axes, arrays)):
        image = ax.imshow(rows, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(dataset, fontsize=9, fontweight="bold", pad=5)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        if panel_idx == 0:
            ax.set_yticks([0, 1], ["+ AttnRes", "+ Specialists\n| AttnRes"], fontsize=7.5)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                text_color = "white" if abs(value) > 0.55 * max_abs else "#222222"
                ax.text(col_idx, row_idx, f"{value:+.1f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    fig.text(0.5, 0.015, "Class-wise recall change (percentage points)",
             ha="center", fontsize=8.5)
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.06, aspect=14)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("Δ recall (pp)", fontsize=7.5, labelpad=3)
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.24, top=0.82, wspace=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.05)
    print(args.output)


if __name__ == "__main__":
    main()
