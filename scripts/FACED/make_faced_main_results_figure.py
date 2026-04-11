#!/usr/bin/env python3
"""Generate the FACED main-results poster figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[2] / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    metrics = ["Balanced Accuracy", "Cohen's kappa", "Weighted F1"]
    series = {
        "Conservative sparse baseline": [0.58374, 0.52830, 0.58375],
        "Broader sparse specialization\n(2 MoE layers)": [0.59863, 0.54423, 0.59695],
        "Broader backbone modification\n(full)": [0.55153, 0.49279, 0.55329],
        "Best selective variant": [0.60548, 0.55276, 0.60721],
    }
    colors = {
        "Conservative sparse baseline": "#9AA1A9",
        "Broader sparse specialization\n(2 MoE layers)": "#C5CBD3",
        "Broader backbone modification\n(full)": "#6F7782",
        "Best selective variant": "#D95F02",
    }

    x = np.arange(len(metrics))
    width = 0.18

    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.titlesize": 19,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(15.5, 5.2), constrained_layout=True)

    offsets = [-1.5, -0.5, 0.5, 1.5]
    for offset, (label, values) in zip(offsets, series.items()):
        bars = ax.bar(
            x + offset * width,
            values,
            width=width,
            label=label,
            color=colors[label],
            edgecolor="#30343B",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                rotation=90,
                color="#20232A",
            )

    ax.set_xticks(x, metrics)
    ax.set_ylim(0.45, 0.65)
    ax.set_ylabel("Test score")
    ax.set_title("Selective Adaptation Beats Broader Alternatives on FACED", pad=12)
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncols=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.5,
        borderaxespad=0.0,
    )

    out_pdf = OUT_DIR / "faced_main_results.pdf"
    out_png = OUT_DIR / "faced_main_results.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
