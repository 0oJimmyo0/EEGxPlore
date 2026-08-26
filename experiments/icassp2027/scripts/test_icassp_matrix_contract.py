"""Dry-run audit for the complete four-dataset, four-method ICASSP matrix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "experiments/icassp2027/configs/run_pilot.sh"
DATASETS = ("SEED-V", "FACED", "ISRUC", "PhysioNet-MI")
METHODS = ("frozen", "depth_aggregation", "upper4", "full")
DATASET_TAGS = {
    "SEED-V": "seedv",
    "FACED": "faced",
    "ISRUC": "isruc",
    "PhysioNet-MI": "physionet_mi",
}


def main() -> None:
    if not LAUNCHER.is_file():
        raise FileNotFoundError(LAUNCHER)

    model_root = REPO_ROOT / "output/icassp2027_depth/health20"
    env = os.environ.copy()
    env.update({
        "ICASSP_REPO_DIR": str(REPO_ROOT),
        "EPOCHS": "20",
        "MODEL_ROOT": str(model_root),
        "SEED": "42",
        "NUM_WORKERS": "0",
        "SELECTED_CHECKPOINT_DIAGNOSTICS": "0",
    })

    completed = 0
    for dataset in DATASETS:
        for method in METHODS:
            result = subprocess.run(
                ["bash", str(LAUNCHER), dataset, method, "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                raise RuntimeError(
                    f"launcher dry-run failed for {dataset}/{method}:\n{output}"
                )

            expected_dir = model_root / DATASET_TAGS[dataset] / method / "seed_42"
            expected_fragments = (
                "--epochs 20",
                f"--model_dir {expected_dir}",
                f"--trainability_mode {method}",
            )
            missing = [fragment for fragment in expected_fragments if fragment not in output]
            if missing:
                raise AssertionError(
                    f"launcher output missing {missing} for {dataset}/{method}:\n{output}"
                )
            if method == "depth_aggregation":
                for fragment in ("--attnres_variant pre_attn", "--attnres_start_layer 8"):
                    if fragment not in output:
                        raise AssertionError(
                            f"DepthAgg launcher output missing {fragment}:\n{output}"
                        )
            elif "--attnres_variant none" not in output:
                raise AssertionError(
                    f"non-DepthAgg launcher enabled AttnRes for {dataset}/{method}:\n{output}"
                )
            completed += 1

    expected = len(DATASETS) * len(METHODS)
    assert completed == expected == 16
    print(f"ICASSP launcher matrix contract: PASS ({completed} active dry-runs)")


if __name__ == "__main__":
    main()
