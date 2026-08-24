#!/usr/bin/env python3
"""Run one frozen pretrained-CBraMod forward per ICASSP dataset.

This is a schema/finite-output gate, not a performance experiment.  It uses
one sample from each frozen manifest and never writes checkpoints or metrics.
ISRUC is checked at the CBraMod epoch-input boundary (one 6-channel epoch,
reshaped from 6000 points into 30 patches of 200 points).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

# Allow the probe to be launched directly from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.faced_dataset import CustomDataset as FacedDataset
from datasets.isruc_dataset import LoadDataset as IsrucLoadDataset
from datasets.physio_dataset import CustomDataset as PhysioDataset
from datasets.seedv_dataset import CustomDataset as SeedVDataset
from models.cbramod import CBraMod


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_backbone(checkpoint: Path) -> CBraMod:
    model = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CBraMod checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _forward(model: CBraMod, array: np.ndarray) -> Dict[str, Any]:
    x = torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        output = model(x)
    return {
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "input_finite": bool(torch.isfinite(x).all().item()),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "output_mean": float(output.mean().item()),
        "output_std": float(output.std(unbiased=False).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("/data/neurogroup/mingyangjiang/data"))
    parser.add_argument("--experiment_root", type=Path, default=Path("experiments/icassp2027"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    manifests = args.experiment_root / "manifests"
    model = _load_backbone(args.checkpoint)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    results: Dict[str, Any] = {}

    seed = SeedVDataset(
        str(args.data_root / "SEED-V_processed_lmdb"),
        mode="train",
        split_manifest_path=str(manifests / "seedv/split_manifest.json"),
    )
    results["SEED-V"] = _forward(model, seed[0][0] / 1.0)

    faced = FacedDataset(
        str(args.data_root / "FACED"),
        mode="train",
        split_manifest_path=str(manifests / "faced/split_manifest.json"),
    )
    results["FACED"] = _forward(model, faced[0][0])

    physio = PhysioDataset(
        str(args.data_root / "PHYSIO_MI"),
        mode="train",
        split_manifest_path=str(manifests / "physionet_mi/split_manifest.json"),
    )
    results["PhysioNet-MI"] = _forward(model, physio[0][0])

    isruc_params = SimpleNamespace(
        datasets_dir=str(args.data_root / "ISRUC"),
        icassp_split_manifest=str(manifests / "isruc/split_manifest.json"),
    )
    isruc_loader = IsrucLoadDataset(isruc_params)
    isruc_pairs = isruc_loader.split_dataset(isruc_loader.seqs_labels_path_pair)[0]
    sequence = np.load(isruc_pairs[0][0], mmap_mode="r")[0] / 100.0
    sequence = sequence.reshape(6, 30, 200)
    results["ISRUC"] = _forward(model, sequence)

    passed = trainable == 0 and all(
        result["input_finite"] and result["output_finite"] for result in results.values()
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "trainable_parameter_count": trainable,
        "datasets": results,
        "passed": passed,
    }
    output_path = args.experiment_root / "audits/frozen_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Frozen probe failed")


if __name__ == "__main__":
    main()
