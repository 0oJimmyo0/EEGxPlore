"""Aggregate typed-specialist routing over complete validation splits.

This is an inference-only artifact diagnostic.  Unlike
``aggregate_routing_validation.py``, it evaluates every validation batch at
the selected checkpoint and aggregates the actual soft routing probabilities.
It does not alter paper metrics, checkpoint selection, or training output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune_main import build_dataset, build_model  # noqa: E402
from utils.tqdm_auto import tqdm_auto  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_summary(run_root: Path) -> Path:
    summaries = sorted(run_root.glob("run_summary*.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected exactly one run_summary*.json in {run_root}, found {len(summaries)}"
        )
    return summaries[0]


def _find_checkpoint(run_root: Path, summary: Dict[str, Any]) -> Path:
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


def _load_state_dict(path: Path) -> Dict[str, Any]:
    state = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}")
    return state


def _forward(model: torch.nn.Module, x: torch.Tensor, batch: Any) -> None:
    batch_meta = None
    if len(batch) >= 4 and isinstance(batch[3], dict):
        batch_meta = {
            key: value.cuda(non_blocking=True)
            for key, value in batch[3].items()
            if torch.is_tensor(value)
        }
    if batch_meta is None:
        model(x)
    else:
        try:
            model(x, batch_meta=batch_meta)
        except TypeError:
            model(x)


def _moe_modules(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    backbone = getattr(model, "backbone", None)
    encoder = getattr(backbone, "encoder", None)
    if encoder is None:
        return []
    modules = []
    for layer_idx, layer in enumerate(getattr(encoder, "layers", [])):
        moe = getattr(layer, "moe_ffn", None)
        if moe is not None and hasattr(moe, "_routing_export_cache"):
            modules.append((layer_idx, moe))
    return modules


def _new_bank(num_experts: int) -> Dict[str, Any]:
    return {
        "num_experts": int(num_experts),
        "probability_sum": [0.0] * int(num_experts),
        "top1_counts": [0] * int(num_experts),
        "assigned_counts": [0] * int(num_experts),
        "entropy_sum": 0.0,
        "sample_count": 0,
        "assigned_sample_count": 0,
    }


def _accumulate_bank(bank: Dict[str, Any], probs: torch.Tensor, cache: Dict[str, Any], prefix: str) -> None:
    if probs.ndim != 2:
        raise RuntimeError(f"expected [batch, experts] routing probabilities, got {tuple(probs.shape)}")
    probs_cpu = probs.detach().float().cpu()
    count = int(probs_cpu.shape[0])
    if int(probs_cpu.shape[1]) != int(bank["num_experts"]):
        raise RuntimeError("routing expert count changed within a validation pass")
    bank["probability_sum"] = [
        float(old + new)
        for old, new in zip(bank["probability_sum"], probs_cpu.sum(dim=0).tolist())
    ]
    bank["top1_counts"] = [
        int(old + new)
        for old, new in zip(
            bank["top1_counts"],
            torch.bincount(probs_cpu.argmax(dim=-1), minlength=bank["num_experts"]).tolist(),
        )
    ]
    entropy = -(probs_cpu * probs_cpu.clamp_min(1e-10).log()).sum(dim=-1)
    bank["entropy_sum"] = float(bank["entropy_sum"] + float(entropy.sum().item()))
    bank["sample_count"] = int(bank["sample_count"] + count)

    assigned = cache.get(f"assigned_{prefix}")
    if assigned is not None:
        assigned_cpu = assigned.detach().cpu().reshape(-1)
        if assigned_cpu.numel() != count:
            raise RuntimeError(f"assigned routing count is not sample-aligned for {prefix}")
        bank["assigned_counts"] = [
            int(old + new)
            for old, new in zip(
                bank["assigned_counts"],
                torch.bincount(assigned_cpu, minlength=bank["num_experts"]).tolist(),
            )
        ]
        bank["assigned_sample_count"] = int(bank["assigned_sample_count"] + count)


def _effective_experts(probabilities: Iterable[float]) -> float:
    values = [float(value) for value in probabilities]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in values)
    return float(math.exp(entropy))


def _finalize_bank(bank: Dict[str, Any]) -> Dict[str, Any]:
    n = int(bank["sample_count"])
    probability_mean = [value / max(n, 1) for value in bank["probability_sum"]]
    result = {
        "num_experts": int(bank["num_experts"]),
        "sample_count": n,
        "mean_soft_probability": probability_mean,
        "mean_soft_entropy": float(bank["entropy_sum"] / max(n, 1)),
        "soft_effective_experts": _effective_experts(probability_mean),
        "top1_counts": list(bank["top1_counts"]),
        "top1_share": [value / max(n, 1) for value in bank["top1_counts"]],
        "top1_effective_experts": _effective_experts(
            [value / max(n, 1) for value in bank["top1_counts"]]
        ),
    }
    if int(bank["assigned_sample_count"]):
        assigned_n = int(bank["assigned_sample_count"])
        result["assigned_counts"] = list(bank["assigned_counts"])
        result["assigned_share"] = [value / assigned_n for value in bank["assigned_counts"]]
        result["assigned_effective_experts"] = _effective_experts(
            [value / assigned_n for value in bank["assigned_counts"]]
        )
    return result


@torch.no_grad()
def diagnose_run(run_root: Path, cuda: int, split: str) -> Dict[str, Any]:
    summary_path = _find_summary(run_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = _find_checkpoint(run_root, summary)
    config = dict(summary.get("config") or {})
    if not config:
        raise RuntimeError(f"run summary has no config: {summary_path}")
    config["cuda"] = int(cuda)
    config["return_sample_keys"] = False
    config["return_domain_ids"] = False
    params = argparse.Namespace(**config)

    data_loader = build_dataset(params)
    if split not in data_loader:
        raise RuntimeError(f"split {split!r} is not available for {run_root}")
    model = build_model(params).cuda()
    model.load_state_dict(_load_state_dict(checkpoint), strict=True)
    model.eval()
    modules = _moe_modules(model)
    if not modules:
        raise RuntimeError(f"no typed specialist MoE module found in {run_root}")

    layer_banks: Dict[str, Dict[str, Dict[str, Any]]] = {}
    batch_count = 0
    for batch in tqdm_auto(data_loader[split], params, desc=f"routing[{run_root.name}/{split}]", mininterval=2):
        x = batch[0].cuda(non_blocking=True)
        _forward(model, x, batch)
        batch_count += 1
        for layer_idx, moe in modules:
            cache = getattr(moe, "_routing_export_cache", None)
            if not isinstance(cache, dict):
                raise RuntimeError(f"MoE layer {layer_idx} did not expose a routing cache")
            layer_key = str(layer_idx)
            layer_banks.setdefault(layer_key, {})
            for bank_name, prefix in (("spatial", "spatial"), ("spectral", "spectral")):
                probs = cache.get(f"probs_{prefix}")
                if probs is None:
                    raise RuntimeError(f"MoE layer {layer_idx} has no {bank_name} probabilities")
                if bank_name not in layer_banks[layer_key]:
                    layer_banks[layer_key][bank_name] = _new_bank(int(probs.shape[-1]))
                _accumulate_bank(layer_banks[layer_key][bank_name], probs, cache, prefix)

    layers = []
    for layer_idx in sorted(layer_banks, key=int):
        layers.append({
            "layer": int(layer_idx),
            "banks": {
                bank_name: _finalize_bank(bank)
                for bank_name, bank in sorted(layer_banks[layer_idx].items())
            },
        })
    result = {
        "dataset": summary.get("dataset") or config.get("downstream_dataset"),
        "condition": summary.get("revision_condition") or summary.get("condition"),
        "seed": config.get("seed"),
        "run_root": str(run_root.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "selected_checkpoint": {
            "best_epoch": summary.get("best_epoch"),
            "selection_metric": config.get("selection_metric"),
        },
        "split": split,
        "validation_batches": batch_count,
        "layers": layers,
    }
    del model, data_loader
    torch.cuda.empty_cache()
    return result


def _run_roots_from_csv(path: Path) -> List[Path]:
    roots: List[Path] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("executable_condition") != "specialist_augmented_full":
                continue
            checkpoint = Path(str(row.get("checkpoint_path", "") or ""))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"frozen evidence checkpoint is missing: {checkpoint}")
            roots.append(checkpoint.parent)
    return sorted(set(roots), key=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    if bool(args.run_root) == bool(args.audit_csv):
        parser.error("provide exactly one of --run-root or --audit-csv")
    roots = [path.resolve() for path in args.run_root] if args.run_root else _run_roots_from_csv(args.audit_csv)

    torch.cuda.set_device(int(args.cuda))
    records = []
    for root in roots:
        print(f"[full-routing] {root}", flush=True)
        records.append(diagnose_run(root, args.cuda, args.split))
    payload = {
        "schema": "icassp2027_full_validation_routing_diagnostics_v1",
        "inference_only": True,
        "scope": "complete_validation_split",
        "full_validation_set": True,
        "routing_probabilities": "actual_soft_probabilities_aggregated_over_all_validation_batches",
        "diagnostic_code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[full-routing] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
