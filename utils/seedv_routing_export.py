"""SEED-V routing export for the ICASSP typed-conditional MoE study."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils.tqdm_auto import tqdm_auto


def find_typed_conditional_moe_modules(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    backbone = getattr(model, "backbone", None)
    encoder = getattr(backbone, "encoder", None)
    if encoder is None:
        return []
    return [
        (idx, layer.moe_ffn)
        for idx, layer in enumerate(getattr(encoder, "layers", []))
        if getattr(getattr(layer, "moe_ffn", None), "moe_kind", None) == "typed_conditional"
    ]


def _subject_from_key(key: str) -> str:
    text = str(key)
    if "_t" in text:
        return text.split("_", 1)[0]
    prefix = text.rsplit("-", 2)[0]
    return prefix.split("_", 1)[0]


@torch.no_grad()
def export_seedv_routing_split(
    model: torch.nn.Module,
    data_loader,
    params: Any,
    split: str,
    epoch_tag: str,
    checkpoint_tag: str,
) -> str:
    out_dir = str(params.routing_export_dir)
    os.makedirs(out_dir, exist_ok=True)
    moe_layers = find_typed_conditional_moe_modules(model)
    if not moe_layers:
        raise RuntimeError("SEED-V routing export: no typed_conditional MoE layers found")

    base = f"seedv_routing_{split}_e{epoch_tag}_{checkpoint_tag}".replace(" ", "_")
    path = os.path.join(out_dir, f"{base}_per_sample.csv")
    rows: List[Dict[str, Any]] = []
    dataset_index = 0

    model.eval()
    for batch in tqdm_auto(data_loader, params, desc=f"seedv-routing[{split}]", mininterval=2):
        if len(batch) < 3 or not isinstance(batch[2], list):
            raise RuntimeError("SEED-V routing export requires return_sample_keys=True")
        x, y, keys = batch[0].cuda(), batch[1].cuda(), batch[2]
        pred = model(x)
        probs = F.softmax(pred, dim=-1)
        conf, pred_cls = probs.max(dim=-1)

        for i, key in enumerate(keys):
            row: Dict[str, Any] = {
                "split": split,
                "dataset_index": dataset_index,
                "epoch_tag": epoch_tag,
                "checkpoint_tag": checkpoint_tag,
                "lmdb_key": str(key),
                "subject_id": _subject_from_key(str(key)),
                "true_label": int(y[i].item()),
                "pred_label": int(pred_cls[i].item()),
                "correct": int(int(y[i].item()) == int(pred_cls[i].item())),
                "max_softmax_confidence": float(conf[i].item()),
            }
            for layer_idx, moe in moe_layers:
                cache = getattr(moe, "_routing_export_cache", None)
                if not cache:
                    raise RuntimeError(f"SEED-V routing export: layer {layer_idx} has no routing cache")
                for bank in ("spatial", "spectral"):
                    logits = cache[f"logits_{bank}"][i]
                    bank_probs = cache[f"probs_{bank}"][i]
                    row[f"layer{layer_idx}_{bank}_logits"] = ",".join(
                        f"{float(v):.8g}" for v in logits.tolist()
                    )
                    row[f"layer{layer_idx}_{bank}_probs"] = ",".join(
                        f"{float(v):.8g}" for v in bank_probs.tolist()
                    )
                    row[f"layer{layer_idx}_{bank}_top1"] = int(bank_probs.argmax().item())
                    row[f"layer{layer_idx}_{bank}_entropy"] = float(
                        cache[f"pre_entropy_{bank}"][i].item()
                    )
            rows.append(row)
            dataset_index += 1

    if not rows:
        raise RuntimeError("SEED-V routing export produced no rows")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[routing_export] wrote {path}", flush=True)
    return path

