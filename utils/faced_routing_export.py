"""FACED per-sample routing export for typed_capacity_domain MoE."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.faced_meta import join_meta_for_key, load_recording_info_csv
from utils.faced_routing_analyze import write_all_analyses
from utils.tqdm_auto import tqdm_auto


def find_typed_capacity_moe_modules(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    bb = getattr(model, "backbone", None)
    if bb is None:
        return []
    enc = getattr(bb, "encoder", None)
    if enc is None:
        return []
    out: List[Tuple[int, torch.nn.Module]] = []
    for li, layer in enumerate(getattr(enc, "layers", [])):
        moe = getattr(layer, "moe_ffn", None)
        if moe is not None and getattr(moe, "moe_kind", None) == "typed_capacity_domain":
            out.append((li, moe))
    return out


@torch.no_grad()
def export_facced_routing_split(
    model: torch.nn.Module,
    data_loader,
    params: Any,
    split: str,
    epoch_tag: str,
    checkpoint_tag: str,
) -> Tuple[str, str]:
    out_dir = params.routing_export_dir
    os.makedirs(out_dir, exist_ok=True)

    meta_path = getattr(params, "faced_meta_csv", "") or ""
    rec_map = load_recording_info_csv(meta_path) if meta_path else {}

    moe_layers = find_typed_capacity_moe_modules(model)
    if not moe_layers:
        raise RuntimeError("routing export: no typed_capacity_domain MoE layers found on model.backbone")

    base = f"faced_routing_{split}_e{epoch_tag}_{checkpoint_tag}".replace(" ", "_")
    per_path = os.path.join(out_dir, f"{base}_per_sample.csv")

    md = getattr(params, "model_dir", "") or ""
    model_dir_basename = os.path.basename(os.path.normpath(md)) if md else ""
    routing_run_name = str(getattr(params, "routing_run_name", "") or "")

    rows: List[Dict[str, Any]] = []
    dataset_index = 0

    model.eval()
    for batch in tqdm_auto(data_loader, params, desc=f"routing[{split}]", mininterval=2):
        keys = None
        batch_meta = None
        if len(batch) >= 3 and isinstance(batch[2], list):
            keys = batch[2]
        if len(batch) >= 4 and isinstance(batch[3], dict):
            batch_meta = {k: v.cuda(non_blocking=True) for k, v in batch[3].items() if torch.is_tensor(v)}
        if keys is None:
            raise RuntimeError("routing export requires return_sample_keys=True (keys must be in batch)")

        x, y = batch[0].cuda(), batch[1].cuda()
        pred = model(x, batch_meta=batch_meta)
        prob = F.softmax(pred, dim=-1)
        conf, pred_cls = prob.max(dim=-1)

        bsz = x.size(0)
        for i in range(bsz):
            key = keys[i]
            meta = join_meta_for_key(key, rec_map)
            row: Dict[str, Any] = {
                "split": split,
                "dataset_index": dataset_index,
                "epoch_tag": epoch_tag,
                "checkpoint_tag": checkpoint_tag,
                "model_dir_basename": model_dir_basename,
                "routing_run_name": routing_run_name,
                "lmdb_key": key,
                "subject_id": str(meta.get("sub_id", "") or ""),
                "true_label": int(y[i].item()),
                "pred_label": int(pred_cls[i].item()),
                "correct": int(int(y[i].item()) == int(pred_cls[i].item())),
                "max_softmax_confidence": float(conf[i].item()),
                "recording_cohort": str(meta.get("cohort", "") or ""),
                "sample_rate_group": str(meta.get("sample_rate_group", "") or ""),
                "age_bucket": str(meta.get("age_bucket", "") or ""),
                "segment_index": meta.get("segment_index", ""),
                "chunk_index": meta.get("chunk_index", ""),
            }

            for li, moe in moe_layers:
                rc = getattr(moe, "_routing_export_cache", None)
                if rc is None:
                    raise RuntimeError(f"MoE layer {li} missing _routing_export_cache after forward")

                row[f"layer{li}_spatial_logits_pre_capacity"] = ",".join(
                    f"{float(rc['logits_spatial'][i, e].item()):.8g}" for e in range(moe.num_specialists)
                )
                row[f"layer{li}_spectral_logits_pre_capacity"] = ",".join(
                    f"{float(rc['logits_spectral'][i, e].item()):.8g}" for e in range(moe.num_specialists)
                )
                row[f"layer{li}_assigned_spatial_expert"] = int(rc["assigned_spatial"][i].item())
                row[f"layer{li}_assigned_spectral_expert"] = int(rc["assigned_spectral"][i].item())
                row[f"layer{li}_spatial_fallback"] = int(bool(rc["fallback_spatial"][i].item()))
                row[f"layer{li}_spectral_fallback"] = int(bool(rc["fallback_spectral"][i].item()))
                row[f"layer{li}_cohort_id"] = int(rc["cohort_id"][i].item())
                row[f"layer{li}_sample_rate_group_id"] = int(rc["sample_rate_group_id"][i].item())
                row[f"layer{li}_age_bucket_id"] = int(rc["age_bucket_id"][i].item())
                row[f"layer{li}_segment_bucket_id"] = int(rc["segment_bucket_id"][i].item())

            rows.append(row)
            dataset_index += 1

    if not rows:
        raise RuntimeError("routing export: no rows")

    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Keep existing analysis pipeline by using the first MoE layer as canonical typed router output.
    num_e = moe_layers[0][1].num_specialists
    canonical_rows = []
    li0 = moe_layers[0][0]
    for r in rows:
        rc = dict(r)
        rc["spatial_top1_expert"] = rc.get(f"layer{li0}_assigned_spatial_expert", -1)
        rc["spectral_top1_expert"] = rc.get(f"layer{li0}_assigned_spectral_expert", -1)
        canonical_rows.append(rc)
    write_all_analyses(canonical_rows, out_dir, base, num_e)

    print(f"[routing_export] wrote {per_path}", flush=True)
    return per_path, base
