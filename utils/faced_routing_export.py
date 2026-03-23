"""FACED typed-MoE per-sample routing CSV export + grouped summaries (analysis only)."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.faced_meta import join_meta_for_key, load_recording_info_csv
from utils.faced_routing_analyze import write_all_analyses
from utils.tqdm_auto import tqdm_auto


def find_typed_moe_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Return first TypedDualBankSharedMoEFFN on the backbone."""
    bb = getattr(model, "backbone", None)
    if bb is None:
        return None
    enc = getattr(bb, "encoder", None)
    if enc is None:
        return None
    for layer in getattr(enc, "layers", []):
        moe = getattr(layer, "moe_ffn", None)
        if moe is not None and getattr(moe, "moe_kind", None) == "typed_shared_specialist":
            return moe
    return None


@torch.no_grad()
def export_facced_routing_split(
    model: torch.nn.Module,
    data_loader,
    params: Any,
    split: str,
    epoch_tag: str,
    checkpoint_tag: str,
) -> Tuple[str, str]:
    """
    One CSV per split with per-sample routing + optional metadata join.
    Returns (per_sample_csv, analysis_manifest_prefix).
    """
    out_dir = params.routing_export_dir
    os.makedirs(out_dir, exist_ok=True)
    meta_path = getattr(params, "faced_meta_csv", "") or ""
    rec_map = load_recording_info_csv(meta_path) if meta_path else {}

    moe = find_typed_moe_module(model)
    if moe is None:
        raise RuntimeError("routing export: no TypedDualBankSharedMoEFFN found on model.backbone")

    num_e = moe.num_specialists
    base = f"faced_routing_{split}_e{epoch_tag}_{checkpoint_tag}".replace(" ", "_")
    per_path = os.path.join(out_dir, f"{base}_per_sample.csv")

    rows: List[Dict[str, Any]] = []
    dataset_index = 0

    model.eval()
    for batch in tqdm_auto(data_loader, params, desc=f"routing[{split}]", mininterval=2):
        if len(batch) == 3:
            x, y, keys = batch
        else:
            raise RuntimeError("routing export requires return_sample_keys=True and keys in batch")

        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        prob = F.softmax(pred, dim=-1)
        conf, pred_cls = prob.max(dim=-1)

        rc = getattr(moe, "_routing_export_cache", None)
        if rc is None:
            raise RuntimeError("MoE did not set _routing_export_cache (forward typed MoE layer?)")

        bsz = x.size(0)
        psd_on = bool(getattr(params, "moe_use_psd_router_features", False))

        for i in range(bsz):
            key = keys[i]
            meta = join_meta_for_key(key, rec_map)
            y_i = int(y[i].item())
            p_i = int(pred_cls[i].item())
            sp_probs = [float(rc["probs_spatial"][i, e].item()) for e in range(num_e)]
            sc_probs = [float(rc["probs_spectral"][i, e].item()) for e in range(num_e)]
            cohort = str(meta.get("cohort", "") or "")
            row: Dict[str, Any] = {
                "split": split,
                "dataset_index": dataset_index,
                "epoch_tag": epoch_tag,
                "checkpoint_tag": checkpoint_tag,
                "lmdb_key": key,
                "subject_id": str(meta.get("sub_id", "") or ""),
                "true_label": y_i,
                "pred_label": p_i,
                "correct": int(y_i == p_i),
                "max_softmax_confidence": float(conf[i].item()),
                "psd_router_enabled": psd_on,
                "spatial_top1_expert": int(rc["expert_spatial"][i].item()),
                "spectral_top1_expert": int(rc["expert_spectral"][i].item()),
                "spatial_entropy": float(rc["entropy_spatial"][i].item()),
                "spectral_entropy": float(rc["entropy_spectral"][i].item()),
                "spatial_probs": ",".join(f"{p:.8g}" for p in sp_probs),
                "spectral_probs": ",".join(f"{p:.8g}" for p in sc_probs),
                "recording_cohort": cohort,
                "cohort": cohort,
                "sample_rate_group": str(meta.get("sample_rate_group", "") or ""),
                "segment_index": meta.get("segment_index", ""),
                "chunk_index": meta.get("chunk_index", ""),
                "age_bucket": str(meta.get("age_bucket", "") or ""),
            }
            for e in range(num_e):
                row[f"spatial_p{e}"] = sp_probs[e]
                row[f"spectral_p{e}"] = sc_probs[e]
            rows.append(row)
            dataset_index += 1

    if not rows:
        raise RuntimeError("routing export: no rows")

    analysis = write_all_analyses(rows, out_dir, base, num_e)
    print(f"[routing_export] analysis CSVs: {analysis}", flush=True)

    fieldnames = list(rows[0].keys())
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[routing_export] wrote {per_path}", flush=True)
    return per_path, base
