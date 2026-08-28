"""Shared schema helpers for the development-only historical candidates.

The candidate recipes deliberately live outside the locked paper protocol.  They
are useful for diagnosing whether the rejected-paper FACED/ISRUC headline
configuration can be recovered, but they must never become paper evidence by
accident.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict


REQUIRED_EXECUTION_FIELDS = (
    "epochs",
    "batch_size",
    "lr",
    "min_lr",
    "warmup_epochs",
    "warmup_start_factor",
    "weight_decay",
    "optimizer",
    "clip_value",
    "dropout",
    "label_smoothing",
    "use_ema",
    "ema_decay",
    "ema_warmup_steps",
    "ema_eval_only",
    "classifier",
    "input_scale_divisor",
    "selection_metric",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "train_drop_last",
    "multi_lr",
    "class_weight_mode",
    "class_weight_clip_min",
    "class_weight_clip_max",
    "effective_num_beta",
    "use_pretrained_weights",
)


REQUIRED_METHOD_FIELDS = (
    "trainability_mode",
    "attnres_variant",
    "attnres_start_layer",
    "attnres_gated",
    "moe",
    "moe_num_layers",
    "moe_num_experts",
    "moe_route_mode",
    "moe_router_policy",
    "moe_capacity_factor",
    "moe_load_balance",
    "moe_domain_bias",
    "moe_domain_emb_dim",
    "moe_domain_bias_reg",
    "moe_router_arch",
    "moe_router_mlp_hidden",
    "moe_use_psd_router_features",
    "moe_use_attnres_depth_router_features",
    "moe_attnres_depth_router_dim",
    "moe_attnres_depth_context_mode",
    "moe_attnres_depth_block_count",
    "moe_attnres_depth_summary_mode",
    "moe_attnres_depth_probe_mlp_for_router",
    "moe_attnres_depth_router_init",
    "moe_attnres_depth_router_norm_gate",
    "moe_attnres_depth_router_gate_init",
    "moe_attnres_depth_router_norm_eps",
    "moe_attnres_depth_block_separation_coef",
    "moe_attnres_depth_block_separation_target_js",
    "moe_attnres_depth_summary_grad_mode",
    "moe_attnres_depth_summary_unfreeze_epoch",
    "moe_router_dispatch_mode",
    "moe_router_temperature",
    "moe_router_entropy_coef",
    "moe_router_balance_kl_coef",
    "moe_router_z_loss_coef",
    "moe_router_jitter_std",
    "moe_router_jitter_final_std",
    "moe_router_jitter_anneal_epochs",
    "moe_router_soft_warmup_epochs",
    "moe_uniform_dispatch_warmup_epochs",
    "moe_shared_blend_warmup_epochs",
    "moe_shared_blend_start",
    "moe_shared_blend_end",
    "moe_shared_output_scale",
    "moe_expert_output_scale",
    "moe_router_base_feature_mode",
    "moe_router_entropy_coef_spatial",
    "moe_router_entropy_coef_spectral",
    "moe_router_balance_kl_coef_spatial",
    "moe_router_balance_kl_coef_spectral",
    "moe_specialist_branch_mode",
    "moe_router_compact_feature_mode",
    "moe_router_compact_feature_dim",
    "moe_router_compact_warmup_epochs",
    "moe_router_compact_gate_init",
    "moe_expert_init_noise_std",
    "use_component_lr",
    "lr_backbone_mult",
    "lr_router_mult",
    "lr_expert_mult",
    "lr_classifier_mult",
    "lr_other_mult",
    "lr_depth_mult",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


def load_recipe(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read historical candidate recipe {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("historical candidate recipe must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("historical candidate recipe schema_version must be 1")
    if payload.get("status") != "development_only":
        raise ValueError("historical candidate recipe must have status='development_only'")
    if payload.get("paper_eligible") is not False:
        raise ValueError("historical candidate recipe must set paper_eligible=false")
    if payload.get("condition") != "historical_candidate":
        raise ValueError("historical candidate recipe must target condition='historical_candidate'")
    if payload.get("dataset") not in {"FACED", "ISRUC"}:
        raise ValueError("historical candidate recipes are currently limited to FACED and ISRUC")
    if payload.get("stage") not in {"opt", "route"}:
        raise ValueError("historical candidate recipe stage must be 'opt' or 'route'")
    for key in ("recipe_id", "historical_family_id", "trainability_basis"):
        if payload.get(key) in (None, ""):
            raise ValueError(f"historical candidate recipe is missing {key}")

    execution = payload.get("execution")
    method = payload.get("method")
    if not isinstance(execution, dict) or not isinstance(method, dict):
        raise ValueError("historical candidate recipe must contain execution and method objects")
    missing_execution = sorted(set(REQUIRED_EXECUTION_FIELDS) - set(execution))
    missing_method = sorted(set(REQUIRED_METHOD_FIELDS) - set(method))
    if missing_execution:
        raise ValueError("historical candidate execution is incomplete: " + ", ".join(missing_execution))
    if missing_method:
        raise ValueError("historical candidate method is incomplete: " + ", ".join(missing_method))
    unknown_execution = sorted(set(execution) - set(REQUIRED_EXECUTION_FIELDS))
    unknown_method = sorted(set(method) - set(REQUIRED_METHOD_FIELDS))
    if unknown_execution:
        raise ValueError("historical candidate has unsupported execution fields: " + ", ".join(unknown_execution))
    if unknown_method:
        raise ValueError("historical candidate has unsupported method fields: " + ", ".join(unknown_method))
    unresolved = [
        f"execution.{key}" for key, value in execution.items() if value is None
    ] + [f"method.{key}" for key, value in method.items() if value is None]
    if unresolved:
        raise ValueError("historical candidate contains unresolved fields: " + ", ".join(unresolved))

    if int(execution["epochs"]) < 1 or int(execution["batch_size"]) < 1:
        raise ValueError("candidate epochs and batch_size must be positive")
    if float(execution["lr"]) <= 0 or float(execution["min_lr"]) <= 0:
        raise ValueError("candidate learning rates must be positive")
    if execution["selection_metric"] != "kappa":
        raise ValueError("candidate selection_metric must be kappa")
    if float(execution["input_scale_divisor"]) != 100.0:
        raise ValueError("candidate input_scale_divisor must be 100.0")
    if method["trainability_mode"] not in {"full", "combined"}:
        raise ValueError("candidate trainability_mode must be full or combined")
    if method["moe"] is not True or method["moe_num_layers"] != 1:
        raise ValueError("candidate recipes require one MoE layer")
    if method["attnres_variant"] != "pre_attn":
        raise ValueError("candidate recipes require attnres_variant='pre_attn'")
    if payload["stage"] == "route":
        if method["moe_use_attnres_depth_router_features"] is not True:
            raise ValueError("route candidate must enable the depth router")
        expected_context = {
            "FACED": "block_shared_typed_proj",
            "ISRUC": "dual_query_block_typed_proj",
        }[payload["dataset"]]
        if method["moe_attnres_depth_context_mode"] != expected_context:
            raise ValueError(
                f"{payload['dataset']} route candidate must use {expected_context}"
            )
    return payload


def apply_method_recipe(args: Any, recipe: Dict[str, Any], path: Path) -> None:
    """Apply the locked method portion before the normal argument validator."""
    method = recipe["method"]
    missing = [key for key in method if not hasattr(args, key)]
    if missing:
        raise ValueError("candidate method fields are not supported by this checkout: " + ", ".join(missing))
    for key, value in method.items():
        setattr(args, key, value)
    args.historical_recipe_path = str(path)
    args.historical_candidate_recipe = str(path)
    args.historical_family_id = str(recipe["historical_family_id"])


def verify_args_against_recipe(args: Any, recipe: Dict[str, Any], *, smoke: bool = False) -> Dict[str, Any]:
    execution = recipe["execution"]
    method = recipe["method"]
    mismatches = {}
    for key, expected in {**execution, **method}.items():
        actual = getattr(args, key, None)
        if smoke and key == "epochs":
            if int(actual) != 1:
                mismatches[key] = {"expected": 1, "actual": actual}
            continue
        if not _equal(expected, actual):
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(
            "resolved arguments do not match historical candidate recipe: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "historical_candidate_recipe_id": str(recipe["recipe_id"]),
        "historical_candidate_recipe_sha256": "",
        "historical_candidate_recipe_status": str(recipe["status"]),
        "historical_candidate_stage": str(recipe["stage"]),
        "historical_family_id": str(recipe["historical_family_id"]),
        "trainability_basis": str(recipe["trainability_basis"]),
    }
