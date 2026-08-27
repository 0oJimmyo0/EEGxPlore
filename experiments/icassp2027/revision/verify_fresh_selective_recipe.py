"""Verify the locked, fresh ICASSP selective-adaptation recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict


REQUIRED_METADATA = ("recipe_id", "condition", "foundation_checkpoint_sha256", "parameters")

PARAMETER_FIELDS = (
    "experiment_profile",
    "revision_condition",
    "backbone",
    "revision_protocol",
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
    "selected_checkpoint_diagnostics",
    "icassp_routing_diagnostic",
    "use_pretrained_weights",
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
    "moe_domain_bias_reg",
    "moe_router_arch",
    "moe_router_mlp_hidden",
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
    "moe_use_psd_router_features",
    "moe_use_attnres_depth_router_features",
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
    "moe_expert_init_noise_std",
    "use_component_lr",
    "lr_backbone_mult",
    "lr_router_mult",
    "lr_expert_mult",
    "lr_classifier_mult",
    "lr_other_mult",
    "lr_depth_mult",
)


def recipe_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actual_values(args: Any) -> Dict[str, Any]:
    return {name: getattr(args, name, None) for name in PARAMETER_FIELDS}


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


def verify_recipe(recipe_path: Path, args: Any) -> Dict[str, str]:
    canonical_path = Path(__file__).with_name("fresh_selective_recipe.json").resolve()
    if recipe_path.resolve() != canonical_path:
        raise ValueError(
            "selective_fresh must use the repository-locked recipe: "
            f"{canonical_path}"
        )
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read fresh selective recipe {recipe_path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("fresh selective recipe must contain a JSON object")

    missing_metadata = [key for key in REQUIRED_METADATA if recipe.get(key) in (None, "", [])]
    if missing_metadata:
        raise ValueError("fresh selective recipe is incomplete: " + ", ".join(missing_metadata))
    if recipe.get("status") != "locked":
        raise ValueError(
            f"fresh selective recipe {recipe.get('recipe_id')!r} is not locked "
            f"(status={recipe.get('status')!r})"
        )
    if recipe.get("condition") != "selective_fresh":
        raise ValueError("fresh selective recipe must target condition 'selective_fresh'")
    if str(getattr(args, "revision_condition", "")) != "selective_fresh":
        raise ValueError(
            "fresh selective recipe can only validate revision_condition='selective_fresh'"
        )

    parameters = recipe.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("fresh selective recipe must contain a non-empty parameters object")
    missing_parameters = sorted(set(PARAMETER_FIELDS) - set(parameters))
    if missing_parameters:
        raise ValueError(
            "fresh selective recipe must lock every supported parameter field: "
            + ", ".join(missing_parameters)
        )
    unknown = sorted(set(parameters) - set(PARAMETER_FIELDS))
    if unknown:
        raise ValueError("recipe contains unsupported argument fields: " + ", ".join(unknown))
    unresolved = sorted(key for key, expected in parameters.items() if expected is None)
    if unresolved:
        raise ValueError("recipe contains unresolved parameters: " + ", ".join(unresolved))
    actual = _actual_values(args)
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in parameters.items()
        if not _equal(expected, actual[key])
    }
    if mismatches:
        raise ValueError(
            "resolved arguments do not match fresh selective recipe: "
            + json.dumps(mismatches, sort_keys=True)
        )

    foundation_path = Path(str(getattr(args, "foundation_dir", "") or ""))
    if not foundation_path.is_file():
        raise ValueError(f"foundation checkpoint is unavailable: {foundation_path}")
    expected_hash = str(recipe["foundation_checkpoint_sha256"])
    actual_hash = recipe_sha256(foundation_path)
    if expected_hash != actual_hash:
        raise ValueError(
            "foundation checkpoint hash mismatch: "
            f"expected {expected_hash}, found {actual_hash}"
        )

    return {
        "fresh_selective_recipe_id": str(recipe["recipe_id"]),
        "fresh_selective_recipe_sha256": recipe_sha256(recipe_path),
        "fresh_selective_recipe_status": str(recipe["status"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    print(json.dumps(verify_recipe(args.recipe, argparse.Namespace(**config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
