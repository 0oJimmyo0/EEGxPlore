"""Verify a resolved training namespace against a locked historical recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping


REQUIRED_METADATA = ("family_id", "source_code_commit", "seeds", "foundation_checkpoint_sha256")


def recipe_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actual_values(args: Any) -> Dict[str, Any]:
    names = (
        "downstream_dataset", "revision_protocol", "num_of_classes", "epochs", "batch_size",
        "optimizer", "lr", "min_lr", "weight_decay", "warmup_epochs", "clip_value",
        "dropout", "label_smoothing", "use_ema", "ema_decay", "ema_warmup_steps",
        "ema_eval_only", "classifier", "input_scale_divisor", "selection_metric",
        "attnres_variant", "attnres_start_layer", "attnres_gated", "moe", "moe_num_layers",
        "moe_num_experts", "moe_route_mode", "moe_router_policy", "moe_capacity_factor",
        "moe_load_balance", "moe_domain_bias", "moe_domain_bias_reg", "moe_router_arch",
        "moe_router_mlp_hidden", "moe_router_dispatch_mode", "moe_router_temperature",
        "moe_router_entropy_coef", "moe_router_balance_kl_coef", "moe_router_z_loss_coef",
        "moe_router_jitter_std", "moe_router_jitter_final_std", "moe_router_jitter_anneal_epochs",
        "moe_router_soft_warmup_epochs", "moe_uniform_dispatch_warmup_epochs",
        "moe_shared_blend_warmup_epochs", "moe_shared_blend_start", "moe_shared_blend_end",
        "moe_shared_output_scale", "moe_expert_output_scale", "moe_router_base_feature_mode",
        "moe_router_compact_feature_mode", "moe_router_compact_feature_dim",
        "moe_specialist_branch_mode", "moe_use_psd_router_features",
        "moe_use_attnres_depth_router_features", "moe_attnres_depth_context_mode",
        "moe_attnres_depth_block_count", "moe_attnres_depth_summary_mode",
        "moe_attnres_depth_probe_mlp_for_router", "trainability_mode",
    )
    return {name: getattr(args, name, None) for name in names}


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


def verify_recipe(recipe_path: Path, args: Any) -> Dict[str, str]:
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read historical recipe {recipe_path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("historical recipe must contain a JSON object")

    if recipe.get("status") != "locked":
        raise ValueError(
            f"historical recipe {recipe.get('family_id')!r} is not locked "
            f"(status={recipe.get('status')!r})"
        )
    requested_family_id = str(getattr(args, "historical_family_id", "") or "")
    if requested_family_id and requested_family_id != str(recipe.get("family_id", "")):
        raise ValueError(
            "historical family id mismatch: "
            f"requested {requested_family_id}, recipe contains {recipe.get('family_id')!r}"
        )
    missing_metadata = [key for key in REQUIRED_METADATA if recipe.get(key) in (None, "", [])]
    if missing_metadata:
        raise ValueError(
            "historical recipe is incomplete; missing metadata: " + ", ".join(missing_metadata)
        )

    seeds = recipe.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("historical recipe seeds must be a non-empty integer list")
    actual_seed = int(getattr(args, "seed", -1))
    if actual_seed not in seeds:
        raise ValueError(f"seed {actual_seed} is not in the audited historical seed list {seeds}")

    parameters = recipe.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("historical recipe must contain a non-empty parameters object")
    actual = _actual_values(args)
    unknown = sorted(set(parameters) - set(actual))
    if unknown:
        raise ValueError("historical recipe contains unsupported argument fields: " + ", ".join(unknown))
    unresolved = sorted(key for key, expected in parameters.items() if expected is None)
    if unresolved:
        raise ValueError("historical recipe still has unresolved parameters: " + ", ".join(unresolved))

    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in parameters.items()
        if not _equal(expected, actual[key])
    }
    if mismatches:
        raise ValueError("resolved arguments do not match historical recipe: " + json.dumps(mismatches, sort_keys=True))

    foundation_hash = str(recipe.get("foundation_checkpoint_sha256"))
    foundation_path = str(getattr(args, "foundation_dir", "") or "")
    if not Path(foundation_path).is_file():
        raise ValueError(f"historical foundation checkpoint is unavailable: {foundation_path}")
    actual_foundation_hash = recipe_sha256(Path(foundation_path))
    if foundation_hash != actual_foundation_hash:
        raise ValueError(
            "historical foundation checkpoint hash mismatch: "
            f"expected {foundation_hash}, found {actual_foundation_hash}"
        )

    return {
        "historical_family_id": str(recipe["family_id"]),
        "historical_recipe_sha256": recipe_sha256(recipe_path),
        "historical_recipe_status": str(recipe["status"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--config-json",
        type=Path,
        required=True,
        help="JSON object containing the resolved arguments, including seed and foundation_dir.",
    )
    args = parser.parse_args()
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    result = verify_recipe(args.recipe, SimpleNamespace(**config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
