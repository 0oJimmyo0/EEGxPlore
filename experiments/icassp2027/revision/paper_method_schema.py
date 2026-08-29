"""Schema and validation helpers for the final ICASSP method recipe.

The final paper method is deliberately separate from the development-only
historical candidates.  It captures the architecture and trainability choice;
dataset-specific optimization settings remain in the locked paper protocols.
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Dict

from experiments.icassp2027.revision.historical_candidate_schema import REQUIRED_METHOD_FIELDS


METHOD_CONDITION = "specialist_augmented_full"
METHOD_DATASETS = {"FACED", "ISRUC"}


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


def load_method_recipe(path: Path) -> Dict[str, Any]:
    canonical_path = Path(__file__).with_name(
        "paper_method_specialist_augmented_full_v1.json"
    ).resolve()
    if path.resolve() != canonical_path:
        raise ValueError(
            "specialist_augmented_full must use the repository-locked method recipe: "
            f"{canonical_path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read paper method recipe {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("paper method recipe must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("paper method recipe schema_version must be 1")
    if payload.get("status") != "locked":
        raise ValueError("paper method recipe must have status='locked'")
    if payload.get("paper_eligible") is not True:
        raise ValueError("paper method recipe must set paper_eligible=true")
    if payload.get("condition") != METHOD_CONDITION:
        raise ValueError(
            f"paper method recipe must target condition={METHOD_CONDITION!r}"
        )
    if payload.get("method_id") in (None, "") or payload.get("paper_label") in (None, ""):
        raise ValueError("paper method recipe must define method_id and paper_label")
    datasets = payload.get("datasets")
    if set(datasets or []) != METHOD_DATASETS:
        raise ValueError(f"paper method recipe datasets must be {sorted(METHOD_DATASETS)}")

    method = payload.get("method")
    if not isinstance(method, dict):
        raise ValueError("paper method recipe must contain a method object")
    missing = sorted(set(REQUIRED_METHOD_FIELDS) - set(method))
    unknown = sorted(set(method) - set(REQUIRED_METHOD_FIELDS))
    if missing:
        raise ValueError("paper method recipe is incomplete: " + ", ".join(missing))
    if unknown:
        raise ValueError("paper method recipe has unsupported fields: " + ", ".join(unknown))
    unresolved = sorted(key for key, value in method.items() if value is None)
    if unresolved:
        raise ValueError("paper method recipe contains unresolved fields: " + ", ".join(unresolved))

    required_values = {
        "trainability_mode": "full",
        "attnres_variant": "pre_attn",
        "attnres_start_layer": 0,
        "attnres_gated": False,
        "moe": True,
        "moe_num_layers": 1,
        "moe_num_experts": 4,
        "moe_route_mode": "typed_capacity_domain",
        "moe_use_psd_router_features": False,
        "moe_use_attnres_depth_router_features": False,
        "moe_attnres_depth_context_mode": "compact_shared",
        "moe_router_compact_feature_mode": "none",
        "moe_domain_bias": False,
        "moe_specialist_branch_mode": "both",
        "use_component_lr": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": method.get(key)}
        for key, expected in required_values.items()
        if not _equal(expected, method.get(key))
    }
    if mismatches:
        raise ValueError(
            "paper method recipe violates the locked specialist-augmented contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload


def apply_method_recipe(args: Any, recipe: Dict[str, Any], path: Path) -> None:
    method = recipe["method"]
    missing = [key for key in method if not hasattr(args, key)]
    if missing:
        raise ValueError("paper method fields are not supported by this checkout: " + ", ".join(missing))
    for key, value in method.items():
        setattr(args, key, value)
    args.paper_method_recipe = str(path)


def verify_args_against_method(args: Any, recipe: Dict[str, Any]) -> Dict[str, Any]:
    method = recipe["method"]
    mismatches = {
        key: {"expected": expected, "actual": getattr(args, key, None)}
        for key, expected in method.items()
        if not _equal(expected, getattr(args, key, None))
    }
    if mismatches:
        raise ValueError(
            "resolved arguments do not match paper method recipe: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "paper_method_id": str(recipe["method_id"]),
        "paper_method_label": str(recipe["paper_label"]),
        "paper_method_status": str(recipe["status"]),
    }


def shell_exports(path: Path, recipe: Dict[str, Any]) -> str:
    values = {
        "PAPER_METHOD_RECIPE_ID": recipe["method_id"],
        "PAPER_METHOD_RECIPE_SHA256": sha256_file(path),
        "PAPER_METHOD_RECIPE_PATH": str(path.resolve()),
    }
    return "\n".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items()
    )
