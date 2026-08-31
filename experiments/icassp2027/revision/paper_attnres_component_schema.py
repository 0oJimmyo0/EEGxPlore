"""Validation helpers for the Full FT + AttnRes component-control recipe."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Dict


CONDITION = "full_attnres_only"
DATASETS = {"FACED", "ISRUC", "SEED-V", "PhysioNet-MI"}
REQUIRED_FIELDS = {
    "trainability_mode",
    "attnres_variant",
    "attnres_start_layer",
    "attnres_gated",
    "moe",
    "use_component_lr",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantics_sha256(recipe: Dict[str, Any]) -> str:
    payload = json.dumps(recipe["method"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    return expected == actual


def load_recipe(path: Path) -> Dict[str, Any]:
    canonical = Path(__file__).with_name("paper_method_attnres_only_v1.json").resolve()
    if path.resolve() != canonical:
        raise ValueError(f"full_attnres_only must use the repository-locked recipe: {canonical}")
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read AttnRes component recipe {path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError("AttnRes component recipe must contain a JSON object")
    if recipe.get("schema_version") != 1 or recipe.get("status") != "locked":
        raise ValueError("AttnRes component recipe must be locked schema_version=1")
    if recipe.get("paper_eligible") is not True or recipe.get("condition") != CONDITION:
        raise ValueError("AttnRes component recipe has an invalid condition or eligibility")
    if set(recipe.get("datasets", [])) != DATASETS:
        raise ValueError(f"AttnRes component recipe datasets must be {sorted(DATASETS)}")
    method = recipe.get("method")
    if not isinstance(method, dict):
        raise ValueError("AttnRes component recipe must contain a method object")
    if set(method) != REQUIRED_FIELDS:
        raise ValueError(
            "AttnRes component recipe fields must be exactly "
            + ", ".join(sorted(REQUIRED_FIELDS))
        )
    expected = {
        "trainability_mode": "full",
        "attnres_variant": "pre_attn",
        "attnres_start_layer": 0,
        "attnres_gated": False,
        "moe": False,
        "use_component_lr": False,
    }
    mismatches = {
        key: {"expected": value, "actual": method.get(key)}
        for key, value in expected.items()
        if not _equal(value, method.get(key))
    }
    if mismatches:
        raise ValueError("AttnRes component recipe mismatch: " + json.dumps(mismatches, sort_keys=True))
    return recipe


def verify_args(args: Any, recipe: Dict[str, Any]) -> Dict[str, Any]:
    mismatches = {
        key: {"expected": expected, "actual": getattr(args, key, None)}
        for key, expected in recipe["method"].items()
        if not _equal(expected, getattr(args, key, None))
    }
    if mismatches:
        raise ValueError("resolved arguments do not match AttnRes component recipe: " + json.dumps(mismatches, sort_keys=True))
    return {
        "paper_component_recipe_id": str(recipe["method_id"]),
        "paper_component_label": str(recipe["paper_label"]),
        "paper_component_semantics_sha256": semantics_sha256(recipe),
    }


def shell_exports(path: Path, recipe: Dict[str, Any]) -> str:
    values = {
        "PAPER_COMPONENT_RECIPE_ID": recipe["method_id"],
        "PAPER_COMPONENT_RECIPE_SHA256": sha256_file(path),
        "PAPER_COMPONENT_SEMANTICS_SHA256": semantics_sha256(recipe),
        "PAPER_COMPONENT_RECIPE_PATH": str(path.resolve()),
    }
    return "\n".join(
        f"export {key}={shlex.quote(str(value))}" for key, value in values.items()
    )
