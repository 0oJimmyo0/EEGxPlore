"""Verify a development-only historical candidate recipe."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from historical_candidate_schema import (
    load_recipe,
    sha256_file,
    verify_args_against_recipe,
)


def _shell_exports(recipe_path: Path, recipe: dict) -> str:
    execution = recipe["execution"]
    values = {
        "HISTORICAL_CANDIDATE_RECIPE_ID": recipe["recipe_id"],
        "HISTORICAL_CANDIDATE_RECIPE_SHA256": sha256_file(recipe_path),
        "HISTORICAL_CANDIDATE_RECIPE_PATH": str(recipe_path.resolve()),
        "HISTORICAL_CANDIDATE_DATASET": recipe["dataset"],
        "HISTORICAL_CANDIDATE_STAGE": recipe["stage"],
        "HISTORICAL_CANDIDATE_HISTORICAL_FAMILY_ID": recipe["historical_family_id"],
    }
    for key, value in execution.items():
        export_name = "HISTORICAL_CANDIDATE_" + key.upper()
        values[export_name] = value
    for key in (
        "use_component_lr",
        "lr_backbone_mult",
        "lr_router_mult",
        "lr_expert_mult",
        "lr_classifier_mult",
        "lr_other_mult",
        "lr_depth_mult",
    ):
        values["HISTORICAL_CANDIDATE_" + key.upper()] = recipe["method"][key]
    return "\n".join(f"{key}={shlex.quote(str(value).lower() if isinstance(value, bool) else str(value))}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset", choices=["FACED", "ISRUC"], required=True)
    parser.add_argument("--stage", choices=["opt", "route"], default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--emit-shell", action="store_true")
    args = parser.parse_args()

    recipe_path = args.recipe.resolve()
    recipe = load_recipe(recipe_path)
    if recipe["dataset"] != args.dataset:
        raise SystemExit(
            f"candidate dataset mismatch: recipe={recipe['dataset']}, requested={args.dataset}"
        )
    if args.stage and recipe["stage"] != args.stage:
        raise SystemExit(
            f"candidate stage mismatch: recipe={recipe['stage']}, requested={args.stage}"
        )
    if args.config_json:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
        info = verify_args_against_recipe(
            argparse.Namespace(**config), recipe, smoke=args.smoke
        )
        info["historical_candidate_recipe_sha256"] = sha256_file(recipe_path)
        print(json.dumps(info, indent=2, sort_keys=True))
    elif args.emit_shell:
        print(_shell_exports(recipe_path, recipe))
    else:
        print(json.dumps(recipe, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
