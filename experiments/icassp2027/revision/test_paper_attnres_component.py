"""Contract test for the locked Full FT + AttnRes recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_attnres_component_schema import load_recipe, verify_args


def main() -> None:
    path = Path(__file__).with_name("paper_method_attnres_only_v1.json")
    recipe = load_recipe(path)
    args = argparse.Namespace(**recipe["method"])
    result = verify_args(args, recipe)
    assert result["paper_component_recipe_id"] == "icassp2027_full_attnres_only_v1"
    print("paper AttnRes component recipe: PASS")


if __name__ == "__main__":
    main()
