"""Verify the locked Full FT + AttnRes component-control recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_attnres_component_schema import load_recipe, shell_exports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--emit-shell", action="store_true")
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    if args.emit_shell:
        print(shell_exports(args.recipe, recipe))
    else:
        print(f"verified {recipe['method_id']} ({recipe['paper_label']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
