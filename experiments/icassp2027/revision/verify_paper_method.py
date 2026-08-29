"""Verify the locked ICASSP paper-facing method recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_method_schema import (
    load_method_recipe,
    sha256_file,
    shell_exports,
    verify_args_against_method,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--emit-shell", action="store_true")
    args = parser.parse_args()

    path = args.recipe.resolve()
    recipe = load_method_recipe(path)
    if args.config_json:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))
        info = verify_args_against_method(argparse.Namespace(**config), recipe)
        info["paper_method_recipe_sha256"] = sha256_file(path)
        print(json.dumps(info, indent=2, sort_keys=True))
    elif args.emit_shell:
        print(shell_exports(path, recipe))
    else:
        print(json.dumps(recipe, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
