"""Contract test ensuring an unfinished historical recipe cannot be used."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from verify_historical_recipe import verify_recipe


RECIPE = Path(__file__).with_name("historical_recipe_1785556.json")


def main() -> None:
    args = SimpleNamespace(seed=42, foundation_dir="/does/not/exist", historical_family_id="1785556")
    try:
        verify_recipe(RECIPE, args)
    except ValueError as exc:
        message = str(exc)
        assert "historical recipe is incomplete" in message or "not locked" in message
    else:
        raise AssertionError("unfinished historical recipe was accepted")
    print("historical recipe guard contract: PASS")


if __name__ == "__main__":
    main()
