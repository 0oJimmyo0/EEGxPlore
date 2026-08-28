"""Static contract test for the locked ARCHIVED_INDEPENDENT recipe."""

from __future__ import annotations

import json
from pathlib import Path

from verify_fresh_selective_recipe import PARAMETER_FIELDS


RECIPE = Path(__file__).with_name("fresh_selective_recipe.json")


def main() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    assert recipe["status"] == "locked"
    assert recipe["condition"] == "selective_fresh"
    parameters = recipe["parameters"]
    assert set(parameters) == set(PARAMETER_FIELDS)
    assert all(value is not None for value in parameters.values())
    assert recipe["foundation_checkpoint_sha256"]
    print("fresh selective recipe contract: PASS")


if __name__ == "__main__":
    main()
