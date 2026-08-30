"""Contract tests for the locked dataset-specific paper protocols."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from verify_paper_protocol import validate_args_against_protocol, verify_protocol


ROOT = Path(__file__).parent


def main() -> None:
    expected = {
        "SEED-V": ("paper_protocol_seedv_v1.json", 25, 64, 3e-5, 3e-2, 0),
        "FACED": ("paper_protocol_faced_v1.json", 40, 32, 2e-4, 2e-2, 4),
        "ISRUC": ("paper_protocol_isruc_v1.json", 30, 16, 3e-5, 2e-2, 4),
        "PhysioNet-MI": ("paper_protocol_physionet_mi_v1.json", 30, 64, 3e-5, 2e-2, 0),
    }
    for dataset, (filename, epochs, batch, lr, weight_decay, num_workers) in expected.items():
        path = ROOT / filename
        info = verify_protocol(path, dataset)
        parameters = info["parameters"]
        assert parameters["epochs"] == epochs
        assert parameters["batch_size"] == batch
        assert parameters["lr"] == lr
        assert parameters["weight_decay"] == weight_decay
        assert parameters["num_workers"] == num_workers
        assert parameters["use_component_lr"] is False
        assert info["sha256"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "locked"
        assert set(payload["parameter_sources"]) == set(parameters)

        resolved = SimpleNamespace(**parameters)
        validate_args_against_protocol(resolved, info)
        resolved.epochs += 1
        try:
            validate_args_against_protocol(resolved, info)
        except ValueError as exc:
            assert "protocol drift" in str(exc)
        else:
            raise AssertionError("protocol drift was not rejected")

        smoke_parameters = dict(parameters)
        smoke_parameters["epochs"] = 1
        smoke = SimpleNamespace(**smoke_parameters, revision_run_mode="smoke")
        validate_args_against_protocol(smoke, info)
        smoke.epochs = 2
        try:
            validate_args_against_protocol(smoke, info)
        except ValueError as exc:
            assert "smoke protocol drift" in str(exc)
        else:
            raise AssertionError("smoke epoch drift was not rejected")
    print("paper protocol contract: PASS")


if __name__ == "__main__":
    main()
