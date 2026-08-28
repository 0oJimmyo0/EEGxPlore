"""Verify and expose a locked dataset-specific ICASSP protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Dict


REQUIRED_PARAMETERS = (
    "epochs",
    "batch_size",
    "lr",
    "min_lr",
    "weight_decay",
    "dropout",
    "label_smoothing",
    "use_ema",
    "ema_decay",
    "ema_warmup_steps",
    "ema_eval_only",
    "classifier",
    "input_scale_divisor",
    "selection_metric",
    "num_workers",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol(path: Path, dataset: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read paper protocol {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("paper protocol must contain a JSON object")
    if payload.get("status") != "locked":
        raise ValueError("paper protocol must have status='locked'")
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"paper protocol dataset mismatch: expected {dataset!r}, got {payload.get('dataset')!r}"
        )
    if payload.get("revision_protocol") != "cbramod_benchmark":
        raise ValueError("paper protocol must target cbramod_benchmark")
    if payload.get("provenance_class") != "paper_derived":
        raise ValueError("paper protocol must have provenance_class='paper_derived'")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("paper protocol must contain a parameters object")
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters]
    if missing:
        raise ValueError("paper protocol is incomplete: " + ", ".join(missing))
    if set(parameters) != set(REQUIRED_PARAMETERS):
        extra = sorted(set(parameters) - set(REQUIRED_PARAMETERS))
        raise ValueError("paper protocol contains unsupported parameters: " + ", ".join(extra))
    sources = payload.get("parameter_sources")
    if not isinstance(sources, dict) or set(sources) != set(REQUIRED_PARAMETERS):
        raise ValueError("paper protocol must provide a source for every execution parameter")
    if int(parameters["epochs"]) < 1 or int(parameters["batch_size"]) < 1:
        raise ValueError("paper protocol epochs and batch_size must be positive")
    if float(parameters["lr"]) <= 0 or float(parameters["min_lr"]) <= 0:
        raise ValueError("paper protocol learning rates must be positive")
    if float(parameters["min_lr"]) > float(parameters["lr"]):
        raise ValueError("paper protocol min_lr cannot exceed lr")
    if parameters["selection_metric"] != "kappa":
        raise ValueError("paper protocol selection_metric must be kappa")
    if float(parameters["input_scale_divisor"]) != 100.0:
        raise ValueError("paper protocol input_scale_divisor must be 100.0")
    return {
        "protocol_id": str(payload["protocol_id"]),
        "dataset": str(payload["dataset"]),
        "revision_protocol": str(payload["revision_protocol"]),
        "provenance_class": str(payload["provenance_class"]),
        "source_location": str(payload.get("source_location", "")),
        "parameters": parameters,
        "sha256": sha256_file(path),
        "path": str(path.resolve()),
    }


def shell_exports(info: Dict[str, Any]) -> str:
    parameters = info["parameters"]
    values = {
        "PAPER_PROTOCOL_ID": info["protocol_id"],
        "PAPER_PROTOCOL_SHA256": info["sha256"],
        "PAPER_PROTOCOL_PATH": info["path"],
        "PAPER_PROTOCOL_EPOCHS": parameters["epochs"],
        "PAPER_PROTOCOL_BATCH_SIZE": parameters["batch_size"],
        "PAPER_PROTOCOL_LR": parameters["lr"],
        "PAPER_PROTOCOL_MIN_LR": parameters["min_lr"],
        "PAPER_PROTOCOL_WEIGHT_DECAY": parameters["weight_decay"],
        "PAPER_PROTOCOL_DROPOUT": parameters["dropout"],
        "PAPER_PROTOCOL_LABEL_SMOOTHING": parameters["label_smoothing"],
        "PAPER_PROTOCOL_USE_EMA": "1" if parameters["use_ema"] else "0",
        "PAPER_PROTOCOL_EMA_DECAY": parameters["ema_decay"],
        "PAPER_PROTOCOL_EMA_WARMUP_STEPS": parameters["ema_warmup_steps"],
        "PAPER_PROTOCOL_EMA_EVAL_ONLY": "1" if parameters["ema_eval_only"] else "0",
        "PAPER_PROTOCOL_CLASSIFIER": parameters["classifier"],
        "PAPER_PROTOCOL_NUM_WORKERS": parameters["num_workers"],
    }
    return "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def validate_args_against_protocol(args: Any, info: Dict[str, Any]) -> None:
    """Reject a resolved paper/smoke command that drifts from its protocol."""
    parameters = info["parameters"]
    numeric_fields = (
        ("epochs", int),
        ("batch_size", int),
        ("lr", float),
        ("min_lr", float),
        ("weight_decay", float),
        ("dropout", float),
        ("label_smoothing", float),
        ("ema_decay", float),
        ("ema_warmup_steps", int),
        ("num_workers", int),
    )
    for field, converter in numeric_fields:
        actual = converter(getattr(args, field))
        expected = converter(parameters[field])
        if field == "epochs" and getattr(args, "revision_run_mode", "paper") == "smoke":
            if actual != 1:
                raise ValueError(
                    f"smoke protocol drift for epochs: expected 1, got {actual}"
                )
            continue
        equal = actual == expected if converter is int else math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        )
        if not equal:
            raise ValueError(
                f"paper protocol drift for {field}: expected {expected}, got {actual}"
            )

    for field in ("use_ema", "ema_eval_only"):
        actual = bool(getattr(args, field))
        expected = bool(parameters[field])
        if actual != expected:
            raise ValueError(
                f"paper protocol drift for {field}: expected {expected}, got {actual}"
            )

    for field in ("classifier", "selection_metric"):
        actual = getattr(args, field)
        expected = parameters[field]
        if actual != expected:
            raise ValueError(
                f"paper protocol drift for {field}: expected {expected!r}, got {actual!r}"
            )

    actual_divisor = float(args.input_scale_divisor)
    expected_divisor = float(parameters["input_scale_divisor"])
    if not math.isclose(actual_divisor, expected_divisor, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "paper protocol drift for input_scale_divisor: "
            f"expected {expected_divisor}, got {actual_divisor}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=["SEED-V", "FACED", "ISRUC"])
    parser.add_argument("--emit-shell", action="store_true")
    args = parser.parse_args()
    info = verify_protocol(args.protocol, args.dataset)
    if args.emit_shell:
        print(shell_exports(info))
    else:
        print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
