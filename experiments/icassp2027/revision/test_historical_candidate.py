"""CPU contract tests for the isolated historical candidate recipes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_main import add_faced_args, add_seedv_args, add_shared_args, validate_args
from historical_candidate_schema import load_recipe, verify_args_against_recipe


def _build_args(recipe_path: Path, smoke: bool = False):
    recipe = load_recipe(recipe_path)
    execution = recipe["execution"]
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_faced_args(parser)
    add_seedv_args(parser)
    argv = [
        "--datasets_dir", "/tmp/icassp_historical_candidate_dataset",
        "--num_of_classes", "9" if recipe["dataset"] == "FACED" else "5",
        "--model_dir", str(REPO_ROOT / "output/icassp2027_historical_candidate/contract_test"),
        "--downstream_dataset", recipe["dataset"],
        "--experiment_profile", "icassp2027_revision",
        "--revision_condition", "historical_candidate",
        "--revision_protocol", "cbramod_benchmark",
        "--revision_run_mode", "internal",
        "--historical_candidate_recipe", str(recipe_path),
        "--historical_recipe_path", str(recipe_path),
        "--epochs", "1" if smoke else str(execution["epochs"]),
        "--batch_size", str(execution["batch_size"]),
        "--lr", str(execution["lr"]),
        "--min_lr", str(execution["min_lr"]),
        "--warmup_epochs", str(execution["warmup_epochs"]),
        "--warmup_start_factor", str(execution["warmup_start_factor"]),
        "--weight_decay", str(execution["weight_decay"]),
        "--optimizer", execution["optimizer"],
        "--clip_value", str(execution["clip_value"]),
        "--dropout", str(execution["dropout"]),
        "--label_smoothing", str(execution["label_smoothing"]),
        "--ema_decay", str(execution["ema_decay"]),
        "--ema_warmup_steps", str(execution["ema_warmup_steps"]),
        "--classifier", execution["classifier"],
        "--input_scale_divisor", str(execution["input_scale_divisor"]),
        "--selection_metric", execution["selection_metric"],
        "--num_workers", str(execution["num_workers"]),
        "--class_weight_mode", execution["class_weight_mode"],
        "--class_weight_clip_min", str(execution["class_weight_clip_min"]),
        "--class_weight_clip_max", str(execution["class_weight_clip_max"]),
        "--effective_num_beta", str(execution["effective_num_beta"]),
        "--use_pretrained_weights",
    ]
    if execution["use_ema"]:
        argv.append("--use_ema")
    if execution["pin_memory"]:
        argv.append("--pin_memory")
    if execution["persistent_workers"]:
        argv.append("--persistent_workers")
    if execution["train_drop_last"]:
        argv.append("--train_drop_last")
    if execution["multi_lr"]:
        argv.append("--multi_lr")
    return parser.parse_args(argv), recipe


def main() -> None:
    for recipe_path in sorted(Path(__file__).parent.glob("historical_candidate_*_v1.json")):
        args, recipe = _build_args(recipe_path)
        validate_args(args)
        verify_args_against_recipe(args, recipe)

        smoke_args, smoke_recipe = _build_args(recipe_path, smoke=True)
        validate_args(smoke_args)
        verify_args_against_recipe(smoke_args, smoke_recipe, smoke=True)
        print(f"{recipe_path.name}: PASS")
    print("historical candidate contract: PASS")


if __name__ == "__main__":
    main()
