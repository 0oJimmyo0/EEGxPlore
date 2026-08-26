"""Strict foundation-loading contract for the ICASSP wrapper models.

The dense wrappers must preserve the pretrained CBraMod parameter schema even
when shared argument defaults contain a typed depth-context mode.  This test
constructs each downstream wrapper with its real strict-loading path, runs a
finite forward pass, and checks that no inactive depth-block parameters leak
into the model.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_for_faced import Model as FACEDModel
from models.model_for_isruc import Model as ISRUCModel
from models.model_for_physio import Model as PhysioModel
from models.model_for_seedv import Model as SeedVModel
from finetune_trainer import configure_trainability, is_depth_aggregation_parameter


def build_params(
    checkpoint: Path,
    num_classes: int,
    *,
    attnres_variant: str = 'none',
    attnres_start_layer: int = 0,
    trainability_mode: str = 'auto',
) -> SimpleNamespace:
    # Intentionally retain the stale block context value while disabling its
    # feature source.  The model constructor must canonicalize this to the
    # compact/no-block schema for dense wrappers.
    return SimpleNamespace(
        backbone="cbramod",
        attnres_variant=attnres_variant,
        attnres_gated=False,
        attnres_gate_init=0.0,
        attnres_start_layer=attnres_start_layer,
        trainability_mode=trainability_mode,
        dropout=0.0,
        moe=False,
        use_pretrained_weights=True,
        foundation_dir=str(checkpoint),
        cuda=0,
        classifier="all_patch_reps",
        num_of_classes=num_classes,
        moe_attnres_depth_context_mode="block_shared_typed_proj",
        moe_use_attnres_depth_router_features=False,
    )


def run_case(name: str, model_cls, x: torch.Tensor, checkpoint: Path, num_classes: int) -> None:
    params = build_params(checkpoint, num_classes)
    model = model_cls(params)
    model.eval()

    depth_keys = [key for key in model.state_dict() if "depth_block_" in key]
    assert not depth_keys, f"{name}: inactive depth-block parameters leaked: {depth_keys[:4]}"

    with torch.inference_mode():
        output = model(x)
    assert output.shape[0] == x.shape[0] and output.shape[-1] == num_classes, (name, output.shape)
    assert torch.isfinite(output).all(), f"{name}: non-finite wrapper output"
    print(f"{name}: PASS (strict load, no depth-block keys, finite forward)")

    del model
    gc.collect()


def run_depth_case(checkpoint: Path) -> None:
    """Exercise the actual SEED-V DepthAgg wrapper with the real foundation checkpoint."""
    params = build_params(
        checkpoint,
        5,
        attnres_variant='pre_attn',
        attnres_start_layer=8,
        trainability_mode='depth_aggregation',
    )
    model = SeedVModel(params)
    model.eval()
    x = torch.zeros(1, 62, 1, 200)
    # Materialize the lazy classifier outside inference_mode; parameters first
    # created there cannot later be switched to requires_grad=True.
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5), output.shape
    assert torch.isfinite(output).all(), 'SEED-V DepthAgg: non-finite wrapper output'

    mode, named_trainable = configure_trainability(model, params)
    assert mode == 'depth_aggregation'
    depth_names = {
        name for name, _ in named_trainable if is_depth_aggregation_parameter(name)
    }
    assert len(depth_names) == 8, sorted(depth_names)
    assert sum(
        parameter.numel()
        for name, parameter in named_trainable
        if is_depth_aggregation_parameter(name)
    ) == 1600
    assert all(
        is_depth_aggregation_parameter(name) or name.startswith('classifier')
        for name, _ in named_trainable
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith('backbone.') and not is_depth_aggregation_parameter(name)
    )
    print('SEED-V DepthAgg: PASS (real checkpoint, 1600 depth params, finite forward)')
    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth"),
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.set_num_threads(1)
    torch.manual_seed(2027)
    run_case("SEED-V", SeedVModel, torch.zeros(1, 62, 1, 200), args.checkpoint, 5)
    run_case("FACED", FACEDModel, torch.zeros(1, 32, 10, 200), args.checkpoint, 9)
    run_case("PhysioNet-MI", PhysioModel, torch.zeros(1, 64, 4, 200), args.checkpoint, 4)
    run_case("ISRUC", ISRUCModel, torch.zeros(1, 1, 6, 6000), args.checkpoint, 5)
    run_depth_case(args.checkpoint)
    print("foundation loading contract: PASS")


if __name__ == "__main__":
    main()
