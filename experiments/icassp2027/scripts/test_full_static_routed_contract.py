"""Full CBraMod Static/Routed schema and trainability audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_trainer import configure_trainability
from models.cbramod import load_foundation_into_backbone
from models.model_for_seedv import Model as SeedVModel


def build_params(policy: str) -> SimpleNamespace:
    return SimpleNamespace(
        backbone="cbramod",
        attnres_variant="none",
        attnres_gated=False,
        attnres_gate_init=0.0,
        attnres_start_layer=0,
        dropout=0.0,
        use_moe=True,
        moe=True,
        moe_num_layers=4,
        moe_num_experts=4,
        moe_route_mode="typed_conditional",
        moe_router_policy=policy,
        moe_router_arch="mlp",
        moe_router_mlp_hidden=128,
        moe_router_dispatch_mode="soft",
        moe_router_temperature=1.0,
        moe_shared_output_scale=1.0,
        moe_expert_output_scale=1.0,
        moe_specialist_branch_mode="both",
        moe_specialist_rand_linear1=False,
        moe_expert_init_noise_std=0.0,
        use_pretrained_weights=False,
        foundation_dir="",
        cuda=0,
        classifier="all_patch_reps",
        num_of_classes=5,
        experiment_profile="icassp2027",
        trainability_mode="typed_conditional",
        frozen=False,
    )


def build(policy: str) -> tuple[SeedVModel, SimpleNamespace]:
    params = build_params(policy)
    return SeedVModel(params), params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth"),
    )
    args = parser.parse_args()

    torch.manual_seed(2027)
    static, static_params = build("static")
    torch.manual_seed(2027)
    routed, routed_params = build("sample")

    # Lazy classifier initialization consumes the global RNG after the
    # policy-specific backbone forward. Materialize both classifiers under
    # the same seed before comparing their state dictionaries.
    for model in (static, routed):
        torch.manual_seed(2028)
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 62, 1, 200))

    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        load_params = SimpleNamespace(
            moe=True, moe_num_layers=4, moe_specialist_rand_linear1=False,
            moe_expert_init_noise_std=0.0, attnres_variant="none",
        )
        load_foundation_into_backbone(static.backbone, load_params, ckpt)
        load_foundation_into_backbone(routed.backbone, load_params, ckpt)

    static.eval()
    routed.eval()
    x = torch.randn(1, 62, 1, 200)
    with torch.no_grad():
        y_static = static(x)
        y_routed = routed(x)
    assert y_static.shape == y_routed.shape == (1, 5)
    assert torch.isfinite(y_static).all() and torch.isfinite(y_routed).all()

    static_mode, static_named_trainable = configure_trainability(static, static_params)
    routed_mode, routed_named_trainable = configure_trainability(routed, routed_params)
    assert static_mode == routed_mode == "typed_conditional"

    static_sd = static.state_dict()
    routed_sd = routed.state_dict()
    assert not any("depth_block_" in name for name in static_sd)
    assert not any("depth_block_" in name for name in routed_sd)
    assert any("moe_ffn.spatial_specialists" in name for name in static_sd)
    assert any("moe_ffn.spatial_router" in name for name in static_sd)
    assert list(static_sd) == list(routed_sd)
    assert all(torch.equal(a, b) for a, b in zip(static_sd.values(), routed_sd.values()))

    static_trainable = {name: parameter.requires_grad for name, parameter in static.named_parameters()}
    routed_trainable = {name: parameter.requires_grad for name, parameter in routed.named_parameters()}
    assert static_trainable == routed_trainable
    assert sum(static_trainable.values()) > 0
    assert {name for name, _ in static_named_trainable} == {
        name for name, trainable in static_trainable.items() if trainable
    }
    assert {name for name, _ in routed_named_trainable} == {
        name for name, trainable in routed_trainable.items() if trainable
    }
    assert all(
        not trainable
        for name, trainable in static_trainable.items()
        if name.startswith("backbone.") and ".moe_ffn." not in name
    )
    assert all(not trainable for name, trainable in static_trainable.items() if ".moe_ffn.shared." in name)
    assert all(
        (".moe_ffn." not in name or int(name.split(".layers.")[1].split(".")[0]) >= 8)
        for name, trainable in static_trainable.items() if trainable
    )

    print("full SEED-V wrapper Static/Routed contract: PASS")
    print(f"trainable_parameter_count={sum(p.numel() for p in static.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    main()
