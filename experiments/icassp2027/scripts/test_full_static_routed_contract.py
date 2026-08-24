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

from finetune_trainer import Trainer
from models.cbramod import CBraMod, load_foundation_into_backbone


def build(policy: str) -> CBraMod:
    return CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8, dropout=0.0,
        attnres_variant="none", use_moe=True, moe_num_layers=4,
        moe_num_experts=4, moe_route_mode="typed_conditional",
        moe_router_policy=policy, moe_router_arch="mlp",
        moe_router_mlp_hidden=128, moe_router_dispatch_mode="soft",
        moe_router_temperature=1.0, moe_shared_output_scale=1.0,
        moe_expert_output_scale=1.0, moe_specialist_branch_mode="both",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth"),
    )
    parser.add_argument("--skip_forward", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(2027)
    static = build("static")
    torch.manual_seed(2027)
    routed = build("sample")

    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        params = SimpleNamespace(
            moe=True, moe_num_layers=4, moe_specialist_rand_linear1=False,
            moe_expert_init_noise_std=0.0, attnres_variant="none",
        )
        load_foundation_into_backbone(static, params, ckpt)
        load_foundation_into_backbone(routed, params, ckpt)

    static_sd = static.state_dict()
    routed_sd = routed.state_dict()
    assert list(static_sd) == list(routed_sd)
    assert all(torch.equal(a, b) for a, b in zip(static_sd.values(), routed_sd.values()))

    static_trainable = {
        name: Trainer._is_icasp_conditional_parameter(f"backbone.{name}")
        for name, _ in static.named_parameters()
    }
    routed_trainable = {
        name: Trainer._is_icasp_conditional_parameter(f"backbone.{name}")
        for name, _ in routed.named_parameters()
    }
    assert static_trainable == routed_trainable
    assert sum(static_trainable.values()) > 0

    # Apply the exact mask used by finetune_trainer.  Without this assignment,
    # the audit would only inspect the intended mask while the model itself
    # would still report every parameter as trainable by default.
    for model, trainable_mask in ((static, static_trainable), (routed, routed_trainable)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad = trainable_mask[name]

    assert {
        name: parameter.requires_grad for name, parameter in static.named_parameters()
    } == static_trainable
    assert {
        name: parameter.requires_grad for name, parameter in routed.named_parameters()
    } == routed_trainable
    assert all(not trainable for name, trainable in static_trainable.items() if ".moe_ffn." not in name)
    assert all(not trainable for name, trainable in static_trainable.items() if ".moe_ffn.shared." in name)
    assert all(
        (".moe_ffn." not in name or int(name.split(".layers.")[1].split(".")[0]) >= 8)
        for name, trainable in static_trainable.items() if trainable
    )

    if not args.skip_forward:
        static.eval()
        routed.eval()
        x = torch.randn(1, 62, 1, 200)
        with torch.no_grad():
            y_static = static(x)
            y_routed = routed(x)
        assert y_static.shape == y_routed.shape == (1, 62, 1, 200)
        assert torch.isfinite(y_static).all() and torch.isfinite(y_routed).all()

    print("full Static/Routed contract: PASS")
    print(f"backbone_trainable_parameter_count={sum(p.numel() for p in static.parameters() if p.requires_grad)}")


if __name__ == "__main__":
    main()
