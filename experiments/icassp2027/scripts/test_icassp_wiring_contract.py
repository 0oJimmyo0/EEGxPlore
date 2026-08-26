"""Small wiring contracts for the ICASSP launcher-facing configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_main import add_faced_args, add_seedv_args, add_shared_args, validate_args
from finetune_trainer import Trainer


MANIFEST = REPO_ROOT / "experiments/icassp2027/manifests/seedv/split_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    add_faced_args(parser)
    add_seedv_args(parser)
    return parser


def _args(*extra: str) -> argparse.Namespace:
    base = [
        "--datasets_dir", "/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb",
        "--num_of_classes", "5",
        "--model_dir", "/tmp/icassp_wiring_contract",
        "--downstream_dataset", "SEED-V",
        "--experiment_profile", "icassp2027",
        "--icassp_split_manifest", str(MANIFEST),
        "--moe",
        "--moe_num_layers", "4",
        "--moe_num_experts", "4",
        "--moe_route_mode", "typed_conditional",
        "--moe_router_policy", "sample",
        "--moe_router_arch", "mlp",
        "--moe_router_mlp_hidden", "128",
        "--moe_router_temperature", "1.0",
        "--moe_shared_output_scale", "1.0",
        "--moe_expert_output_scale", "1.0",
        "--moe_router_dispatch_mode", "soft",
        "--moe_attnres_depth_context_mode", "compact_shared",
        "--moe_specialist_branch_mode", "both",
        "--moe_router_compact_feature_mode", "none",
        "--moe_load_balance", "0",
        "--moe_router_entropy_coef", "0",
        "--moe_router_balance_kl_coef", "0",
        "--moe_router_z_loss_coef", "0",
        "--moe_router_jitter_std", "0",
        "--moe_router_jitter_final_std", "0",
        "--moe_router_soft_warmup_epochs", "0",
        "--moe_uniform_dispatch_warmup_epochs", "0",
        "--moe_shared_blend_warmup_epochs", "0",
        "--trainability_mode", "typed_conditional",
    ]
    return _parser().parse_args(base + list(extra))


def _depth_args(*extra: str) -> argparse.Namespace:
    base = [
        "--datasets_dir", "/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb",
        "--num_of_classes", "5",
        "--model_dir", "/tmp/icassp_depth_wiring_contract",
        "--downstream_dataset", "SEED-V",
        "--experiment_profile", "icassp2027",
        "--icassp_split_manifest", str(MANIFEST),
        "--trainability_mode", "depth_aggregation",
        "--attnres_variant", "pre_attn",
        "--attnres_start_layer", "8",
        "--use_component_lr",
        "--lr_depth_mult", "1.0",
    ]
    return _parser().parse_args(base + list(extra))


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    primary = _args("--moe_expert_init_noise_std", "0")
    validate_args(primary)

    diagnostic = _args(
        "--moe_expert_init_noise_std", "0.001",
        "--icassp_routing_diagnostic",
    )
    validate_args(diagnostic)

    rejected = _args("--moe_expert_init_noise_std", "0.001")
    try:
        validate_args(rejected)
    except ValueError as exc:
        assert "diagnostic-only" in str(exc)
    else:
        raise AssertionError("nonzero expert-init noise was accepted without diagnostic opt-in")

    assert Trainer._component_name_for_param("backbone.encoder.layers.0.self_attn.in_proj_weight") == "backbone"
    assert Trainer._component_name_for_param("backbone.encoder.layers.8.moe_ffn.spatial_router.weight") == "router"
    assert Trainer._component_name_for_param("backbone.encoder.layers.8.moe_ffn.spatial_specialists.0.linear1.weight") == "experts"
    assert Trainer._component_name_for_param("backbone.adapter.0.weight") == "other"
    assert Trainer._component_name_for_param("backbone.encoder.layers.8.pre_attn_res.query") == "depth"

    depth = _depth_args()
    validate_args(depth)
    invalid_depth = _depth_args("--attnres_start_layer", "7")
    try:
        validate_args(invalid_depth)
    except ValueError as exc:
        assert "start_layer=8" in str(exc)
    else:
        raise AssertionError("invalid DepthAgg start layer was accepted")

    invalid_optimizer = _depth_args("--optimizer", "SGD")
    try:
        validate_args(invalid_optimizer)
    except ValueError as exc:
        assert "locked profile" in str(exc)
    else:
        raise AssertionError("non-AdamW DepthAgg optimizer was accepted")

    print("ICASSP wiring contract: PASS")


if __name__ == "__main__":
    main()
