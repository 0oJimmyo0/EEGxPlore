"""Contract tests for the ICASSP Static/Routed typed-conditional MoE."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.moe import TypedConditionalMoEFFN  # noqa: E402


def main() -> None:
    kwargs = dict(
        d_model=12,
        dim_feedforward=24,
        num_specialists=4,
        dropout=0.0,
        activation="gelu",
        router_arch="linear",
        router_mlp_hidden=16,
    )

    torch.manual_seed(2027)
    static = TypedConditionalMoEFFN(router_policy="static", **kwargs)
    torch.manual_seed(2027)
    routed = TypedConditionalMoEFFN(router_policy="sample", **kwargs)

    static_sd = static.state_dict()
    routed_sd = routed.state_dict()
    assert list(static_sd) == list(routed_sd), "Static/Routed state-dict keys differ"
    assert all(a.shape == b.shape for a, b in zip(static_sd.values(), routed_sd.values()))
    assert all(torch.equal(a, b) for a, b in zip(static_sd.values(), routed_sd.values()))
    assert sum(p.numel() for p in static.parameters()) == sum(p.numel() for p in routed.parameters())

    sample_a = torch.randn(3, 2, 5, kwargs["d_model"])
    sample_b = sample_a + 0.75
    static_a = static._router_input(sample_a, static.router_constant_spatial)
    static_b = static._router_input(sample_b, static.router_constant_spatial)
    routed_a = routed._router_input(sample_a, routed.router_constant_spatial)
    routed_b = routed._router_input(sample_b, routed.router_constant_spatial)
    assert torch.equal(static_a, static_b), "Static router input changed with the sample"
    assert not torch.allclose(routed_a, routed_b), "Routed router input ignored the sample"

    static_logits_a = static.spatial_router(static_a)
    static_logits_b = static.spatial_router(static_b)
    routed_logits_a = routed.spatial_router(routed_a)
    routed_logits_b = routed.spatial_router(routed_b)
    assert torch.equal(static_logits_a, static_logits_b), "Static router probabilities are not batch-invariant"
    assert not torch.allclose(routed_logits_a, routed_logits_b), "Routed router logits did not change"

    sample_grad = sample_a.detach().clone().requires_grad_(True)
    routed_logits = routed.spatial_router(
        routed._router_input(sample_grad, routed.router_constant_spatial)
    )
    routed_logits.square().mean().backward()
    assert sample_grad.grad is not None and torch.isfinite(sample_grad.grad).all()
    assert routed.spatial_router.weight.grad is not None

    print("typed_conditional contract: PASS")
    print(f"parameter_count={sum(p.numel() for p in static.parameters())}")


if __name__ == "__main__":
    main()

