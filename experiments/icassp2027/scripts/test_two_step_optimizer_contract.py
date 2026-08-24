"""Two-step optimizer and routing-connectivity audit for the ICASSP wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_trainer import configure_trainability, is_icasp_conditional_parameter
from models.cbramod import load_foundation_into_backbone
from test_full_static_routed_contract import build


def _is_specialist_output(name: str) -> bool:
    return (
        (".spatial_specialists." in name or ".spectral_specialists." in name)
        and ".linear2." in name
    )


def _is_router(name: str) -> bool:
    return (
        ".spatial_router." in name
        or ".spectral_router." in name
        or ".router_constant_spatial" in name
        or ".router_constant_spectral" in name
    )


def _finite_nonzero_gradients(model: torch.nn.Module, predicate) -> int:
    count = 0
    for name, parameter in model.named_parameters():
        if not predicate(name) or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"non-finite gradient in {name}")
        if float(parameter.grad.detach().abs().sum()) > 0.0:
            count += 1
    return count


def _snapshot_frozen(model: torch.nn.Module):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and not is_icasp_conditional_parameter(name)
    }


def _assert_frozen_unchanged(model: torch.nn.Module, snapshot, step: int) -> None:
    for name, before in snapshot.items():
        after = dict(model.named_parameters())[name].detach().cpu()
        if not torch.equal(before, after):
            raise AssertionError(f"frozen pretrained parameter changed at step {step}: {name}")


def _routing_checks(model: torch.nn.Module, policy: str) -> dict:
    spatial_rows = []
    spectral_rows = []
    for layer in model.backbone.encoder.layers:
        moe = getattr(layer, "moe_ffn", None)
        cache = getattr(moe, "_routing_export_cache", {}) if moe is not None else {}
        if cache:
            spatial_rows.append(cache["probs_spatial"])
            spectral_rows.append(cache["probs_spectral"])
    if not spatial_rows:
        raise AssertionError("no typed-conditional routing cache was populated")

    if policy == "static":
        max_spatial_delta = max(float((rows - rows[:1]).abs().max()) for rows in spatial_rows)
        max_spectral_delta = max(float((rows - rows[:1]).abs().max()) for rows in spectral_rows)
        if max(max_spatial_delta, max_spectral_delta) > 1e-7:
            raise AssertionError(
                f"Static routing is not batch-invariant: spatial={max_spatial_delta} "
                f"spectral={max_spectral_delta}"
            )
        return {
            "static_max_spatial_batch_delta": max_spatial_delta,
            "static_max_spectral_batch_delta": max_spectral_delta,
        }

    max_spatial_variance = max(float(rows.var(dim=0).max()) for rows in spatial_rows)
    max_spectral_variance = max(float(rows.var(dim=0).max()) for rows in spectral_rows)
    if max(max_spatial_variance, max_spectral_variance) <= 1e-12:
        raise AssertionError(
            f"Routed probabilities are sample-invariant: spatial={max_spatial_variance} "
            f"spectral={max_spectral_variance}"
        )
    return {
        "routed_max_spatial_sample_variance": max_spatial_variance,
        "routed_max_spectral_sample_variance": max_spectral_variance,
    }


def audit_policy(policy: str, checkpoint: Path, device: torch.device) -> dict:
    torch.manual_seed(2027)
    model, params = build(policy)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["state_dict"]
    load_params = type("LoadParams", (), {
        "moe": True,
        "moe_num_layers": 4,
        "moe_specialist_rand_linear1": False,
        "moe_expert_init_noise_std": 0.0,
        "attnres_variant": "none",
    })()
    load_foundation_into_backbone(model.backbone, load_params, checkpoint_state)
    model.to(device)
    x = torch.randn(2, 62, 1, 200, device=device)
    model.eval()
    with torch.no_grad():
        materialized_output = model(x)
    if materialized_output.shape != (2, 5):
        raise AssertionError(f"unexpected SEED-V wrapper output shape: {tuple(materialized_output.shape)}")
    model.train()
    params.trainability_mode = "typed_conditional"
    params.frozen = False
    configure_trainability(model, params)

    y = torch.tensor([0, 1], dtype=torch.long, device=device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    frozen_snapshot = _snapshot_frozen(model)
    specialist_updates = []
    router_grad_counts = []
    routing_results = {}

    for step in range(1, 3):
        before_specialists = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if _is_specialist_output(name)
        }
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss for {policy} at step {step}")
        loss.backward()
        specialist_grad_count = _finite_nonzero_gradients(model, _is_specialist_output)
        router_grad_count = _finite_nonzero_gradients(model, _is_router)
        if specialist_grad_count == 0:
            raise AssertionError(f"specialist output gradients absent for {policy} at step {step}")
        router_grad_counts.append(router_grad_count)
        optimizer.step()

        specialist_changed = any(
            not torch.equal(before, dict(model.named_parameters())[name].detach())
            for name, before in before_specialists.items()
        )
        if not specialist_changed:
            raise AssertionError(f"specialist output weights did not update for {policy} at step {step}")
        specialist_updates.append(specialist_changed)
        _assert_frozen_unchanged(model, frozen_snapshot, step)
        if step == 2:
            routing_results = _routing_checks(model, policy)

    if router_grad_counts[0] != 0:
        raise AssertionError(
            f"expected zero router gradient before specialist activation for {policy}, "
            f"got {router_grad_counts[0]}"
        )
    if router_grad_counts[1] == 0:
        raise AssertionError(f"router gradients remained disconnected for {policy} at step 2")

    return {
        "policy": policy,
        "step1_specialist_update": specialist_updates[0],
        "step2_specialist_update": specialist_updates[1],
        "step1_router_grad_parameter_count": router_grad_counts[0],
        "step2_router_grad_parameter_count": router_grad_counts[1],
        **routing_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    results = [audit_policy(policy, args.checkpoint, device) for policy in ("static", "sample")]
    print("two-step optimizer contract: PASS")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
