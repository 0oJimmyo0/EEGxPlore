"""Contract tests for the ICASSP CBraMod depth-aggregation profile."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_trainer import (  # noqa: E402
    Trainer,
    configure_trainability,
    is_depth_aggregation_parameter,
    is_depth_parameter,
)
from models.attn_res import FullAttnRes  # noqa: E402
from models.cbramod import CBraMod, load_foundation_into_backbone  # noqa: E402


class Probe(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(200, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features.mean(dim=(1, 2)))


def _depth_backbone() -> CBraMod:
    return CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        dropout=0.0,
        attnres_variant='pre_attn',
        attnres_gated=False,
        attnres_start_layer=8,
        use_moe=False,
    )


def _params() -> SimpleNamespace:
    return SimpleNamespace(
        trainability_mode='depth_aggregation',
        frozen=False,
        experiment_profile='icassp2027',
        moe=False,
        attnres_variant='pre_attn',
        attnres_start_layer=8,
        attnres_gated=False,
        lr=1e-4,
        lr_backbone_mult=0.5,
        lr_router_mult=2.0,
        lr_expert_mult=1.5,
        lr_classifier_mult=3.5,
        lr_other_mult=1.0,
        lr_depth_mult=1.0,
        weight_decay=5e-2,
    )


def test_attnres_initialization() -> None:
    torch.manual_seed(7)
    module = FullAttnRes(200)
    assert torch.equal(module.query, torch.zeros_like(module.query))
    assert torch.equal(module.norm.weight, torch.ones_like(module.norm.weight))
    sources = [torch.randn(1, 1, 1, 200), torch.randn(1, 1, 1, 200)]
    _, alpha = module(sources, return_alpha=True)
    expected = torch.full_like(alpha, 0.5)
    assert torch.allclose(alpha, expected)
    assert torch.allclose(alpha.sum(dim=0), torch.ones_like(alpha.sum(dim=0)))


def test_depth_mask_load_and_optimizer_contract() -> None:
    torch.manual_seed(11)
    dense = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        dropout=0.0,
        attnres_variant='none',
        use_moe=False,
    )
    depth = _depth_backbone()
    params = _params()

    dense_state = {key: value.detach().clone() for key, value in dense.state_dict().items()}
    loaded = load_foundation_into_backbone(depth, params, dense_state)
    depth_state = depth.state_dict()
    depth_keys = {key for key in depth_state if '.pre_attn_res.' in key}
    assert depth_keys
    assert set(loaded) == set(depth_state) - depth_keys
    for key in depth_state:
        if key not in depth_keys:
            assert torch.equal(depth_state[key], dense_state[key]), key

    model = Probe(depth)
    mode, named_trainable = configure_trainability(model, params)
    assert mode == 'depth_aggregation'
    names = {name for name, _ in named_trainable}
    depth_names = {name for name in names if is_depth_parameter(name)}
    assert len(depth_names) == 8
    assert sum(parameter.numel() for name, parameter in named_trainable if is_depth_parameter(name)) == 1600
    assert all(is_depth_aggregation_parameter(name) for name in depth_names)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if 'backbone' in name and not is_depth_aggregation_parameter(name)
    )
    assert all(name.startswith('classifier') for name in names if not is_depth_parameter(name))

    grouped = {'backbone': [], 'router': [], 'experts': [], 'classifier': [], 'other': [], 'depth': []}
    for name, parameter in named_trainable:
        grouped[Trainer._component_name_for_param(name)].append(parameter)
    fake_trainer = object.__new__(Trainer)
    fake_trainer.params = params
    optimizer = fake_trainer._build_component_optimizer(grouped, kind='adamw')
    lr_by_name = {group['name']: group['lr'] for group in optimizer.param_groups}
    assert lr_by_name['depth'] == 1e-4
    assert lr_by_name['classifier'] == 3.5e-4


def test_depth_forward_scope_and_gradient_connectivity() -> None:
    torch.manual_seed(13)
    model = Probe(_depth_backbone()).eval()
    params = _params()
    configure_trainability(model, params)
    x = torch.randn(1, 2, 1, 200)

    with torch.inference_mode():
        baseline = model(x)
        with torch.no_grad():
            model.backbone.encoder.layers[7].pre_attn_res.query.fill_(0.7)
        below_start = model(x)
        with torch.no_grad():
            model.backbone.encoder.layers[7].pre_attn_res.query.zero_()
            model.backbone.encoder.layers[8].pre_attn_res.query.fill_(0.7)
        active_layer = model(x)
    assert torch.equal(baseline, below_start)
    assert not torch.equal(baseline, active_layer)

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in model.named_parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if 'backbone' in name and not is_depth_parameter(name)
    }
    logits = model(x)
    loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long))
    loss.backward()
    query_grad = model.backbone.encoder.layers[8].pre_attn_res.query.grad
    norm_grad = model.backbone.encoder.layers[8].pre_attn_res.norm.weight.grad
    assert query_grad is not None and torch.isfinite(query_grad).all() and query_grad.abs().sum() > 0
    assert norm_grad is not None and torch.isfinite(norm_grad).all()
    optimizer.step()
    for name, before in original.items():
        assert torch.equal(model.state_dict()[name], before), name


def main() -> None:
    test_attnres_initialization()
    test_depth_mask_load_and_optimizer_contract()
    test_depth_forward_scope_and_gradient_connectivity()
    print('depth aggregation contract: PASS')


if __name__ == '__main__':
    main()
