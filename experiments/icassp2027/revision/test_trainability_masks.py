"""Unit contracts for the focused revision trainability masks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_main import resolve_revision_condition
from finetune_trainer import configure_trainability


class _TinyLayer(nn.Module):
    def __init__(self, with_components: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        if with_components:
            self.pre_attn_res = nn.Linear(2, 2, bias=False)
            self.moe_ffn = nn.Module()
            self.moe_ffn.shared = nn.Linear(2, 2, bias=False)
            self.moe_ffn.spatial_router = nn.Linear(2, 2, bias=False)
            self.moe_ffn.spectral_router = nn.Linear(2, 2, bias=False)
            self.moe_ffn.spatial_specialists = nn.ModuleList([nn.Linear(2, 2, bias=False)])
            self.moe_ffn.spectral_specialists = nn.ModuleList([nn.Linear(2, 2, bias=False)])


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layers = nn.ModuleList(
            [_TinyLayer(with_components=(idx == 11)) for idx in range(12)]
        )
        self.classifier = nn.Linear(2, 2)


def _params(mode: str):
    return SimpleNamespace(
        trainability_mode=mode,
        frozen=False,
        experiment_profile='icassp2027_revision',
        revision_condition=mode,
    )


def _names(model):
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def main() -> None:
    model = _TinyModel()

    mode, _ = configure_trainability(model, _params('upper1'))
    assert mode == 'upper1'
    upper1 = _names(model)
    assert 'classifier.weight' in upper1
    assert 'backbone.encoder.layers.11.weight' in upper1
    assert 'backbone.encoder.layers.10.weight' not in upper1

    mode, _ = configure_trainability(model, _params('specialist_only'))
    assert mode == 'specialist_only'
    specialist = _names(model)
    assert 'classifier.weight' in specialist
    assert any('spatial_specialists' in name for name in specialist)
    assert any('spatial_router' in name for name in specialist)
    assert not any('.pre_attn_res.' in name for name in specialist)
    assert not any('.moe_ffn.shared.' in name for name in specialist)

    mode, _ = configure_trainability(model, _params('combined'))
    assert mode == 'combined'
    combined = _names(model)
    assert any('.pre_attn_res.' in name for name in combined)
    assert any('spatial_specialists' in name for name in combined)
    assert any('spatial_router' in name for name in combined)
    assert not any('.moe_ffn.shared.' in name for name in combined)

    fresh_params = _params('selective_fresh')
    resolve_revision_condition(fresh_params)
    assert fresh_params.trainability_mode == 'combined'
    mode, _ = configure_trainability(model, fresh_params)
    assert mode == 'combined'
    fresh = _names(model)
    assert fresh == combined

    paper_params = _params('selective_paper')
    resolve_revision_condition(paper_params)
    assert paper_params.trainability_mode == 'combined'
    mode, _ = configure_trainability(model, paper_params)
    assert mode == 'combined'
    paper = _names(model)
    assert paper == combined

    historical_params = _params('historical_selective')
    resolve_revision_condition(historical_params)
    assert historical_params.trainability_mode == 'combined'
    mode, _ = configure_trainability(model, historical_params)
    assert mode == 'combined'
    historical = _names(model)
    assert historical == combined

    print('trainability mask contract: PASS')


if __name__ == '__main__':
    main()
