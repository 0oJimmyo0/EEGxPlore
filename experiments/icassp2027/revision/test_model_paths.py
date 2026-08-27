"""Small forward-path contracts for the focused specialist variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cbramod import CBraMod


def _model(attnres_variant: str, router_base_feature_mode: str) -> CBraMod:
    return CBraMod(
        in_dim=200,
        out_dim=4,
        # CBraMod's patch embedding produces 25 x 8 = 200 features per
        # patch; the native width is therefore part of this smoke fixture.
        d_model=200,
        dim_feedforward=16,
        seq_len=1,
        n_layer=2,
        nhead=2,
        attnres_variant=attnres_variant,
        attnres_start_layer=0,
        use_moe=True,
        moe_num_layers=1,
        moe_num_experts=2,
        moe_route_mode='typed_capacity_domain',
        moe_router_arch='linear',
        moe_router_dispatch_mode='soft',
        moe_router_temperature=1.0,
        moe_router_base_feature_mode=router_base_feature_mode,
        moe_load_balance=0.0,
        moe_router_entropy_coef=0.0,
        moe_router_balance_kl_coef=0.0,
        moe_router_z_loss_coef=0.0,
        moe_router_jitter_std=0.0,
        moe_router_jitter_final_std=0.0,
        moe_router_soft_warmup_epochs=0,
        moe_uniform_dispatch_warmup_epochs=0,
        moe_shared_blend_warmup_epochs=0,
        moe_specialist_branch_mode='both',
    )


def main() -> None:
    x = torch.randn(2, 2, 1, 200)
    specialist_only = _model('none', 'baseline_only')
    with torch.no_grad():
        specialist_output = specialist_only(x)
    assert specialist_output.shape == (2, 2, 1, 4)

    combined = _model('pre_attn', 'full')
    with torch.no_grad():
        combined_output = combined(x)
    assert combined_output.shape == (2, 2, 1, 4)

    print('model path contract: PASS')


if __name__ == '__main__':
    main()
