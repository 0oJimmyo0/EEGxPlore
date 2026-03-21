from typing import Any, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from models.moe import SparseMoEFFN, warm_start_moe_from_dense_ckpt


class CBraMod(nn.Module):
    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        attnres_variant='none',
        attnres_gated=False,
        attnres_gate_init=0.0,
        attnres_start_layer=0,
        dropout=0.1,
        use_moe=False,
        moe_num_layers=2,
        moe_num_experts=4,
        moe_top_k=1,
        moe_router_noise_std=0.0,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        if use_moe:
            moe_num_layers = min(max(1, moe_num_layers), n_layer)
            moe_start = n_layer - moe_num_layers
            layers_list = []
            for idx in range(n_layer):
                moe_mod = None
                if idx >= moe_start:
                    moe_mod = SparseMoEFFN(
                        d_model=d_model,
                        dim_feedforward=dim_feedforward,
                        num_experts=moe_num_experts,
                        top_k=moe_top_k,
                        dropout=dropout,
                        activation=F.gelu,
                        router_noise_std=moe_router_noise_std,
                    )
                layers_list.append(
                    TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation=F.gelu,
                        attnres_variant=attnres_variant,
                        attnres_gated=attnres_gated,
                        attnres_gate_init=attnres_gate_init,
                        attnres_start_layer=attnres_start_layer,
                        moe_ffn=moe_mod,
                    )
                )
            encoder_layers = nn.ModuleList(layers_list)
            self.encoder = TransformerEncoder(
                None,
                n_layer,
                enable_nested_tensor=False,
                attnres_variant=attnres_variant,
                d_model=d_model,
                attnres_start_layer=attnres_start_layer,
                layers=encoder_layers,
            )
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation=F.gelu,
                attnres_variant=attnres_variant,
                attnres_gated=attnres_gated,
                attnres_gate_init=attnres_gate_init,
                attnres_start_layer=attnres_start_layer,
            )

            self.encoder = TransformerEncoder(
                encoder_layer,
                num_layers=n_layer,
                enable_nested_tensor=False,
                attnres_variant=attnres_variant,
                d_model=d_model,
                attnres_start_layer=attnres_start_layer,
            )

        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out

    def moe_auxiliary_loss(self) -> torch.Tensor:
        """Sum of per-layer Switch-style load-balancing terms (multiply by --moe_load_balance in trainer)."""
        device = next(self.parameters()).device
        tot = torch.zeros((), device=device, dtype=torch.float32)
        for layer in self.encoder.layers:
            moe = getattr(layer, 'moe_ffn', None)
            if moe is None:
                continue
            tot = tot + moe._last_lb_loss.to(dtype=tot.dtype)
        return tot

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def backbone_finetune_kwargs(param) -> Dict[str, Any]:
    """CBraMod kwargs shared across downstream `model_for_*` wrappers."""
    return {
        'attnres_variant': getattr(param, 'attnres_variant', 'none'),
        'attnres_gated': getattr(param, 'attnres_gated', False),
        'attnres_gate_init': getattr(param, 'attnres_gate_init', 0.0),
        'attnres_start_layer': getattr(param, 'attnres_start_layer', 0),
        'use_moe': getattr(param, 'moe', False),
        'moe_num_layers': getattr(param, 'moe_num_layers', 2),
        'moe_num_experts': getattr(param, 'moe_num_experts', 4),
        'moe_top_k': getattr(param, 'moe_top_k', 1),
        'moe_router_noise_std': getattr(param, 'moe_router_noise', 0.0),
    }


def _moe_warm_started_expert_keys(backbone: nn.Module) -> Set[str]:
    keys = set()
    for idx, layer in enumerate(backbone.encoder.layers):
        if layer.moe_ffn is None:
            continue
        prefix = f'encoder.layers.{idx}.moe_ffn.'
        for k in backbone.state_dict().keys():
            if k.startswith(prefix) and '.experts.' in k:
                keys.add(k)
    return keys


def load_foundation_into_backbone(backbone: nn.Module, param, ckpt_state: Dict[str, Any]) -> Set[str]:
    """
    Load pretrained CBraMod weights. Handles baseline strict load, AttnRes partial load,
    and MoE (warm-start experts from dense linear1/2 in checkpoint, then partial load).
    Returns backbone state_dict keys to treat as foundation-initialized (for LR grouping).
    """
    if isinstance(ckpt_state, dict) and 'state_dict' in ckpt_state:
        ckpt_state = ckpt_state['state_dict']

    use_moe = getattr(param, 'moe', False)
    attnres = getattr(param, 'attnres_variant', 'none')
    backbone_sd = backbone.state_dict()

    if use_moe:
        n = backbone.encoder.num_layers
        moe_n = min(getattr(param, 'moe_num_layers', 2), n)
        copy_all = not getattr(param, 'moe_expert_zero_only', False)
        start = max(0, n - moe_n)
        for idx in range(start, n):
            layer = backbone.encoder.layers[idx]
            if layer.moe_ffn is None:
                continue
            warm_start_moe_from_dense_ckpt(layer.moe_ffn, ckpt_state, idx, copy_all)

    if attnres == 'none' and not use_moe:
        backbone.load_state_dict(ckpt_state, strict=True)
        return set(backbone_sd.keys())

    loadable = {
        k: v for k, v in ckpt_state.items()
        if k in backbone_sd and backbone_sd[k].shape == v.shape
    }
    backbone.load_state_dict(loadable, strict=False)
    pretrained = set(loadable.keys())
    if use_moe:
        pretrained |= _moe_warm_started_expert_keys(backbone)
    return pretrained


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)
