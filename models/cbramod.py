from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from models.moe import (
    TypedCapacityDomainMoEFFN,
    compact_psd_bandpowers,
    reset_moe_eeg_router_summary,
    reset_moe_faced_metadata,
    reset_moe_psd_router_features,
    set_moe_eeg_router_summary,
    set_moe_faced_metadata,
    set_moe_psd_router_features,
    warm_start_moe_from_dense_ckpt,
)


def _compact_eeg_summary(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Compact per-sample EEG summary [B, out_dim] from raw [B,C,S,T]."""
    if x.dim() != 4:
        raise ValueError(f"Expected raw EEG [B,C,S,T], got {tuple(x.shape)}")
    if out_dim <= 0:
        raise ValueError("out_dim must be > 0")
    # Mean over sequence/time, keep compact channel profile, then pool to out_dim.
    v = x.mean(dim=(2, 3))
    return F.adaptive_avg_pool1d(v.unsqueeze(1), out_dim).squeeze(1)


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
        moe_route_mode: str = "typed_capacity_domain",
        moe_capacity_factor: float = 1.0,
        moe_domain_bias: bool = False,
        moe_domain_emb_dim: int = 16,
        moe_specialist_linear1_from_dense=True,
        moe_router_arch: str = "linear",
        moe_router_mlp_hidden: int = 128,
        moe_use_psd_router_features: bool = False,
        moe_use_attnres_depth_router_features: bool = False,
        moe_attnres_depth_router_dim: int = 26,
                    moe_attnres_depth_context_mode: str = "compact_shared",
                    moe_attnres_depth_block_count: int = 4,
        moe_attnres_depth_summary_mode: str = "auto",
        moe_attnres_depth_probe_mlp_for_router: bool = False,
        moe_attnres_depth_router_init: str = "xavier",
        moe_attnres_depth_router_norm_gate: bool = True,
        moe_attnres_depth_router_gate_init: float = 0.075,
        moe_attnres_depth_router_norm_eps: float = 1e-6,
        moe_attnres_depth_block_separation_coef: float = 0.0,
        moe_attnres_depth_summary_grad_mode: str = "delayed_unfreeze",
        moe_attnres_depth_summary_unfreeze_epoch: int = 8,
        moe_router_dispatch_mode: str = "hard_capacity",
        moe_router_temperature: float = 1.0,
        moe_router_entropy_coef: float = 0.0,
        moe_router_balance_kl_coef: float = 0.0,
        moe_router_z_loss_coef: float = 0.0,
        moe_router_jitter_std: float = 0.0,
        moe_router_jitter_final_std: float = 0.0,
        moe_router_jitter_anneal_epochs: int = 0,
        moe_router_soft_warmup_epochs: int = 0,
        moe_uniform_dispatch_warmup_epochs: int = 0,
        moe_shared_blend_warmup_epochs: int = 0,
        moe_shared_blend_start: float = 1.0,
        moe_shared_blend_end: float = 0.0,
        moe_router_entropy_coef_spatial: Optional[float] = None,
        moe_router_entropy_coef_spectral: Optional[float] = None,
        moe_router_balance_kl_coef_spatial: Optional[float] = None,
        moe_router_balance_kl_coef_spectral: Optional[float] = None,
        moe_specialist_branch_mode: str = "both",
        moe_router_compact_feature_mode: str = "none",
        moe_router_compact_feature_dim: int = 8,
        moe_expert_init_noise_std: float = 0.0,
        moe_load_balance: float = 0.0,
        moe_domain_bias_reg: float = 0.0,
    ):
        super().__init__()

        self.use_moe = use_moe
        self.moe_use_psd_router_features = bool(moe_use_psd_router_features)
        self.moe_route_mode = moe_route_mode
        self.moe_router_compact_feature_mode = str(moe_router_compact_feature_mode)
        self.moe_router_compact_feature_dim = int(moe_router_compact_feature_dim)
        valid_compact = {"none", "eeg_summary", "psd_summary"}
        if self.moe_router_compact_feature_mode not in valid_compact:
            raise ValueError(
                f"moe_router_compact_feature_mode must be one of {sorted(valid_compact)}, "
                f"got {self.moe_router_compact_feature_mode!r}"
            )
        if self.moe_router_compact_feature_mode != "none" and self.moe_router_compact_feature_dim <= 0:
            raise ValueError("moe_router_compact_feature_dim must be > 0 when compact router features are enabled")
        valid_depth_context_modes = {"compact_shared", "block_shared_typed_proj", "dual_query_block_typed_proj"}
        if str(moe_attnres_depth_context_mode) not in valid_depth_context_modes:
            raise ValueError(
                f"moe_attnres_depth_context_mode must be one of {sorted(valid_depth_context_modes)}, "
                f"got {moe_attnres_depth_context_mode!r}"
            )
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        if use_moe:
            moe_num_layers = min(max(1, moe_num_layers), n_layer)
            moe_start = n_layer - moe_num_layers
            if moe_route_mode != "typed_capacity_domain":
                raise ValueError("Only moe_route_mode=typed_capacity_domain is supported")
            if attnres_variant not in ("pre_attn", "full"):
                raise ValueError(
                    "typed_capacity_domain requires attnres_variant pre_attn or full "
                    "(pre-attn AttnRes path for router inputs)."
                )
            if attnres_start_layer > moe_start:
                raise ValueError(
                    f"attnres_start_layer must be <= first MoE layer index ({moe_start}); "
                    f"got {attnres_start_layer}. Otherwise MoE runs without baseline/attnres."
                )
            layers_list = []
            for idx in range(n_layer):
                moe_mod = None
                if idx >= moe_start:
                    use_spatial_specialists = True
                    use_spectral_specialists = True
                    if moe_specialist_branch_mode == "spatial_only":
                        use_spectral_specialists = False
                    elif moe_specialist_branch_mode == "spectral_only":
                        use_spatial_specialists = False
                    elif moe_specialist_branch_mode != "both":
                        raise ValueError(
                            f"moe_specialist_branch_mode must be one of ['both','spatial_only','spectral_only'], "
                            f"got {moe_specialist_branch_mode!r}"
                        )

                    use_compact_router_summary = self.moe_router_compact_feature_mode != "none"
                    moe_mod = TypedCapacityDomainMoEFFN(
                        d_model=d_model,
                        dim_feedforward=dim_feedforward,
                        num_specialists=moe_num_experts,
                        dropout=dropout,
                        activation=F.gelu,
                        route_mode=moe_route_mode,
                        capacity_factor=moe_capacity_factor,
                        domain_bias=moe_domain_bias,
                        domain_emb_dim=moe_domain_emb_dim,
                        router_arch=moe_router_arch,
                        router_mlp_hidden=moe_router_mlp_hidden,
                        use_psd_router_features=moe_use_psd_router_features,
                        use_attnres_depth_router_concat=moe_use_attnres_depth_router_features,
                        attnres_depth_router_dim=moe_attnres_depth_router_dim,
                                                attnres_depth_router_init=moe_attnres_depth_router_init,
                                                attnres_depth_router_norm_gate=moe_attnres_depth_router_norm_gate,
                                                attnres_depth_router_gate_init=moe_attnres_depth_router_gate_init,
                                                attnres_depth_router_norm_eps=moe_attnres_depth_router_norm_eps,
                        attnres_depth_block_separation_coef=moe_attnres_depth_block_separation_coef,
                        router_dispatch_mode=moe_router_dispatch_mode,
                        router_temperature=moe_router_temperature,
                        router_entropy_coef=moe_router_entropy_coef,
                        router_balance_kl_coef=moe_router_balance_kl_coef,
                        router_z_loss_coef=moe_router_z_loss_coef,
                        router_jitter_std=moe_router_jitter_std,
                        router_jitter_final_std=moe_router_jitter_final_std,
                        router_jitter_anneal_epochs=moe_router_jitter_anneal_epochs,
                        router_soft_warmup_epochs=moe_router_soft_warmup_epochs,
                        uniform_dispatch_warmup_epochs=moe_uniform_dispatch_warmup_epochs,
                        shared_blend_warmup_epochs=moe_shared_blend_warmup_epochs,
                        shared_blend_start=moe_shared_blend_start,
                        shared_blend_end=moe_shared_blend_end,
                        router_entropy_coef_spatial=moe_router_entropy_coef_spatial,
                        router_entropy_coef_spectral=moe_router_entropy_coef_spectral,
                        router_balance_kl_coef_spatial=moe_router_balance_kl_coef_spatial,
                        router_balance_kl_coef_spectral=moe_router_balance_kl_coef_spectral,
                        use_spatial_specialists=use_spatial_specialists,
                        use_spectral_specialists=use_spectral_specialists,
                        use_eeg_summary_router_concat_spatial=use_compact_router_summary,
                        use_eeg_summary_router_concat_spectral=use_compact_router_summary,
                        eeg_summary_router_dim=self.moe_router_compact_feature_dim,
                        expert_init_noise_std=moe_expert_init_noise_std,
                        load_balance_coef=moe_load_balance,
                        domain_bias_reg_coef=moe_domain_bias_reg,
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
                                                    moe_attnres_depth_context_mode=moe_attnres_depth_context_mode,
                                                    moe_attnres_depth_block_count=moe_attnres_depth_block_count,
                        moe_attnres_depth_router_dim=moe_attnres_depth_router_dim,
                        moe_attnres_depth_summary_mode=moe_attnres_depth_summary_mode,
                        moe_attnres_depth_probe_mlp_for_router=moe_attnres_depth_probe_mlp_for_router,
                        moe_attnres_depth_summary_grad_mode=moe_attnres_depth_summary_grad_mode,
                        moe_attnres_depth_summary_unfreeze_epoch=moe_attnres_depth_summary_unfreeze_epoch,
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
                                    moe_attnres_depth_context_mode=moe_attnres_depth_context_mode,
                                    moe_attnres_depth_block_count=moe_attnres_depth_block_count,
                moe_attnres_depth_router_dim=moe_attnres_depth_router_dim,
                moe_attnres_depth_summary_mode=moe_attnres_depth_summary_mode,
                moe_attnres_depth_probe_mlp_for_router=moe_attnres_depth_probe_mlp_for_router,
                moe_attnres_depth_summary_grad_mode=moe_attnres_depth_summary_grad_mode,
                moe_attnres_depth_summary_unfreeze_epoch=moe_attnres_depth_summary_unfreeze_epoch,
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
        if use_moe:
            for li, layer in enumerate(self.encoder.layers):
                m = getattr(layer, 'moe_ffn', None)
                if isinstance(m, TypedCapacityDomainMoEFFN):
                    m._zero_specialist_output_weights()

    def forward(self, x, mask=None, batch_meta=None):
        tok_psd = None
        tok_meta = None
        tok_eeg = None
        if self.use_moe and self.moe_use_psd_router_features:
            tok_psd = set_moe_psd_router_features(compact_psd_bandpowers(x))
        if self.use_moe and self.moe_router_compact_feature_mode != "none":
            if self.moe_router_compact_feature_mode == "eeg_summary":
                compact = _compact_eeg_summary(x, self.moe_router_compact_feature_dim)
            elif self.moe_router_compact_feature_mode == "psd_summary":
                compact = compact_psd_bandpowers(x, n_bands=self.moe_router_compact_feature_dim)
            else:
                raise ValueError(f"Unsupported compact router mode: {self.moe_router_compact_feature_mode!r}")
            tok_eeg = set_moe_eeg_router_summary(compact)
        if self.use_moe and self.moe_route_mode == "typed_capacity_domain":
            tok_meta = set_moe_faced_metadata(batch_meta)
        try:
            patch_emb = self.patch_embedding(x, mask)
            feats = self.encoder(patch_emb)
            out = self.proj_out(feats)
            return out
        finally:
            if tok_psd is not None:
                reset_moe_psd_router_features(tok_psd)
            if tok_eeg is not None:
                reset_moe_eeg_router_summary(tok_eeg)
            if tok_meta is not None:
                reset_moe_faced_metadata(tok_meta)

    def moe_auxiliary_loss(self) -> torch.Tensor:
        """Combined MoE auxiliary loss from all active MoE layers."""
        device = next(self.parameters()).device
        tot = torch.zeros((), device=device, dtype=torch.float32)
        for layer in self.encoder.layers:
            moe = getattr(layer, 'moe_ffn', None)
            if moe is None:
                continue
            if hasattr(moe, 'auxiliary_loss'):
                tot = tot + moe.auxiliary_loss().to(dtype=tot.dtype)
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
        'moe_route_mode': getattr(param, 'moe_route_mode', 'typed_capacity_domain'),
        'moe_capacity_factor': getattr(param, 'moe_capacity_factor', 1.0),
        'moe_domain_bias': getattr(param, 'moe_domain_bias', False),
        'moe_domain_emb_dim': getattr(param, 'moe_domain_emb_dim', 16),
        'moe_specialist_linear1_from_dense': not getattr(param, 'moe_specialist_rand_linear1', False),
        'moe_router_arch': getattr(param, 'moe_router_arch', 'linear'),
        'moe_router_mlp_hidden': getattr(param, 'moe_router_mlp_hidden', 128),
        'moe_use_psd_router_features': getattr(param, 'moe_use_psd_router_features', False),
        'moe_use_attnres_depth_router_features': getattr(param, 'moe_use_attnres_depth_router_features', False),
        'moe_attnres_depth_router_dim': getattr(param, 'moe_attnres_depth_router_dim', 26),
        'moe_attnres_depth_context_mode': getattr(param, 'moe_attnres_depth_context_mode', 'compact_shared'),
        'moe_attnres_depth_block_count': getattr(param, 'moe_attnres_depth_block_count', 4),
        'moe_attnres_depth_summary_mode': getattr(param, 'moe_attnres_depth_summary_mode', 'auto'),
        'moe_attnres_depth_probe_mlp_for_router': getattr(param, 'moe_attnres_depth_probe_mlp_for_router', False),
        'moe_attnres_depth_router_init': getattr(param, 'moe_attnres_depth_router_init', 'xavier'),
        'moe_attnres_depth_router_norm_gate': getattr(param, 'moe_attnres_depth_router_norm_gate', True),
        'moe_attnres_depth_router_gate_init': getattr(param, 'moe_attnres_depth_router_gate_init', 0.075),
        'moe_attnres_depth_router_norm_eps': getattr(param, 'moe_attnres_depth_router_norm_eps', 1e-6),
        'moe_attnres_depth_block_separation_coef': getattr(param, 'moe_attnres_depth_block_separation_coef', 0.0),
        'moe_attnres_depth_summary_grad_mode': getattr(param, 'moe_attnres_depth_summary_grad_mode', 'delayed_unfreeze'),
        'moe_attnres_depth_summary_unfreeze_epoch': getattr(param, 'moe_attnres_depth_summary_unfreeze_epoch', 8),
        'moe_router_dispatch_mode': getattr(param, 'moe_router_dispatch_mode', 'hard_capacity'),
        'moe_router_temperature': getattr(param, 'moe_router_temperature', 1.0),
        'moe_router_entropy_coef': getattr(param, 'moe_router_entropy_coef', 0.0),
        'moe_router_balance_kl_coef': getattr(param, 'moe_router_balance_kl_coef', 0.0),
        'moe_router_z_loss_coef': getattr(param, 'moe_router_z_loss_coef', 0.0),
        'moe_router_jitter_std': getattr(param, 'moe_router_jitter_std', 0.0),
        'moe_router_jitter_final_std': getattr(param, 'moe_router_jitter_final_std', 0.0),
        'moe_router_jitter_anneal_epochs': getattr(param, 'moe_router_jitter_anneal_epochs', 0),
        'moe_router_soft_warmup_epochs': getattr(param, 'moe_router_soft_warmup_epochs', 0),
        'moe_uniform_dispatch_warmup_epochs': getattr(param, 'moe_uniform_dispatch_warmup_epochs', 0),
        'moe_shared_blend_warmup_epochs': getattr(param, 'moe_shared_blend_warmup_epochs', 0),
        'moe_shared_blend_start': getattr(param, 'moe_shared_blend_start', 1.0),
        'moe_shared_blend_end': getattr(param, 'moe_shared_blend_end', 0.0),
        'moe_router_entropy_coef_spatial': (
            None if getattr(param, 'moe_router_entropy_coef_spatial', -1.0) < 0
            else getattr(param, 'moe_router_entropy_coef_spatial', -1.0)
        ),
        'moe_router_entropy_coef_spectral': (
            None if getattr(param, 'moe_router_entropy_coef_spectral', -1.0) < 0
            else getattr(param, 'moe_router_entropy_coef_spectral', -1.0)
        ),
        'moe_router_balance_kl_coef_spatial': (
            None if getattr(param, 'moe_router_balance_kl_coef_spatial', -1.0) < 0
            else getattr(param, 'moe_router_balance_kl_coef_spatial', -1.0)
        ),
        'moe_router_balance_kl_coef_spectral': (
            None if getattr(param, 'moe_router_balance_kl_coef_spectral', -1.0) < 0
            else getattr(param, 'moe_router_balance_kl_coef_spectral', -1.0)
        ),
        'moe_specialist_branch_mode': getattr(param, 'moe_specialist_branch_mode', 'both'),
        'moe_router_compact_feature_mode': getattr(param, 'moe_router_compact_feature_mode', 'none'),
        'moe_router_compact_feature_dim': getattr(param, 'moe_router_compact_feature_dim', 8),
        'moe_expert_init_noise_std': getattr(param, 'moe_expert_init_noise_std', 0.0),
        'moe_load_balance': getattr(param, 'moe_load_balance', 0.0),
        'moe_domain_bias_reg': getattr(param, 'moe_domain_bias_reg', 0.0),
    }


def _moe_warm_started_expert_keys(backbone: nn.Module) -> Set[str]:
    keys = set()
    for idx, layer in enumerate(backbone.encoder.layers):
        if layer.moe_ffn is None:
            continue
        prefix = f'encoder.layers.{idx}.moe_ffn.'
        for k in backbone.state_dict().keys():
            if not k.startswith(prefix):
                continue
            if '.experts.' in k:
                keys.add(k)
            if '.shared.' in k:
                keys.add(k)
            if '.spatial_specialists.' in k and '.linear1.' in k:
                keys.add(k)
            if '.spectral_specialists.' in k and '.linear1.' in k:
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
        start = max(0, n - moe_n)
        spec_l1 = not getattr(param, 'moe_specialist_rand_linear1', False)
        for idx in range(start, n):
            layer = backbone.encoder.layers[idx]
            if layer.moe_ffn is None:
                continue
            warm_start_moe_from_dense_ckpt(
                layer.moe_ffn,
                ckpt_state,
                idx,
                copy_specialist_linear1_from_dense=spec_l1,
                expert_init_noise_std=getattr(param, 'moe_expert_init_noise_std', 0.0),
            )

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
