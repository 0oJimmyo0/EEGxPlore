from typing import Any, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from models.adapters import (
    EEGChannelContextEncoder,
    SubjectDomainAdapter,
    load_channel_context_file,
    reset_adapter_batch_meta,
    set_adapter_batch_meta,
)
from models.moe import (
    TypedCapacityDomainMoEFFN,
    compact_psd_bandpowers,
    reset_moe_faced_metadata,
    reset_moe_psd_router_features,
    set_moe_faced_metadata,
    set_moe_psd_router_features,
    warm_start_moe_from_dense_ckpt,
)


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
        moe_load_balance: float = 0.0,
        moe_domain_bias_reg: float = 0.0,
        use_subject_adapters: bool = False,
        adapter_num_layers: int = 2,
        adapter_rank: int = 16,
        adapter_cond_dim: int = 32,
        adapter_scale: float = 0.2,
        use_eeg_channel_context: bool = False,
        channel_context_file: str = "",
        use_subject_summary: bool = False,
        subject_summary_handling: str = "project",
        metadata_debug: bool = True,
    ):
        super().__init__()

        self.use_moe = use_moe
        self.moe_use_psd_router_features = bool(moe_use_psd_router_features)
        self.moe_route_mode = moe_route_mode
        self.use_subject_adapters = bool(use_subject_adapters)
        self.metadata_debug = bool(metadata_debug)
        self._metadata_runtime_logged = False
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)

        channel_ctx_data = {}
        if channel_context_file:
            try:
                channel_ctx_data = load_channel_context_file(channel_context_file)
            except FileNotFoundError:
                print(
                    f"[CBraMod] channel_context_file not found: {channel_context_file}. "
                    "Continue without explicit channel context file.",
                    flush=True,
                )
        self.channel_context_encoder = EEGChannelContextEncoder(
            d_model=d_model,
            num_channels=in_dim,
            use_channel_context=use_eeg_channel_context,
            channel_context_data=channel_ctx_data,
            metadata_debug=metadata_debug,
        )

        adapter_num_layers = min(max(1, adapter_num_layers), n_layer)
        adapter_start = n_layer - adapter_num_layers

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
                adapter_mod = None
                if idx >= moe_start:
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
                        load_balance_coef=moe_load_balance,
                        domain_bias_reg_coef=moe_domain_bias_reg,
                    )
                if self.use_subject_adapters and idx >= adapter_start and moe_mod is None:
                    adapter_mod = SubjectDomainAdapter(
                        d_model=d_model,
                        rank=adapter_rank,
                        cond_dim=adapter_cond_dim,
                        adapter_scale=adapter_scale,
                        use_subject_summary=use_subject_summary,
                        subject_summary_handling=subject_summary_handling,
                        metadata_debug=metadata_debug,
                        log_usage=(idx == adapter_start),
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
                        subject_adapter=adapter_mod,
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
                subject_adapter=None,
            )

            if self.use_subject_adapters:
                layers = []
                for idx in range(n_layer):
                    layer = TransformerEncoderLayer(
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
                        subject_adapter=(
                            SubjectDomainAdapter(
                                d_model=d_model,
                                rank=adapter_rank,
                                cond_dim=adapter_cond_dim,
                                adapter_scale=adapter_scale,
                                use_subject_summary=use_subject_summary,
                                subject_summary_handling=subject_summary_handling,
                                metadata_debug=metadata_debug,
                                log_usage=(idx == adapter_start),
                            )
                            if idx >= adapter_start
                            else None
                        ),
                    )
                    layers.append(layer)
                self.encoder = TransformerEncoder(
                    None,
                    n_layer,
                    enable_nested_tensor=False,
                    attnres_variant=attnres_variant,
                    d_model=d_model,
                    attnres_start_layer=attnres_start_layer,
                    layers=nn.ModuleList(layers),
                )
            else:
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
        tok_adapter = set_adapter_batch_meta(batch_meta if isinstance(batch_meta, dict) else None)
        if self.metadata_debug and not self._metadata_runtime_logged:
            present = sorted(list(batch_meta.keys())) if isinstance(batch_meta, dict) else []
            print(
                "[metadata-runtime] "
                f"batch_meta_present={present} "
                f"channel_context={self.channel_context_encoder.use_channel_context} "
                f"subject_adapters={self.use_subject_adapters}",
                flush=True,
            )
            self._metadata_runtime_logged = True
        if self.use_moe and self.moe_use_psd_router_features:
            tok_psd = set_moe_psd_router_features(compact_psd_bandpowers(x))
        if self.use_moe and self.moe_route_mode == "typed_capacity_domain":
            tok_meta = set_moe_faced_metadata(batch_meta)
        try:
            patch_emb = self.patch_embedding(x, mask)
            patch_emb = self.channel_context_encoder(patch_emb, batch_meta=batch_meta)
            feats = self.encoder(patch_emb)
            out = self.proj_out(feats)
            return out
        finally:
            if tok_psd is not None:
                reset_moe_psd_router_features(tok_psd)
            if tok_meta is not None:
                reset_moe_faced_metadata(tok_meta)
            reset_adapter_batch_meta(tok_adapter)

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
        'moe_load_balance': getattr(param, 'moe_load_balance', 0.0),
        'moe_domain_bias_reg': getattr(param, 'moe_domain_bias_reg', 0.0),
        'use_subject_adapters': (
            getattr(param, 'subject_adapter', False)
            or getattr(param, 'adapter_mode', 'none') != 'none'
        ),
        'adapter_num_layers': getattr(param, 'adapter_num_layers', 2),
        'adapter_rank': getattr(param, 'adapter_rank', 16),
        'adapter_cond_dim': getattr(param, 'adapter_cond_dim', 32),
        'adapter_scale': getattr(param, 'adapter_scale', 0.2),
        'use_eeg_channel_context': getattr(param, 'eeg_channel_context', False),
        'channel_context_file': getattr(param, 'channel_context_file', ''),
        'use_subject_summary': getattr(param, 'use_subject_summary', False),
        'subject_summary_handling': getattr(param, 'subject_summary_handling', 'project'),
        'metadata_debug': getattr(param, 'metadata_debug', True),
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

    loadable = {
        k: v for k, v in ckpt_state.items()
        if k in backbone_sd and backbone_sd[k].shape == v.shape
    }

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
            )

    if attnres == 'none' and not use_moe:
        strict_compatible = (len(loadable) == len(backbone_sd) == len(ckpt_state))
        if strict_compatible:
            backbone.load_state_dict(ckpt_state, strict=True)
            return set(backbone_sd.keys())

        backbone.load_state_dict(loadable, strict=False)
        print(
            "[CBraMod] Foundation checkpoint is not strict-compatible with current backbone; "
            "loaded overlapping tensors only. This is expected when new modules "
            "(e.g., EEG channel context/adapters) were added after pretraining.",
            flush=True,
        )
        return set(loadable.keys())

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
