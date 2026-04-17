"""Typed MoE FFN for CBraMod with capacity-limited routing and optional domain-aware bias."""

from __future__ import annotations

import contextvars
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MOE_ROUTE_MODES = ("typed_capacity_domain",)
MOE_ROUTER_ARCHS = ("linear", "mlp")
MOE_ROUTER_DISPATCH_MODES = ("hard_capacity", "soft")
PSD_ROUTER_DIM = 5

# Optional multiclass labels [B] for diagnostics.
_MOE_DIAG_LABELS: contextvars.ContextVar[Optional[Tensor]] = contextvars.ContextVar(
    "moe_diag_labels", default=None
)
# Optional [B, PSD_ROUTER_DIM] log1p band powers from raw EEG.
_MOE_PSD_ROUTER: contextvars.ContextVar[Optional[Tensor]] = contextvars.ContextVar(
    "moe_psd_router", default=None
)
# Optional FACED metadata dict with integer id tensors.
_MOE_FACED_METADATA: contextvars.ContextVar[Optional[Dict[str, Tensor]]] = contextvars.ContextVar(
    "moe_faced_metadata", default=None
)
# Optional compact EEG summary [B, D] for router concat.
_MOE_EEG_ROUTER_SUMMARY: contextvars.ContextVar[Optional[Tensor]] = contextvars.ContextVar(
    "moe_eeg_router_summary", default=None
)
# Optional current training epoch (1-based) used for warmup scheduling.
_MOE_TRAIN_EPOCH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "moe_train_epoch", default=0
)


def set_moe_diagnostic_labels(labels: Optional[Tensor]) -> Any:
    return _MOE_DIAG_LABELS.set(labels)


def reset_moe_diagnostic_labels(token: Any) -> None:
    _MOE_DIAG_LABELS.reset(token)


def set_moe_psd_router_features(psd: Optional[Tensor]) -> Any:
    return _MOE_PSD_ROUTER.set(psd)


def reset_moe_psd_router_features(token: Any) -> None:
    _MOE_PSD_ROUTER.reset(token)


def set_moe_faced_metadata(meta: Optional[Dict[str, Tensor]]) -> Any:
    return _MOE_FACED_METADATA.set(meta)


def reset_moe_faced_metadata(token: Any) -> None:
    _MOE_FACED_METADATA.reset(token)


def set_moe_eeg_router_summary(summary: Optional[Tensor]) -> Any:
    return _MOE_EEG_ROUTER_SUMMARY.set(summary)


def reset_moe_eeg_router_summary(token: Any) -> None:
    _MOE_EEG_ROUTER_SUMMARY.reset(token)


def set_moe_train_epoch(epoch: int) -> Any:
    return _MOE_TRAIN_EPOCH.set(int(epoch))


def reset_moe_train_epoch(token: Any) -> None:
    _MOE_TRAIN_EPOCH.reset(token)


def get_moe_train_epoch() -> int:
    return int(_MOE_TRAIN_EPOCH.get())


def compact_psd_bandpowers(x: Tensor, n_bands: int = PSD_ROUTER_DIM) -> Tensor:
    """Compact per-sample PSD: mean log1p power in contiguous rfft bins, x=[B,C,S,T]."""
    if x.dim() != 4:
        raise ValueError(f"compact_psd_bandpowers expects [B,C,S,T], got {tuple(x.shape)}")
    b, c, s, tt = x.shape
    x2 = x.reshape(b, c * s, tt)
    spec = torch.fft.rfft(x2, dim=-1, norm="ortho")
    power = spec.abs().pow(2).mean(dim=1)
    nbin = power.shape[-1]
    edges = torch.linspace(0, nbin, n_bands + 1, device=x.device).long().clamp(0, nbin)
    bands: List[Tensor] = []
    for i in range(n_bands):
        a, e = int(edges[i]), int(edges[i + 1])
        a, e = min(a, nbin), min(e, nbin)
        if e > a:
            bands.append(power[:, a:e].mean(dim=-1))
        else:
            bands.append(torch.zeros(b, device=x.device, dtype=power.dtype))
    return torch.log1p(torch.stack(bands, dim=-1))


def _make_router_head(d_in: int, num_experts: int, arch: str, hidden: int, factory_kwargs) -> nn.Module:
    if arch not in MOE_ROUTER_ARCHS:
        raise ValueError(f"router arch must be one of {MOE_ROUTER_ARCHS}, got {arch!r}")
    if arch == "linear":
        m = nn.Linear(d_in, num_experts, bias=True, **factory_kwargs)
        nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)
        return m
    seq = nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, hidden, bias=True, **factory_kwargs),
        nn.GELU(),
        nn.Linear(hidden, num_experts, bias=True, **factory_kwargs),
    )
    for mod in seq:
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.02)
            nn.init.zeros_(mod.bias)
    return seq


def _activation_from_arg(activation: Union[str, Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    if isinstance(activation, str):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    return activation


def _spatial_mean_bc_sd(x: Tensor) -> Tensor:
    return x.mean(dim=(1, 2))


def _attnres_base_features(baseline: Tensor, attnres: Tensor) -> Tensor:
    return torch.cat(
        [
            _spatial_mean_bc_sd(baseline),
            _spatial_mean_bc_sd(attnres),
            _spatial_mean_bc_sd(attnres - baseline),
        ],
        dim=-1,
    )


class ExpertFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: Callable[[Tensor], Tensor],
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TypedCapacityDomainMoEFFN(nn.Module):
    """
    Shared dense FFN + typed specialists (spatial bank + spectral bank):
    y = shared(x) + spatial_residual(x) + spectral_residual(x)

    Routing: top-1 per bank with per-expert capacity and fallback to shared-only when saturated.
    Optional domain-aware additive logit bias is zero-initialized.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_specialists: int,
        dropout: float,
        activation: Union[str, Callable[[Tensor], Tensor]],
        bias: bool = True,
        route_mode: str = "typed_capacity_domain",
        capacity_factor: float = 1.0,
        domain_bias: bool = False,
        domain_emb_dim: int = 16,
        router_arch: str = "linear",
        router_mlp_hidden: int = 128,
        use_psd_router_features: bool = False,
        use_adapter_cond_bias: bool = False,
        adapter_cond_dim: int = 0,
        use_subject_summary_router_concat: bool = False,
        subject_summary_router_dim: int = 0,
        use_eeg_summary_router_concat_spatial: bool = False,
        use_eeg_summary_router_concat_spectral: bool = False,
        use_attnres_depth_router_concat: bool = False,
        attnres_depth_router_dim: int = 0,
        eeg_summary_router_dim: int = 0,
        compact_router_warmup_epochs: int = 0,
        compact_router_gate_init: float = 1.0,
        router_concat_proj_dim: int = 16,
        linear_router_input_norm: bool = False,
        router_dispatch_mode: str = "hard_capacity",
        router_temperature: float = 1.0,
        router_entropy_coef: float = 0.0,
        router_balance_kl_coef: float = 0.0,
        router_z_loss_coef: float = 0.0,
        router_jitter_std: float = 0.0,
        router_jitter_final_std: float = 0.0,
        router_jitter_anneal_epochs: int = 0,
        router_soft_warmup_epochs: int = 0,
        uniform_dispatch_warmup_epochs: int = 0,
        shared_blend_warmup_epochs: int = 0,
        shared_blend_start: float = 1.0,
        shared_blend_end: float = 0.0,
        router_entropy_coef_spatial: Optional[float] = None,
        router_entropy_coef_spectral: Optional[float] = None,
        router_balance_kl_coef_spatial: Optional[float] = None,
        router_balance_kl_coef_spectral: Optional[float] = None,
        use_spatial_specialists: bool = True,
        use_spectral_specialists: bool = True,
                attnres_depth_router_init: str = "xavier",
                attnres_depth_router_norm_gate: bool = True,
                attnres_depth_router_norm_eps: float = 1e-6,
                attnres_depth_router_gate_init: float = 0.075,
        attnres_depth_block_separation_coef: float = 0.0,
            attnres_depth_block_separation_target_js: float = 0.03,
        expert_init_noise_std: float = 0.0,
        load_balance_coef: float = 0.0,
        domain_bias_reg_coef: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if route_mode not in MOE_ROUTE_MODES:
            raise ValueError(f"route_mode must be one of {MOE_ROUTE_MODES}, got {route_mode!r}")
        if router_arch not in MOE_ROUTER_ARCHS:
            raise ValueError(f"router_arch must be one of {MOE_ROUTER_ARCHS}, got {router_arch!r}")
        if router_dispatch_mode not in MOE_ROUTER_DISPATCH_MODES:
            raise ValueError(
                f"router_dispatch_mode must be one of {MOE_ROUTER_DISPATCH_MODES}, got {router_dispatch_mode!r}"
            )
        if num_specialists < 1:
            raise ValueError("num_specialists must be >= 1")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0")

        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn = _activation_from_arg(activation)

        self.moe_kind = "typed_capacity_domain"
        self.route_mode = route_mode
        self.num_specialists = num_specialists
        self.num_experts = num_specialists
        self.d_model = d_model
        self.capacity_factor = float(capacity_factor)
        self.domain_bias = bool(domain_bias)
        self.domain_emb_dim = int(domain_emb_dim)
        self.router_arch = router_arch
        self.router_mlp_hidden = int(router_mlp_hidden)
        self.use_psd_router_features = bool(use_psd_router_features)
        self.use_adapter_cond_bias = bool(use_adapter_cond_bias)
        self.adapter_cond_dim = int(adapter_cond_dim)
        if self.use_adapter_cond_bias and self.adapter_cond_dim <= 0:
            raise ValueError("use_adapter_cond_bias=True requires adapter_cond_dim > 0")
        self.use_subject_summary_router_concat = bool(use_subject_summary_router_concat)
        self.subject_summary_router_dim = int(subject_summary_router_dim)
        self.use_eeg_summary_router_concat_spatial = bool(use_eeg_summary_router_concat_spatial)
        self.use_eeg_summary_router_concat_spectral = bool(use_eeg_summary_router_concat_spectral)
        self.use_attnres_depth_router_concat = bool(use_attnres_depth_router_concat)
        self.attnres_depth_router_dim = int(attnres_depth_router_dim)
        self.eeg_summary_router_dim = int(eeg_summary_router_dim)
        self.compact_router_warmup_epochs = int(max(0, compact_router_warmup_epochs))
        self.compact_router_gate_init = float(compact_router_gate_init)
        self.router_concat_proj_dim = int(router_concat_proj_dim)
        self.linear_router_input_norm = bool(linear_router_input_norm)
        self.router_dispatch_mode = str(router_dispatch_mode)
        self.router_temperature = float(max(1e-4, router_temperature))
        self.router_entropy_coef = float(router_entropy_coef)
        self.router_balance_kl_coef = float(router_balance_kl_coef)
        self.router_z_loss_coef = float(router_z_loss_coef)
        self.router_jitter_std = float(max(0.0, router_jitter_std))
        self.router_jitter_final_std = float(max(0.0, router_jitter_final_std))
        self.router_jitter_anneal_epochs = int(max(0, router_jitter_anneal_epochs))
        self.router_soft_warmup_epochs = int(max(0, router_soft_warmup_epochs))
        self.uniform_dispatch_warmup_epochs = int(max(0, uniform_dispatch_warmup_epochs))
        self.shared_blend_warmup_epochs = int(max(0, shared_blend_warmup_epochs))
        self.shared_blend_start = float(shared_blend_start)
        self.shared_blend_end = float(shared_blend_end)
        if not (0.0 <= self.shared_blend_start <= 1.0 and 0.0 <= self.shared_blend_end <= 1.0):
            raise ValueError("shared_blend_start/end must be in [0, 1]")
        self.router_entropy_coef_spatial = (
            None if router_entropy_coef_spatial is None else float(router_entropy_coef_spatial)
        )
        self.router_entropy_coef_spectral = (
            None if router_entropy_coef_spectral is None else float(router_entropy_coef_spectral)
        )
        self.router_balance_kl_coef_spatial = (
            None if router_balance_kl_coef_spatial is None else float(router_balance_kl_coef_spatial)
        )
        self.router_balance_kl_coef_spectral = (
            None if router_balance_kl_coef_spectral is None else float(router_balance_kl_coef_spectral)
        )
        self.use_spatial_specialists = bool(use_spatial_specialists)
        self.use_spectral_specialists = bool(use_spectral_specialists)
        self.attnres_depth_router_init = str(attnres_depth_router_init).strip().lower()
        self.attnres_depth_router_norm_gate = bool(attnres_depth_router_norm_gate)
        self.attnres_depth_router_norm_eps = float(attnres_depth_router_norm_eps)
        self.attnres_depth_router_gate_init = float(attnres_depth_router_gate_init)
        self.attnres_depth_block_separation_coef = float(attnres_depth_block_separation_coef)
        self.attnres_depth_block_separation_target_js = float(attnres_depth_block_separation_target_js)
        if self.attnres_depth_router_init not in {"zero", "xavier"}:
            raise ValueError("attnres_depth_router_init must be one of {'zero','xavier'}")
        if self.attnres_depth_router_norm_eps <= 0.0:
            raise ValueError("attnres_depth_router_norm_eps must be > 0")
        if self.attnres_depth_router_gate_init <= 0.0:
            raise ValueError("attnres_depth_router_gate_init must be > 0")
        if self.attnres_depth_block_separation_coef < 0.0:
            raise ValueError("attnres_depth_block_separation_coef must be >= 0")
        if self.attnres_depth_block_separation_target_js < 0.0:
            raise ValueError("attnres_depth_block_separation_target_js must be >= 0")
        if not self.use_spatial_specialists and not self.use_spectral_specialists:
            raise ValueError("At least one specialist bank must be enabled.")
        self.expert_init_noise_std = float(max(0.0, expert_init_noise_std))
        if self.use_subject_summary_router_concat and self.subject_summary_router_dim <= 0:
            raise ValueError("use_subject_summary_router_concat=True requires subject_summary_router_dim > 0")
        if (
            (self.use_eeg_summary_router_concat_spatial or self.use_eeg_summary_router_concat_spectral)
            and self.eeg_summary_router_dim <= 0
        ):
            raise ValueError("EEG router concat requires eeg_summary_router_dim > 0")
        if self.compact_router_gate_init <= 0:
            raise ValueError("compact_router_gate_init must be > 0")
        if self.use_attnres_depth_router_concat and self.attnres_depth_router_dim <= 0:
            raise ValueError("use_attnres_depth_router_concat=True requires attnres_depth_router_dim > 0")
        self.load_balance_coef = float(load_balance_coef)
        self.domain_bias_reg_coef = float(domain_bias_reg_coef)

        self.shared = ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)

        self.subject_summary_router_proj_spatial = None
        self.subject_summary_router_proj_spectral = None
        self.eeg_summary_router_proj_spatial = None
        self.eeg_summary_router_proj_spectral = None
        self.eeg_summary_router_gate_spatial = None
        self.eeg_summary_router_gate_spectral = None
        self.attnres_depth_router_proj_spatial = None
        self.attnres_depth_router_proj_spectral = None
        self.attnres_depth_router_gate_spatial = None
        self.attnres_depth_router_gate_spectral = None
        spatial_extra = 0
        spectral_extra = 0
        if self.use_subject_summary_router_concat:
            self.subject_summary_router_proj_spatial = nn.Linear(
                self.subject_summary_router_dim,
                self.router_concat_proj_dim,
                bias=True,
                **factory_kwargs,
            )
            self.subject_summary_router_proj_spectral = nn.Linear(
                self.subject_summary_router_dim,
                self.router_concat_proj_dim,
                bias=True,
                **factory_kwargs,
            )
            spatial_extra += self.router_concat_proj_dim
            spectral_extra += self.router_concat_proj_dim
        if self.use_eeg_summary_router_concat_spatial:
            self.eeg_summary_router_proj_spatial = nn.Linear(
                self.eeg_summary_router_dim,
                self.router_concat_proj_dim,
                bias=True,
                **factory_kwargs,
            )
            gate_init = max(1e-8, self.compact_router_gate_init)
            gate_raw_init = math.log(math.expm1(gate_init))
            self.eeg_summary_router_gate_spatial = nn.Parameter(
                torch.tensor(gate_raw_init, **factory_kwargs)
            )
            spatial_extra += self.router_concat_proj_dim
        if self.use_eeg_summary_router_concat_spectral:
            self.eeg_summary_router_proj_spectral = nn.Linear(
                self.eeg_summary_router_dim,
                self.router_concat_proj_dim,
                bias=True,
                **factory_kwargs,
            )
            gate_init = max(1e-8, self.compact_router_gate_init)
            gate_raw_init = math.log(math.expm1(gate_init))
            self.eeg_summary_router_gate_spectral = nn.Parameter(
                torch.tensor(gate_raw_init, **factory_kwargs)
            )
            spectral_extra += self.router_concat_proj_dim
        if self.use_attnres_depth_router_concat:
            self.attnres_depth_router_proj_spatial = nn.Linear(
                self.attnres_depth_router_dim,
                self.router_concat_proj_dim,
                bias=True,
                **factory_kwargs,
            )
            self.attnres_depth_router_proj_spectral = nn.Linear(
                self.attnres_depth_router_dim,
                self.router_concat_proj_dim,
                bias=True,
                **factory_kwargs,
            )
            gate_init = max(1e-8, self.attnres_depth_router_gate_init)
            gate_raw_init = math.log(math.expm1(gate_init))
            self.attnres_depth_router_gate_spatial = nn.Parameter(
                torch.tensor(gate_raw_init, **factory_kwargs)
            )
            self.attnres_depth_router_gate_spectral = nn.Parameter(
                torch.tensor(gate_raw_init, **factory_kwargs)
            )
            spatial_extra += self.router_concat_proj_dim
            spectral_extra += self.router_concat_proj_dim

        spatial_in = 3 * d_model + spatial_extra
        spectral_in = 3 * d_model + (PSD_ROUTER_DIM if self.use_psd_router_features else 0) + spectral_extra
        self.spatial_router = _make_router_head(
            spatial_in, num_specialists, router_arch, self.router_mlp_hidden, factory_kwargs
        )
        self.spectral_router = _make_router_head(
            spectral_in, num_specialists, router_arch, self.router_mlp_hidden, factory_kwargs
        )
        self.spatial_router_input_norm = None
        self.spectral_router_input_norm = None
        if self.router_arch == "linear" and self.linear_router_input_norm:
            self.spatial_router_input_norm = nn.LayerNorm(spatial_in, **factory_kwargs)
            self.spectral_router_input_norm = nn.LayerNorm(spectral_in, **factory_kwargs)
        self.adapter_cond_proj_spatial = None
        self.adapter_cond_proj_spectral = None
        if self.use_adapter_cond_bias:
            self.adapter_cond_proj_spatial = nn.Linear(self.adapter_cond_dim, num_specialists, bias=True, **factory_kwargs)
            self.adapter_cond_proj_spectral = nn.Linear(self.adapter_cond_dim, num_specialists, bias=True, **factory_kwargs)

        self.spatial_specialists = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_specialists)
        )
        self.spectral_specialists = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_specialists)
        )

        # Metadata ids include 0 as unknown.
        n_vocab = 64
        self.domain_emb_cohort = nn.Embedding(n_vocab, self.domain_emb_dim, **factory_kwargs)
        self.domain_emb_sample_rate = nn.Embedding(n_vocab, self.domain_emb_dim, **factory_kwargs)
        self.domain_emb_age = nn.Embedding(n_vocab, self.domain_emb_dim, **factory_kwargs)
        self.domain_emb_segment = nn.Embedding(n_vocab, self.domain_emb_dim, **factory_kwargs)
        self.domain_proj_spatial = nn.Linear(self.domain_emb_dim, num_specialists, bias=True, **factory_kwargs)
        self.domain_proj_spectral = nn.Linear(self.domain_emb_dim, num_specialists, bias=True, **factory_kwargs)

        self._zero_specialist_output_weights()
        self._zero_domain_bias_path()
        self._zero_adapter_cond_path()
        self._init_attnres_depth_router_path()

        z = torch.zeros((), device=device, dtype=torch.float32)
        self._last_lb_loss = z
        self._last_overflow_penalty = z
        self._last_domain_bias_reg = z
        self._last_entropy_reg = z
        self._last_balance_kl = z
        self._last_z_loss = z
        self._last_aux_loss = z

        self.last_diagnostics: Optional[Dict[str, Any]] = None
        self._routing_export_cache: Dict[str, Tensor] = {}

    def _shared_blend_value(self, cur_epoch: int) -> float:
        if self.shared_blend_warmup_epochs <= 0:
            return self.shared_blend_end
        if cur_epoch <= 0:
            return self.shared_blend_start
        if cur_epoch >= self.shared_blend_warmup_epochs:
            return self.shared_blend_end
        denom = float(max(1, self.shared_blend_warmup_epochs - 1))
        t = float(cur_epoch - 1) / denom
        return (1.0 - t) * self.shared_blend_start + t * self.shared_blend_end

    def apply_expert_init_noise_(self, std: float) -> None:
        std = float(max(0.0, std))
        if std <= 0.0:
            return
        for bank in (self.spatial_specialists, self.spectral_specialists):
            for spec in bank:
                spec.linear1.weight.add_(torch.randn_like(spec.linear1.weight) * std)
                if spec.linear1.bias is not None:
                    spec.linear1.bias.add_(torch.randn_like(spec.linear1.bias) * std)

    def _zero_specialist_output_weights(self) -> None:
        for bank in (self.spatial_specialists, self.spectral_specialists):
            for s in bank:
                nn.init.zeros_(s.linear2.weight)
                if s.linear2.bias is not None:
                    nn.init.zeros_(s.linear2.bias)

    def _zero_domain_bias_path(self) -> None:
        for emb in (
            self.domain_emb_cohort,
            self.domain_emb_sample_rate,
            self.domain_emb_age,
            self.domain_emb_segment,
        ):
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.domain_proj_spatial.weight)
        nn.init.zeros_(self.domain_proj_spatial.bias)
        nn.init.zeros_(self.domain_proj_spectral.weight)
        nn.init.zeros_(self.domain_proj_spectral.bias)

    def _zero_adapter_cond_path(self) -> None:
        if self.adapter_cond_proj_spatial is not None:
            nn.init.zeros_(self.adapter_cond_proj_spatial.weight)
            nn.init.zeros_(self.adapter_cond_proj_spatial.bias)
        if self.adapter_cond_proj_spectral is not None:
            nn.init.zeros_(self.adapter_cond_proj_spectral.weight)
            nn.init.zeros_(self.adapter_cond_proj_spectral.bias)

    def _init_attnres_depth_router_path(self) -> None:
        if self.attnres_depth_router_proj_spatial is not None:
            if self.attnres_depth_router_init == "zero":
                nn.init.zeros_(self.attnres_depth_router_proj_spatial.weight)
            else:
                nn.init.xavier_uniform_(self.attnres_depth_router_proj_spatial.weight)
            nn.init.zeros_(self.attnres_depth_router_proj_spatial.bias)
        if self.attnres_depth_router_proj_spectral is not None:
            if self.attnres_depth_router_init == "zero":
                nn.init.zeros_(self.attnres_depth_router_proj_spectral.weight)
            else:
                nn.init.xavier_uniform_(self.attnres_depth_router_proj_spectral.weight)
            nn.init.zeros_(self.attnres_depth_router_proj_spectral.bias)

    def _adapter_cond_logit_bias(self, router_context: Optional[Dict[str, Tensor]], batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        z = torch.zeros(batch_size, self.num_specialists, device=device)
        if not self.use_adapter_cond_bias:
            return z, z
        if router_context is None:
            return z, z
        cond = router_context.get("adapter_cond")
        if cond is None:
            return z, z
        if cond.dim() != 2 or cond.shape[0] != batch_size or cond.shape[1] != self.adapter_cond_dim:
            raise ValueError(
                "router_context['adapter_cond'] must be [B, adapter_cond_dim], got "
                f"{tuple(cond.shape)} with adapter_cond_dim={self.adapter_cond_dim}"
            )
        cond = cond.to(device=device)
        return self.adapter_cond_proj_spatial(cond), self.adapter_cond_proj_spectral(cond)

    def _subject_summary_router_features(
        self,
        router_context: Optional[Dict[str, Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not self.use_subject_summary_router_concat:
            return None, None
        z = torch.zeros(batch_size, self.subject_summary_router_dim, device=device, dtype=dtype)
        ssum = z
        if router_context is not None and torch.is_tensor(router_context.get("subject_summary")):
            raw = router_context.get("subject_summary")
            if raw.dim() == 1:
                raw = raw.unsqueeze(0)
            if raw.dim() == 2 and raw.shape[0] == batch_size and raw.shape[1] == self.subject_summary_router_dim:
                ssum = raw.to(device=device, dtype=dtype)
        return self.subject_summary_router_proj_spatial(ssum), self.subject_summary_router_proj_spectral(ssum)

    def _eeg_summary_router_features(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not (self.use_eeg_summary_router_concat_spatial or self.use_eeg_summary_router_concat_spectral):
            return None, None
        z = torch.zeros(batch_size, self.eeg_summary_router_dim, device=device, dtype=dtype)
        eeg = z
        raw = _MOE_EEG_ROUTER_SUMMARY.get()
        if torch.is_tensor(raw) and raw.dim() == 2 and raw.shape[0] == batch_size and raw.shape[1] == self.eeg_summary_router_dim:
            eeg = raw.to(device=device, dtype=dtype)
        sp = self.eeg_summary_router_proj_spatial(eeg) if self.use_eeg_summary_router_concat_spatial else None
        sc = self.eeg_summary_router_proj_spectral(eeg) if self.use_eeg_summary_router_concat_spectral else None
        return sp, sc

    def _attnres_depth_router_features(
        self,
        router_context: Optional[Dict[str, Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not self.use_attnres_depth_router_concat:
            return None, None
        z = torch.zeros(batch_size, self.attnres_depth_router_dim, device=device, dtype=dtype)

        def _summary_rms_norm(x: Tensor) -> Tensor:
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.attnres_depth_router_norm_eps).sqrt()
            return x / rms

        def _coerce_summary(raw: Optional[Tensor]) -> Optional[Tensor]:
            if not torch.is_tensor(raw):
                return None
            if raw.dim() == 1:
                raw = raw.unsqueeze(0)
            if raw.dim() != 2:
                return None
            if raw.shape[0] != batch_size or raw.shape[1] != self.attnres_depth_router_dim:
                return None
            return raw.to(device=device, dtype=dtype)

        shared_summary = _coerce_summary(None if router_context is None else router_context.get("attnres_depth_summary"))
        spatial_summary = _coerce_summary(None if router_context is None else router_context.get("attnres_depth_summary_spatial"))
        spectral_summary = _coerce_summary(None if router_context is None else router_context.get("attnres_depth_summary_spectral"))

        # Prefer bank-specific summaries; fall back to shared summary for backward compatibility.
        ds_sp = spatial_summary if spatial_summary is not None else (shared_summary if shared_summary is not None else z)
        ds_sc = spectral_summary if spectral_summary is not None else (shared_summary if shared_summary is not None else z)

        ctx_mode = ""
        summary_mode = ""
        if router_context is not None and isinstance(router_context.get("attnres_depth_context_mode"), str):
            ctx_mode = str(router_context.get("attnres_depth_context_mode")).strip().lower()
        if router_context is not None and isinstance(router_context.get("attnres_depth_summary_mode"), str):
            summary_mode = str(router_context.get("attnres_depth_summary_mode")).strip().lower()
        dual_query_mode_active = (
            ctx_mode == "dual_query_block_typed_proj"
            or summary_mode == "dual_query_block_typed_learned"
        )
        norm_gate_active = (
            self.attnres_depth_router_norm_gate
            and dual_query_mode_active
            and self.attnres_depth_router_gate_spatial is not None
            and self.attnres_depth_router_gate_spectral is not None
        )
        gate_sp = None
        gate_sc = None
        normed_sp_norm = None
        normed_sc_norm = None
        if self.attnres_depth_router_gate_spatial is not None and self.attnres_depth_router_gate_spectral is not None:
            gate_sp_t = F.softplus(self.attnres_depth_router_gate_spatial).to(device=device, dtype=dtype)
            gate_sc_t = F.softplus(self.attnres_depth_router_gate_spectral).to(device=device, dtype=dtype)
            gate_sp = float(gate_sp_t.detach().float().item())
            gate_sc = float(gate_sc_t.detach().float().item())
            if norm_gate_active:
                ds_sp = gate_sp_t * _summary_rms_norm(ds_sp)
                ds_sc = gate_sc_t * _summary_rms_norm(ds_sc)
                normed_sp_norm = float(ds_sp.detach().float().norm(dim=-1).mean().item())
                normed_sc_norm = float(ds_sc.detach().float().norm(dim=-1).mean().item())

        if router_context is not None:
            router_context["attnres_depth_router_norm_gate_active"] = bool(norm_gate_active)
            router_context["attnres_depth_router_norm_eps"] = float(self.attnres_depth_router_norm_eps)
            router_context["attnres_depth_router_gate_spatial"] = gate_sp
            router_context["attnres_depth_router_gate_spectral"] = gate_sc
            router_context["attnres_depth_router_normed_spatial_norm"] = normed_sp_norm
            router_context["attnres_depth_router_normed_spectral_norm"] = normed_sc_norm

        return self.attnres_depth_router_proj_spatial(ds_sp), self.attnres_depth_router_proj_spectral(ds_sc)

    def _domain_logit_bias(self, batch_size: int, bank: str, device: torch.device) -> Tensor:
        if not self.domain_bias:
            return torch.zeros(batch_size, self.num_specialists, device=device)
        meta = _MOE_FACED_METADATA.get()
        if meta is None:
            return torch.zeros(batch_size, self.num_specialists, device=device)

        def _id(name: str) -> Tensor:
            t = meta.get(name)
            if t is None:
                return torch.zeros(batch_size, device=device, dtype=torch.long)
            t = t.to(device=device, dtype=torch.long)
            if t.dim() != 1:
                t = t.view(-1)
            if t.numel() != batch_size:
                return torch.zeros(batch_size, device=device, dtype=torch.long)
            return t.clamp_min(0).clamp_max(63)

        emb = (
            self.domain_emb_cohort(_id("cohort_id"))
            + self.domain_emb_sample_rate(_id("sample_rate_group_id"))
            + self.domain_emb_age(_id("age_bucket_id"))
            + self.domain_emb_segment(_id("segment_bucket_id"))
        )
        if bank == "spatial":
            return self.domain_proj_spatial(emb)
        return self.domain_proj_spectral(emb)

    @staticmethod
    def _capacity_assign_top1(
        logits: Tensor,
        capacity: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Use raw argmax as primary route; reroute only overflow samples."""
        b, e = logits.shape
        raw_top1 = logits.argmax(dim=-1)
        assigned = raw_top1.clone()
        if b == 0:
            z = torch.zeros(0, device=logits.device, dtype=torch.bool)
            c = torch.zeros(e, device=logits.device, dtype=torch.long)
            return assigned, z, z, c, c

        counts_pre = torch.bincount(raw_top1, minlength=e).to(dtype=torch.long)
        counts_work = counts_pre.clone()
        ranked = logits.argsort(dim=1, descending=True)
        rerouted = torch.zeros(b, device=logits.device, dtype=torch.bool)
        fallback = torch.zeros(b, device=logits.device, dtype=torch.bool)

        for ex in range(e):
            excess = int(max(0, int(counts_work[ex].item()) - capacity))
            if excess <= 0:
                continue
            idx_e = torch.nonzero(raw_top1 == ex, as_tuple=False).flatten()
            if idx_e.numel() == 0:
                continue
            conf_e = logits[idx_e, ex]
            # Keep the most confident assignments on the overloaded expert.
            drop_order = conf_e.argsort(descending=False)
            overflow_idx = idx_e[drop_order[:excess]]
            for bi in overflow_idx.tolist():
                moved = False
                for cand in ranked[bi].tolist():
                    if cand == ex:
                        continue
                    if int(counts_work[cand].item()) < capacity:
                        assigned[bi] = cand
                        counts_work[ex] -= 1
                        counts_work[cand] += 1
                        rerouted[bi] = True
                        moved = True
                        break
                if not moved:
                    assigned[bi] = -1
                    counts_work[ex] -= 1
                    rerouted[bi] = True
                    fallback[bi] = True

        valid = assigned >= 0
        if valid.any():
            counts_post = torch.bincount(assigned[valid], minlength=e).to(dtype=torch.long)
        else:
            counts_post = torch.zeros(e, device=logits.device, dtype=torch.long)
        return assigned, rerouted, fallback, counts_pre, counts_post

    @staticmethod
    def _sample_entropy(probs: Tensor) -> float:
        p = probs.clamp_min(1e-10)
        return float((-(p * p.log()).sum(dim=-1)).mean().item())

    @staticmethod
    def _hist_entropy(counts: Tensor) -> float:
        total = float(counts.sum().item())
        if total <= 0:
            return 0.0
        p = counts.float() / total
        p = p.clamp_min(1e-10)
        return float((-(p * p.log()).sum()).item())

    @staticmethod
    def _mean_js_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
        m = 0.5 * (p + q)
        return 0.5 * (p * (p.log() - m.log())).sum(dim=-1).mean() + 0.5 * (q * (q.log() - m.log())).sum(dim=-1).mean()

    def _bank_load_balance(self, probs: Tensor, raw_top1: Tensor) -> Tensor:
        if probs.numel() == 0:
            return probs.new_zeros(())
        e = probs.size(1)
        onehot = F.one_hot(raw_top1, num_classes=e).to(dtype=probs.dtype)
        fi = onehot.mean(dim=0).detach()
        pi = probs.mean(dim=0)
        return e * (fi * pi).sum()

    @staticmethod
    def _batch_uniform_kl(probs: Tensor) -> Tensor:
        if probs.numel() == 0:
            return probs.new_zeros(())
        e = probs.size(1)
        pi = probs.mean(dim=0).clamp_min(1e-10)
        log_u = math.log(1.0 / float(max(e, 1)))
        return (pi * (pi.log() - log_u)).sum()

    def _bank_residual(self, x_flat: Tensor, sample_assign: Tensor, batch_idx: Tensor, bank: nn.ModuleList) -> Tensor:
        res = torch.zeros_like(x_flat)
        per_token = sample_assign[batch_idx]
        for e in range(self.num_specialists):
            m = per_token == e
            if m.any():
                res[m] = bank[e](x_flat[m])
        return res

    def _bank_residual_soft(self, x_flat: Tensor, sample_probs: Tensor, batch_idx: Tensor, bank: nn.ModuleList) -> Tensor:
        res = torch.zeros_like(x_flat)
        for e in range(self.num_specialists):
            out_e = bank[e](x_flat)
            w = sample_probs[:, e].index_select(0, batch_idx).unsqueeze(-1)
            res = res + w * out_e
        return res

    def _effective_router_jitter_std(self, cur_epoch: int) -> float:
        start = float(max(0.0, self.router_jitter_std))
        final = float(max(0.0, self.router_jitter_final_std))
        n = int(max(0, self.router_jitter_anneal_epochs))
        if n <= 0:
            return start
        if cur_epoch <= 0:
            return start
        if n == 1:
            return final
        t = float(max(0.0, min(1.0, (float(cur_epoch) - 1.0) / float(n - 1))))
        return (1.0 - t) * start + t * final

    def forward(self, x: Tensor, router_context: Optional[Dict[str, Tensor]] = None) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        h_shared = self.shared(x_flat)
        cur_epoch = int(_MOE_TRAIN_EPOCH.get())

        if router_context is None or router_context.get("baseline") is None or router_context.get("attnres") is None:
            raise ValueError(
                "typed_capacity_domain requires router_context with 'baseline' and 'attnres' [B,C,S,D]."
            )
        baseline = router_context["baseline"]
        attnres = router_context["attnres"]
        if baseline.shape != x.shape or attnres.shape != x.shape:
            raise ValueError("typed_capacity_domain: baseline/attnres must match [B,C,S,D]")

        b, c, s, _ = x.shape
        t = c * s
        batch_idx = torch.arange(b, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, t).reshape(-1)

        depth_summary_mean = None
        depth_summary_std = None
        depth_summary_mode = "NA"
        depth_context_mode = "compact_shared"
        depth_block_count = 0
        depth_block_mean = None
        depth_block_std = None
        depth_block_summary_norms = None
        depth_block_layer_counts = None
        depth_block_pooling = None
        depth_family_mode = None
        depth_shared_context_norm = None
        depth_spatial_context_norm = None
        depth_spectral_context_norm = None
        depth_block_layer_counts_pre_attn = None
        depth_block_layer_counts_pre_mlp = None
        depth_block_peak_weight_pre_attn = None
        depth_block_peak_weight_pre_mlp = None
        depth_block_peak_weight_spatial_pre_attn = None
        depth_block_peak_weight_spatial_pre_mlp = None
        depth_block_peak_weight_spectral_pre_attn = None
        depth_block_peak_weight_spectral_pre_mlp = None
        depth_block_weight_dist_spatial = None
        depth_block_weight_dist_spectral = None
        depth_block_weight_dist_cosine = None
        depth_block_weight_dist_js_div = None
        depth_block_weight_dist_spatial_raw: Optional[Tensor] = None
        depth_block_weight_dist_spectral_raw: Optional[Tensor] = None
        depth_probe_mlp_for_router = False
        depth_proj_spatial_norm = None
        depth_proj_spectral_norm = None
        depth_proj_cosine = None
        depth_proj_l2 = None
        depth_summary_grad_mode = "detached"
        depth_summary_grad_active = False
        depth_summary_detached = True
        depth_summary_cur_epoch = int(cur_epoch)
        depth_summary_unfreeze_epoch = 1
        depth_summary_unfreeze_reached = False
        depth_summary_spatial_norm = None
        depth_summary_spectral_norm = None
        depth_router_norm_gate_active = False
        depth_router_norm_eps = None
        depth_router_gate_spatial = None
        depth_router_gate_spectral = None
        depth_router_normed_spatial_norm = None
        depth_router_normed_spectral_norm = None
        if self.use_attnres_depth_router_concat and router_context is not None:
            ds = router_context.get("attnres_depth_summary")
            ds_sp = router_context.get("attnres_depth_summary_spatial")
            ds_sc = router_context.get("attnres_depth_summary_spectral")
            if not torch.is_tensor(ds):
                ds = ds_sp if torch.is_tensor(ds_sp) else ds_sc
            if torch.is_tensor(ds):
                dsf = ds.detach().float()
                depth_summary_mean = float(dsf.mean().item())
                depth_summary_std = float(dsf.std(unbiased=False).item())
                if torch.is_tensor(ds_sp):
                    dsf_sp = ds_sp.detach().float()
                    depth_summary_spatial_norm = float(dsf_sp.norm(dim=-1).mean().item())
                if torch.is_tensor(ds_sc):
                    dsf_sc = ds_sc.detach().float()
                    depth_summary_spectral_norm = float(dsf_sc.norm(dim=-1).mean().item())
                if isinstance(router_context.get("attnres_depth_summary_mode"), str):
                    depth_summary_mode = str(router_context.get("attnres_depth_summary_mode"))
                if isinstance(router_context.get("attnres_depth_context_mode"), str):
                    depth_context_mode = str(router_context.get("attnres_depth_context_mode"))
                if "attnres_depth_block_count" in router_context:
                    depth_block_count = int(router_context.get("attnres_depth_block_count"))
                if "attnres_depth_block_mean" in router_context:
                    depth_block_mean = list(router_context.get("attnres_depth_block_mean"))
                if "attnres_depth_block_std" in router_context:
                    depth_block_std = list(router_context.get("attnres_depth_block_std"))
                if "attnres_depth_block_summary_norms" in router_context:
                    depth_block_summary_norms = list(router_context.get("attnres_depth_block_summary_norms"))
                if "attnres_depth_block_layer_counts" in router_context:
                    depth_block_layer_counts = list(router_context.get("attnres_depth_block_layer_counts"))
                if "attnres_depth_block_pooling" in router_context:
                    depth_block_pooling = str(router_context.get("attnres_depth_block_pooling"))
                if "attnres_depth_family_mode" in router_context:
                    depth_family_mode = str(router_context.get("attnres_depth_family_mode"))
                if "attnres_depth_shared_context_norm" in router_context:
                    depth_shared_context_norm = float(router_context.get("attnres_depth_shared_context_norm"))
                if "attnres_depth_spatial_context_norm" in router_context:
                    depth_spatial_context_norm = float(router_context.get("attnres_depth_spatial_context_norm"))
                if "attnres_depth_spectral_context_norm" in router_context:
                    depth_spectral_context_norm = float(router_context.get("attnres_depth_spectral_context_norm"))
                if "attnres_depth_block_layer_counts_pre_attn" in router_context:
                    depth_block_layer_counts_pre_attn = list(router_context.get("attnres_depth_block_layer_counts_pre_attn"))
                if "attnres_depth_block_layer_counts_pre_mlp" in router_context:
                    depth_block_layer_counts_pre_mlp = list(router_context.get("attnres_depth_block_layer_counts_pre_mlp"))
                if "attnres_depth_block_peak_weight_pre_attn" in router_context:
                    depth_block_peak_weight_pre_attn = list(router_context.get("attnres_depth_block_peak_weight_pre_attn"))
                if "attnres_depth_block_peak_weight_pre_mlp" in router_context:
                    depth_block_peak_weight_pre_mlp = list(router_context.get("attnres_depth_block_peak_weight_pre_mlp"))
                if "attnres_depth_block_peak_weight_spatial_pre_attn" in router_context:
                    depth_block_peak_weight_spatial_pre_attn = list(router_context.get("attnres_depth_block_peak_weight_spatial_pre_attn"))
                if "attnres_depth_block_peak_weight_spatial_pre_mlp" in router_context:
                    depth_block_peak_weight_spatial_pre_mlp = list(router_context.get("attnres_depth_block_peak_weight_spatial_pre_mlp"))
                if "attnres_depth_block_peak_weight_spectral_pre_attn" in router_context:
                    depth_block_peak_weight_spectral_pre_attn = list(router_context.get("attnres_depth_block_peak_weight_spectral_pre_attn"))
                if "attnres_depth_block_peak_weight_spectral_pre_mlp" in router_context:
                    depth_block_peak_weight_spectral_pre_mlp = list(router_context.get("attnres_depth_block_peak_weight_spectral_pre_mlp"))
                p_dist_mean = None
                q_dist_mean = None
                if torch.is_tensor(router_context.get("attnres_depth_block_weight_dist_spatial")):
                    p_dist_raw = router_context.get("attnres_depth_block_weight_dist_spatial")
                    if p_dist_raw.dim() == 1:
                        p_dist_raw = p_dist_raw.unsqueeze(0)
                    if p_dist_raw.dim() == 2 and p_dist_raw.shape[0] > 0:
                        depth_block_weight_dist_spatial_raw = p_dist_raw.to(device=x.device, dtype=x_flat.dtype)
                    p_dist = p_dist_raw.detach().float()
                    if p_dist.dim() == 1:
                        p_dist = p_dist.unsqueeze(0)
                    if p_dist.dim() == 2 and p_dist.shape[0] > 0:
                        p_dist_mean = p_dist.mean(dim=0)
                        p_dist_mean = p_dist_mean / p_dist_mean.sum().clamp_min(1e-8)
                        depth_block_weight_dist_spatial = p_dist_mean.cpu().tolist()
                if torch.is_tensor(router_context.get("attnres_depth_block_weight_dist_spectral")):
                    q_dist_raw = router_context.get("attnres_depth_block_weight_dist_spectral")
                    if q_dist_raw.dim() == 1:
                        q_dist_raw = q_dist_raw.unsqueeze(0)
                    if q_dist_raw.dim() == 2 and q_dist_raw.shape[0] > 0:
                        depth_block_weight_dist_spectral_raw = q_dist_raw.to(device=x.device, dtype=x_flat.dtype)
                    q_dist = q_dist_raw.detach().float()
                    if q_dist.dim() == 1:
                        q_dist = q_dist.unsqueeze(0)
                    if q_dist.dim() == 2 and q_dist.shape[0] > 0:
                        q_dist_mean = q_dist.mean(dim=0)
                        q_dist_mean = q_dist_mean / q_dist_mean.sum().clamp_min(1e-8)
                        depth_block_weight_dist_spectral = q_dist_mean.cpu().tolist()
                if p_dist_mean is not None and q_dist_mean is not None:
                    depth_block_weight_dist_cosine = float(
                        F.cosine_similarity(p_dist_mean.unsqueeze(0), q_dist_mean.unsqueeze(0), dim=-1).item()
                    )
                    pm = p_dist_mean.clamp_min(1e-8)
                    qm = q_dist_mean.clamp_min(1e-8)
                    mm = 0.5 * (pm + qm)
                    depth_block_weight_dist_js_div = float(
                        0.5 * (pm * (pm.log() - mm.log())).sum().item()
                        + 0.5 * (qm * (qm.log() - mm.log())).sum().item()
                    )
                if "attnres_depth_probe_mlp_for_router" in router_context:
                    depth_probe_mlp_for_router = bool(router_context.get("attnres_depth_probe_mlp_for_router"))
                if isinstance(router_context.get("attnres_depth_summary_grad_mode"), str):
                    depth_summary_grad_mode = str(router_context.get("attnres_depth_summary_grad_mode"))
                if "attnres_depth_summary_grad_active" in router_context:
                    depth_summary_grad_active = bool(router_context.get("attnres_depth_summary_grad_active"))
                if "attnres_depth_summary_detached" in router_context:
                    depth_summary_detached = bool(router_context.get("attnres_depth_summary_detached"))
                if "attnres_depth_summary_cur_epoch" in router_context:
                    depth_summary_cur_epoch = int(router_context.get("attnres_depth_summary_cur_epoch"))
                if "attnres_depth_summary_unfreeze_epoch" in router_context:
                    depth_summary_unfreeze_epoch = int(router_context.get("attnres_depth_summary_unfreeze_epoch"))
                depth_summary_unfreeze_reached = depth_summary_cur_epoch >= depth_summary_unfreeze_epoch
                if "attnres_depth_router_norm_gate_active" in router_context:
                    depth_router_norm_gate_active = bool(router_context.get("attnres_depth_router_norm_gate_active"))
                if "attnres_depth_router_norm_eps" in router_context:
                    depth_router_norm_eps = float(router_context.get("attnres_depth_router_norm_eps"))
                if "attnres_depth_router_gate_spatial" in router_context:
                    v = router_context.get("attnres_depth_router_gate_spatial")
                    depth_router_gate_spatial = None if v is None else float(v)
                if "attnres_depth_router_gate_spectral" in router_context:
                    v = router_context.get("attnres_depth_router_gate_spectral")
                    depth_router_gate_spectral = None if v is None else float(v)
                if "attnres_depth_router_normed_spatial_norm" in router_context:
                    v = router_context.get("attnres_depth_router_normed_spatial_norm")
                    depth_router_normed_spatial_norm = None if v is None else float(v)
                if "attnres_depth_router_normed_spectral_norm" in router_context:
                    v = router_context.get("attnres_depth_router_normed_spectral_norm")
                    depth_router_normed_spectral_norm = None if v is None else float(v)

        base_feat = _attnres_base_features(baseline, attnres)
        subj_sp, subj_sc = self._subject_summary_router_features(router_context, b, x.device, base_feat.dtype)
        eeg_sp, eeg_sc = self._eeg_summary_router_features(b, x.device, base_feat.dtype)
        depth_sp, depth_sc = self._attnres_depth_router_features(router_context, b, x.device, base_feat.dtype)
        compact_path_scale = 1.0
        compact_router_gate_spatial = None
        compact_router_gate_spectral = None
        if (self.use_eeg_summary_router_concat_spatial or self.use_eeg_summary_router_concat_spectral):
            if self.compact_router_warmup_epochs > 0:
                if cur_epoch <= 0:
                    compact_path_scale = 0.0
                else:
                    compact_path_scale = min(1.0, float(cur_epoch) / float(self.compact_router_warmup_epochs))
            if eeg_sp is not None and self.eeg_summary_router_gate_spatial is not None:
                compact_router_gate_spatial = float(F.softplus(self.eeg_summary_router_gate_spatial).detach().item())
                eeg_sp = eeg_sp * (compact_path_scale * compact_router_gate_spatial)
            if eeg_sc is not None and self.eeg_summary_router_gate_spectral is not None:
                compact_router_gate_spectral = float(F.softplus(self.eeg_summary_router_gate_spectral).detach().item())
                eeg_sc = eeg_sc * (compact_path_scale * compact_router_gate_spectral)
        depth_path_scale = 1.0
        if self.use_attnres_depth_router_concat and self.router_soft_warmup_epochs > 0:
            if cur_epoch <= 0:
                depth_path_scale = 0.0
            else:
                depth_path_scale = min(1.0, float(cur_epoch) / float(self.router_soft_warmup_epochs))
            if depth_sp is not None:
                depth_sp = depth_sp * depth_path_scale
            if depth_sc is not None:
                depth_sc = depth_sc * depth_path_scale

        # Read post-transform gate diagnostics after depth features are built.
        if self.use_attnres_depth_router_concat and router_context is not None:
            if "attnres_depth_router_norm_gate_active" in router_context:
                depth_router_norm_gate_active = bool(router_context.get("attnres_depth_router_norm_gate_active"))
            if "attnres_depth_router_norm_eps" in router_context:
                depth_router_norm_eps = float(router_context.get("attnres_depth_router_norm_eps"))
            if "attnres_depth_router_gate_spatial" in router_context:
                v = router_context.get("attnres_depth_router_gate_spatial")
                depth_router_gate_spatial = None if v is None else float(v)
            if "attnres_depth_router_gate_spectral" in router_context:
                v = router_context.get("attnres_depth_router_gate_spectral")
                depth_router_gate_spectral = None if v is None else float(v)
            if "attnres_depth_router_normed_spatial_norm" in router_context:
                v = router_context.get("attnres_depth_router_normed_spatial_norm")
                depth_router_normed_spatial_norm = None if v is None else float(v)
            if "attnres_depth_router_normed_spectral_norm" in router_context:
                v = router_context.get("attnres_depth_router_normed_spectral_norm")
                depth_router_normed_spectral_norm = None if v is None else float(v)

        if depth_sp is not None and depth_sc is not None:
            dsp = depth_sp.detach().float()
            dsc = depth_sc.detach().float()
            depth_proj_spatial_norm = float(dsp.norm(dim=-1).mean().item())
            depth_proj_spectral_norm = float(dsc.norm(dim=-1).mean().item())
            depth_proj_cosine = float(F.cosine_similarity(dsp, dsc, dim=-1).mean().item())
            depth_proj_l2 = float((dsp - dsc).norm(dim=-1).mean().item())

        shared_blend = self._shared_blend_value(cur_epoch)
        expert_residual_scale = 1.0 - shared_blend

        spatial_parts = [base_feat]
        spectral_parts = [base_feat]
        if subj_sp is not None:
            spatial_parts.append(subj_sp)
        if eeg_sp is not None:
            spatial_parts.append(eeg_sp)
        if subj_sc is not None:
            spectral_parts.append(subj_sc)
        if eeg_sc is not None:
            spectral_parts.append(eeg_sc)
        if depth_sp is not None:
            spatial_parts.append(depth_sp)
        if depth_sc is not None:
            spectral_parts.append(depth_sc)

        spatial_in = torch.cat(spatial_parts, dim=-1)
        spectral_in = torch.cat(spectral_parts, dim=-1)
        psd_stats: Optional[Tuple[List[float], List[float]]] = None
        if self.use_psd_router_features:
            psd = _MOE_PSD_ROUTER.get()
            if psd is None:
                raise ValueError("moe_use_psd_router_features=True requires PSD context from backbone")
            if psd.shape[0] != b or psd.shape[-1] != PSD_ROUTER_DIM:
                raise ValueError(f"PSD expected [B,{PSD_ROUTER_DIM}], got {tuple(psd.shape)}")
            spectral_in = torch.cat([spectral_in, psd], dim=-1)
            pd = psd.detach().float()
            psd_stats = (pd.mean(0).cpu().tolist(), pd.std(0).cpu().tolist())

        if self.spatial_router_input_norm is not None:
            spatial_in = self.spatial_router_input_norm(spatial_in)
        if self.spectral_router_input_norm is not None:
            spectral_in = self.spectral_router_input_norm(spectral_in)

        logits_sp_content = self.spatial_router(spatial_in)
        logits_sc_content = self.spectral_router(spectral_in)

        bias_sp = self._domain_logit_bias(b, "spatial", x.device)
        bias_sc = self._domain_logit_bias(b, "spectral", x.device)
        cond_bias_sp, cond_bias_sc = self._adapter_cond_logit_bias(router_context, b, x.device)

        eff_jitter_std = self._effective_router_jitter_std(cur_epoch)

        logits_sp_domain = (logits_sp_content + bias_sp) / self.router_temperature
        logits_sc_domain = (logits_sc_content + bias_sc) / self.router_temperature
        logits_sp = (logits_sp_content + bias_sp + cond_bias_sp) / self.router_temperature
        logits_sc = (logits_sc_content + bias_sc + cond_bias_sc) / self.router_temperature
        if self.training and eff_jitter_std > 0:
            logits_sp = logits_sp + torch.randn_like(logits_sp) * eff_jitter_std
            logits_sc = logits_sc + torch.randn_like(logits_sc) * eff_jitter_std

        probs_sp = F.softmax(logits_sp, dim=-1)
        probs_sc = F.softmax(logits_sc, dim=-1)

        uniform_dispatch_active = (
            self.training
            and self.uniform_dispatch_warmup_epochs > 0
            and cur_epoch > 0
            and cur_epoch <= self.uniform_dispatch_warmup_epochs
        )

        raw_top1_sp = logits_sp.argmax(dim=-1)
        raw_top1_sc = logits_sc.argmax(dim=-1)
        raw_top1_sp_domain = logits_sp_domain.argmax(dim=-1)
        raw_top1_sc_domain = logits_sc_domain.argmax(dim=-1)
        raw_top1_sp_content = logits_sp_content.argmax(dim=-1)
        raw_top1_sc_content = logits_sc_content.argmax(dim=-1)

        top2_sp_l, _ = torch.topk(logits_sp, k=min(2, self.num_specialists), dim=-1)
        top2_sc_l, _ = torch.topk(logits_sc, k=min(2, self.num_specialists), dim=-1)
        top2_sp_p, _ = torch.topk(probs_sp, k=min(2, self.num_specialists), dim=-1)
        top2_sc_p, _ = torch.topk(probs_sc, k=min(2, self.num_specialists), dim=-1)

        margin_sp_logit = top2_sp_l[:, 0] - (top2_sp_l[:, 1] if top2_sp_l.size(1) > 1 else 0.0)
        margin_sc_logit = top2_sc_l[:, 0] - (top2_sc_l[:, 1] if top2_sc_l.size(1) > 1 else 0.0)
        margin_sp_prob = top2_sp_p[:, 0] - (top2_sp_p[:, 1] if top2_sp_p.size(1) > 1 else 0.0)
        margin_sc_prob = top2_sc_p[:, 0] - (top2_sc_p[:, 1] if top2_sc_p.size(1) > 1 else 0.0)

        capacity = max(1, int(math.ceil(self.capacity_factor * b / float(self.num_specialists))))
        soft_warmup_applicable = (
            self.router_soft_warmup_epochs > 0
            and cur_epoch > 0
            and cur_epoch <= self.router_soft_warmup_epochs
        )
        soft_dispatch_active = (self.router_dispatch_mode == "soft")
        soft_warmup_active = (
            self.training
            and not soft_dispatch_active
            and soft_warmup_applicable
        )
        soft_dispatch_warmup_active = (
            self.training
            and soft_dispatch_active
            and soft_warmup_applicable
        )
        soft_warmup_alpha = 1.0
        if soft_dispatch_warmup_active:
            soft_warmup_alpha = min(1.0, float(cur_epoch) / float(max(1, self.router_soft_warmup_epochs)))
        early_reg_epochs = max(self.router_soft_warmup_epochs, 12)
        early_reg_boost = 1.0
        if self.training and cur_epoch > 0 and cur_epoch <= early_reg_epochs:
            early_reg_boost = 1.5
        base_entropy_coef_sp = (
            self.router_entropy_coef
            if self.router_entropy_coef_spatial is None
            else self.router_entropy_coef_spatial
        )
        base_entropy_coef_sc = (
            self.router_entropy_coef
            if self.router_entropy_coef_spectral is None
            else self.router_entropy_coef_spectral
        )
        base_balance_kl_coef_sp = (
            self.router_balance_kl_coef
            if self.router_balance_kl_coef_spatial is None
            else self.router_balance_kl_coef_spatial
        )
        base_balance_kl_coef_sc = (
            self.router_balance_kl_coef
            if self.router_balance_kl_coef_spectral is None
            else self.router_balance_kl_coef_spectral
        )
        if not self.use_spatial_specialists:
            base_entropy_coef_sp = 0.0
            base_balance_kl_coef_sp = 0.0
        if not self.use_spectral_specialists:
            base_entropy_coef_sc = 0.0
            base_balance_kl_coef_sc = 0.0

        eff_entropy_coef_sp = base_entropy_coef_sp * early_reg_boost
        eff_entropy_coef_sc = base_entropy_coef_sc * early_reg_boost
        eff_balance_kl_coef_sp = base_balance_kl_coef_sp * early_reg_boost
        eff_balance_kl_coef_sc = base_balance_kl_coef_sc * early_reg_boost

        if uniform_dispatch_active:
            probs_sp_dispatch = torch.full_like(probs_sp, 1.0 / float(self.num_specialists))
            probs_sc_dispatch = torch.full_like(probs_sc, 1.0 / float(self.num_specialists))
            assign_sp = raw_top1_sp
            assign_sc = raw_top1_sc
            rerouted_sp = torch.zeros_like(assign_sp, dtype=torch.bool)
            rerouted_sc = torch.zeros_like(assign_sc, dtype=torch.bool)
            fallback_sp = torch.zeros_like(assign_sp, dtype=torch.bool)
            fallback_sc = torch.zeros_like(assign_sc, dtype=torch.bool)
            counts_sp_pre = torch.bincount(raw_top1_sp, minlength=self.num_specialists).to(dtype=torch.long)
            counts_sc_pre = torch.bincount(raw_top1_sc, minlength=self.num_specialists).to(dtype=torch.long)
            counts_sp = torch.round(probs_sp_dispatch.sum(dim=0)).to(dtype=torch.long)
            counts_sc = torch.round(probs_sc_dispatch.sum(dim=0)).to(dtype=torch.long)
            if self.use_spatial_specialists:
                res_sp = self._bank_residual_soft(x_flat, probs_sp_dispatch, batch_idx, self.spatial_specialists)
            else:
                res_sp = torch.zeros_like(x_flat)
            if self.use_spectral_specialists:
                res_sc = self._bank_residual_soft(x_flat, probs_sc_dispatch, batch_idx, self.spectral_specialists)
            else:
                res_sc = torch.zeros_like(x_flat)
        elif soft_warmup_active or soft_dispatch_active:
            probs_sp_dispatch = probs_sp
            probs_sc_dispatch = probs_sc
            if soft_dispatch_warmup_active:
                uniform_sp = torch.full_like(probs_sp, 1.0 / float(self.num_specialists))
                uniform_sc = torch.full_like(probs_sc, 1.0 / float(self.num_specialists))
                probs_sp_dispatch = soft_warmup_alpha * probs_sp + (1.0 - soft_warmup_alpha) * uniform_sp
                probs_sc_dispatch = soft_warmup_alpha * probs_sc + (1.0 - soft_warmup_alpha) * uniform_sc
            assign_sp = raw_top1_sp
            assign_sc = raw_top1_sc
            rerouted_sp = torch.zeros_like(assign_sp, dtype=torch.bool)
            rerouted_sc = torch.zeros_like(assign_sc, dtype=torch.bool)
            fallback_sp = torch.zeros_like(assign_sp, dtype=torch.bool)
            fallback_sc = torch.zeros_like(assign_sc, dtype=torch.bool)
            counts_sp_pre = torch.bincount(raw_top1_sp, minlength=self.num_specialists).to(dtype=torch.long)
            counts_sc_pre = torch.bincount(raw_top1_sc, minlength=self.num_specialists).to(dtype=torch.long)
            if soft_dispatch_warmup_active:
                counts_sp = torch.round(probs_sp_dispatch.sum(dim=0)).to(dtype=torch.long)
                counts_sc = torch.round(probs_sc_dispatch.sum(dim=0)).to(dtype=torch.long)
            else:
                counts_sp = counts_sp_pre.clone()
                counts_sc = counts_sc_pre.clone()
            if self.use_spatial_specialists:
                res_sp = self._bank_residual_soft(x_flat, probs_sp_dispatch, batch_idx, self.spatial_specialists)
            else:
                res_sp = torch.zeros_like(x_flat)
            if self.use_spectral_specialists:
                res_sc = self._bank_residual_soft(x_flat, probs_sc_dispatch, batch_idx, self.spectral_specialists)
            else:
                res_sc = torch.zeros_like(x_flat)
        else:
            assign_sp, rerouted_sp, fallback_sp, counts_sp_pre, counts_sp = self._capacity_assign_top1(logits_sp, capacity)
            assign_sc, rerouted_sc, fallback_sc, counts_sc_pre, counts_sc = self._capacity_assign_top1(logits_sc, capacity)
            if self.use_spatial_specialists:
                res_sp = self._bank_residual(x_flat, assign_sp, batch_idx, self.spatial_specialists)
            else:
                res_sp = torch.zeros_like(x_flat)
            if self.use_spectral_specialists:
                res_sc = self._bank_residual(x_flat, assign_sc, batch_idx, self.spectral_specialists)
            else:
                res_sc = torch.zeros_like(x_flat)

        out = h_shared + expert_residual_scale * (res_sp + res_sc)

        lb_sp = self._bank_load_balance(probs_sp, raw_top1_sp) if self.use_spatial_specialists else probs_sp.new_zeros(())
        lb_sc = self._bank_load_balance(probs_sc, raw_top1_sc) if self.use_spectral_specialists else probs_sc.new_zeros(())
        lb = lb_sp + lb_sc

        reroute_frac = rerouted_sp.float().mean() + rerouted_sc.float().mean()
        fallback_frac = fallback_sp.float().mean() + fallback_sc.float().mean()
        overflow_penalty = reroute_frac + fallback_frac
        if soft_warmup_active or soft_dispatch_active:
            overflow_penalty = overflow_penalty * 0.0
        domain_reg = bias_sp.pow(2).mean() + bias_sc.pow(2).mean()
        entropy_sp = (-(probs_sp * probs_sp.clamp_min(1e-10).log()).sum(dim=-1)).mean()
        entropy_sc = (-(probs_sc * probs_sc.clamp_min(1e-10).log()).sum(dim=-1)).mean()
        entropy_reg = -(entropy_sp + entropy_sc)
        entropy_reg_sp = -entropy_sp
        entropy_reg_sc = -entropy_sc
        balance_kl_sp = self._batch_uniform_kl(probs_sp) if self.use_spatial_specialists else probs_sp.new_zeros(())
        balance_kl_sc = self._batch_uniform_kl(probs_sc) if self.use_spectral_specialists else probs_sc.new_zeros(())
        balance_kl = balance_kl_sp + balance_kl_sc
        depth_block_sep_js = probs_sp.new_zeros(())
        depth_block_sep_hinge = probs_sp.new_zeros(())
        if (
            self.attnres_depth_block_separation_coef > 0.0
            and depth_context_mode == "dual_query_block_typed_proj"
            and depth_block_weight_dist_spatial_raw is not None
            and depth_block_weight_dist_spectral_raw is not None
            and depth_block_weight_dist_spatial_raw.shape == depth_block_weight_dist_spectral_raw.shape
            and depth_block_weight_dist_spatial_raw.dim() == 2
            and depth_block_weight_dist_spatial_raw.shape[0] > 0
        ):
            p_sep = depth_block_weight_dist_spatial_raw.to(device=probs_sp.device, dtype=probs_sp.dtype)
            q_sep = depth_block_weight_dist_spectral_raw.to(device=probs_sp.device, dtype=probs_sp.dtype)
            depth_block_sep_js = self._mean_js_divergence(p_sep, q_sep)
            depth_block_sep_hinge = F.relu(
                depth_block_sep_js.new_tensor(self.attnres_depth_block_separation_target_js) - depth_block_sep_js
            )
        depth_block_sep_term = self.attnres_depth_block_separation_coef * depth_block_sep_hinge
        z_loss = torch.logsumexp(logits_sp, dim=-1).pow(2).mean() + torch.logsumexp(logits_sc, dim=-1).pow(2).mean()

        self._last_lb_loss = lb
        self._last_overflow_penalty = overflow_penalty
        self._last_domain_bias_reg = domain_reg
        self._last_entropy_reg = entropy_reg
        self._last_balance_kl = balance_kl
        self._last_z_loss = z_loss
        self._last_aux_loss = (
            self.load_balance_coef * lb
            + overflow_penalty
            + self.domain_bias_reg_coef * domain_reg
            + eff_entropy_coef_sp * entropy_reg_sp
            + eff_entropy_coef_sc * entropy_reg_sc
            + eff_balance_kl_coef_sp * balance_kl_sp
            + eff_balance_kl_coef_sc * balance_kl_sc
            + depth_block_sep_term
            + self.router_z_loss_coef * z_loss
        )

        pre_ent_sp = self._sample_entropy(probs_sp)
        pre_ent_sc = self._sample_entropy(probs_sc)
        post_ent_sp = self._hist_entropy(counts_sp)
        post_ent_sc = self._hist_entropy(counts_sc)
        eff_exp_post_sp = float(math.exp(max(post_ent_sp, 0.0)))
        eff_exp_post_sc = float(math.exp(max(post_ent_sc, 0.0)))
        pre_max_sp = float((counts_sp_pre.float().max() / max(float(b), 1.0)).item())
        pre_max_sc = float((counts_sc_pre.float().max() / max(float(b), 1.0)).item())
        pre_max_gap = abs(pre_max_sp - pre_max_sc)
        dom_norm = float(torch.cat([bias_sp, bias_sc], dim=1).float().norm(dim=-1).mean().item())
        cond_norm = float(torch.cat([cond_bias_sp, cond_bias_sc], dim=1).float().norm(dim=-1).mean().item())
        sp_content_abs = float(logits_sp_content.abs().mean().item())
        sc_content_abs = float(logits_sc_content.abs().mean().item())
        sp_domain_abs = float(bias_sp.abs().mean().item())
        sc_domain_abs = float(bias_sc.abs().mean().item())
        sp_adapter_abs = float(cond_bias_sp.abs().mean().item())
        sc_adapter_abs = float(cond_bias_sc.abs().mean().item())

        self.last_diagnostics = {
            "moe_kind": self.moe_kind,
            "capacity": int(capacity),
            "num_experts": int(self.num_specialists),
            "domain_bias_enabled": bool(self.domain_bias),
            "domain_bias_norm": dom_norm,
            "adapter_cond_bias_enabled": bool(self.use_adapter_cond_bias),
            "adapter_cond_bias_norm": cond_norm,
            "content_logit_abs_spatial": sp_content_abs,
            "content_logit_abs_spectral": sc_content_abs,
            "adapter_cond_bias_abs_spatial": sp_adapter_abs,
            "adapter_cond_bias_abs_spectral": sc_adapter_abs,
            "adapter_cond_bias_rel_spatial": float(sp_adapter_abs / max(sp_content_abs, 1e-8)),
            "adapter_cond_bias_rel_spectral": float(sc_adapter_abs / max(sc_content_abs, 1e-8)),
            "router_temperature": float(self.router_temperature),
            "router_soft_warmup_epochs": int(self.router_soft_warmup_epochs),
            "router_soft_warmup_active": bool(soft_warmup_active or soft_dispatch_warmup_active),
            "router_soft_warmup_applicable_epoch": bool(soft_warmup_applicable),
            "router_soft_dispatch_warmup_active": bool(soft_dispatch_warmup_active),
            "router_soft_warmup_alpha": float(soft_warmup_alpha),
            "router_dispatch_mode": str(self.router_dispatch_mode),
            "route_strategy": (
                "uniform_dispatch_warmup"
                if uniform_dispatch_active
                else (
                "soft_dispatch_warmup"
                if soft_dispatch_warmup_active
                else (
                    "soft_dispatch"
                    if soft_dispatch_active
                    else ("soft_warmup" if soft_warmup_active else "hard_capacity")
                )
                )
            ),
            "router_entropy_coef": float(self.router_entropy_coef),
            "router_entropy_coef_spatial": float(base_entropy_coef_sp),
            "router_entropy_coef_spectral": float(base_entropy_coef_sc),
            "router_entropy_coef_effective": float(eff_entropy_coef_sp),
            "router_entropy_coef_effective_spectral": float(eff_entropy_coef_sc),
            "router_balance_kl_coef": float(self.router_balance_kl_coef),
            "router_balance_kl_coef_spatial": float(base_balance_kl_coef_sp),
            "router_balance_kl_coef_spectral": float(base_balance_kl_coef_sc),
            "router_balance_kl_coef_effective": float(eff_balance_kl_coef_sp),
            "router_balance_kl_coef_effective_spectral": float(eff_balance_kl_coef_sc),
            "attnres_depth_block_separation_coef": float(self.attnres_depth_block_separation_coef),
            "attnres_depth_block_separation_target_js": float(self.attnres_depth_block_separation_target_js),
            "router_early_reg_boost": float(early_reg_boost),
            "router_early_reg_epochs": int(early_reg_epochs),
            "router_z_loss_coef": float(self.router_z_loss_coef),
            "router_jitter_std": float(self.router_jitter_std),
            "router_jitter_std_effective": float(eff_jitter_std),
            "router_jitter_final_std": float(self.router_jitter_final_std),
            "router_jitter_anneal_epochs": int(self.router_jitter_anneal_epochs),
            "subject_summary_router_concat": bool(self.use_subject_summary_router_concat),
            "eeg_summary_router_concat_spatial": bool(self.use_eeg_summary_router_concat_spatial),
            "eeg_summary_router_concat_spectral": bool(self.use_eeg_summary_router_concat_spectral),
            "compact_router_warmup_epochs": int(self.compact_router_warmup_epochs),
            "compact_router_path_scale": float(compact_path_scale),
            "compact_router_gate_init": float(self.compact_router_gate_init),
            "compact_router_gate_spatial": compact_router_gate_spatial,
            "compact_router_gate_spectral": compact_router_gate_spectral,
            "attnres_depth_router_concat": bool(self.use_attnres_depth_router_concat),
            "attnres_depth_router_init": self.attnres_depth_router_init,
            "attnres_depth_router_norm_gate": bool(self.attnres_depth_router_norm_gate),
            "attnres_depth_router_norm_gate_active": bool(depth_router_norm_gate_active),
            "attnres_depth_router_norm_eps": depth_router_norm_eps,
            "attnres_depth_router_gate_init": float(self.attnres_depth_router_gate_init),
            "attnres_depth_router_gate_spatial": depth_router_gate_spatial,
            "attnres_depth_router_gate_spectral": depth_router_gate_spectral,
            "attnres_depth_router_normed_spatial_norm": depth_router_normed_spatial_norm,
            "attnres_depth_router_normed_spectral_norm": depth_router_normed_spectral_norm,
            "attnres_depth_path_scale": float(depth_path_scale),
            "attnres_depth_context_mode": depth_context_mode,
            "attnres_depth_block_count": int(depth_block_count),
            "attnres_depth_block_mean": depth_block_mean,
            "attnres_depth_block_std": depth_block_std,
            "attnres_depth_block_summary_norms": depth_block_summary_norms,
            "attnres_depth_block_layer_counts": depth_block_layer_counts,
            "attnres_depth_block_layer_counts_pre_attn": depth_block_layer_counts_pre_attn,
            "attnres_depth_block_layer_counts_pre_mlp": depth_block_layer_counts_pre_mlp,
            "attnres_depth_block_pooling": depth_block_pooling,
            "attnres_depth_family_mode": depth_family_mode,
            "attnres_depth_block_peak_weight_pre_attn": depth_block_peak_weight_pre_attn,
            "attnres_depth_block_peak_weight_pre_mlp": depth_block_peak_weight_pre_mlp,
            "attnres_depth_block_peak_weight_spatial_pre_attn": depth_block_peak_weight_spatial_pre_attn,
            "attnres_depth_block_peak_weight_spatial_pre_mlp": depth_block_peak_weight_spatial_pre_mlp,
            "attnres_depth_block_peak_weight_spectral_pre_attn": depth_block_peak_weight_spectral_pre_attn,
            "attnres_depth_block_peak_weight_spectral_pre_mlp": depth_block_peak_weight_spectral_pre_mlp,
            "attnres_depth_block_weight_dist_spatial": depth_block_weight_dist_spatial,
            "attnres_depth_block_weight_dist_spectral": depth_block_weight_dist_spectral,
            "attnres_depth_block_weight_dist_cosine": depth_block_weight_dist_cosine,
            "attnres_depth_block_weight_dist_js_div": depth_block_weight_dist_js_div,
            "attnres_depth_shared_context_norm": depth_shared_context_norm,
            "attnres_depth_spatial_context_norm": depth_spatial_context_norm,
            "attnres_depth_spectral_context_norm": depth_spectral_context_norm,
            "attnres_depth_summary_mean": depth_summary_mean,
            "attnres_depth_summary_std": depth_summary_std,
            "attnres_depth_summary_spatial_norm": depth_summary_spatial_norm,
            "attnres_depth_summary_spectral_norm": depth_summary_spectral_norm,
            "attnres_depth_summary_mode": depth_summary_mode,
            "attnres_depth_probe_mlp_for_router": depth_probe_mlp_for_router,
            "attnres_depth_proj_spatial_norm": depth_proj_spatial_norm,
            "attnres_depth_proj_spectral_norm": depth_proj_spectral_norm,
            "attnres_depth_proj_cosine": depth_proj_cosine,
            "attnres_depth_proj_l2": depth_proj_l2,
            "attnres_depth_summary_grad_mode": depth_summary_grad_mode,
            "attnres_depth_summary_grad_active": depth_summary_grad_active,
            "attnres_depth_summary_detached": bool(depth_summary_detached),
            "attnres_depth_summary_cur_epoch": int(depth_summary_cur_epoch),
            "attnres_depth_summary_unfreeze_epoch": depth_summary_unfreeze_epoch,
            "attnres_depth_summary_unfreeze_reached": bool(depth_summary_unfreeze_reached),
            "spatial_bank_enabled": bool(self.use_spatial_specialists),
            "spectral_bank_enabled": bool(self.use_spectral_specialists),
            "uniform_dispatch_warmup_epochs": int(self.uniform_dispatch_warmup_epochs),
            "uniform_dispatch_warmup_active": bool(uniform_dispatch_active),
            "shared_blend": float(shared_blend),
            "expert_residual_scale": float(expert_residual_scale),
            "mean_shared_output_norm": float(h_shared.float().norm(dim=-1).mean().item()),
            "mean_spatial_residual_norm": float(res_sp.float().norm(dim=-1).mean().item()),
            "mean_spectral_residual_norm": float(res_sc.float().norm(dim=-1).mean().item()),
            "aux_load_balance": float(lb.detach().item()),
            "aux_overflow": float(overflow_penalty.detach().item()),
            "aux_domain_bias_reg": float(domain_reg.detach().item()),
            "aux_entropy_reg": float(entropy_reg.detach().item()),
            "aux_balance_kl": float(balance_kl.detach().item()),
            "aux_depth_block_sep_js": float(depth_block_sep_js.detach().item()),
            "aux_depth_block_sep_hinge": float(depth_block_sep_hinge.detach().item()),
            "aux_depth_block_sep_term": float(depth_block_sep_term.detach().item()),
            "aux_z_loss": float(z_loss.detach().item()),
            "aux_total": float(self._last_aux_loss.detach().item()),
            "domain_bias_abs_spatial": sp_domain_abs,
            "domain_bias_abs_spectral": sc_domain_abs,
            "domain_bias_rel_spatial": float(sp_domain_abs / max(sp_content_abs, 1e-8)),
            "domain_bias_rel_spectral": float(sc_domain_abs / max(sc_content_abs, 1e-8)),
            "domain_shift_rate_spatial": float((raw_top1_sp_domain != raw_top1_sp_content).float().mean().item()),
            "domain_shift_rate_spectral": float((raw_top1_sc_domain != raw_top1_sc_content).float().mean().item()),
            "adapter_shift_rate_spatial": float((raw_top1_sp != raw_top1_sp_domain).float().mean().item()),
            "adapter_shift_rate_spectral": float((raw_top1_sc != raw_top1_sc_domain).float().mean().item()),
            "full_shift_rate_spatial": float((raw_top1_sp != raw_top1_sp_content).float().mean().item()),
            "full_shift_rate_spectral": float((raw_top1_sc != raw_top1_sc_content).float().mean().item()),
            "specialization_effective_experts_post_spatial": eff_exp_post_sp,
            "specialization_effective_experts_post_spectral": eff_exp_post_sc,
            "specialization_pre_max_frac_gap": pre_max_gap,
            "spatial": {
                "pre_top1_histogram": counts_sp_pre.detach().cpu().tolist(),
                "pre_max_expert_fraction": pre_max_sp,
                "pre_entropy": pre_ent_sp,
                "pre_margin_logit": float(margin_sp_logit.mean().item()),
                "pre_margin_prob": float(margin_sp_prob.mean().item()),
                "assigned_count_per_expert": counts_sp.detach().cpu().tolist(),
                "overflow_count": int(fallback_sp.sum().item()),
                "reroute_rate": float(rerouted_sp.float().mean().item()),
                "shared_only_fraction": float(fallback_sp.float().mean().item()),
                "routing_entropy_pre_capacity": pre_ent_sp,
                "routing_entropy_post_assignment": post_ent_sp,
            },
            "spectral": {
                "pre_top1_histogram": counts_sc_pre.detach().cpu().tolist(),
                "pre_max_expert_fraction": pre_max_sc,
                "pre_entropy": pre_ent_sc,
                "pre_margin_logit": float(margin_sc_logit.mean().item()),
                "pre_margin_prob": float(margin_sc_prob.mean().item()),
                "assigned_count_per_expert": counts_sc.detach().cpu().tolist(),
                "overflow_count": int(fallback_sc.sum().item()),
                "reroute_rate": float(rerouted_sc.float().mean().item()),
                "shared_only_fraction": float(fallback_sc.float().mean().item()),
                "routing_entropy_pre_capacity": pre_ent_sc,
                "routing_entropy_post_assignment": post_ent_sc,
            },
        }
        if psd_stats is not None:
            self.last_diagnostics["psd_feature_mean"] = psd_stats[0]
            self.last_diagnostics["psd_feature_std"] = psd_stats[1]

        self._routing_export_cache = {
            "logits_spatial": logits_sp.detach().cpu(),
            "logits_spectral": logits_sc.detach().cpu(),
            "probs_spatial": probs_sp.detach().cpu(),
            "probs_spectral": probs_sc.detach().cpu(),
            "raw_top1_spatial": raw_top1_sp.detach().cpu(),
            "raw_top1_spectral": raw_top1_sc.detach().cpu(),
            "pre_entropy_spatial": (-(probs_sp * probs_sp.clamp_min(1e-10).log()).sum(dim=-1)).detach().cpu(),
            "pre_entropy_spectral": (-(probs_sc * probs_sc.clamp_min(1e-10).log()).sum(dim=-1)).detach().cpu(),
            "pre_margin_logit_spatial": margin_sp_logit.detach().cpu(),
            "pre_margin_logit_spectral": margin_sc_logit.detach().cpu(),
            "pre_margin_prob_spatial": margin_sp_prob.detach().cpu(),
            "pre_margin_prob_spectral": margin_sc_prob.detach().cpu(),
            "assigned_spatial": assign_sp.detach().cpu(),
            "assigned_spectral": assign_sc.detach().cpu(),
            "rerouted_spatial": rerouted_sp.detach().cpu(),
            "rerouted_spectral": rerouted_sc.detach().cpu(),
            "fallback_spatial": fallback_sp.detach().cpu(),
            "fallback_spectral": fallback_sc.detach().cpu(),
            "cohort_id": (
                _MOE_FACED_METADATA.get().get("cohort_id").detach().cpu()
                if _MOE_FACED_METADATA.get() is not None and _MOE_FACED_METADATA.get().get("cohort_id") is not None
                else torch.zeros(b, dtype=torch.long)
            ),
            "sample_rate_group_id": (
                _MOE_FACED_METADATA.get().get("sample_rate_group_id").detach().cpu()
                if _MOE_FACED_METADATA.get() is not None and _MOE_FACED_METADATA.get().get("sample_rate_group_id") is not None
                else torch.zeros(b, dtype=torch.long)
            ),
            "age_bucket_id": (
                _MOE_FACED_METADATA.get().get("age_bucket_id").detach().cpu()
                if _MOE_FACED_METADATA.get() is not None and _MOE_FACED_METADATA.get().get("age_bucket_id") is not None
                else torch.zeros(b, dtype=torch.long)
            ),
            "segment_bucket_id": (
                _MOE_FACED_METADATA.get().get("segment_bucket_id").detach().cpu()
                if _MOE_FACED_METADATA.get() is not None and _MOE_FACED_METADATA.get().get("segment_bucket_id") is not None
                else torch.zeros(b, dtype=torch.long)
            ),
        }

        return out.view(*leading, d)

    def auxiliary_loss(self) -> Tensor:
        return self._last_aux_loss


@torch.no_grad()
def warm_start_typed_capacity_domain_from_dense_ckpt(
    moe: TypedCapacityDomainMoEFFN,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_dense_into_specialist_linear1: bool,
    expert_init_noise_std: float = 0.0,
) -> None:
    p1w = f"encoder.layers.{layer_idx}.linear1.weight"
    p1b = f"encoder.layers.{layer_idx}.linear1.bias"
    p2w = f"encoder.layers.{layer_idx}.linear2.weight"
    p2b = f"encoder.layers.{layer_idx}.linear2.bias"
    for key in (p1w, p2w):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key} for typed_capacity_domain warm-start at layer {layer_idx}")

    w1, w2 = ckpt[p1w], ckpt[p2w]
    b1, b2 = ckpt.get(p1b), ckpt.get(p2b)

    moe.shared.linear1.weight.copy_(w1)
    moe.shared.linear2.weight.copy_(w2)
    if b1 is not None and moe.shared.linear1.bias is not None:
        moe.shared.linear1.bias.copy_(b1)
    if b2 is not None and moe.shared.linear2.bias is not None:
        moe.shared.linear2.bias.copy_(b2)

    if copy_dense_into_specialist_linear1:
        for bank in (moe.spatial_specialists, moe.spectral_specialists):
            for spec in bank:
                spec.linear1.weight.copy_(w1)
                if b1 is not None and spec.linear1.bias is not None:
                    spec.linear1.bias.copy_(b1)

    moe.apply_expert_init_noise_(expert_init_noise_std)
    moe._zero_specialist_output_weights()


def warm_start_moe_from_dense_ckpt(
    moe: nn.Module,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_specialist_linear1_from_dense: bool = True,
    expert_init_noise_std: float = 0.0,
) -> None:
    if not isinstance(moe, TypedCapacityDomainMoEFFN):
        raise TypeError(f"Expected TypedCapacityDomainMoEFFN, got {type(moe)}")
    warm_start_typed_capacity_domain_from_dense_ckpt(
        moe,
        ckpt,
        layer_idx,
        copy_dense_into_specialist_linear1=copy_specialist_linear1_from_dense,
        expert_init_noise_std=expert_init_noise_std,
    )


def format_moe_diagnostics_lines(layer_idx: int, diag: Dict[str, Any]) -> List[str]:
    if diag.get("moe_kind") != "typed_capacity_domain":
        return [f"  [MoE L{layer_idx}] unsupported diagnostics payload: {diag.get('moe_kind')}"]
    sp = diag.get("spatial", {})
    sc = diag.get("spectral", {})
    lines = [
        (
            f"  [MoE L{layer_idx}] kind=typed_capacity_domain  capacity={diag.get('capacity')}  "
            f"experts={diag.get('num_experts')}  domain_bias={diag.get('domain_bias_enabled')}  "
            f"domain_bias_norm={diag.get('domain_bias_norm', 0.0):.6f}  "
            f"|bias|_sp={diag.get('domain_bias_abs_spatial', 0.0):.6f}  "
            f"|bias|_sc={diag.get('domain_bias_abs_spectral', 0.0):.6f}"
        ),
        (
            f"    |content_logit|_sp={diag.get('content_logit_abs_spatial', 0.0):.6f}  "
            f"|content_logit|_sc={diag.get('content_logit_abs_spectral', 0.0):.6f}  "
            f"|adapter_bias|_sp={diag.get('adapter_cond_bias_abs_spatial', 0.0):.6f}  "
            f"|adapter_bias|_sc={diag.get('adapter_cond_bias_abs_spectral', 0.0):.6f}"
        ),
        (
            f"    rel_bias_vs_content: domain(sp/sc)=({diag.get('domain_bias_rel_spatial', 0.0):.4f}/"
            f"{diag.get('domain_bias_rel_spectral', 0.0):.4f})  "
            f"adapter(sp/sc)=({diag.get('adapter_cond_bias_rel_spatial', 0.0):.4f}/"
            f"{diag.get('adapter_cond_bias_rel_spectral', 0.0):.4f})"
        ),
        (
            f"    route={diag.get('route_strategy', 'hard_capacity')}  "
            f"dispatch_mode={diag.get('router_dispatch_mode', 'hard_capacity')}  "
            f"temp(sp/sc)=({diag.get('router_temperature', 1.0):.4f}/"
            f"{diag.get('router_temperature', 1.0):.4f})  "
            f"soft_warmup_active={diag.get('router_soft_warmup_active', False)}  "
            f"soft_dispatch_warmup_active={diag.get('router_soft_dispatch_warmup_active', False)}  "
            f"soft_warmup_alpha={diag.get('router_soft_warmup_alpha', 1.0):.4f}  "
            f"soft_warmup_applicable_epoch={diag.get('router_soft_warmup_applicable_epoch', False)}  "
            f"soft_warmup_epochs={diag.get('router_soft_warmup_epochs', 0)}"
        ),
        (
            f"    banks_enabled(sp/sc)=({diag.get('spatial_bank_enabled', True)}/"
            f"{diag.get('spectral_bank_enabled', True)})  "
            f"uniform_warmup(active/epochs)=({diag.get('uniform_dispatch_warmup_active', False)}/"
            f"{diag.get('uniform_dispatch_warmup_epochs', 0)})  "
            f"shared_blend={diag.get('shared_blend', 0.0):.4f}  "
            f"expert_residual_scale={diag.get('expert_residual_scale', 1.0):.4f}"
        ),
        (
            f"    router_concat: subject_summary={diag.get('subject_summary_router_concat', False)}  "
            f"eeg_spatial={diag.get('eeg_summary_router_concat_spatial', False)}  "
            f"eeg_spectral={diag.get('eeg_summary_router_concat_spectral', False)}  "
            f"attnres_depth={diag.get('attnres_depth_router_concat', False)}"
        ),
        (
            f"    compact_summary: scale={diag.get('compact_router_path_scale', 1.0):.4f}  "
            f"warmup_epochs={diag.get('compact_router_warmup_epochs', 0)}  "
            f"gate_init={diag.get('compact_router_gate_init', 1.0):.4f}  "
            f"gate_sp={diag.get('compact_router_gate_spatial', 'NA')}  "
            f"gate_sc={diag.get('compact_router_gate_spectral', 'NA')}"
        ),
        (
            f"    depth_summary: scale={diag.get('attnres_depth_path_scale', 1.0):.4f}  "
            f"ctx_mode={diag.get('attnres_depth_context_mode', 'compact_shared')}  "
            f"proj_init={diag.get('attnres_depth_router_init', 'xavier')}  "
            f"norm_gate={diag.get('attnres_depth_router_norm_gate', False)}  "
            f"norm_gate_active={diag.get('attnres_depth_router_norm_gate_active', False)}  "
            f"norm_eps={diag.get('attnres_depth_router_norm_eps', 'NA')}  "
            f"gate_sp={diag.get('attnres_depth_router_gate_spatial', 'NA')}  "
            f"gate_sc={diag.get('attnres_depth_router_gate_spectral', 'NA')}  "
            f"blocks={diag.get('attnres_depth_block_count', 0)}  "
            f"pool={diag.get('attnres_depth_block_pooling', 'NA')}  "
            f"layer_counts={diag.get('attnres_depth_block_layer_counts', 'NA')}  "
            f"mean={diag.get('attnres_depth_summary_mean', 0.0) if diag.get('attnres_depth_summary_mean') is not None else 'NA'}  "
            f"std={diag.get('attnres_depth_summary_std', 0.0) if diag.get('attnres_depth_summary_std') is not None else 'NA'}  "
            f"summary_norm_sp={diag.get('attnres_depth_summary_spatial_norm', 'NA')}  "
            f"summary_norm_sc={diag.get('attnres_depth_summary_spectral_norm', 'NA')}  "
            f"normed_summary_norm_sp={diag.get('attnres_depth_router_normed_spatial_norm', 'NA')}  "
            f"normed_summary_norm_sc={diag.get('attnres_depth_router_normed_spectral_norm', 'NA')}  "
            f"mode={diag.get('attnres_depth_summary_mode', 'NA')}  "
            f"probe_mlp={diag.get('attnres_depth_probe_mlp_for_router', False)}  "
            f"shared_ctx_norm={diag.get('attnres_depth_shared_context_norm', 'NA')}  "
            f"proj_norm_sp={diag.get('attnres_depth_proj_spatial_norm', 'NA')}  "
            f"proj_norm_sc={diag.get('attnres_depth_proj_spectral_norm', 'NA')}  "
            f"proj_cos={diag.get('attnres_depth_proj_cosine', 'NA')}  "
            f"proj_l2={diag.get('attnres_depth_proj_l2', 'NA')}  "
            f"grad_mode={diag.get('attnres_depth_summary_grad_mode', 'detached')}  "
            f"grad_active={diag.get('attnres_depth_summary_grad_active', False)}  "
            f"detached={diag.get('attnres_depth_summary_detached', True)}  "
            f"cur_epoch={diag.get('attnres_depth_summary_cur_epoch', 0)}  "
            f"unfreeze_epoch={diag.get('attnres_depth_summary_unfreeze_epoch', 1)}  "
            f"unfreeze_reached={diag.get('attnres_depth_summary_unfreeze_reached', False)}"
        ),
        (
            f"    depth_block_dist: spatial={diag.get('attnres_depth_block_weight_dist_spatial', 'NA')}  "
            f"spectral={diag.get('attnres_depth_block_weight_dist_spectral', 'NA')}  "
            f"cos={diag.get('attnres_depth_block_weight_dist_cosine', 'NA')}  "
            f"js={diag.get('attnres_depth_block_weight_dist_js_div', 'NA')}"
        ),
        (
            f"    aux_total={diag.get('aux_total', 0.0):.6f}  lb={diag.get('aux_load_balance', 0.0):.6f}  "
            f"overflow={diag.get('aux_overflow', 0.0):.6f}  domain_reg={diag.get('aux_domain_bias_reg', 0.0):.6f}  "
            f"entropy_reg={diag.get('aux_entropy_reg', 0.0):.6f}  entropy_coef_sp(base/eff)=({diag.get('router_entropy_coef', 0.0):.6f}/"
            f"{diag.get('router_entropy_coef_effective', diag.get('router_entropy_coef', 0.0)):.6f})  "
            f"entropy_coef_spatial={diag.get('router_entropy_coef_spatial', diag.get('router_entropy_coef', 0.0)):.6f}  "
            f"entropy_coef_sc(base/eff)=({diag.get('router_entropy_coef', 0.0):.6f}/"
            f"{diag.get('router_entropy_coef_effective_spectral', diag.get('router_entropy_coef_effective', diag.get('router_entropy_coef', 0.0))):.6f})  "
            f"entropy_coef_spectral={diag.get('router_entropy_coef_spectral', diag.get('router_entropy_coef', 0.0)):.6f}  "
            f"balance_kl={diag.get('aux_balance_kl', 0.0):.6f}  balance_kl_coef(base/eff)=({diag.get('router_balance_kl_coef', 0.0):.6f}/"
            f"{diag.get('router_balance_kl_coef_effective', diag.get('router_balance_kl_coef', 0.0)):.6f})  "
            f"balance_kl_coef_spatial={diag.get('router_balance_kl_coef_spatial', diag.get('router_balance_kl_coef', 0.0)):.6f}  "
            f"balance_kl_coef_sc(base/eff)=({diag.get('router_balance_kl_coef', 0.0):.6f}/"
            f"{diag.get('router_balance_kl_coef_effective_spectral', diag.get('router_balance_kl_coef_effective', diag.get('router_balance_kl_coef', 0.0))):.6f})  "
            f"balance_kl_coef_spectral={diag.get('router_balance_kl_coef_spectral', diag.get('router_balance_kl_coef', 0.0)):.6f}  "
            f"depth_sep_js={diag.get('aux_depth_block_sep_js', 0.0):.6f}  "
            f"depth_sep_target={diag.get('attnres_depth_block_separation_target_js', 0.0):.6f}  "
            f"depth_sep_hinge={diag.get('aux_depth_block_sep_hinge', 0.0):.6f}  "
            f"depth_sep_coef={diag.get('attnres_depth_block_separation_coef', 0.0):.6f}  "
            f"depth_sep_term={diag.get('aux_depth_block_sep_term', 0.0):.6f}  "
            f"early_reg_boost={diag.get('router_early_reg_boost', 1.0):.2f}<=ep{diag.get('router_early_reg_epochs', 0)}  "
            f"z_loss={diag.get('aux_z_loss', 0.0):.6f}  z_loss_coef={diag.get('router_z_loss_coef', 0.0):.6f}"
        ),
        (
            f"    router_jitter_std(start/effective/final)=({diag.get('router_jitter_std', 0.0):.4f}/"
            f"{diag.get('router_jitter_std_effective', diag.get('router_jitter_std', 0.0)):.4f}/"
            f"{diag.get('router_jitter_final_std', 0.0):.4f})  "
            f"jitter_anneal_epochs={diag.get('router_jitter_anneal_epochs', 0)}"
        ),
        (
            f"    route_shift_rate: domain(sp/sc)=({diag.get('domain_shift_rate_spatial', 0.0):.4f}/"
            f"{diag.get('domain_shift_rate_spectral', 0.0):.4f})  "
            f"adapter(sp/sc)=({diag.get('adapter_shift_rate_spatial', 0.0):.4f}/"
            f"{diag.get('adapter_shift_rate_spectral', 0.0):.4f})  "
            f"full(sp/sc)=({diag.get('full_shift_rate_spatial', 0.0):.4f}/"
            f"{diag.get('full_shift_rate_spectral', 0.0):.4f})"
        ),
        (
            f"    specialization_quality: eff_exp_post(sp/sc)=({diag.get('specialization_effective_experts_post_spatial', 0.0):.3f}/"
            f"{diag.get('specialization_effective_experts_post_spectral', 0.0):.3f})  "
            f"pre_max_frac_gap={diag.get('specialization_pre_max_frac_gap', 0.0):.4f}"
        ),
        (
            f"    [spatial] pre_hist={sp.get('pre_top1_histogram')}  pre_max_frac={sp.get('pre_max_expert_fraction', 0.0):.4f}  "
            f"pre_H={sp.get('pre_entropy', 0.0):.4f}  pre_margin(logit/prob)=({sp.get('pre_margin_logit', 0.0):.4f}/{sp.get('pre_margin_prob', 0.0):.4f})"
        ),
        (
            f"      assigned={sp.get('assigned_count_per_expert')}  reroute_rate={sp.get('reroute_rate', 0.0):.4f}  "
            f"overflow={sp.get('overflow_count')}  "
            f"shared_only_frac={sp.get('shared_only_fraction', 0.0):.4f}  "
            f"H_pre={sp.get('routing_entropy_pre_capacity', 0.0):.4f}  "
            f"H_post={sp.get('routing_entropy_post_assignment', 0.0):.4f}"
        ),
        (
            f"    [spectral] pre_hist={sc.get('pre_top1_histogram')}  pre_max_frac={sc.get('pre_max_expert_fraction', 0.0):.4f}  "
            f"pre_H={sc.get('pre_entropy', 0.0):.4f}  pre_margin(logit/prob)=({sc.get('pre_margin_logit', 0.0):.4f}/{sc.get('pre_margin_prob', 0.0):.4f})"
        ),
        (
            f"      assigned={sc.get('assigned_count_per_expert')}  reroute_rate={sc.get('reroute_rate', 0.0):.4f}  "
            f"overflow={sc.get('overflow_count')}  "
            f"shared_only_frac={sc.get('shared_only_fraction', 0.0):.4f}  "
            f"H_pre={sc.get('routing_entropy_pre_capacity', 0.0):.4f}  "
            f"H_post={sc.get('routing_entropy_post_assignment', 0.0):.4f}"
        ),
    ]
    if diag.get("psd_feature_mean") is not None and diag.get("psd_feature_std") is not None:
        lines.append(
            f"    psd_mean={[round(v, 4) for v in diag['psd_feature_mean']]}  "
            f"psd_std={[round(v, 4) for v in diag['psd_feature_std']]}"
        )
    return lines


# Backward-compatible alias while keeping a single routed implementation.
TypedDualBankSharedMoEFFN = TypedCapacityDomainMoEFFN
