"""MoE FFN variants for CBraMod: replacement sparse MoE, or shared dense + specialist residuals."""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MOE_ROUTER_MODES = ("token", "sample_hidden", "sample_attnres")
MOE_ROUTER_ARCHS = ("linear", "mlp")
PSD_ROUTER_DIM = 5

# Optional multiclass labels [B] for extended diagnostics (set from trainer via context).
_MOE_DIAG_LABELS: contextvars.ContextVar[Optional[Tensor]] = contextvars.ContextVar(
    "moe_diag_labels", default=None
)
# Optional [B, PSD_ROUTER_DIM] log1p band powers from raw EEG (set in CBraMod.forward).
_MOE_PSD_ROUTER: contextvars.ContextVar[Optional[Tensor]] = contextvars.ContextVar(
    "moe_psd_router", default=None
)
# Future FACED sample metadata (cohort, segment idx, age bucket, …) — not used by routers yet.
_MOE_FACED_METADATA: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "moe_faced_metadata", default=None
)



def set_moe_diagnostic_labels(labels: Optional[Tensor]) -> Any:
    """Trainer: wrap val batch forward so MoE can log class-conditional stats (optional)."""
    return _MOE_DIAG_LABELS.set(labels)


def reset_moe_diagnostic_labels(token: Any) -> None:
    _MOE_DIAG_LABELS.reset(token)


def set_moe_psd_router_features(psd: Optional[Tensor]) -> Any:
    return _MOE_PSD_ROUTER.set(psd)


def reset_moe_psd_router_features(token: Any) -> None:
    _MOE_PSD_ROUTER.reset(token)


def set_moe_faced_metadata(meta: Optional[Dict[str, Any]]) -> Any:
    """Placeholder for future cohort / segment index / age bucket (not consumed by routers yet)."""
    return _MOE_FACED_METADATA.set(meta)


def reset_moe_faced_metadata(token: Any) -> None:
    _MOE_FACED_METADATA.reset(token)


def compact_psd_bandpowers(x: Tensor, n_bands: int = PSD_ROUTER_DIM) -> Tensor:
    """Compact per-sample PSD: mean log1p power in n_bands contiguous bins over rfft axis. x: [B,C,S,T]."""
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
        bands.append(power[:, a:e].mean(dim=-1) if e > a else torch.zeros(b, device=x.device, dtype=power.dtype))
    out = torch.stack(bands, dim=-1)
    return torch.log1p(out)


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


def _attnres_base_features(baseline: Tensor, attnres: Tensor) -> Tensor:
    """[B, 3*d_model] sample-level AttnRes routing features (no PSD)."""
    return torch.cat(
        [
            _spatial_mean_bc_sd(baseline),
            _spatial_mean_bc_sd(attnres),
            _spatial_mean_bc_sd(attnres - baseline),
        ],
        dim=-1,
    )


def _attnres_router_features(
    baseline: Tensor,
    attnres: Tensor,
    use_psd: bool,
) -> Tuple[Tensor, Optional[Tuple[List[float], List[float]]]]:
    feat = _attnres_base_features(baseline, attnres)
    psd_stats: Optional[Tuple[List[float], List[float]]] = None
    if use_psd:
        psd = _MOE_PSD_ROUTER.get()
        if psd is None:
            raise ValueError(
                "moe_use_psd_router_features=True requires raw-input PSD; ensure CBraMod.forward sets context."
            )
        if psd.shape[0] != feat.shape[0] or psd.shape[-1] != PSD_ROUTER_DIM:
            raise ValueError(
                f"PSD features expected [B,{PSD_ROUTER_DIM}], got {tuple(psd.shape)} for batch {feat.shape[0]}"
            )
        feat = torch.cat([feat, psd], dim=-1)
        pd = psd.detach().float()
        psd_stats = (pd.mean(0).cpu().tolist(), pd.std(0).cpu().tolist())
    return feat, psd_stats


def _spatial_mean_bc_sd(x: Tensor) -> Tensor:
    """[B, C, S, D] -> [B, D] mean over C,S."""
    return x.mean(dim=(1, 2))


def _reshape_to_btd(x: Tensor, d: int) -> Tuple[Tensor, int, int]:
    """x [..., D] -> x_bt [B, T, D], batch B, tokens T per batch."""
    leading = x.shape[:-1]
    b = int(leading[0])
    if len(leading) == 1:
        t = 1
    else:
        rest = 1
        for s in leading[1:]:
            rest *= int(s)
        t = rest
    x_bt = x.reshape(b, t, d)
    return x_bt, b, t


class ExpertFFN(nn.Module):
    """One dense FFN: d_model -> dim_feedforward -> d_model (same as encoder linear1/linear2 path)."""

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


def _activation_from_arg(activation: Union[str, Callable[[Tensor], Tensor]], factory_kwargs) -> Callable[[Tensor], Tensor]:
    if isinstance(activation, str):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
    return activation


@torch.no_grad()
def _optional_class_conditional_moe_stats(
    routing_weights: Tensor,
    top1_assign: Tensor,
    num_experts: int,
) -> Dict[str, Any]:
    """routing_weights [U,E], top1 [U]; labels from context [U] if aligned."""
    labels = _MOE_DIAG_LABELS.get()
    out: Dict[str, Any] = {}
    if labels is None or labels.numel() != routing_weights.size(0):
        return out
    labels = labels.long().flatten()
    if labels.numel() != routing_weights.size(0):
        return out
    p = routing_weights.clamp_min(1e-10)
    ent = (-(p * p.log()).sum(dim=-1))
    n_cls = int(labels.max().item()) + 1
    sum_e = torch.zeros(n_cls, device=ent.device, dtype=ent.dtype)
    cnt = torch.zeros(n_cls, device=ent.device, dtype=ent.dtype)
    sum_e.scatter_add_(0, labels, ent)
    cnt.scatter_add_(0, labels, torch.ones_like(ent))
    mean_by_c = (sum_e / (cnt + 1e-9)).cpu().tolist()
    out["mean_entropy_by_class"] = [float(x) for x in mean_by_c]

    hist_ec = torch.zeros(num_experts, n_cls, device=top1_assign.device, dtype=torch.float32)
    for e in range(num_experts):
        m = top1_assign == e
        if not m.any():
            continue
        lbl_e = labels[m]
        for c in range(n_cls):
            hist_ec[e, c] = (lbl_e == c).float().sum()
    out["per_expert_class_histogram"] = hist_ec.cpu().tolist()
    return out


@torch.no_grad()
def build_moe_diagnostic_dict(
    router_logits: Tensor,
    routing_weights: Tensor,
    sel_idx: Tensor,
    sel_weights: Tensor,
    num_experts: int,
    top_k: int,
    last_lb_loss: Tensor,
    routing_scope: str = "token",
    mean_shared_output_norm: Optional[float] = None,
    mean_specialist_residual_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Detached stats for logging (one forward). routing_scope: token (per position) or sample (per batch item)."""
    n_units = router_logits.size(0)
    p = routing_weights.clamp_min(1e-10)
    mean_entropy = float((-(p * p.log()).sum(dim=-1)).mean().item())

    top1_assign = sel_idx[:, 0]
    hist = torch.bincount(top1_assign, minlength=num_experts).float()
    frac = hist / max(float(n_units), 1.0)
    max_frac = float(frac.max().item())

    out: Dict[str, Any] = {
        "routing_scope": routing_scope,
        "n_routing_units": int(n_units),
        "n_tokens": int(n_units),
        "mean_entropy": mean_entropy,
        "top1_histogram": hist.cpu().tolist(),
        "fraction_per_expert": frac.cpu().tolist(),
        "max_expert_fraction": max_frac,
        "router_logit_var_mean": float(router_logits.float().var(dim=0).mean().item()),
        "lb_loss": float(last_lb_loss.item()) if last_lb_loss.numel() == 1 else float(last_lb_loss.mean().item()),
    }
    if top_k >= 2:
        distinct = (sel_idx[:, 0] != sel_idx[:, 1]).float().mean().item()
        out["top1_top2_distinct_fraction"] = float(distinct)
        out["mean_weight_top1"] = float(sel_weights[:, 0].mean().item())
        out["mean_weight_top2"] = float(sel_weights[:, 1].mean().item())
    else:
        out["top1_top2_distinct_fraction"] = None
        out["mean_weight_top1"] = float(sel_weights[:, 0].mean().item())
        out["mean_weight_top2"] = None
    if mean_shared_output_norm is not None:
        out["mean_shared_output_norm"] = mean_shared_output_norm
    if mean_specialist_residual_norm is not None:
        out["mean_specialist_residual_norm"] = mean_specialist_residual_norm
    out.update(_optional_class_conditional_moe_stats(routing_weights, top1_assign, num_experts))
    return out


class SparseMoEFFN(nn.Module):
    """Sparse MoE: router picks top-k experts; output is a weighted mixture (replaces dense FFN)."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        activation: Union[str, Callable[[Tensor], Tensor]],
        bias: bool = True,
        router_noise_std: float = 0.0,
        router_mode: str = "token",
        router_arch: str = "linear",
        router_mlp_hidden: int = 128,
        use_psd_router_features: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        if router_mode not in MOE_ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {MOE_ROUTER_MODES}, got {router_mode!r}")
        if router_arch not in MOE_ROUTER_ARCHS:
            raise ValueError(f"router_arch must be one of {MOE_ROUTER_ARCHS}, got {router_arch!r}")
        if use_psd_router_features and router_mode != "sample_attnres":
            raise ValueError("moe_use_psd_router_features requires moe_router_mode=sample_attnres")
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn = _activation_from_arg(activation, factory_kwargs)

        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.router_noise_std = router_noise_std
        self.router_mode = router_mode
        self.router_arch = router_arch
        self.router_mlp_hidden = router_mlp_hidden
        self.use_psd_router_features = use_psd_router_features
        if router_mode in ("token", "sample_hidden"):
            self.router = _make_router_head(d_model, num_experts, router_arch, router_mlp_hidden, factory_kwargs)
        else:
            self.router = None
        if router_mode == "sample_attnres":
            ain = 3 * d_model + (PSD_ROUTER_DIM if use_psd_router_features else 0)
            self.router_attnres = _make_router_head(ain, num_experts, router_arch, router_mlp_hidden, factory_kwargs)
        else:
            self.router_attnres = None
        self.experts = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_experts)
        )
        self.last_diagnostics: Optional[Dict[str, Any]] = None
        self.moe_kind = "replace"

    def forward(self, x: Tensor, router_context: Optional[Dict[str, Tensor]] = None) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.size(0)
        x_bt, b, t = _reshape_to_btd(x, d)
        psd_stats: Optional[Tuple[List[float], List[float]]] = None

        if self.router_mode == "token":
            assert self.router is not None
            router_logits_flat = self.router(x_flat)
            routing_scope = "token"
            logits_for_diag = router_logits_flat
            router_in_dim_meta = d
        elif self.router_mode == "sample_hidden":
            assert self.router is not None
            pooled = x_bt.mean(dim=1)
            logits_s = self.router(pooled)
            logits_for_diag = logits_s
            router_logits_flat = logits_s.unsqueeze(1).expand(-1, t, -1).reshape(-1, self.num_experts)
            routing_scope = "sample"
            router_in_dim_meta = d
        elif self.router_mode == "sample_attnres":
            if router_context is None or router_context.get("baseline") is None or router_context.get("attnres") is None:
                raise ValueError(
                    "MoE router_mode='sample_attnres' requires encoder to pass router_context with "
                    "'baseline' and 'attnres' [B,C,S,D] (pre-attn AttnRes path)."
                )
            baseline = router_context["baseline"]
            attnres = router_context["attnres"]
            if baseline.shape != x.shape or attnres.shape != x.shape:
                raise ValueError(
                    "sample_attnres: baseline/attnres must match MoE input shape [B,C,S,D]"
                )
            feat, psd_stats = _attnres_router_features(baseline, attnres, self.use_psd_router_features)
            logits_s = self.router_attnres(feat)
            logits_for_diag = logits_s
            router_logits_flat = logits_s.unsqueeze(1).expand(-1, t, -1).reshape(-1, self.num_experts)
            routing_scope = "sample"
            router_in_dim_meta = int(feat.shape[-1])
        else:
            raise RuntimeError(f"unknown router_mode {self.router_mode}")

        if self.training and self.router_noise_std > 0:
            if routing_scope == "token":
                router_logits_flat = router_logits_flat + torch.randn_like(router_logits_flat) * self.router_noise_std
                logits_for_diag = router_logits_flat
            else:
                noise = torch.randn_like(logits_for_diag) * self.router_noise_std
                logits_for_diag = logits_for_diag + noise
                router_logits_flat = (
                    logits_for_diag.unsqueeze(1).expand(-1, t, -1).reshape(-1, self.num_experts)
                )

        routing_weights_flat = F.softmax(router_logits_flat, dim=-1)
        sel_weights, sel_idx = torch.topk(routing_weights_flat, self.top_k, dim=-1)
        sel_weights = sel_weights / (sel_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training and n_tokens > 0:
            if routing_scope == "token":
                dispatch = torch.zeros(n_tokens, self.num_experts, device=x.device, dtype=routing_weights_flat.dtype)
                for k in range(self.top_k):
                    dispatch.scatter_(1, sel_idx[:, k : k + 1], 1.0)
                fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
                pi = routing_weights_flat.mean(0)
            else:
                rw_s = F.softmax(logits_for_diag, dim=-1)
                _, sel_s = torch.topk(rw_s, self.top_k, dim=-1)
                dispatch = torch.zeros(b, self.num_experts, device=x.device, dtype=rw_s.dtype)
                for k in range(self.top_k):
                    dispatch.scatter_(1, sel_s[:, k : k + 1], 1.0)
                fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
                pi = rw_s.mean(0)
            self._last_lb_loss = self.num_experts * (fi * pi).sum()
        else:
            self._last_lb_loss = x_flat.new_zeros(())

        if routing_scope == "token":
            diag_rw = routing_weights_flat
            diag_sel_idx = sel_idx
            diag_sel_w = sel_weights
        else:
            diag_rw = F.softmax(logits_for_diag, dim=-1)
            diag_sel_w, diag_sel_idx = torch.topk(diag_rw, self.top_k, dim=-1)
            diag_sel_w = diag_sel_w / (diag_sel_w.sum(dim=-1, keepdim=True) + 1e-9)

        self.last_diagnostics = build_moe_diagnostic_dict(
            logits_for_diag,
            diag_rw,
            diag_sel_idx,
            diag_sel_w,
            self.num_experts,
            self.top_k,
            self._last_lb_loss.detach(),
            routing_scope=routing_scope,
        )
        self.last_diagnostics["moe_kind"] = "replace"
        self.last_diagnostics["router_arch"] = self.router_arch
        self.last_diagnostics["router_in_dim"] = router_in_dim_meta
        self.last_diagnostics["moe_use_psd_router_features"] = self.use_psd_router_features
        if psd_stats is not None:
            self.last_diagnostics["psd_feature_mean"] = psd_stats[0]
            self.last_diagnostics["psd_feature_std"] = psd_stats[1]

        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = sel_idx[:, k] == e
                if not mask.any():
                    continue
                y = self.experts[e](x_flat[mask])
                out_flat[mask] = out_flat[mask] + sel_weights[mask, k : k + 1] * y
        return out_flat.view(*leading, d)


class SharedSpecialistMoEFFN(nn.Module):
    """
    Always-on shared FFN (pretrained dense) + routed specialist FFNs summed as residual:
    output = shared(x) + sum_k w_k * specialist_{e_k}(x).

    Specialists start with zero output (linear2 zero) so initial behavior matches dense-only;
    they learn task-specific deviations.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_specialists: int,
        top_k: int,
        dropout: float,
        activation: Union[str, Callable[[Tensor], Tensor]],
        bias: bool = True,
        router_noise_std: float = 0.0,
        router_mode: str = "token",
        router_arch: str = "linear",
        router_mlp_hidden: int = 128,
        use_psd_router_features: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_specialists:
            raise ValueError(f"top_k must be in [1, num_specialists], got {top_k}, {num_specialists}")
        if router_mode not in MOE_ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {MOE_ROUTER_MODES}, got {router_mode!r}")
        if router_arch not in MOE_ROUTER_ARCHS:
            raise ValueError(f"router_arch must be one of {MOE_ROUTER_ARCHS}, got {router_arch!r}")
        if use_psd_router_features and router_mode != "sample_attnres":
            raise ValueError("moe_use_psd_router_features requires moe_router_mode=sample_attnres")
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn = _activation_from_arg(activation, factory_kwargs)

        self.num_specialists = num_specialists
        self.num_experts = num_specialists  # alias for diagnostics code paths
        self.top_k = top_k
        self.d_model = d_model
        self.router_noise_std = router_noise_std
        self.router_mode = router_mode
        self.router_arch = router_arch
        self.router_mlp_hidden = router_mlp_hidden
        self.use_psd_router_features = use_psd_router_features
        self.shared = ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
        if router_mode in ("token", "sample_hidden"):
            self.router = _make_router_head(d_model, num_specialists, router_arch, router_mlp_hidden, factory_kwargs)
        else:
            self.router = None
        if router_mode == "sample_attnres":
            ain = 3 * d_model + (PSD_ROUTER_DIM if use_psd_router_features else 0)
            self.router_attnres = _make_router_head(ain, num_specialists, router_arch, router_mlp_hidden, factory_kwargs)
        else:
            self.router_attnres = None
        self.specialists = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_specialists)
        )
        self._zero_specialist_output_weights()
        self.last_diagnostics: Optional[Dict[str, Any]] = None
        self.moe_kind = "shared_specialist"

    def _zero_specialist_output_weights(self) -> None:
        for s in self.specialists:
            nn.init.zeros_(s.linear2.weight)
            if s.linear2.bias is not None:
                nn.init.zeros_(s.linear2.bias)

    def forward(self, x: Tensor, router_context: Optional[Dict[str, Tensor]] = None) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.size(0)
        x_bt, b, t = _reshape_to_btd(x, d)
        h_shared = self.shared(x_flat)
        psd_stats: Optional[Tuple[List[float], List[float]]] = None

        if self.router_mode == "token":
            assert self.router is not None
            router_logits_flat = self.router(x_flat)
            routing_scope = "token"
            logits_for_diag = router_logits_flat
            router_in_dim_meta = d
        elif self.router_mode == "sample_hidden":
            assert self.router is not None
            pooled = x_bt.mean(dim=1)
            logits_s = self.router(pooled)
            logits_for_diag = logits_s
            router_logits_flat = logits_s.unsqueeze(1).expand(-1, t, -1).reshape(-1, self.num_specialists)
            routing_scope = "sample"
            router_in_dim_meta = d
        elif self.router_mode == "sample_attnres":
            if router_context is None or router_context.get("baseline") is None or router_context.get("attnres") is None:
                raise ValueError(
                    "MoE router_mode='sample_attnres' requires encoder to pass router_context with "
                    "'baseline' and 'attnres' [B,C,S,D] (pre-attn AttnRes path)."
                )
            baseline = router_context["baseline"]
            attnres = router_context["attnres"]
            if baseline.shape != x.shape or attnres.shape != x.shape:
                raise ValueError(
                    "sample_attnres: baseline/attnres must match MoE input shape [B,C,S,D]"
                )
            feat, psd_stats = _attnres_router_features(baseline, attnres, self.use_psd_router_features)
            logits_s = self.router_attnres(feat)
            logits_for_diag = logits_s
            router_logits_flat = logits_s.unsqueeze(1).expand(-1, t, -1).reshape(-1, self.num_specialists)
            routing_scope = "sample"
            router_in_dim_meta = int(feat.shape[-1])
        else:
            raise RuntimeError(f"unknown router_mode {self.router_mode}")

        if self.training and self.router_noise_std > 0:
            if routing_scope == "token":
                router_logits_flat = router_logits_flat + torch.randn_like(router_logits_flat) * self.router_noise_std
                logits_for_diag = router_logits_flat
            else:
                noise = torch.randn_like(logits_for_diag) * self.router_noise_std
                logits_for_diag = logits_for_diag + noise
                router_logits_flat = (
                    logits_for_diag.unsqueeze(1).expand(-1, t, -1).reshape(-1, self.num_specialists)
                )

        routing_weights_flat = F.softmax(router_logits_flat, dim=-1)
        sel_weights, sel_idx = torch.topk(routing_weights_flat, self.top_k, dim=-1)
        sel_weights = sel_weights / (sel_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training and n_tokens > 0:
            if routing_scope == "token":
                dispatch = torch.zeros(n_tokens, self.num_specialists, device=x.device, dtype=routing_weights_flat.dtype)
                for k in range(self.top_k):
                    dispatch.scatter_(1, sel_idx[:, k : k + 1], 1.0)
                fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
                pi = routing_weights_flat.mean(0)
            else:
                rw_s = F.softmax(logits_for_diag, dim=-1)
                _, sel_s = torch.topk(rw_s, self.top_k, dim=-1)
                dispatch = torch.zeros(b, self.num_specialists, device=x.device, dtype=rw_s.dtype)
                for k in range(self.top_k):
                    dispatch.scatter_(1, sel_s[:, k : k + 1], 1.0)
                fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
                pi = rw_s.mean(0)
            self._last_lb_loss = self.num_specialists * (fi * pi).sum()
        else:
            self._last_lb_loss = x_flat.new_zeros(())

        if routing_scope == "token":
            diag_rw = routing_weights_flat
            diag_sel_idx = sel_idx
            diag_sel_w = sel_weights
        else:
            diag_rw = F.softmax(logits_for_diag, dim=-1)
            diag_sel_w, diag_sel_idx = torch.topk(diag_rw, self.top_k, dim=-1)
            diag_sel_w = diag_sel_w / (diag_sel_w.sum(dim=-1, keepdim=True) + 1e-9)

        residual = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_specialists):
                mask = sel_idx[:, k] == e
                if not mask.any():
                    continue
                y = self.specialists[e](x_flat[mask])
                residual[mask] = residual[mask] + sel_weights[mask, k : k + 1] * y

        out = h_shared + residual
        mean_shared_norm = float(h_shared.float().norm(dim=-1).mean().item())
        mean_res_norm = float(residual.float().norm(dim=-1).mean().item())

        self.last_diagnostics = build_moe_diagnostic_dict(
            logits_for_diag,
            diag_rw,
            diag_sel_idx,
            diag_sel_w,
            self.num_specialists,
            self.top_k,
            self._last_lb_loss.detach(),
            routing_scope=routing_scope,
            mean_shared_output_norm=mean_shared_norm,
            mean_specialist_residual_norm=mean_res_norm,
        )
        self.last_diagnostics["moe_kind"] = "shared_specialist"
        self.last_diagnostics["router_arch"] = self.router_arch
        self.last_diagnostics["router_in_dim"] = router_in_dim_meta
        self.last_diagnostics["moe_use_psd_router_features"] = self.use_psd_router_features
        if psd_stats is not None:
            self.last_diagnostics["psd_feature_mean"] = psd_stats[0]
            self.last_diagnostics["psd_feature_std"] = psd_stats[1]

        return out.view(*leading, d)


def _spectral_router_input_feats(
    base_feat: Tensor,
    use_psd: bool,
) -> Tuple[Tensor, Optional[Tuple[List[float], List[float]]]]:
    """Spectral bank: same AttnRes features as spatial, optionally + PSD (only spectral router)."""
    if not use_psd:
        return base_feat, None
    psd = _MOE_PSD_ROUTER.get()
    if psd is None:
        raise ValueError(
            "Spectral router with PSD requires moe_use_psd_router_features and CBraMod.forward PSD context."
        )
    if psd.shape[0] != base_feat.shape[0] or psd.shape[-1] != PSD_ROUTER_DIM:
        raise ValueError(f"PSD expected [B,{PSD_ROUTER_DIM}], got {tuple(psd.shape)}")
    out = torch.cat([base_feat, psd], dim=-1)
    pd = psd.detach().float()
    return out, (pd.mean(0).cpu().tolist(), pd.std(0).cpu().tolist())


class TypedDualBankSharedMoEFFN(nn.Module):
    """
    Shared dense FFN + two specialist banks (spatial / spectral), one expert per bank per sample:
    y = shared(x) + spatial_e(x) + spectral_e'(x). Routers: sample-level AttnRes features; PSD only on spectral.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_specialists: int,
        dropout: float,
        activation: Union[str, Callable[[Tensor], Tensor]],
        bias: bool = True,
        router_noise_std: float = 0.0,
        router_arch: str = "linear",
        router_mlp_hidden: int = 128,
        use_psd_router_features: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if router_arch not in MOE_ROUTER_ARCHS:
            raise ValueError(f"router_arch must be one of {MOE_ROUTER_ARCHS}, got {router_arch!r}")
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn = _activation_from_arg(activation, factory_kwargs)

        self.d_model = d_model
        self.num_specialists = num_specialists
        self.num_experts = num_specialists
        self.top_k = 1
        self.router_noise_std = router_noise_std
        self.router_arch = router_arch
        self.router_mlp_hidden = router_mlp_hidden
        self.use_psd_router_features = use_psd_router_features
        self.shared = ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)

        spatial_in = 3 * d_model
        spectral_in = spatial_in + (PSD_ROUTER_DIM if use_psd_router_features else 0)
        self.spatial_router = _make_router_head(
            spatial_in, num_specialists, router_arch, router_mlp_hidden, factory_kwargs
        )
        self.spectral_router = _make_router_head(
            spectral_in, num_specialists, router_arch, router_mlp_hidden, factory_kwargs
        )
        self.spatial_specialists = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_specialists)
        )
        self.spectral_specialists = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_specialists)
        )
        self._zero_specialist_output_weights()
        self.last_diagnostics: Optional[Dict[str, Any]] = None
        self.moe_kind = "typed_shared_specialist"

    def _zero_specialist_output_weights(self) -> None:
        for bank in (self.spatial_specialists, self.spectral_specialists):
            for s in bank:
                nn.init.zeros_(s.linear2.weight)
                if s.linear2.bias is not None:
                    nn.init.zeros_(s.linear2.bias)

    def forward(self, x: Tensor, router_context: Optional[Dict[str, Tensor]] = None) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.size(0)
        x_bt, b, t = _reshape_to_btd(x, d)
        h_shared = self.shared(x_flat)

        if router_context is None or router_context.get("baseline") is None or router_context.get("attnres") is None:
            raise ValueError(
                "TypedDualBankSharedMoEFFN requires router_context with 'baseline' and 'attnres' [B,C,S,D]."
            )
        baseline = router_context["baseline"]
        attnres = router_context["attnres"]
        if baseline.shape != x.shape or attnres.shape != x.shape:
            raise ValueError("typed MoE: baseline/attnres must match FFN input shape [B,C,S,D]")

        base_feat = _attnres_base_features(baseline, attnres)
        spec_in, psd_stats = _spectral_router_input_feats(base_feat, self.use_psd_router_features)

        logits_sp = self.spatial_router(base_feat)
        logits_spec = self.spectral_router(spec_in)

        if self.training and self.router_noise_std > 0:
            logits_sp = logits_sp + torch.randn_like(logits_sp) * self.router_noise_std
            logits_spec = logits_spec + torch.randn_like(logits_spec) * self.router_noise_std

        rw_sp = F.softmax(logits_sp, dim=-1)
        rw_spec = F.softmax(logits_spec, dim=-1)
        sel_w_sp, sel_idx_sp = torch.topk(rw_sp, 1, dim=-1)
        sel_w_spec, sel_idx_spec = torch.topk(rw_spec, 1, dim=-1)

        E = self.num_specialists
        lb_sp = x_flat.new_zeros(())
        lb_s = x_flat.new_zeros(())
        if self.training and b > 0:
            dispatch_sp = torch.zeros(b, E, device=x.device, dtype=rw_sp.dtype)
            dispatch_sp.scatter_(1, sel_idx_sp, 1.0)
            fi_sp = dispatch_sp.mean(0).detach()
            pi_sp = rw_sp.mean(0)
            lb_sp = E * (fi_sp * pi_sp).sum()

            dispatch_spec = torch.zeros(b, E, device=x.device, dtype=rw_spec.dtype)
            dispatch_spec.scatter_(1, sel_idx_spec, 1.0)
            fi_s = dispatch_spec.mean(0).detach()
            pi_s = rw_spec.mean(0)
            lb_s = E * (fi_s * pi_s).sum()
            self._last_lb_loss = lb_sp + lb_s
        else:
            self._last_lb_loss = x_flat.new_zeros(())

        batch_idx = torch.arange(b, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, t).reshape(-1)
        per_tok_sp = sel_idx_sp.squeeze(-1)[batch_idx]
        per_tok_spec = sel_idx_spec.squeeze(-1)[batch_idx]

        res_sp = torch.zeros_like(x_flat)
        res_spec = torch.zeros_like(x_flat)
        for e in range(E):
            m = per_tok_sp == e
            if m.any():
                res_sp[m] = self.spatial_specialists[e](x_flat[m])
        for e in range(E):
            m = per_tok_spec == e
            if m.any():
                res_spec[m] = self.spectral_specialists[e](x_flat[m])

        out = h_shared + res_sp + res_spec
        mean_shared = float(h_shared.float().norm(dim=-1).mean().item())
        mean_n_sp = float(res_sp.float().norm(dim=-1).mean().item())
        mean_n_spec = float(res_spec.float().norm(dim=-1).mean().item())

        diag_sp = build_moe_diagnostic_dict(
            logits_sp,
            rw_sp,
            sel_idx_sp,
            sel_w_sp,
            E,
            1,
            lb_sp.detach() if self.training else lb_sp,
            routing_scope="sample",
            mean_shared_output_norm=None,
            mean_specialist_residual_norm=mean_n_sp,
        )
        diag_sp["moe_kind"] = "typed_bank_spatial"

        diag_spec = build_moe_diagnostic_dict(
            logits_spec,
            rw_spec,
            sel_idx_spec,
            sel_w_spec,
            E,
            1,
            lb_s.detach() if self.training else lb_s,
            routing_scope="sample",
            mean_shared_output_norm=None,
            mean_specialist_residual_norm=mean_n_spec,
        )
        diag_spec["moe_kind"] = "typed_bank_spectral"

        self.last_diagnostics = {
            "moe_kind": "typed_shared_specialist",
            "routing_scope": "sample",
            "router_arch": self.router_arch,
            "spatial_router_in_dim": 3 * self.d_model,
            "spectral_router_in_dim": int(spec_in.shape[-1]),
            "psd_on_spectral_router": self.use_psd_router_features,
            "spatial": diag_sp,
            "spectral": diag_spec,
            "mean_shared_output_norm": mean_shared,
            "mean_spatial_residual_norm": mean_n_sp,
            "mean_spectral_residual_norm": mean_n_spec,
            "moe_use_psd_router_features": self.use_psd_router_features,
        }
        if psd_stats is not None:
            self.last_diagnostics["psd_feature_mean"] = psd_stats[0]
            self.last_diagnostics["psd_feature_std"] = psd_stats[1]

        with torch.no_grad():
            eps = 1e-10
            ent_sp = -(rw_sp * (rw_sp.clamp_min(eps)).log()).sum(dim=-1)
            ent_sc = -(rw_spec * (rw_spec.clamp_min(eps)).log()).sum(dim=-1)
        self._routing_export_cache = {
            "logits_spatial": logits_sp.detach().cpu(),
            "logits_spectral": logits_spec.detach().cpu(),
            "probs_spatial": rw_sp.detach().cpu(),
            "probs_spectral": rw_spec.detach().cpu(),
            "entropy_spatial": ent_sp.cpu(),
            "entropy_spectral": ent_sc.cpu(),
            "expert_spatial": sel_idx_sp.squeeze(-1).detach().cpu(),
            "expert_spectral": sel_idx_spec.squeeze(-1).detach().cpu(),
        }

        return out.view(*leading, d)


@torch.no_grad()
def warm_start_sparse_moe_from_dense_ckpt(
    moe: SparseMoEFFN,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_to_all_experts: bool,
) -> None:
    """Copy pretrained encoder.layers.{i}.linear1/2 into each expert (replacement MoE)."""
    p1w = f"encoder.layers.{layer_idx}.linear1.weight"
    p1b = f"encoder.layers.{layer_idx}.linear1.bias"
    p2w = f"encoder.layers.{layer_idx}.linear2.weight"
    p2b = f"encoder.layers.{layer_idx}.linear2.bias"
    for key in (p1w, p2w):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key} for MoE warm-start at layer {layer_idx}")
    w1, w2 = ckpt[p1w], ckpt[p2w]
    b1 = ckpt.get(p1b)
    b2 = ckpt.get(p2b)
    for e_idx, expert in enumerate(moe.experts):
        if copy_to_all_experts or e_idx == 0:
            expert.linear1.weight.copy_(w1)
            expert.linear2.weight.copy_(w2)
            if b1 is not None and expert.linear1.bias is not None:
                expert.linear1.bias.copy_(b1)
            if b2 is not None and expert.linear2.bias is not None:
                expert.linear2.bias.copy_(b2)


@torch.no_grad()
def warm_start_shared_specialist_from_dense_ckpt(
    moe: SharedSpecialistMoEFFN,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_dense_into_specialist_linear1: bool,
) -> None:
    """
    Shared path: full dense FFN from checkpoint.
    Specialists: optional copy of linear1 from dense; linear2 stays zero (residual specialists).
    """
    p1w = f"encoder.layers.{layer_idx}.linear1.weight"
    p1b = f"encoder.layers.{layer_idx}.linear1.bias"
    p2w = f"encoder.layers.{layer_idx}.linear2.weight"
    p2b = f"encoder.layers.{layer_idx}.linear2.bias"
    for key in (p1w, p2w):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key} for shared-specialist warm-start at layer {layer_idx}")
    w1, w2 = ckpt[p1w], ckpt[p2w]
    b1 = ckpt.get(p1b)
    b2 = ckpt.get(p2b)

    for mod in (moe.shared,):
        mod.linear1.weight.copy_(w1)
        mod.linear2.weight.copy_(w2)
        if b1 is not None and mod.linear1.bias is not None:
            mod.linear1.bias.copy_(b1)
        if b2 is not None and mod.linear2.bias is not None:
            mod.linear2.bias.copy_(b2)

    # Tiny iid noise breaks symmetry when the same dense linear1 is copied to every specialist —
    # otherwise hidden transforms match and the router collapses (diagnostics: max_expert_frac=1).
    _sym_eps = 1e-4
    for spec in moe.specialists:
        if copy_dense_into_specialist_linear1:
            spec.linear1.weight.copy_(w1)
            if b1 is not None and spec.linear1.bias is not None:
                spec.linear1.bias.copy_(b1)
            spec.linear1.weight.add_(torch.randn_like(spec.linear1.weight) * _sym_eps)
            if spec.linear1.bias is not None:
                spec.linear1.bias.add_(torch.randn_like(spec.linear1.bias) * _sym_eps)
    moe._zero_specialist_output_weights()


@torch.no_grad()
def warm_start_typed_dual_bank_from_dense_ckpt(
    moe: TypedDualBankSharedMoEFFN,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_dense_into_specialist_linear1: bool,
) -> None:
    """Shared path from dense; both banks get optional linear1 copy + symmetry noise; linear2 zero."""
    p1w = f"encoder.layers.{layer_idx}.linear1.weight"
    p1b = f"encoder.layers.{layer_idx}.linear1.bias"
    p2w = f"encoder.layers.{layer_idx}.linear2.weight"
    p2b = f"encoder.layers.{layer_idx}.linear2.bias"
    for key in (p1w, p2w):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing {key} for typed MoE warm-start at layer {layer_idx}")
    w1, w2 = ckpt[p1w], ckpt[p2w]
    b1, b2 = ckpt.get(p1b), ckpt.get(p2b)

    for mod in (moe.shared,):
        mod.linear1.weight.copy_(w1)
        mod.linear2.weight.copy_(w2)
        if b1 is not None and mod.linear1.bias is not None:
            mod.linear1.bias.copy_(b1)
        if b2 is not None and mod.linear2.bias is not None:
            mod.linear2.bias.copy_(b2)

    _sym_eps = 1e-4
    for bank in (moe.spatial_specialists, moe.spectral_specialists):
        for spec in bank:
            if copy_dense_into_specialist_linear1:
                spec.linear1.weight.copy_(w1)
                if b1 is not None and spec.linear1.bias is not None:
                    spec.linear1.bias.copy_(b1)
                spec.linear1.weight.add_(torch.randn_like(spec.linear1.weight) * _sym_eps)
                if spec.linear1.bias is not None:
                    spec.linear1.bias.add_(torch.randn_like(spec.linear1.bias) * _sym_eps)
    moe._zero_specialist_output_weights()


def warm_start_moe_from_dense_ckpt(
    moe: nn.Module,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_to_all_experts: bool,
    moe_shared_specialist: bool,
    copy_specialist_linear1_from_dense: bool = True,
) -> None:
    """Dispatch warm-start for replacement MoE vs shared+specialist MoE."""
    if moe_shared_specialist:
        if isinstance(moe, TypedDualBankSharedMoEFFN):
            warm_start_typed_dual_bank_from_dense_ckpt(
                moe, ckpt, layer_idx, copy_dense_into_specialist_linear1=copy_specialist_linear1_from_dense
            )
        else:
            warm_start_shared_specialist_from_dense_ckpt(
                moe, ckpt, layer_idx, copy_dense_into_specialist_linear1=copy_specialist_linear1_from_dense
            )
    else:
        warm_start_sparse_moe_from_dense_ckpt(moe, ckpt, layer_idx, copy_to_all_experts)


def _format_typed_moe_diagnostics_lines(layer_idx: int, diag: Dict[str, Any]) -> List[str]:
    lines = [
        f"  [MoE L{layer_idx}] kind=typed_shared_specialist  scope=sample  "
        f"router_arch={diag.get('router_arch')}  "
        f"spatial_in_dim={diag.get('spatial_router_in_dim')}  "
        f"spectral_in_dim={diag.get('spectral_router_in_dim')}  "
        f"psd_on_spectral={diag.get('psd_on_spectral_router')}  "
        f"||shared||={diag.get('mean_shared_output_norm', 0):.4f}  "
        f"||spatial_res||={diag.get('mean_spatial_residual_norm', 0):.4f}  "
        f"||spectral_res||={diag.get('mean_spectral_residual_norm', 0):.4f}",
    ]
    if diag.get("psd_feature_mean") is not None and diag.get("psd_feature_std") is not None:
        lines.append(
            f"    psd_mean={ [round(v, 4) for v in diag['psd_feature_mean']]}  "
            f"psd_std={ [round(v, 4) for v in diag['psd_feature_std']]}"
        )
    for label, key in (("spatial bank", "spatial"), ("spectral bank", "spectral")):
        sub = diag.get(key)
        if not isinstance(sub, dict):
            continue
        lines.append(
            f"    [{label}] H={sub.get('mean_entropy', 0):.4f}  max_expert_frac={sub.get('max_expert_fraction', 0):.4f}  "
            f"lb={sub.get('lb_loss', 0):.6f}  ||res||={sub.get('mean_specialist_residual_norm', 0):.4f}"
        )
        lines.append(f"      hist_top1: {sub.get('top1_histogram')}  frac: {sub.get('fraction_per_expert')}")
        if sub.get("mean_entropy_by_class") is not None:
            lines.append(f"      H_mean_by_class: {[round(h, 4) for h in sub['mean_entropy_by_class']]}")
        if sub.get("per_expert_class_histogram") is not None:
            lines.append(f"      per_expert_class_hist: {sub['per_expert_class_histogram']}")
    return lines


def format_moe_diagnostics_lines(layer_idx: int, diag: Dict[str, Any]) -> List[str]:
    """Human-readable lines for logging."""
    if diag.get("moe_kind") == "typed_shared_specialist":
        return _format_typed_moe_diagnostics_lines(layer_idx, diag)
    scope = diag.get("routing_scope", "token")
    n_u = diag.get("n_routing_units", diag.get("n_tokens", 0))
    r_arch = diag.get("router_arch", "linear")
    r_in = diag.get("router_in_dim", "?")
    use_psd = diag.get("moe_use_psd_router_features", False)
    lines = [
        f"  [MoE L{layer_idx}] kind={diag.get('moe_kind', 'replace')}  scope={scope}  n_units={n_u}  "
        f"router_arch={r_arch}  router_in_dim={r_in}  psd_feats={use_psd}  "
        f"H_mean={diag['mean_entropy']:.4f}  max_expert_frac={diag['max_expert_fraction']:.4f}  "
        f"lb={diag['lb_loss']:.6f}  logit_var={diag['router_logit_var_mean']:.6f}",
        f"    histogram_top1: {diag['top1_histogram']}  frac: {[round(f, 4) for f in diag['fraction_per_expert']]}",
    ]
    if diag.get("psd_feature_mean") is not None and diag.get("psd_feature_std") is not None:
        lines.append(
            f"    psd_mean={ [round(v, 4) for v in diag['psd_feature_mean']]}  "
            f"psd_std={ [round(v, 4) for v in diag['psd_feature_std']]}"
        )
    if diag.get("top1_top2_distinct_fraction") is not None:
        lines.append(
            f"    top1!=top2_frac={diag['top1_top2_distinct_fraction']:.4f}  "
            f"w_top1={diag.get('mean_weight_top1', 0):.4f}  w_top2={diag.get('mean_weight_top2', 0):.4f}"
        )
    if diag.get("mean_shared_output_norm") is not None:
        lines.append(
            f"    ||shared||_mean={diag['mean_shared_output_norm']:.4f}  "
            f"||spec_residual||_mean={diag.get('mean_specialist_residual_norm', 0):.4f}"
        )
    if diag.get("mean_entropy_by_class") is not None:
        lines.append(f"    H_mean_by_class: {[round(h, 4) for h in diag['mean_entropy_by_class']]}")
    if diag.get("per_expert_class_histogram") is not None:
        lines.append(f"    per_expert_class_hist: {diag['per_expert_class_histogram']}")
    return lines
