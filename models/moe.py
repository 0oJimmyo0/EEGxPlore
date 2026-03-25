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
        self.load_balance_coef = float(load_balance_coef)
        self.domain_bias_reg_coef = float(domain_bias_reg_coef)

        self.shared = ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)

        spatial_in = 3 * d_model
        spectral_in = spatial_in + (PSD_ROUTER_DIM if self.use_psd_router_features else 0)
        self.spatial_router = _make_router_head(
            spatial_in, num_specialists, router_arch, self.router_mlp_hidden, factory_kwargs
        )
        self.spectral_router = _make_router_head(
            spectral_in, num_specialists, router_arch, self.router_mlp_hidden, factory_kwargs
        )

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

        z = torch.zeros((), device=device, dtype=torch.float32)
        self._last_lb_loss = z
        self._last_overflow_penalty = z
        self._last_domain_bias_reg = z
        self._last_aux_loss = z

        self.last_diagnostics: Optional[Dict[str, Any]] = None
        self._routing_export_cache: Dict[str, Tensor] = {}

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
    def _capacity_assign_top1(logits: Tensor, capacity: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Greedy top-1 with per-expert capacity and next-best fallback."""
        b, e = logits.shape
        assigned = torch.full((b,), -1, device=logits.device, dtype=torch.long)
        counts = torch.zeros(e, device=logits.device, dtype=torch.long)
        if b == 0:
            return assigned, torch.zeros(0, device=logits.device, dtype=torch.bool), counts

        order = logits.max(dim=1).values.argsort(descending=True)
        ranked = logits.argsort(dim=1, descending=True)
        for i in order.tolist():
            for k in range(e):
                ex = int(ranked[i, k].item())
                if int(counts[ex].item()) < capacity:
                    assigned[i] = ex
                    counts[ex] += 1
                    break
        fallback = assigned < 0
        return assigned, fallback, counts

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

    def _bank_load_balance(self, probs: Tensor, assigned: Tensor) -> Tensor:
        if probs.numel() == 0:
            return probs.new_zeros(())
        e = probs.size(1)
        valid = assigned >= 0
        if not valid.any():
            fi = probs.new_zeros((e,))
        else:
            onehot = F.one_hot(assigned[valid], num_classes=e).to(dtype=probs.dtype)
            fi = onehot.mean(dim=0).detach()
        pi = probs.mean(dim=0)
        return e * (fi * pi).sum()

    def _bank_residual(self, x_flat: Tensor, sample_assign: Tensor, batch_idx: Tensor, bank: nn.ModuleList) -> Tensor:
        res = torch.zeros_like(x_flat)
        per_token = sample_assign[batch_idx]
        for e in range(self.num_specialists):
            m = per_token == e
            if m.any():
                res[m] = bank[e](x_flat[m])
        return res

    def forward(self, x: Tensor, router_context: Optional[Dict[str, Tensor]] = None) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        h_shared = self.shared(x_flat)

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

        base_feat = _attnres_base_features(baseline, attnres)
        spectral_in = base_feat
        psd_stats: Optional[Tuple[List[float], List[float]]] = None
        if self.use_psd_router_features:
            psd = _MOE_PSD_ROUTER.get()
            if psd is None:
                raise ValueError("moe_use_psd_router_features=True requires PSD context from backbone")
            if psd.shape[0] != b or psd.shape[-1] != PSD_ROUTER_DIM:
                raise ValueError(f"PSD expected [B,{PSD_ROUTER_DIM}], got {tuple(psd.shape)}")
            spectral_in = torch.cat([base_feat, psd], dim=-1)
            pd = psd.detach().float()
            psd_stats = (pd.mean(0).cpu().tolist(), pd.std(0).cpu().tolist())

        logits_sp_content = self.spatial_router(base_feat)
        logits_sc_content = self.spectral_router(spectral_in)

        bias_sp = self._domain_logit_bias(b, "spatial", x.device)
        bias_sc = self._domain_logit_bias(b, "spectral", x.device)

        logits_sp = logits_sp_content + bias_sp
        logits_sc = logits_sc_content + bias_sc

        probs_sp = F.softmax(logits_sp, dim=-1)
        probs_sc = F.softmax(logits_sc, dim=-1)

        capacity = max(1, int(math.ceil(self.capacity_factor * b / float(self.num_specialists))))

        assign_sp, fallback_sp, counts_sp = self._capacity_assign_top1(logits_sp, capacity)
        assign_sc, fallback_sc, counts_sc = self._capacity_assign_top1(logits_sc, capacity)

        res_sp = self._bank_residual(x_flat, assign_sp, batch_idx, self.spatial_specialists)
        res_sc = self._bank_residual(x_flat, assign_sc, batch_idx, self.spectral_specialists)

        out = h_shared + res_sp + res_sc

        lb_sp = self._bank_load_balance(probs_sp, assign_sp)
        lb_sc = self._bank_load_balance(probs_sc, assign_sc)
        lb = lb_sp + lb_sc

        overflow = fallback_sp.float().mean() + fallback_sc.float().mean()
        domain_reg = bias_sp.pow(2).mean() + bias_sc.pow(2).mean()

        self._last_lb_loss = lb
        self._last_overflow_penalty = overflow
        self._last_domain_bias_reg = domain_reg
        self._last_aux_loss = (
            self.load_balance_coef * lb
            + overflow
            + self.domain_bias_reg_coef * domain_reg
        )

        pre_ent_sp = self._sample_entropy(probs_sp)
        pre_ent_sc = self._sample_entropy(probs_sc)
        post_ent_sp = self._hist_entropy(counts_sp)
        post_ent_sc = self._hist_entropy(counts_sc)
        dom_norm = float(torch.cat([bias_sp, bias_sc], dim=1).float().norm(dim=-1).mean().item())

        self.last_diagnostics = {
            "moe_kind": self.moe_kind,
            "capacity": int(capacity),
            "num_experts": int(self.num_specialists),
            "domain_bias_enabled": bool(self.domain_bias),
            "domain_bias_norm": dom_norm,
            "mean_shared_output_norm": float(h_shared.float().norm(dim=-1).mean().item()),
            "mean_spatial_residual_norm": float(res_sp.float().norm(dim=-1).mean().item()),
            "mean_spectral_residual_norm": float(res_sc.float().norm(dim=-1).mean().item()),
            "aux_load_balance": float(lb.detach().item()),
            "aux_overflow": float(overflow.detach().item()),
            "aux_domain_bias_reg": float(domain_reg.detach().item()),
            "aux_total": float(self._last_aux_loss.detach().item()),
            "spatial": {
                "assigned_count_per_expert": counts_sp.detach().cpu().tolist(),
                "overflow_count": int(fallback_sp.sum().item()),
                "shared_only_fraction": float(fallback_sp.float().mean().item()),
                "routing_entropy_pre_capacity": pre_ent_sp,
                "routing_entropy_post_assignment": post_ent_sp,
            },
            "spectral": {
                "assigned_count_per_expert": counts_sc.detach().cpu().tolist(),
                "overflow_count": int(fallback_sc.sum().item()),
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
            "assigned_spatial": assign_sp.detach().cpu(),
            "assigned_spectral": assign_sc.detach().cpu(),
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

    sym_eps = 1e-4
    if copy_dense_into_specialist_linear1:
        for bank in (moe.spatial_specialists, moe.spectral_specialists):
            for spec in bank:
                spec.linear1.weight.copy_(w1)
                if b1 is not None and spec.linear1.bias is not None:
                    spec.linear1.bias.copy_(b1)
                spec.linear1.weight.add_(torch.randn_like(spec.linear1.weight) * sym_eps)
                if spec.linear1.bias is not None:
                    spec.linear1.bias.add_(torch.randn_like(spec.linear1.bias) * sym_eps)

    moe._zero_specialist_output_weights()


def warm_start_moe_from_dense_ckpt(
    moe: nn.Module,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_specialist_linear1_from_dense: bool = True,
) -> None:
    if not isinstance(moe, TypedCapacityDomainMoEFFN):
        raise TypeError(f"Expected TypedCapacityDomainMoEFFN, got {type(moe)}")
    warm_start_typed_capacity_domain_from_dense_ckpt(
        moe,
        ckpt,
        layer_idx,
        copy_dense_into_specialist_linear1=copy_specialist_linear1_from_dense,
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
            f"domain_bias_norm={diag.get('domain_bias_norm', 0.0):.6f}"
        ),
        (
            f"    aux_total={diag.get('aux_total', 0.0):.6f}  lb={diag.get('aux_load_balance', 0.0):.6f}  "
            f"overflow={diag.get('aux_overflow', 0.0):.6f}  domain_reg={diag.get('aux_domain_bias_reg', 0.0):.6f}"
        ),
        (
            f"    [spatial] assigned={sp.get('assigned_count_per_expert')}  overflow={sp.get('overflow_count')}  "
            f"shared_only_frac={sp.get('shared_only_fraction', 0.0):.4f}  "
            f"H_pre={sp.get('routing_entropy_pre_capacity', 0.0):.4f}  "
            f"H_post={sp.get('routing_entropy_post_assignment', 0.0):.4f}"
        ),
        (
            f"    [spectral] assigned={sc.get('assigned_count_per_expert')}  overflow={sc.get('overflow_count')}  "
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
