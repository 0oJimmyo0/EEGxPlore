"""MoE FFN variants for CBraMod: replacement sparse MoE, or shared dense + specialist residuals."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
def build_moe_diagnostic_dict(
    router_logits: Tensor,
    routing_weights: Tensor,
    sel_idx: Tensor,
    sel_weights: Tensor,
    num_experts: int,
    top_k: int,
    last_lb_loss: Tensor,
) -> Dict[str, Any]:
    """Detached stats for logging (one forward). Answers: collapse, traffic, dominance, entropy."""
    n_tokens = router_logits.size(0)
    p = routing_weights.clamp_min(1e-10)
    mean_entropy = float((-(p * p.log()).sum(dim=-1)).mean().item())

    # Top-1 assignment = primary expert in top-k selection
    top1_assign = sel_idx[:, 0]
    hist = torch.bincount(top1_assign, minlength=num_experts).float()
    frac = hist / max(float(n_tokens), 1.0)
    max_frac = float(frac.max().item())

    out: Dict[str, Any] = {
        "n_tokens": int(n_tokens),
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
    return out


class SparseMoEFFN(nn.Module):
    """Token-wise sparse MoE: router picks top-k experts; output is a weighted mixture (replaces dense FFN)."""

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
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got top_k={top_k}, num_experts={num_experts}")
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn = _activation_from_arg(activation, factory_kwargs)

        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.router_noise_std = router_noise_std
        self.router = nn.Linear(d_model, num_experts, bias=True, **factory_kwargs)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
            for _ in range(num_experts)
        )
        self.last_diagnostics: Optional[Dict[str, Any]] = None
        self.moe_kind = "replace"

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.size(0)
        router_logits = self.router(x_flat)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std
        routing_weights = F.softmax(router_logits, dim=-1)
        sel_weights, sel_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        sel_weights = sel_weights / (sel_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training and n_tokens > 0:
            dispatch = torch.zeros(n_tokens, self.num_experts, device=x.device, dtype=routing_weights.dtype)
            for k in range(self.top_k):
                dispatch.scatter_(1, sel_idx[:, k : k + 1], 1.0)
            fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
            pi = routing_weights.mean(0)
            self._last_lb_loss = self.num_experts * (fi * pi).sum()
        else:
            self._last_lb_loss = x_flat.new_zeros(())

        self.last_diagnostics = build_moe_diagnostic_dict(
            router_logits, routing_weights, sel_idx, sel_weights,
            self.num_experts, self.top_k, self._last_lb_loss.detach(),
        )
        self.last_diagnostics["moe_kind"] = "replace"

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
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_specialists:
            raise ValueError(f"top_k must be in [1, num_specialists], got {top_k}, {num_specialists}")
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_fn = _activation_from_arg(activation, factory_kwargs)

        self.num_specialists = num_specialists
        self.num_experts = num_specialists  # alias for diagnostics code paths
        self.top_k = top_k
        self.d_model = d_model
        self.router_noise_std = router_noise_std
        self.shared = ExpertFFN(d_model, dim_feedforward, dropout, activation_fn, bias=bias, **factory_kwargs)
        self.router = nn.Linear(d_model, num_specialists, bias=True, **factory_kwargs)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
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

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        d = x.shape[-1]
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.size(0)
        h_shared = self.shared(x_flat)

        router_logits = self.router(x_flat)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std
        routing_weights = F.softmax(router_logits, dim=-1)
        sel_weights, sel_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        sel_weights = sel_weights / (sel_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if self.training and n_tokens > 0:
            dispatch = torch.zeros(n_tokens, self.num_specialists, device=x.device, dtype=routing_weights.dtype)
            for k in range(self.top_k):
                dispatch.scatter_(1, sel_idx[:, k : k + 1], 1.0)
            fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
            pi = routing_weights.mean(0)
            self._last_lb_loss = self.num_specialists * (fi * pi).sum()
        else:
            self._last_lb_loss = x_flat.new_zeros(())

        self.last_diagnostics = build_moe_diagnostic_dict(
            router_logits, routing_weights, sel_idx, sel_weights,
            self.num_specialists, self.top_k, self._last_lb_loss.detach(),
        )
        self.last_diagnostics["moe_kind"] = "shared_specialist"

        residual = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_specialists):
                mask = sel_idx[:, k] == e
                if not mask.any():
                    continue
                y = self.specialists[e](x_flat[mask])
                residual[mask] = residual[mask] + sel_weights[mask, k : k + 1] * y

        out = h_shared + residual
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
        warm_start_shared_specialist_from_dense_ckpt(
            moe, ckpt, layer_idx, copy_dense_into_specialist_linear1=copy_specialist_linear1_from_dense
        )
    else:
        warm_start_sparse_moe_from_dense_ckpt(moe, ckpt, layer_idx, copy_to_all_experts)


def format_moe_diagnostics_lines(layer_idx: int, diag: Dict[str, Any]) -> List[str]:
    """Human-readable lines for logging."""
    lines = [
        f"  [MoE L{layer_idx}] kind={diag.get('moe_kind', 'replace')}  n_tokens={diag['n_tokens']}  "
        f"H_mean={diag['mean_entropy']:.4f}  max_expert_frac={diag['max_expert_fraction']:.4f}  "
        f"lb={diag['lb_loss']:.6f}  logit_var={diag['router_logit_var_mean']:.6f}",
        f"    histogram_top1: {diag['top1_histogram']}  frac: {[round(f, 4) for f in diag['fraction_per_expert']]}",
    ]
    if diag.get("top1_top2_distinct_fraction") is not None:
        lines.append(
            f"    top1!=top2_frac={diag['top1_top2_distinct_fraction']:.4f}  "
            f"w_top1={diag.get('mean_weight_top1', 0):.4f}  w_top2={diag.get('mean_weight_top2', 0):.4f}"
        )
    return lines
