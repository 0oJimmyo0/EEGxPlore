"""Sparse top-k MoE FFN matching CBraMod TransformerEncoderLayer FFN (GELU, same dims)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

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


class SparseMoEFFN(nn.Module):
    """Token-wise sparse MoE: router picks top-k experts; outputs are weighted sums."""

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
        if isinstance(activation, str):
            if activation == "relu":
                activation = F.relu
            elif activation == "gelu":
                activation = F.gelu
            else:
                raise RuntimeError(f"activation should be relu/gelu, not {activation}")

        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.router_noise_std = router_noise_std
        self.router = nn.Linear(d_model, num_experts, bias=True, **factory_kwargs)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList(
            ExpertFFN(d_model, dim_feedforward, dropout, activation, bias=bias, **factory_kwargs)
            for _ in range(num_experts)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, S, D]
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

        # Switch-style load-balancing auxiliary (trainer multiplies by --moe_load_balance).
        # With top_k>1 each token is counted in up to k experts, so mean(dispatch,0) sums to top_k;
        # normalize by top_k so expert loads sum to ~1 and loss scale matches top_k=1.
        if self.training and n_tokens > 0:
            dispatch = torch.zeros(n_tokens, self.num_experts, device=x.device, dtype=routing_weights.dtype)
            for k in range(self.top_k):
                dispatch.scatter_(1, sel_idx[:, k : k + 1], 1.0)
            fi = dispatch.mean(0).detach() / float(max(self.top_k, 1))
            pi = routing_weights.mean(0)
            self._last_lb_loss = self.num_experts * (fi * pi).sum()
        else:
            self._last_lb_loss = x_flat.new_zeros(())

        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = sel_idx[:, k] == e
                if not mask.any():
                    continue
                y = self.experts[e](x_flat[mask])
                out_flat[mask] = out_flat[mask] + sel_weights[mask, k : k + 1] * y
        return out_flat.view(*leading, d)


@torch.no_grad()
def warm_start_moe_from_dense_ckpt(
    moe: SparseMoEFFN,
    ckpt: Dict[str, Any],
    layer_idx: int,
    copy_to_all_experts: bool,
) -> None:
    """Copy pretrained encoder.layers.{i}.linear1/2 into expert(s)."""
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
