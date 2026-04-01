import copy
from typing import Dict, Optional, Any, Union, Callable

import torch
import torch.nn as nn
# import torch.nn.functional as F
import warnings
from torch import Tensor
from torch.nn import functional as F
from models.attn_res import FullAttnRes
from models.adapters import get_adapter_batch_meta


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True,use_attnres=False, d_model=None,
             final_output_mode='attnres', attnres_variant='none', attnres_start_layer=0, layers=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        if layers is not None:
            self.layers = layers
            self.num_layers = len(layers)
        else:
            if encoder_layer is None:
                raise ValueError("TransformerEncoder requires encoder_layer when layers is None")
            self.layers = _get_clones(encoder_layer, num_layers)
            self.num_layers = num_layers
        for idx, layer in enumerate(self.layers):
            layer.layer_idx = idx
            layer.attnres_start_layer = attnres_start_layer
        self.norm = norm
        self.use_attnres = use_attnres
        self.final_output_mode = final_output_mode
        self.attnres_variant = attnres_variant
        self.use_final_attnres = attnres_variant in ['final', 'full']
        if self.use_final_attnres:
            self.final_attnres = FullAttnRes(d_model)

        if self.use_attnres and self.final_output_mode == 'attnres':
            self.final_attnres = FullAttnRes(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        if self.attnres_variant == 'none':
            output = src
            for mod in self.layers:
                output, _ = mod(output, src_mask=mask)
            if self.norm is not None:
                output = self.norm(output)
            return output

        sources = [src]
        output = src

        for mod in self.layers:
            output, new_sources = mod(
                output,
                sources=sources,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal if is_causal is not None else False,
            )
            sources.extend(new_sources)

        if self.use_final_attnres:
            output = self.final_attnres(sources)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, use_attnres: bool = False,
                 attnres_variant: str = 'none', attnres_gated: bool = False,
                 attnres_gate_init: float = 0.0, attnres_start_layer: int = 0,
                 attnres_subject_gates: bool = False,
                 attnres_eeg_cond_gates: bool = False,
                 attnres_eeg_context_dim: int = 0,
                 attnres_eeg_gate_scale: float = 0.1,
                 attnres_eeg_depth_cond: bool = False,
                 attnres_eeg_depth_cond_scale: float = 1.0,
                 moe_ffn: Optional[nn.Module] = None,
                 subject_adapter: Optional[nn.Module] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)

        self.moe_ffn = moe_ffn
        self.subject_adapter = subject_adapter
        # Feedforward: dense linear1/linear2, or MoE replaces both.
        if moe_ffn is None:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        else:
            self.linear1 = None  # noqa: unused in MoE path
            self.dropout = nn.Dropout(dropout)
            self.linear2 = None

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_attnres = use_attnres
        self.attnres_variant = attnres_variant
        self.attnres_gated = attnres_gated
        self.attnres_start_layer = attnres_start_layer
        self.layer_idx = -1 
        self.use_pre_attnres = attnres_variant in ['pre_attn', 'full']
        self.use_pre_mlpres = attnres_variant in ['pre_mlp', 'full']
        self.attnres_subject_gates = bool(attnres_subject_gates)
        self.attnres_eeg_cond_gates = bool(attnres_eeg_cond_gates)
        self.attnres_eeg_context_dim = int(attnres_eeg_context_dim)
        self.attnres_eeg_gate_scale = float(attnres_eeg_gate_scale)
        self.attnres_eeg_depth_cond = bool(attnres_eeg_depth_cond)
        self.attnres_eeg_depth_cond_scale = float(attnres_eeg_depth_cond_scale)
        self.last_gate_stats: Dict[str, float] = {}

        if self.use_pre_attnres:
            self.pre_attn_res = FullAttnRes(d_model)
            if self.attnres_gated:
                self.pre_attn_gate = nn.Parameter(torch.tensor(float(attnres_gate_init)))
                self.pre_attn_gate_cond = None
                if self.attnres_subject_gates and self.subject_adapter is not None:
                    cond_dim = int(getattr(self.subject_adapter, 'cond_dim', 32))
                    self.pre_attn_gate_cond = nn.Linear(cond_dim, 1, bias=True)
                    nn.init.zeros_(self.pre_attn_gate_cond.weight)
                    nn.init.zeros_(self.pre_attn_gate_cond.bias)
                self.pre_attn_eeg_gate_cond = None
                if self.attnres_eeg_cond_gates and self.attnres_eeg_context_dim > 0:
                    self.pre_attn_eeg_gate_cond = nn.Linear(self.attnres_eeg_context_dim, 1, bias=True)
                    nn.init.zeros_(self.pre_attn_eeg_gate_cond.weight)
                    nn.init.zeros_(self.pre_attn_eeg_gate_cond.bias)
                self.pre_attn_eeg_depth_cond = None
                if self.attnres_eeg_depth_cond and self.attnres_eeg_context_dim > 0:
                    self.pre_attn_eeg_depth_cond = nn.Linear(self.attnres_eeg_context_dim, 2, bias=True)
                    nn.init.zeros_(self.pre_attn_eeg_depth_cond.weight)
                    nn.init.zeros_(self.pre_attn_eeg_depth_cond.bias)

        if self.use_pre_mlpres:
            self.pre_mlp_res = FullAttnRes(d_model)
            if self.attnres_gated:
                self.pre_mlp_gate = nn.Parameter(torch.tensor(float(attnres_gate_init)))
                self.pre_mlp_gate_cond = None
                if self.attnres_subject_gates and self.subject_adapter is not None:
                    cond_dim = int(getattr(self.subject_adapter, 'cond_dim', 32))
                    self.pre_mlp_gate_cond = nn.Linear(cond_dim, 1, bias=True)
                    nn.init.zeros_(self.pre_mlp_gate_cond.weight)
                    nn.init.zeros_(self.pre_mlp_gate_cond.bias)
                self.pre_mlp_eeg_gate_cond = None
                if self.attnres_eeg_cond_gates and self.attnres_eeg_context_dim > 0:
                    self.pre_mlp_eeg_gate_cond = nn.Linear(self.attnres_eeg_context_dim, 1, bias=True)
                    nn.init.zeros_(self.pre_mlp_eeg_gate_cond.weight)
                    nn.init.zeros_(self.pre_mlp_eeg_gate_cond.bias)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def _adapter_condition_tensor(self, x_ref: Tensor) -> Optional[Tensor]:
        if self.subject_adapter is None:
            return None
        batch_meta = get_adapter_batch_meta()
        if not isinstance(batch_meta, dict):
            return None
        cond_fn = getattr(self.subject_adapter, '_condition', None)
        if cond_fn is None:
            return None
        return cond_fn(batch_meta, x_ref.device, x_ref.dtype)

    def _subject_summary_tensor(self, x_ref: Tensor) -> Optional[Tensor]:
        batch_meta = get_adapter_batch_meta()
        if not isinstance(batch_meta, dict):
            return None
        ssum = batch_meta.get("subject_summary")
        if not torch.is_tensor(ssum):
            return None
        if ssum.ndim == 1:
            ssum = ssum.unsqueeze(0)
        if ssum.ndim != 2:
            return None
        return ssum.to(device=x_ref.device, dtype=x_ref.dtype)

    def _subject_gate_delta(self, gate_proj: Optional[nn.Module], x_ref: Tensor) -> Optional[Tensor]:
        if gate_proj is None or self.subject_adapter is None:
            return None
        cond = self._adapter_condition_tensor(x_ref)
        if cond is None:
            return None
        delta = gate_proj(cond).view(-1, 1, 1, 1)
        return self.attnres_eeg_gate_scale * torch.tanh(delta)

    def _eeg_context_tensor(self, x_ref: Tensor) -> Optional[Tensor]:
        batch_meta = get_adapter_batch_meta()
        if not isinstance(batch_meta, dict):
            return None
        ctx = batch_meta.get("eeg_context_summary")
        if not torch.is_tensor(ctx):
            return None
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        if ctx.ndim != 2:
            return None
        if self.attnres_eeg_context_dim <= 0:
            return None
        if int(ctx.shape[-1]) != int(self.attnres_eeg_context_dim):
            return None
        return ctx.to(device=x_ref.device, dtype=x_ref.dtype)

    def _eeg_gate_delta(self, gate_proj: Optional[nn.Module], x_ref: Tensor) -> Optional[Tensor]:
        if gate_proj is None:
            return None
        ctx = self._eeg_context_tensor(x_ref)
        if ctx is None:
            return None
        delta = gate_proj(ctx).view(-1, 1, 1, 1)
        return self.attnres_eeg_gate_scale * torch.tanh(delta)

    def _record_gate_stats(self, prefix: str, gate: Tensor, subject_delta: Optional[Tensor], eeg_delta: Optional[Tensor]) -> None:
        gd = gate.detach().float()
        self.last_gate_stats[f'{prefix}_mean'] = float(gd.mean().item())
        self.last_gate_stats[f'{prefix}_std'] = float(gd.std(unbiased=False).item())
        self.last_gate_stats[f'{prefix}_min'] = float(gd.min().item())
        self.last_gate_stats[f'{prefix}_max'] = float(gd.max().item())
        if subject_delta is not None:
            sd = subject_delta.detach().float()
            self.last_gate_stats[f'{prefix}_subject_delta_mean'] = float(sd.mean().item())
            self.last_gate_stats[f'{prefix}_subject_delta_std'] = float(sd.std(unbiased=False).item())
            self.last_gate_stats[f'{prefix}_subject_delta_norm'] = float(sd.norm().item())
        if eeg_delta is not None:
            ed = eeg_delta.detach().float()
            self.last_gate_stats[f'{prefix}_eeg_delta_mean'] = float(ed.mean().item())
            self.last_gate_stats[f'{prefix}_eeg_delta_std'] = float(ed.std(unbiased=False).item())
            self.last_gate_stats[f'{prefix}_eeg_delta_norm'] = float(ed.norm().item())

    def _apply_eeg_depth_condition(self, alpha: Tensor, x_ref: Tensor) -> Tensor:
        if not self.attnres_eeg_depth_cond:
            return alpha
        depth_proj = getattr(self, 'pre_attn_eeg_depth_cond', None)
        if depth_proj is None:
            return alpha
        ctx = self._eeg_context_tensor(x_ref)
        if ctx is None:
            return alpha
        n_depth = int(alpha.shape[0])
        depth_bias = depth_proj(ctx)
        if depth_bias.ndim != 2:
            return alpha
        if depth_bias.shape[-1] != 2:
            return alpha

        pos = torch.linspace(0.0, 1.0, steps=n_depth, device=alpha.device, dtype=alpha.dtype)
        pos_center = pos - 0.5
        coeff = self.attnres_eeg_depth_cond_scale * torch.tanh(depth_bias).to(dtype=alpha.dtype)
        depth_curve = (
            coeff[:, 0:1] * pos.view(1, -1)
            + coeff[:, 1:2] * (pos_center.view(1, -1) ** 2)
        )
        depth_bias = depth_curve.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)

        log_alpha = torch.log(alpha.clamp_min(1e-10))
        alpha_cond = torch.softmax(log_alpha + depth_bias, dim=0)

        db = depth_bias.detach().float()
        self.last_gate_stats['pre_attn_depth_cond_mean'] = float(db.mean().item())
        self.last_gate_stats['pre_attn_depth_cond_std'] = float(db.std(unbiased=False).item())
        self.last_gate_stats['pre_attn_depth_cond_norm'] = float(db.norm().item())
        return alpha_cond

    @staticmethod
    def _attnres_depth_summary(alpha: Tensor) -> Tensor:
        # alpha: [N, B, C, S] softmax over depth N
        if alpha.dim() != 4:
            raise ValueError(f"attnres alpha must be [N,B,C,S], got {tuple(alpha.shape)}")
        n, b, _, _ = alpha.shape
        p = alpha.permute(1, 0, 2, 3).reshape(b, n, -1).mean(dim=-1)
        p = p.clamp_min(1e-10)
        ent = -(p * p.log()).sum(dim=-1)
        if n > 1:
            ent = ent / float(torch.log(torch.tensor(float(n), device=alpha.device)).item())
        pmax = p.max(dim=-1).values
        plast = p[:, -1]
        if n > 1:
            top2 = torch.topk(p, k=2, dim=-1).values
            pgap = top2[:, 0] - top2[:, 1]
        else:
            pgap = pmax
        return torch.stack([ent, pmax, plast, pgap], dim=-1)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def _forward_baseline(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x), router_context=None, adapter_context=None)
        return x

    def _forward_attnres(self, sources, src_mask=None, src_key_padding_mask=None, is_causal=False):
    # sources is a Python list:
    # [patch_emb, out_1, out_2, ..., out_k]
    # each tensor shape [B, C, S, D]

    # ---- pre-attention depth aggregation ----
        h_attn = self.pre_attn_res(sources)
        attn_out = self._sa_block(self.norm1(h_attn), src_mask, src_key_padding_mask, is_causal=is_causal)
        sources = sources + [attn_out]

        # ---- pre-MLP depth aggregation ----
        h_mlp = self.pre_mlp_res(sources)
        ff_out = self._ff_block(self.norm2(h_mlp), router_context=None)
        sources = sources + [ff_out]

        return sources

    # in TransformerEncoderLayer.forward
    def forward(self, x, sources=None, src_mask=None, src_key_padding_mask=None, is_causal=False):

        if self.attnres_variant == 'none':
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x), router_context=None)
            return x, [x]

        new_sources = []

        # Pre-attention input: pure AttnRes agg, or gated blend with residual (x) when attnres_gated, top k layers only
        baseline_in = None
        attnres_in = None
        use_pre_attn_here = (
            self.use_pre_attnres and
            self.layer_idx >= self.attnres_start_layer
        )

        if use_pre_attn_here:
            baseline_in = x
            attnres_in, attnres_alpha = self.pre_attn_res(sources, return_alpha=True)
            attnres_alpha = self._apply_eeg_depth_condition(attnres_alpha, baseline_in)
            if self.attnres_eeg_depth_cond:
                src_stack = torch.stack(sources, dim=0)
                attnres_in = torch.sum(attnres_alpha.unsqueeze(-1) * src_stack, dim=0)
            attnres_depth_summary = self._attnres_depth_summary(attnres_alpha).to(
                device=baseline_in.device,
                dtype=baseline_in.dtype,
            )

            if self.attnres_gated:
                base_gate = self.pre_attn_gate.view(1, 1, 1, 1)
                subj_delta = None
                eeg_delta = None
                if self.attnres_subject_gates:
                    subj_delta = self._subject_gate_delta(getattr(self, 'pre_attn_gate_cond', None), baseline_in)
                if self.attnres_eeg_cond_gates:
                    eeg_delta = self._eeg_gate_delta(getattr(self, 'pre_attn_eeg_gate_cond', None), baseline_in)
                gate_logits = base_gate
                if subj_delta is not None:
                    gate_logits = gate_logits + subj_delta
                if eeg_delta is not None:
                    gate_logits = gate_logits + eeg_delta
                gate = torch.sigmoid(gate_logits)
                self._record_gate_stats('pre_attn', gate, subj_delta, eeg_delta)
                attn_in = baseline_in + gate * (attnres_in - baseline_in)
            else:
                attn_in = attnres_in
        else:
            attn_in = x

        x_attn = attn_in + self._sa_block(self.norm1(attn_in), src_mask, src_key_padding_mask, is_causal=is_causal)

        if self.attnres_variant in ['pre_attn', 'pre_mlp', 'full']:
            new_sources.append(x_attn)

        use_pre_mlp_here = (
            self.use_pre_mlpres and
            self.layer_idx >= self.attnres_start_layer
        )
        mlp_source_pool = sources + new_sources if sources is not None else [x_attn]
        if use_pre_mlp_here:
            mlpres_agg = self.pre_mlp_res(mlp_source_pool)
            if self.attnres_gated:
                base_gate_m = self.pre_mlp_gate.view(1, 1, 1, 1)
                subj_delta_m = None
                eeg_delta_m = None
                if self.attnres_subject_gates:
                    subj_delta_m = self._subject_gate_delta(getattr(self, 'pre_mlp_gate_cond', None), x_attn)
                if self.attnres_eeg_cond_gates:
                    eeg_delta_m = self._eeg_gate_delta(getattr(self, 'pre_mlp_eeg_gate_cond', None), x_attn)
                gate_logits_m = base_gate_m
                if subj_delta_m is not None:
                    gate_logits_m = gate_logits_m + subj_delta_m
                if eeg_delta_m is not None:
                    gate_logits_m = gate_logits_m + eeg_delta_m
                gate_m = torch.sigmoid(gate_logits_m)
                self._record_gate_stats('pre_mlp', gate_m, subj_delta_m, eeg_delta_m)
                mlp_in = x_attn + gate_m * (mlpres_agg - x_attn)
            else:
                mlp_in = mlpres_agg
        else:
            mlp_in = x_attn
        ffn_in = self.norm2(mlp_in)
        # typed_capacity_domain MoE needs pre-attn baseline/attnres [B,C,S,D] (PSD is set from backbone context).
        router_ctx: Optional[Dict[str, Tensor]] = None
        moe = self.moe_ffn
        needs_pre_attn_ctx = moe is not None and getattr(moe, "moe_kind", "") == "typed_capacity_domain"
        if needs_pre_attn_ctx:
            if baseline_in is None or attnres_in is None:
                raise ValueError(
                    "MoE typed_capacity_domain requires pre-attn AttnRes on this layer: "
                    "use attnres_variant pre_attn or full, attnres_start_layer <= layer index, and ensure "
                    "the MoE layer is not above the last layer with pre-attn (see CBraMod typed MoE guard)."
                )
            router_ctx = {"baseline": baseline_in, "attnres": attnres_in}
            if use_pre_attn_here:
                router_ctx["attnres_depth_summary"] = attnres_depth_summary
            adapter_cond = self._adapter_condition_tensor(mlp_in)
            if adapter_cond is not None:
                router_ctx["adapter_cond"] = adapter_cond
            subject_summary = self._subject_summary_tensor(mlp_in)
            if subject_summary is not None:
                router_ctx["subject_summary"] = subject_summary
        adapter_ctx = {"layer_idx": self.layer_idx}
        x_out = mlp_in + self._ff_block(ffn_in, router_context=router_ctx, adapter_context=adapter_ctx)

        # for final-only, collect only block outputs
        if self.attnres_variant == 'final':
            new_sources = [x_out]
        else:
            new_sources.append(x_out)

        return x_out, new_sources

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(bz*patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz*ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(xs, xs, xs,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(
        self,
        x: Tensor,
        router_context: Optional[Dict[str, Tensor]] = None,
        adapter_context: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        batch_meta = get_adapter_batch_meta() if self.subject_adapter is not None else None
        if self.moe_ffn is not None:
            x = self.moe_ffn(x, router_context=router_context)
        else:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if self.subject_adapter is not None:
            x = x + self.subject_adapter(x, batch_meta=batch_meta)
        return self.dropout2(x)



def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=1024, batch_first=True, norm_first=True,
        activation=F.gelu
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
    encoder = encoder.cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()
    b = encoder(a)
    print(a.shape, b.shape)