import copy
from typing import Dict, Optional, Any, Union, Callable

import torch
import torch.nn as nn
# import torch.nn.functional as F
import warnings
from torch import Tensor
from torch.nn import functional as F
from models.attn_res import FullAttnRes


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
                 moe_ffn: Optional[nn.Module] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)

        self.moe_ffn = moe_ffn
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

        if self.use_pre_attnres:
            self.pre_attn_res = FullAttnRes(d_model)
            if self.attnres_gated:
                self.pre_attn_gate = nn.Parameter(torch.tensor(float(attnres_gate_init)))

        if self.use_pre_mlpres:
            self.pre_mlp_res = FullAttnRes(d_model)
            if self.attnres_gated:
                self.pre_mlp_gate = nn.Parameter(torch.tensor(float(attnres_gate_init)))

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

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def _forward_baseline(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x), router_context=None)
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
            attnres_in = self.pre_attn_res(sources)

            if self.attnres_gated:
                gate = torch.sigmoid(self.pre_attn_gate)
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
                gate_m = torch.sigmoid(self.pre_mlp_gate)
                mlp_in = x_attn + gate_m * (mlpres_agg - x_attn)
            else:
                mlp_in = mlpres_agg
        else:
            mlp_in = x_attn
        ffn_in = self.norm2(mlp_in)
        # sample_attnres + typed dual-bank MoE need pre-attn baseline/attnres [B,C,S,D] (not here: PSD from backbone).
        router_ctx: Optional[Dict[str, Tensor]] = None
        moe = self.moe_ffn
        needs_pre_attn_ctx = moe is not None and (
            getattr(moe, "router_mode", None) == "sample_attnres"
            or getattr(moe, "moe_kind", "") == "typed_shared_specialist"
        )
        if needs_pre_attn_ctx:
            if baseline_in is None or attnres_in is None:
                raise ValueError(
                    "MoE (sample_attnres or typed_shared_specialist) requires pre-attn AttnRes on this layer: "
                    "use attnres_variant pre_attn or full, attnres_start_layer <= layer index, and ensure "
                    "the MoE layer is not above the last layer with pre-attn (see CBraMod typed MoE guard)."
                )
            router_ctx = {"baseline": baseline_in, "attnres": attnres_in}
        x_out = mlp_in + self._ff_block(ffn_in, router_context=router_ctx)

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
    def _ff_block(self, x: Tensor, router_context: Optional[Dict[str, Tensor]] = None) -> Tensor:
        if self.moe_ffn is not None:
            return self.dropout2(self.moe_ffn(x, router_context=router_context))
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
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