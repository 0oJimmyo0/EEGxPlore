import copy
from typing import Dict, Optional, Any, Union, Callable

import torch
import torch.nn as nn
# import torch.nn.functional as F
import warnings
from torch import Tensor
from torch.nn import functional as F
from models.attn_res import FullAttnRes
from models.moe import get_moe_train_epoch


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
                  moe_attnres_depth_context_mode: str = 'compact_shared',
                  moe_attnres_depth_block_count: int = 4,
                 moe_attnres_depth_router_dim: int = 26,
                 moe_attnres_depth_summary_mode: str = 'auto',
                 moe_attnres_depth_probe_mlp_for_router: bool = False,
                  moe_attnres_depth_summary_grad_mode: str = 'delayed_unfreeze',
                  moe_attnres_depth_summary_unfreeze_epoch: int = 8,
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
        self.moe_attnres_depth_context_mode = str(moe_attnres_depth_context_mode).strip().lower()
        self.moe_attnres_depth_block_count = int(moe_attnres_depth_block_count)
        self.moe_attnres_depth_router_dim = int(moe_attnres_depth_router_dim)
        self.moe_attnres_depth_summary_mode = str(moe_attnres_depth_summary_mode).strip().lower()
        self.moe_attnres_depth_probe_mlp_for_router = bool(moe_attnres_depth_probe_mlp_for_router)
        self.moe_attnres_depth_summary_grad_mode = str(moe_attnres_depth_summary_grad_mode).strip().lower()
        self.moe_attnres_depth_summary_unfreeze_epoch = int(moe_attnres_depth_summary_unfreeze_epoch)
        valid_context_modes = {'compact_shared', 'block_shared_typed_proj'}
        valid_depth_modes = {'auto', 'attn_delta4', 'attn_mlp_balanced', 'attn_mlp_latemix'}
        valid_grad_modes = {'detached', 'delayed_unfreeze', 'trainable'}
        if self.moe_attnres_depth_context_mode not in valid_context_modes:
            raise ValueError(
                f"Unsupported moe_attnres_depth_context_mode={self.moe_attnres_depth_context_mode!r}; "
                f"expected one of {sorted(valid_context_modes)}"
            )
        if self.moe_attnres_depth_block_count < 1:
            raise ValueError("moe_attnres_depth_block_count must be >= 1")
        if self.moe_attnres_depth_router_dim <= 0:
            raise ValueError("moe_attnres_depth_router_dim must be > 0")
        if self.moe_attnres_depth_summary_mode not in valid_depth_modes:
            raise ValueError(
                f"Unsupported moe_attnres_depth_summary_mode={self.moe_attnres_depth_summary_mode!r}; "
                f"expected one of {sorted(valid_depth_modes)}"
            )
        if self.moe_attnres_depth_summary_grad_mode not in valid_grad_modes:
            raise ValueError(
                f"Unsupported moe_attnres_depth_summary_grad_mode={self.moe_attnres_depth_summary_grad_mode!r}; "
                f"expected one of {sorted(valid_grad_modes)}"
            )
        if self.moe_attnres_depth_summary_unfreeze_epoch < 1:
            raise ValueError("moe_attnres_depth_summary_unfreeze_epoch must be >= 1")

        if self.use_pre_attnres:
            self.pre_attn_res = FullAttnRes(d_model)
            if self.attnres_gated:
                self.pre_attn_gate = nn.Parameter(torch.tensor(float(attnres_gate_init)))

        if self.use_pre_mlpres or self.moe_attnres_depth_probe_mlp_for_router:
            self.pre_mlp_res = FullAttnRes(d_model)
            if self.attnres_gated and self.use_pre_mlpres:
                self.pre_mlp_gate = nn.Parameter(torch.tensor(float(attnres_gate_init)))

        self.depth_block_score_pre_attn = None
        self.depth_block_score_pre_mlp = None
        self.depth_block_readout_spatial = None
        self.depth_block_readout_spectral = None
        if self.moe_attnres_depth_context_mode == 'block_shared_typed_proj':
            self.depth_block_score_pre_attn = nn.Linear(d_model, 1, bias=True, **factory_kwargs)
            self.depth_block_score_pre_mlp = nn.Linear(d_model, 1, bias=True, **factory_kwargs)
            readout_in_dim = 2 * self.moe_attnres_depth_block_count * d_model
            self.depth_block_readout_spatial = nn.Linear(
                readout_in_dim,
                self.moe_attnres_depth_router_dim,
                bias=True,
                **factory_kwargs,
            )
            self.depth_block_readout_spectral = nn.Linear(
                readout_in_dim,
                self.moe_attnres_depth_router_dim,
                bias=True,
                **factory_kwargs,
            )

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

    @staticmethod
    def _attnres_alpha_features(alpha: Optional[Tensor]) -> Optional[Tensor]:
        """Per-sample depth-selection statistics from AttnRes alpha.

        Returns [B, 11]:
        [
          entropy,
          top1_mass,
          top2_mass,
          top3_mass,
          early_mass,
          middle_mass,
          late_mass,
          source0_mass,
          sink_mass,
          attn_family_mass,
          mlp_family_mass,
        ]
        """
        if alpha is None:
            return None
        if alpha.dim() != 4:
            raise ValueError(f"AttnRes alpha must be [N,B,C,S], got {tuple(alpha.shape)}")
        n, b, _, _ = alpha.shape
        if n <= 0:
            return torch.zeros((b, 11), device=alpha.device, dtype=alpha.dtype)

        # Average over channel/patch axes: [B, N], still a depth distribution per sample.
        p = alpha.mean(dim=(2, 3)).transpose(0, 1)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        top1_mass = p.max(dim=-1).values
        k2 = min(2, n)
        k3 = min(3, n)
        top2_mass = p.topk(k=k2, dim=-1).values.sum(dim=-1)
        top3_mass = p.topk(k=k3, dim=-1).values.sum(dim=-1)
        entropy = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1)
        ent_norm = torch.log(torch.tensor(float(max(n, 2)), device=alpha.device, dtype=alpha.dtype)).clamp_min(1e-8)
        entropy = entropy / ent_norm

        k1 = max(1, n // 3)
        k2cut = max(k1 + 1, (2 * n) // 3)
        early_mass = p[:, :k1].sum(dim=-1)
        middle_mass = p[:, k1:k2cut].sum(dim=-1)
        late_mass = p[:, k2cut:].sum(dim=-1)

        sink_mass = p[:, -1]
        source0_mass = p[:, 0]

        # In current source construction: index 0 is embedding; odd indices are pre-attn outputs,
        # even indices (>0) are post-MLP/block outputs.
        idx = torch.arange(n, device=alpha.device)
        attn_mask = ((idx % 2) == 1)
        mlp_mask = ((idx % 2) == 0) & (idx > 0)
        attn_family_mass = p[:, attn_mask].sum(dim=-1) if attn_mask.any() else torch.zeros_like(top1_mass)
        mlp_family_mass = p[:, mlp_mask].sum(dim=-1) if mlp_mask.any() else torch.zeros_like(top1_mass)

        return torch.stack(
            [
                entropy,
                top1_mass,
                top2_mass,
                top3_mass,
                early_mass,
                middle_mass,
                late_mass,
                source0_mass,
                sink_mass,
                attn_family_mass,
                mlp_family_mass,
            ],
            dim=-1,
        )

    @staticmethod
    def _compose_attnres_depth_summary(
        attn_alpha: Optional[Tensor],
        mlp_alpha: Optional[Tensor],
        out_dim: int,
        fallback_ref: Tensor,
        summary_mode: str = 'auto',
    ) -> Optional[Tensor]:
        """Build depth-summary vector for MoE router_context['attnres_depth_summary'].

        Base feature layout before pad/truncate (26 dims):
        [attn(11), mlp(11), delta_entropy, delta_late, delta_sink, delta_attn_family].

        Explicit supported interfaces:
        - out_dim == 11: attn-only summary.
        - out_dim == 15: configurable via summary_mode.
        - out_dim == 26: full attn+mlp+deltas summary.
        - out_dim == 13: legacy compat = attn(11) + {delta_entropy, delta_late}.
        """
        if out_dim <= 0:
            return None

        b = fallback_ref.shape[0]
        dev = fallback_ref.device
        dtype = fallback_ref.dtype

        attn_f = TransformerEncoderLayer._attnres_alpha_features(attn_alpha)
        mlp_f = TransformerEncoderLayer._attnres_alpha_features(mlp_alpha)

        if attn_f is None:
            attn_f = torch.zeros((b, 11), device=dev, dtype=dtype)
        if mlp_f is None:
            mlp_f = torch.zeros((b, 11), device=dev, dtype=dtype)

        attn_f = attn_f.to(device=dev, dtype=dtype)
        mlp_f = mlp_f.to(device=dev, dtype=dtype)
        delta_entropy = (attn_f[:, 0] - mlp_f[:, 0]).unsqueeze(-1)
        delta_late = (attn_f[:, 6] - mlp_f[:, 6]).unsqueeze(-1)
        delta_sink = (attn_f[:, 8] - mlp_f[:, 8]).unsqueeze(-1)
        delta_attn_family = (attn_f[:, 9] - mlp_f[:, 9]).unsqueeze(-1)
        deltas = torch.cat([delta_entropy, delta_late, delta_sink, delta_attn_family], dim=-1)
        raw = torch.cat([attn_f, mlp_f, deltas], dim=-1)

        # Use explicit semantic packings for common router dimensions.
        if out_dim == 11:
            return attn_f
        if out_dim == 15:
            mode = str(summary_mode)
            if mode in ('auto', 'attn_delta4'):
                return torch.cat([attn_f, deltas], dim=-1)
            if mode == 'attn_mlp_balanced':
                return torch.cat([attn_f[:, :7], mlp_f[:, :4], deltas], dim=-1)
            if mode == 'attn_mlp_latemix':
                return torch.cat([attn_f[:, :7], mlp_f[:, 4:8], deltas], dim=-1)
            raise ValueError(f"Unsupported summary_mode for out_dim=15: {mode!r}")
        if out_dim == 26:
            return raw
        if out_dim == 13:
            return torch.cat([attn_f, delta_entropy, delta_late], dim=-1)

        if raw.size(-1) > out_dim:
            return raw[:, :out_dim]
        if raw.size(-1) < out_dim:
            pad = torch.zeros((b, out_dim - raw.size(-1)), device=dev, dtype=dtype)
            return torch.cat([raw, pad], dim=-1)
        return raw

    @staticmethod
    def _split_depth_family_sources(valid_sources):
        pre_attn_sources = []
        pre_mlp_sources = []
        # Current source ordering: index 0 is embedding, odd indices are pre-attn outputs,
        # even indices (>0) are post-MLP/block outputs.
        for i, src in enumerate(valid_sources):
            if i == 0:
                continue
            if i % 2 == 1:
                pre_attn_sources.append(src)
            else:
                pre_mlp_sources.append(src)
        return pre_attn_sources, pre_mlp_sources

    def _learned_block_weighted_vectors(self, depth_hidden: Tensor, scorer: nn.Linear, block_count: int):
        b, _, d = depth_hidden.shape
        dev = depth_hidden.device
        dtype = depth_hidden.dtype
        idx_blocks = torch.tensor_split(torch.arange(depth_hidden.shape[1], device=dev), block_count)

        block_vecs = []
        block_layer_counts = []
        block_weight_mass = []
        for ids in idx_blocks:
            block_layer_counts.append(int(ids.numel()))
            if ids.numel() == 0:
                block_vecs.append(torch.zeros((b, d), device=dev, dtype=dtype))
                block_weight_mass.append(torch.zeros((b,), device=dev, dtype=dtype))
                continue
            blk = depth_hidden.index_select(1, ids)
            logits = scorer(blk).squeeze(-1)
            weights = F.softmax(logits, dim=-1)
            vec = torch.einsum('bl,bld->bd', weights, blk)
            block_vecs.append(vec)
            block_weight_mass.append(weights.max(dim=-1).values)

        block_stack = torch.stack(block_vecs, dim=1)
        block_weight_peak = torch.stack(block_weight_mass, dim=1)
        return block_stack, block_layer_counts, block_weight_peak

    def _compose_block_typed_depth_summary(
        self,
        source_pool,
        out_dim: int,
        fallback_ref: Tensor,
        block_count: int,
    ) -> Dict[str, Any]:
        """Learned block summary with typed readouts and pre-attn/pre-MLP family separation."""
        b = fallback_ref.shape[0]
        dev = fallback_ref.device
        dtype = fallback_ref.dtype
        n_blocks = max(1, int(block_count))

        valid_sources = []
        for src in (source_pool or []):
            if not torch.is_tensor(src):
                continue
            if src.dim() != 4 or src.shape[0] != b:
                continue
            valid_sources.append(src.to(device=dev, dtype=dtype))

        d = int(fallback_ref.shape[-1])
        z_blocks = torch.zeros((b, n_blocks), device=dev, dtype=dtype)
        z_summary = torch.zeros((b, max(0, int(out_dim))), device=dev, dtype=dtype)
        if not valid_sources:
            return {
                'summary': z_summary,
                'summary_spatial': z_summary,
                'summary_spectral': z_summary,
                'block_means': z_blocks,
                'block_stds': z_blocks,
                'block_norms': z_blocks,
                'block_layer_counts': [0 for _ in range(n_blocks)],
                'block_layer_counts_pre_attn': [0 for _ in range(n_blocks)],
                'block_layer_counts_pre_mlp': [0 for _ in range(n_blocks)],
                'block_pooling': 'learned_softmax',
                'depth_family_mode': 'pre_attn_pre_mlp',
                'block_peak_weight_pre_attn': z_blocks,
                'block_peak_weight_pre_mlp': z_blocks,
            }

        all_hidden = torch.stack([src.mean(dim=(1, 2)) for src in valid_sources], dim=1)
        all_counts = [int(x.numel()) for x in torch.tensor_split(torch.arange(all_hidden.shape[1], device=dev), n_blocks)]

        pre_attn_sources, pre_mlp_sources = self._split_depth_family_sources(valid_sources)
        if not pre_attn_sources:
            pre_attn_sources = valid_sources
        if not pre_mlp_sources:
            pre_mlp_sources = valid_sources

        pre_attn_hidden = torch.stack([src.mean(dim=(1, 2)) for src in pre_attn_sources], dim=1)
        pre_mlp_hidden = torch.stack([src.mean(dim=(1, 2)) for src in pre_mlp_sources], dim=1)

        attn_stack, attn_counts, attn_peak = self._learned_block_weighted_vectors(
            pre_attn_hidden,
            self.depth_block_score_pre_attn,
            n_blocks,
        )
        mlp_stack, mlp_counts, mlp_peak = self._learned_block_weighted_vectors(
            pre_mlp_hidden,
            self.depth_block_score_pre_mlp,
            n_blocks,
        )

        fused = torch.cat([attn_stack, mlp_stack], dim=1).reshape(b, -1)
        if fused.shape[1] != (2 * n_blocks * d):
            raise ValueError(
                f'Unexpected fused block summary shape={tuple(fused.shape)}; expected last dim={2 * n_blocks * d}'
            )

        summary_spatial = self.depth_block_readout_spatial(fused)
        summary_spectral = self.depth_block_readout_spectral(fused)
        summary = 0.5 * (summary_spatial + summary_spectral)

        shared_block = 0.5 * (attn_stack + mlp_stack)
        return {
            'summary': summary,
            'summary_spatial': summary_spatial,
            'summary_spectral': summary_spectral,
            'block_means': shared_block.mean(dim=-1),
            'block_stds': shared_block.std(dim=-1, unbiased=False),
            'block_norms': torch.linalg.vector_norm(shared_block, ord=2, dim=-1),
            'block_layer_counts': all_counts,
            'block_layer_counts_pre_attn': attn_counts,
            'block_layer_counts_pre_mlp': mlp_counts,
            'block_pooling': 'learned_softmax',
            'depth_family_mode': 'pre_attn_pre_mlp',
            'block_peak_weight_pre_attn': attn_peak,
            'block_peak_weight_pre_mlp': mlp_peak,
        }


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
        attn_alpha = None
        use_pre_attn_here = (
            self.use_pre_attnres and
            self.layer_idx >= self.attnres_start_layer
        )

        if use_pre_attn_here:
            baseline_in = x
            attnres_in, attn_alpha = self.pre_attn_res(sources, return_alpha=True)

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
        mlp_alpha = None
        if use_pre_mlp_here:
            mlpres_agg, mlp_alpha = self.pre_mlp_res(mlp_source_pool, return_alpha=True)
            if self.attnres_gated:
                gate_m = torch.sigmoid(self.pre_mlp_gate)
                mlp_in = x_attn + gate_m * (mlpres_agg - x_attn)
            else:
                mlp_in = mlpres_agg
        else:
            mlp_in = x_attn
            if self.moe_attnres_depth_probe_mlp_for_router and hasattr(self, 'pre_mlp_res'):
                _, mlp_alpha = self.pre_mlp_res(mlp_source_pool, return_alpha=True)
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
            if getattr(moe, "use_attnres_depth_router_concat", False):
                depth_dim = int(getattr(moe, "attnres_depth_router_dim", 0))
                block_diag = None
                if self.moe_attnres_depth_context_mode == 'block_shared_typed_proj':
                    block_diag = self._compose_block_typed_depth_summary(
                        source_pool=mlp_source_pool,
                        out_dim=depth_dim,
                        fallback_ref=baseline_in,
                        block_count=self.moe_attnres_depth_block_count,
                    )
                    depth_summary = block_diag['summary']
                    depth_summary_spatial = block_diag['summary_spatial']
                    depth_summary_spectral = block_diag['summary_spectral']
                else:
                    depth_summary = self._compose_attnres_depth_summary(
                        attn_alpha=attn_alpha,
                        mlp_alpha=mlp_alpha,
                        out_dim=depth_dim,
                        fallback_ref=baseline_in,
                        summary_mode=self.moe_attnres_depth_summary_mode,
                    )
                    depth_summary_spatial = depth_summary
                    depth_summary_spectral = depth_summary
                if depth_summary is not None:
                    grad_mode = str(self.moe_attnres_depth_summary_grad_mode).strip().lower()
                    cur_epoch = int(get_moe_train_epoch())
                    unfreeze_epoch = int(self.moe_attnres_depth_summary_unfreeze_epoch)
                    if grad_mode == 'trainable':
                        grad_active = True
                    elif grad_mode == 'delayed_unfreeze':
                        grad_active = cur_epoch >= unfreeze_epoch
                    else:
                        grad_active = False
                    depth_summary_detached = not grad_active
                    depth_summary_for_router = depth_summary if grad_active else depth_summary.detach()
                    depth_summary_spatial_for_router = (
                        depth_summary_spatial if grad_active else depth_summary_spatial.detach()
                    )
                    depth_summary_spectral_for_router = (
                        depth_summary_spectral if grad_active else depth_summary_spectral.detach()
                    )
                    router_ctx["attnres_depth_summary"] = depth_summary_for_router
                    router_ctx["attnres_depth_summary_spatial"] = depth_summary_spatial_for_router
                    router_ctx["attnres_depth_summary_spectral"] = depth_summary_spectral_for_router
                    router_ctx["attnres_depth_context_mode"] = self.moe_attnres_depth_context_mode
                    router_ctx["attnres_depth_block_count"] = int(self.moe_attnres_depth_block_count)
                    if block_diag is not None:
                        router_ctx["attnres_depth_summary_mode"] = "block_typed_learned"
                        router_ctx["attnres_depth_probe_mlp_for_router"] = False
                    else:
                        router_ctx["attnres_depth_summary_mode"] = self.moe_attnres_depth_summary_mode
                        router_ctx["attnres_depth_probe_mlp_for_router"] = bool(self.moe_attnres_depth_probe_mlp_for_router)
                    router_ctx["attnres_depth_summary_grad_mode"] = grad_mode
                    router_ctx["attnres_depth_summary_grad_active"] = bool(grad_active)
                    router_ctx["attnres_depth_summary_detached"] = bool(depth_summary_detached)
                    router_ctx["attnres_depth_summary_unfreeze_epoch"] = unfreeze_epoch
                    router_ctx["attnres_depth_summary_cur_epoch"] = cur_epoch
                    router_ctx["attnres_depth_shared_context_norm"] = float(
                        depth_summary_for_router.detach().float().norm(dim=-1).mean().item()
                    )
                    router_ctx["attnres_depth_spatial_context_norm"] = float(
                        depth_summary_spatial_for_router.detach().float().norm(dim=-1).mean().item()
                    )
                    router_ctx["attnres_depth_spectral_context_norm"] = float(
                        depth_summary_spectral_for_router.detach().float().norm(dim=-1).mean().item()
                    )
                    if block_diag is not None:
                        router_ctx["attnres_depth_block_layer_counts"] = list(block_diag['block_layer_counts'])
                        router_ctx["attnres_depth_block_layer_counts_pre_attn"] = list(block_diag['block_layer_counts_pre_attn'])
                        router_ctx["attnres_depth_block_layer_counts_pre_mlp"] = list(block_diag['block_layer_counts_pre_mlp'])
                        router_ctx["attnres_depth_block_pooling"] = str(block_diag['block_pooling'])
                        router_ctx["attnres_depth_family_mode"] = str(block_diag['depth_family_mode'])
                        router_ctx["attnres_depth_block_mean"] = (
                            block_diag['block_means'].detach().float().mean(dim=0).cpu().tolist()
                        )
                        router_ctx["attnres_depth_block_std"] = (
                            block_diag['block_stds'].detach().float().mean(dim=0).cpu().tolist()
                        )
                        router_ctx["attnres_depth_block_summary_norms"] = (
                            block_diag['block_norms'].detach().float().mean(dim=0).cpu().tolist()
                        )
                        router_ctx["attnres_depth_block_peak_weight_pre_attn"] = (
                            block_diag['block_peak_weight_pre_attn'].detach().float().mean(dim=0).cpu().tolist()
                        )
                        router_ctx["attnres_depth_block_peak_weight_pre_mlp"] = (
                            block_diag['block_peak_weight_pre_mlp'].detach().float().mean(dim=0).cpu().tolist()
                        )
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