import importlib
import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbramod import CBraMod, backbone_finetune_kwargs
from .moe import (
    compact_psd_bandpowers,
    reset_moe_eeg_router_summary,
    reset_moe_faced_metadata,
    reset_moe_psd_router_features,
    set_moe_eeg_router_summary,
    set_moe_faced_metadata,
    set_moe_psd_router_features,
)


def _compact_eeg_summary(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected raw EEG [B,C,S,T], got {tuple(x.shape)}")
    if out_dim <= 0:
        raise ValueError("out_dim must be > 0")
    v = x.mean(dim=(2, 3))
    return F.adaptive_avg_pool1d(v.unsqueeze(1), out_dim).squeeze(1)


class _EncoderView(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers)


def _import_labram_ctor(repo_dir: str, model_name: str):
    repo_dir = os.path.abspath(repo_dir)
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"LaBraM repo dir not found: {repo_dir}")
    module_path = os.path.join(repo_dir, "modeling_finetune.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"LaBraM modeling_finetune.py not found: {module_path}")
    try:
        spec = importlib.util.spec_from_file_location("labram_modeling_finetune", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise ImportError(
            "Failed to import LaBraM modeling_finetune. Ensure the active EEGxPlore "
            "environment includes LaBraM dependencies such as 'timm'."
        ) from exc
    try:
        return getattr(module, model_name)
    except AttributeError as exc:
        raise AttributeError(f"LaBraM model constructor not found: {model_name}") from exc


def _import_labram_utils(repo_dir: str):
    repo_dir = os.path.abspath(repo_dir)
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"LaBraM repo dir not found: {repo_dir}")
    module_path = os.path.join(repo_dir, "utils.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"LaBraM utils.py not found: {module_path}")
    try:
        spec = importlib.util.spec_from_file_location("labram_repo_utils", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    except ImportError as exc:
        raise ImportError("Failed to import LaBraM utils for input_chans resolution.") from exc


def _extract_labram_state_dict(ckpt_state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ckpt_state, dict):
        raise TypeError(f"Expected LaBraM checkpoint to be a dict, got {type(ckpt_state)!r}")

    if "model" in ckpt_state and isinstance(ckpt_state["model"], dict):
        state = ckpt_state["model"]
    elif "module" in ckpt_state and isinstance(ckpt_state["module"], dict):
        state = ckpt_state["module"]
    else:
        state = ckpt_state

    extracted = {}
    for key, value in state.items():
        if key.startswith("student."):
            extracted[key[len("student."):]] = value
        elif not key.startswith("teacher."):
            extracted[key] = value
    if not extracted:
        raise RuntimeError("No loadable LaBraM student/foundation tensors found in checkpoint.")
    return extracted


def load_labram_foundation_into_backbone(backbone: "LaBraMBackbone", ckpt_state: Dict[str, Any]) -> Set[str]:
    state = _extract_labram_state_dict(ckpt_state)
    foundation_sd = backbone.foundation.state_dict()
    loadable = {
        k: v for k, v in state.items()
        if k in foundation_sd and foundation_sd[k].shape == v.shape
    }
    if not loadable:
        raise RuntimeError("No matching LaBraM tensors could be loaded into the EEGxPlore LaBraM backbone.")
    backbone.foundation.load_state_dict(loadable, strict=False)
    return {f"foundation.{k}" for k in loadable.keys()}


class LaBraMBackbone(nn.Module):
    def __init__(self, param):
        super().__init__()
        model_name = getattr(param, "labram_model_name", "labram_base_patch200_200")
        repo_dir = getattr(param, "labram_repo_dir", "")
        ctor = _import_labram_ctor(repo_dir, model_name)
        self.foundation = ctor(
            EEG_size=200,
            in_chans=1,
            num_classes=0,
            use_mean_pooling=True,
            init_values=float(getattr(param, "labram_layer_scale_init_value", 0.1)),
            drop_path_rate=float(getattr(param, "labram_drop_path_rate", 0.1)),
        )
        self.proj_out = nn.Identity()
        self.output_mode = "pooled"
        self.feature_dim = int(getattr(self.foundation, "embed_dim", 200))
        self.channel_names = list(getattr(param, "labram_channel_names", []) or [])
        self.input_chans = None
        if self.channel_names:
            labram_utils = _import_labram_utils(repo_dir)
            self.input_chans = labram_utils.get_input_chans(self.channel_names)
            print(
                f"[LaBraM] input_chans resolved from manifest: n={len(self.channel_names)} "
                f"tail_names={self.channel_names[-4:]} tail_indices={self.input_chans[-4:]}",
                flush=True,
            )
        else:
            print("[LaBraM] no channel manifest provided; using stored tensor slot positions.", flush=True)

        selective_requested = bool(getattr(param, "moe", False)) or getattr(param, "attnres_variant", "none") != "none"
        adapter_layers = int(getattr(param, "labram_adapter_layers", 4))
        self.adapter: Optional[CBraMod]
        if selective_requested and adapter_layers > 0:
            self.adapter = CBraMod(
                in_dim=200,
                out_dim=200,
                d_model=200,
                dim_feedforward=800,
                seq_len=1,
                n_layer=adapter_layers,
                nhead=8,
                **backbone_finetune_kwargs(param),
            )
            self.adapter.proj_out = nn.Identity()
            for p in self.adapter.patch_embedding.parameters():
                p.requires_grad = False
            self.encoder = self.adapter.encoder
            print(
                f"[LaBraM] selective adapter enabled: layers={adapter_layers}, "
                f"attnres_variant={getattr(param, 'attnres_variant', 'none')}, "
                f"moe={getattr(param, 'moe', False)}"
            )
        else:
            self.adapter = None
            self.encoder = _EncoderView(self.foundation.blocks)
            if selective_requested and adapter_layers <= 0:
                print(
                    "[LaBraM] selective adaptation flags were requested, but "
                    "labram_adapter_layers <= 0 so the run will fall back to dense LaBraM features."
                )

    def _foundation_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, patches, _ = x.shape
        if self.input_chans is not None and len(self.input_chans) != channels + 1:
            raise ValueError(
                f"LaBraM input_chans length mismatch: expected {channels + 1}, got {len(self.input_chans)}"
            )
        patch_tokens = self.foundation.forward_features(
            x,
            input_chans=self.input_chans,
            return_patch_tokens=True,
        )
        if patch_tokens.shape[1] != channels * patches:
            raise ValueError(
                f"LaBraM patch token count mismatch: expected {channels * patches}, "
                f"got {patch_tokens.shape[1]}"
            )
        return patch_tokens.reshape(batch_size, channels, patches, patch_tokens.shape[-1])

    def _pool_token_grid(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 4:
            raise ValueError(f"Expected token grid [B,C,S,D], got {tuple(feats.shape)}")
        pooled = feats.reshape(feats.shape[0], -1, feats.shape[-1]).mean(dim=1)
        if getattr(self.foundation, "fc_norm", None) is not None:
            pooled = self.foundation.fc_norm(pooled)
        return pooled

    def forward(self, x, mask=None, batch_meta=None):
        del mask
        tok_psd = None
        tok_meta = None
        tok_eeg = None
        if self.adapter is not None and self.adapter.use_moe and self.adapter.moe_use_psd_router_features:
            tok_psd = set_moe_psd_router_features(compact_psd_bandpowers(x))
        if self.adapter is not None and self.adapter.use_moe and self.adapter.moe_router_compact_feature_mode != "none":
            if self.adapter.moe_router_compact_feature_mode == "eeg_summary":
                compact = _compact_eeg_summary(x, self.adapter.moe_router_compact_feature_dim)
            elif self.adapter.moe_router_compact_feature_mode == "psd_summary":
                compact = compact_psd_bandpowers(x, n_bands=self.adapter.moe_router_compact_feature_dim)
            else:
                raise ValueError(
                    f"Unsupported compact router mode: {self.adapter.moe_router_compact_feature_mode!r}"
                )
            tok_eeg = set_moe_eeg_router_summary(compact)
        if self.adapter is not None and self.adapter.use_moe and self.adapter.moe_route_mode == "typed_capacity_domain":
            tok_meta = set_moe_faced_metadata(batch_meta)
        try:
            if self.adapter is None:
                # Dense LaBraM path: mirror the original finetune recipe and return
                # the pooled fc_norm feature that normally feeds LaBraM's linear head.
                if self.input_chans is not None and len(self.input_chans) != x.shape[1] + 1:
                    raise ValueError(
                        f"LaBraM input_chans length mismatch: expected {x.shape[1] + 1}, got {len(self.input_chans)}"
                    )
                return self.foundation.forward_features(
                    x,
                    input_chans=self.input_chans,
                    return_patch_tokens=False,
                )

            feats = self._foundation_features(x)
            feats = self.adapter.encoder(feats)
            return self._pool_token_grid(feats)
        finally:
            if tok_psd is not None:
                reset_moe_psd_router_features(tok_psd)
            if tok_eeg is not None:
                reset_moe_eeg_router_summary(tok_eeg)
            if tok_meta is not None:
                reset_moe_faced_metadata(tok_meta)

    def moe_auxiliary_loss(self) -> torch.Tensor:
        if self.adapter is None:
            device = next(self.foundation.parameters()).device
            return torch.zeros((), device=device, dtype=torch.float32)
        return self.adapter.moe_auxiliary_loss()
