from __future__ import annotations

import contextvars
import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_ADAPTER_BATCH_META: contextvars.ContextVar[Optional[Dict[str, torch.Tensor]]] = contextvars.ContextVar(
    "adapter_batch_meta", default=None
)


def set_adapter_batch_meta(batch_meta: Optional[Dict[str, torch.Tensor]]):
    return _ADAPTER_BATCH_META.set(batch_meta)


def reset_adapter_batch_meta(token):
    _ADAPTER_BATCH_META.reset(token)


def get_adapter_batch_meta() -> Optional[Dict[str, torch.Tensor]]:
    return _ADAPTER_BATCH_META.get()


def _to_tensor(x, dtype=torch.float32):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def load_channel_context_file(path: str, expected_channels: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Loads optional EEG channel context data from .pt/.pth/.json.

    Expected keys (all optional):
      - channel_ids: [C]
      - coords: [C,2] or [C,3]
      - montage_mask: [C]
      - region_ids: [C]
    """
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"channel_context_file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".pt", ".pth"}:
        blob = torch.load(path, map_location="cpu")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    else:
        raise ValueError("channel_context_file must be .pt/.pth/.json")

    if not isinstance(blob, dict):
        raise ValueError("channel_context_file must contain a dictionary")

    out: Dict[str, torch.Tensor] = {}
    inferred_channels: Optional[int] = int(expected_channels) if expected_channels is not None else None

    def _ensure_len(name: str, n: int):
        nonlocal inferred_channels
        if inferred_channels is None:
            inferred_channels = int(n)
            return
        if int(n) != int(inferred_channels):
            raise ValueError(f"{name} length {n} is inconsistent with inferred channel count {inferred_channels}")

    channel_ids = blob.get("channel_ids")
    if channel_ids is not None:
        ch = torch.as_tensor(channel_ids, dtype=torch.long).view(-1)
        _ensure_len("channel_ids", int(ch.numel()))
        out["channel_ids"] = ch

    coords = blob.get("coords")
    if coords is not None:
        co = torch.as_tensor(coords, dtype=torch.float32)
        if co.ndim != 2 or co.shape[1] not in (2, 3):
            raise ValueError("coords must be [C,2] or [C,3]")
        _ensure_len("coords", int(co.shape[0]))
        out["coords"] = co

    montage_mask = blob.get("montage_mask")
    if montage_mask is not None:
        mm = torch.as_tensor(montage_mask, dtype=torch.float32).view(-1)
        _ensure_len("montage_mask", int(mm.numel()))
        out["montage_mask"] = mm

    region_ids = blob.get("region_ids")
    if region_ids is not None:
        rg = torch.as_tensor(region_ids, dtype=torch.long).view(-1)
        _ensure_len("region_ids", int(rg.numel()))
        out["region_ids"] = rg

    return out


class EEGChannelContextEncoder(nn.Module):
    """Early channel-wise context injection for EEG tokens."""

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        use_channel_context: bool = False,
        channel_context_data: Optional[Dict[str, torch.Tensor]] = None,
        coord_dim: int = 3,
        region_vocab_size: int = 64,
        metadata_debug: bool = True,
    ):
        super().__init__()
        self.use_channel_context = bool(use_channel_context)
        channel_context_data = channel_context_data or {}
        inferred_c = None
        if "channel_ids" in channel_context_data:
            inferred_c = int(channel_context_data["channel_ids"].numel())
        elif "coords" in channel_context_data:
            inferred_c = int(channel_context_data["coords"].shape[0])
        elif "montage_mask" in channel_context_data:
            inferred_c = int(channel_context_data["montage_mask"].numel())
        elif "region_ids" in channel_context_data:
            inferred_c = int(channel_context_data["region_ids"].numel())
        self.num_channels = int(inferred_c if inferred_c is not None else num_channels)
        self.metadata_debug = bool(metadata_debug)
        self._usage_logged = False
        self._has_file_channel_ids = "channel_ids" in channel_context_data
        self._has_file_coords = "coords" in channel_context_data
        self._has_file_montage_mask = "montage_mask" in channel_context_data
        self._has_file_region_ids = "region_ids" in channel_context_data
        if "channel_ids" in channel_context_data:
            channel_ids = channel_context_data["channel_ids"].long().view(-1)
        else:
            channel_ids = torch.arange(self.num_channels, dtype=torch.long)

        coords = channel_context_data.get("coords")
        if coords is None:
            coords = torch.zeros(self.num_channels, coord_dim, dtype=torch.float32)
        else:
            coords = coords.float()
            if coords.shape[1] < coord_dim:
                pad = torch.zeros(self.num_channels, coord_dim - coords.shape[1], dtype=torch.float32)
                coords = torch.cat([coords, pad], dim=1)
            elif coords.shape[1] > coord_dim:
                coords = coords[:, :coord_dim]

        montage_mask = channel_context_data.get("montage_mask")
        if montage_mask is None:
            montage_mask = torch.ones(self.num_channels, dtype=torch.float32)
        else:
            montage_mask = montage_mask.float().view(-1)

        region_ids = channel_context_data.get("region_ids")
        if region_ids is None:
            region_ids = torch.zeros(self.num_channels, dtype=torch.long)
        else:
            region_ids = region_ids.long().view(-1)

        self.register_buffer("channel_ids", channel_ids, persistent=True)
        self.register_buffer("coords", coords, persistent=True)
        self.register_buffer("montage_mask", montage_mask, persistent=True)
        self.register_buffer("region_ids", region_ids, persistent=True)

        ch_vocab = max(128, int(channel_ids.max().item()) + 16)
        self.channel_emb = nn.Embedding(ch_vocab, d_model)
        self.region_emb = nn.Embedding(max(region_vocab_size, int(region_ids.max().item()) + 8), d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.mask_proj = nn.Linear(1, d_model)
        self.count_proj = nn.Linear(1, d_model)

        # Weak init: metadata should guide, not dominate.
        nn.init.zeros_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        nn.init.zeros_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)
        nn.init.zeros_(self.count_proj.weight)
        nn.init.zeros_(self.count_proj.bias)

    def forward(self, x: torch.Tensor, batch_meta: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # x: [B, C, S, D]
        if (not self.use_channel_context) or x.ndim != 4:
            return x

        bsz, ch, _, _ = x.shape
        if ch != self.num_channels:
            return x

        dev = x.device
        dtype = x.dtype

        ch_ids = self.channel_ids.to(device=dev)
        rg_ids = self.region_ids.to(device=dev)
        coords = self.coords.to(device=dev, dtype=dtype)
        mm = self.montage_mask.to(device=dev, dtype=dtype).unsqueeze(-1)

        channel_bias = self.channel_emb(ch_ids).to(dtype=dtype)
        region_bias = self.region_emb(rg_ids).to(dtype=dtype)
        coord_bias = self.coord_proj(coords)
        mask_bias = self.mask_proj(mm)

        # Optional per-batch channel count from metadata.
        if isinstance(batch_meta, dict) and torch.is_tensor(batch_meta.get("channel_count")):
            cc = batch_meta["channel_count"].to(device=dev, dtype=dtype).view(bsz, 1, 1)
            used_batch_fields = ["channel_count"]
        else:
            cc = torch.full((bsz, 1, 1), float(ch), device=dev, dtype=dtype)
            used_batch_fields = []
        cc = cc / max(float(self.num_channels), 1.0)
        count_bias = self.count_proj(cc).expand(bsz, ch, -1)

        bias = (channel_bias + region_bias + coord_bias + mask_bias).unsqueeze(0).expand(bsz, ch, -1)
        bias = bias + count_bias
        if self.metadata_debug and not self._usage_logged:
            present_fields = sorted(list(batch_meta.keys())) if isinstance(batch_meta, dict) else []
            static_fields: List[str] = []
            if self._has_file_channel_ids:
                static_fields.append("channel_ids")
            if self._has_file_coords:
                static_fields.append("coords")
            if self._has_file_montage_mask:
                static_fields.append("montage_mask")
            if self._has_file_region_ids:
                static_fields.append("region_ids")
            print(
                "[channel-context] "
                f"present_batch_fields={present_fields} "
                f"used_batch_fields={used_batch_fields} "
                f"used_static_fields={static_fields} "
                f"effective_dim={x.shape[-1]}",
                flush=True,
            )
            self._usage_logged = True
        return x + bias.unsqueeze(2)


class SubjectDomainAdapter(nn.Module):
    """Subject/domain-conditioned low-rank residual adapter.

    Shared backbone remains dominant; adapter path is weak-initialized.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 16,
        cond_dim: int = 32,
        subject_vocab: int = 8192,
        domain_vocab: int = 256,
        adapter_scale: float = 0.2,
        use_subject_summary: bool = False,
        subject_summary_handling: str = "project",
        metadata_debug: bool = True,
        log_usage: bool = True,
    ):
        super().__init__()
        self.rank = int(rank)
        self.cond_dim = int(cond_dim)
        self.adapter_scale = float(adapter_scale)
        self.use_subject_summary = bool(use_subject_summary)
        if subject_summary_handling not in {"project", "error"}:
            raise ValueError("subject_summary_handling must be one of: project, error")
        self.subject_summary_handling = subject_summary_handling
        self.metadata_debug = bool(metadata_debug)
        self.log_usage = bool(log_usage)
        self._usage_logged = False

        self.subject_emb = nn.Embedding(subject_vocab, self.cond_dim)
        self.cohort_emb = nn.Embedding(domain_vocab, self.cond_dim)
        self.dataset_emb = nn.Embedding(domain_vocab, self.cond_dim)
        self.sr_group_emb = nn.Embedding(domain_vocab, self.cond_dim)
        self.age_bucket_emb = nn.Embedding(domain_vocab, self.cond_dim)
        self.segment_bucket_emb = nn.Embedding(domain_vocab, self.cond_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, 2 * self.rank),
        )
        self.subject_summary_proj = nn.ModuleDict()

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, d_model, bias=False)

        # Weak-start: adapter initially near identity/no-op.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.cond_mlp[2].weight)
        nn.init.zeros_(self.cond_mlp[2].bias)

    def _project_subject_summary(self, ssum: torch.Tensor, device, dtype) -> torch.Tensor:
        if ssum.ndim == 1:
            ssum = ssum.unsqueeze(0)
        if ssum.ndim != 2:
            raise ValueError(f"subject_summary must be [B,D] or [D], got shape {tuple(ssum.shape)}")
        ssum = ssum.to(device=device, dtype=dtype)
        in_dim = int(ssum.shape[1])
        if in_dim == self.cond_dim:
            return ssum
        msg = (
            "subject_summary feature dimension mismatch: "
            f"got {in_dim}, expected {self.cond_dim}. "
            "Use --subject_summary_handling project (recommended) "
            "or --subject_summary_handling error."
        )
        if self.subject_summary_handling == "error":
            raise ValueError(msg)

        key = f"in_{in_dim}"
        if key not in self.subject_summary_proj:
            proj = nn.Linear(in_dim, self.cond_dim, bias=True)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.subject_summary_proj[key] = proj
            if self.metadata_debug and self.log_usage:
                print(
                    "[adapter-meta] created subject_summary projector "
                    f"{in_dim}->{self.cond_dim} (zero-init)",
                    flush=True,
                )
        proj = self.subject_summary_proj[key].to(device=device, dtype=dtype)
        return proj(ssum)

    def _log_usage_once(
        self,
        batch_meta: Optional[Dict[str, torch.Tensor]],
        used_fields: List[str],
        summary_status: str,
    ) -> None:
        if not (self.metadata_debug and self.log_usage) or self._usage_logged:
            return
        present_fields = sorted(list(batch_meta.keys())) if isinstance(batch_meta, dict) else []
        print(
            "[adapter-meta] "
            f"present_fields={present_fields} "
            f"used_fields={used_fields} "
            f"cond_dim={self.cond_dim} "
            f"subject_summary={summary_status}",
            flush=True,
        )
        self._usage_logged = True

    def _condition(self, batch_meta: Optional[Dict[str, torch.Tensor]], device, dtype) -> Optional[torch.Tensor]:
        if not isinstance(batch_meta, dict):
            self._log_usage_once(batch_meta, [], "disabled")
            return None

        used_fields: List[str] = []
        sid = batch_meta.get("subject_id")
        if not torch.is_tensor(sid):
            self._log_usage_once(batch_meta, [], "disabled")
            return None

        sid = sid.to(device=device, dtype=torch.long)
        used_fields.append("subject_id")
        cohort = batch_meta.get("cohort_id")
        cohort = cohort.to(device=device, dtype=torch.long) if torch.is_tensor(cohort) else torch.zeros_like(sid)
        if torch.is_tensor(batch_meta.get("cohort_id")):
            used_fields.append("cohort_id")
        did = batch_meta.get("dataset_id")
        did = did.to(device=device, dtype=torch.long) if torch.is_tensor(did) else torch.zeros_like(sid)
        if torch.is_tensor(batch_meta.get("dataset_id")):
            used_fields.append("dataset_id")
        srg = batch_meta.get("sample_rate_group_id")
        srg = srg.to(device=device, dtype=torch.long) if torch.is_tensor(srg) else torch.zeros_like(sid)
        if torch.is_tensor(batch_meta.get("sample_rate_group_id")):
            used_fields.append("sample_rate_group_id")
        age = batch_meta.get("age_bucket_id")
        age = age.to(device=device, dtype=torch.long) if torch.is_tensor(age) else torch.zeros_like(sid)
        if torch.is_tensor(batch_meta.get("age_bucket_id")):
            used_fields.append("age_bucket_id")
        seg = batch_meta.get("segment_bucket_id")
        seg = seg.to(device=device, dtype=torch.long) if torch.is_tensor(seg) else torch.zeros_like(sid)
        if torch.is_tensor(batch_meta.get("segment_bucket_id")):
            used_fields.append("segment_bucket_id")

        cond = self.subject_emb(sid)
        cond = (
            cond
            + self.cohort_emb(cohort)
            + self.dataset_emb(did)
            + self.sr_group_emb(srg)
            + self.age_bucket_emb(age)
            + self.segment_bucket_emb(seg)
        )

        summary_status = "disabled"
        ssum = batch_meta.get("subject_summary")
        if self.use_subject_summary:
            if torch.is_tensor(ssum):
                ssum_cond = self._project_subject_summary(ssum, device=device, dtype=cond.dtype)
                if ssum_cond.shape[0] != cond.shape[0]:
                    raise ValueError(
                        "subject_summary batch size mismatch: "
                        f"summary batch={ssum_cond.shape[0]} vs model batch={cond.shape[0]}"
                    )
                cond = cond + 0.1 * ssum_cond
                used_fields.append("subject_summary")
                summary_status = f"used({int(ssum.shape[-1])}->{self.cond_dim})"
            else:
                summary_status = "enabled_but_missing"
        self._log_usage_once(batch_meta, used_fields, summary_status)

        return cond.to(dtype=dtype)

    def forward(self, x: torch.Tensor, batch_meta: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # x: [B, C, S, D]
        cond = self._condition(batch_meta, x.device, x.dtype)
        if cond is None:
            return torch.zeros_like(x)

        bsz = x.shape[0]
        gamma_beta = self.cond_mlp(cond)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)

        h = self.down(self.norm(x))
        gamma = gamma.view(bsz, 1, 1, self.rank)
        beta = beta.view(bsz, 1, 1, self.rank)
        h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        out = self.up(h)
        return self.adapter_scale * out
