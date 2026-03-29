from __future__ import annotations

import contextvars
import json
import os
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from typing import Any, Dict, Iterable, List, Optional

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


_XLSX_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_NS = {"a": _XLSX_NS_MAIN, "r": _XLSX_NS_REL}


def _xlsx_col_to_index(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    if not letters:
        return -1
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _xlsx_cell_text(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", _XLSX_NS))

    v = cell.find("a:v", _XLSX_NS)
    if v is None:
        return ""
    text = v.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (ValueError, IndexError):
            return text
    return text


def _xlsx_read_shared_strings(zf: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out: List[str] = []
    for si in root.findall("a:si", _XLSX_NS):
        out.append("".join(t.text or "" for t in si.findall(".//a:t", _XLSX_NS)))
    return out


def _xlsx_parse_sheet_rows(zf: ZipFile, target: str, shared_strings: List[str]) -> List[List[str]]:
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    root = ET.fromstring(zf.read(target))
    rows: List[List[str]] = []
    for row in root.findall("a:sheetData/a:row", _XLSX_NS):
        sparse: Dict[int, str] = {}
        for cell in row.findall("a:c", _XLSX_NS):
            cell_ref = cell.attrib.get("r", "")
            idx = _xlsx_col_to_index(cell_ref)
            if idx < 0:
                continue
            sparse[idx] = _xlsx_cell_text(cell, shared_strings)
        if not sparse:
            continue
        max_col = max(sparse.keys())
        rows.append([sparse.get(i, "") for i in range(max_col + 1)])
    return rows


def _parse_positive_int(val: str) -> Optional[int]:
    s = str(val).strip()
    if not s:
        return None
    try:
        num = float(s)
    except ValueError:
        return None
    if not num.is_integer():
        return None
    out = int(num)
    return out if out > 0 else None


def _load_channel_context_xlsx(path: str) -> Dict[str, Any]:
    """Parses FACED Electrode_Location.xlsx style sheets into channel context."""
    with ZipFile(path, "r") as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared_strings = _xlsx_read_shared_strings(zf)

        id_to_name: Dict[int, str] = {}
        sheets = wb.findall("a:sheets/a:sheet", _XLSX_NS)
        for sheet in sheets:
            rid = sheet.attrib.get(f"{{{_XLSX_NS_REL}}}id")
            if not rid or rid not in rel_map:
                continue
            rows = _xlsx_parse_sheet_rows(zf, rel_map[rid], shared_strings)
            for row in rows:
                # FACED layout is repeated [channel_id, channel_name] pairs across columns.
                for col in range(0, max(0, len(row) - 1), 2):
                    channel_id = _parse_positive_int(row[col])
                    if channel_id is None:
                        continue
                    name = re.sub(r"\s+", " ", str(row[col + 1]).strip())
                    if not name:
                        continue
                    if not re.search(r"[A-Za-z]", name):
                        continue
                    id_to_name[channel_id] = name

    if not id_to_name:
        raise ValueError(
            "Failed to parse channel id/name pairs from .xlsx. "
            "Expected FACED Electrode_Location layout with [id, name] pairs."
        )

    ordered_ids = sorted(id_to_name.keys())
    return {
        "channel_ids": ordered_ids,
        "channel_names": [id_to_name[i] for i in ordered_ids],
    }


def load_channel_context_file(
    path: str,
    expected_channels: Optional[int] = None,
    expected_channel_ids: Optional[Iterable[int]] = None,
    align_mode: str = "auto",
) -> Dict[str, torch.Tensor]:
    """Loads optional EEG channel context data from .pt/.pth/.json/.xlsx.

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
    elif ext == ".xlsx":
        blob = _load_channel_context_xlsx(path)
    else:
        raise ValueError("channel_context_file must be .pt/.pth/.json/.xlsx")

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

    if inferred_channels is not None and expected_channels is not None and int(inferred_channels) != int(expected_channels):
        raise ValueError(
            f"channel context inferred channels={inferred_channels} does not match expected={expected_channels}"
        )

    if align_mode not in {"auto", "strict", "off"}:
        raise ValueError("align_mode must be one of: auto, strict, off")

    if "channel_ids" in out:
        ch = out["channel_ids"]
        if int(torch.unique(ch).numel()) != int(ch.numel()):
            raise ValueError("channel_ids contains duplicate values")
        if expected_channel_ids is not None and align_mode != "off":
            exp = torch.as_tensor(list(expected_channel_ids), dtype=torch.long).view(-1)
            if exp.numel() != ch.numel():
                raise ValueError(
                    "channel_ids length mismatch with expected channels "
                    f"(expected={int(exp.numel())}, got={int(ch.numel())})"
                )

            ch_cpu = ch.cpu()
            if align_mode == "auto" and not torch.equal(ch_cpu, exp):
                # Common convention: 1..C instead of 0..C-1.
                one_based = exp + 1
                if torch.equal(ch_cpu, one_based):
                    ch_cpu = exp.clone()
                    out["channel_ids"] = ch_cpu

            if torch.equal(ch_cpu, exp):
                return out

            if align_mode == "strict":
                raise ValueError(
                    "channel_ids in context file do not match expected labels/order "
                    f"(expected={exp.tolist()}, got={ch_cpu.tolist()})"
                )

            # Same labels but different order: reorder arrays to expected order.
            exp_set = set(exp.tolist())
            ch_set = set(ch_cpu.tolist())
            if ch_set == exp_set:
                idx_map = {int(v): i for i, v in enumerate(ch_cpu.tolist())}
                perm = torch.as_tensor([idx_map[int(v)] for v in exp.tolist()], dtype=torch.long)
                for key in ("channel_ids", "coords", "montage_mask", "region_ids"):
                    if key in out:
                        out[key] = out[key].index_select(0, perm)
                out["channel_ids"] = exp.clone()
            else:
                raise ValueError(
                    "channel_ids set mismatch with expected labels "
                    f"(expected_set={sorted(exp_set)}, got_set={sorted(ch_set)})"
                )

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
        self.context_scale = nn.Parameter(torch.tensor(0.0))

        # Weak init: metadata should guide, not dominate.
        nn.init.zeros_(self.channel_emb.weight)
        nn.init.zeros_(self.region_emb.weight)
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
                f"context_scale_init={float(self.context_scale.detach().item()):.4f} "
                f"effective_dim={x.shape[-1]}",
                flush=True,
            )
            if (
                self._has_file_channel_ids
                and not self._has_file_coords
                and not self._has_file_montage_mask
                and not self._has_file_region_ids
            ):
                print(
                    "[channel-context][warning] only channel_ids are provided; "
                    "no coords/montage/region metadata found. Context signal may be weak.",
                    flush=True,
                )
            self._usage_logged = True
        return x + (self.context_scale * bias).unsqueeze(2)


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
        subject_summary_dim: int = 0,
        subject_summary_handling: str = "project",
        metadata_debug: bool = True,
        log_usage: bool = True,
        use_subject_id: bool = True,
        use_age_bucket: bool = True,
        use_segment_bucket: bool = False,
    ):
        super().__init__()
        self.rank = int(rank)
        self.cond_dim = int(cond_dim)
        self.adapter_scale = float(adapter_scale)
        self.use_subject_summary = bool(use_subject_summary)
        self.subject_summary_dim = int(subject_summary_dim)
        self.use_subject_id = bool(use_subject_id)
        self.use_age_bucket = bool(use_age_bucket)
        self.use_segment_bucket = bool(use_segment_bucket)
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
        self.subject_summary_proj = None
        if self.use_subject_summary:
            if self.subject_summary_dim <= 0:
                raise ValueError(
                    "use_subject_summary=True requires a positive subject_summary_dim at model init."
                )
            if self.subject_summary_dim == self.cond_dim:
                self.subject_summary_proj = nn.Identity()
            else:
                self.subject_summary_proj = nn.Linear(self.subject_summary_dim, self.cond_dim, bias=True)
                nn.init.zeros_(self.subject_summary_proj.weight)
                nn.init.zeros_(self.subject_summary_proj.bias)

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
        if in_dim != self.subject_summary_dim:
            msg = (
                "subject_summary feature dimension mismatch: "
                f"got {in_dim}, expected {self.subject_summary_dim}."
            )
            raise ValueError(msg)
        if self.subject_summary_proj is None:
            raise ValueError("subject_summary projection is not initialized")
        proj = self.subject_summary_proj.to(device=device, dtype=dtype)
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
        if self.use_subject_id and not torch.is_tensor(sid):
            self._log_usage_once(batch_meta, [], "disabled")
            return None
        if torch.is_tensor(sid):
            sid = sid.to(device=device, dtype=torch.long)
            if self.use_subject_id:
                used_fields.append("subject_id")
        else:
            # fall back to zeros so domain-only adapters can still run
            ref = batch_meta.get("dataset_id")
            if not torch.is_tensor(ref):
                self._log_usage_once(batch_meta, [], "disabled")
                return None
            sid = torch.zeros_like(ref.to(device=device, dtype=torch.long))
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
        if self.use_age_bucket and torch.is_tensor(batch_meta.get("age_bucket_id")):
            used_fields.append("age_bucket_id")
        seg = batch_meta.get("segment_bucket_id")
        seg = seg.to(device=device, dtype=torch.long) if torch.is_tensor(seg) else torch.zeros_like(sid)
        if self.use_segment_bucket and torch.is_tensor(batch_meta.get("segment_bucket_id")):
            used_fields.append("segment_bucket_id")

        cond = self.cohort_emb(cohort) + self.dataset_emb(did) + self.sr_group_emb(srg)
        if self.use_subject_id:
            cond = cond + self.subject_emb(sid)
        if self.use_age_bucket:
            cond = cond + self.age_bucket_emb(age)
        if self.use_segment_bucket:
            cond = cond + self.segment_bucket_emb(seg)

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
                summary_status = f"used({self.subject_summary_dim}->{self.cond_dim})"
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
