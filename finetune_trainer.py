import gc
import os
import random
import shutil
import traceback
from collections import deque
from timeit import default_timer as timer
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from finetune_evaluator import Evaluator
from utils.tqdm_auto import tqdm_auto
from models.moe import (
    format_moe_diagnostics_lines,
    reset_moe_diagnostic_labels,
    set_moe_train_epoch,
    set_moe_diagnostic_labels,
)
from models.prototype_alignment import PrototypeBank

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _mem_report(tag: str, model_dir: Optional[str]) -> None:
    """Process RSS, CUDA alloc/reserved/max, free disk on model_dir filesystem."""
    parts = [f"[mem] {tag}"]
    if _HAS_PSUTIL:
        try:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            parts.append(f"RSS_MB={rss_mb:.1f}")
        except Exception:
            parts.append("RSS_MB=?")
    if torch.cuda.is_available():
        try:
            parts.append(f"cuda_alloc_MB={torch.cuda.memory_allocated() / 1e6:.2f}")
            parts.append(f"cuda_reserved_MB={torch.cuda.memory_reserved() / 1e6:.2f}")
            parts.append(f"cuda_max_alloc_MB={torch.cuda.max_memory_allocated() / 1e6:.2f}")
        except Exception:
            parts.append("cuda_mem=?")
    path = model_dir or "."
    try:
        root = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)) or "."
        du = shutil.disk_usage(root)
        parts.append(f"disk_free_GB={du.free / (1024 ** 3):.2f}")
    except Exception:
        parts.append("disk_free=?")
    print(" ".join(parts), flush=True)


def _estimate_state_dict_cpu_bytes(sd: Dict[str, Any]) -> int:
    n = 0
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            n += v.numel() * v.element_size()
    return n


def _state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, Any]:
    """Checkpoint snapshot without deepcopy: avoids duplicating GPU weights (OOM on large models)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _move_meta_to_cuda(batch_meta):
    if not isinstance(batch_meta, dict):
        return None
    out = {}
    for k, v in batch_meta.items():
        if torch.is_tensor(v):
            out[k] = v.cuda(non_blocking=True)
    return out


def _forward_with_optional_meta(model, x, batch_meta):
    if batch_meta is None:
        return model(x)
    try:
        return model(x, batch_meta=batch_meta)
    except TypeError:
        return model(x)


def _forward_with_shared_feature(model, x, batch_meta):
    try:
        if batch_meta is None:
            out = model(x, return_features=True)
        else:
            out = model(x, batch_meta=batch_meta, return_features=True)
        if isinstance(out, tuple) and len(out) == 2:
            return out
    except TypeError:
        pass
    logits = _forward_with_optional_meta(model, x, batch_meta)
    return logits, None


def _extract_batch_meta(batch):
    if len(batch) >= 3 and isinstance(batch[2], dict):
        return batch[2]
    if len(batch) >= 4 and isinstance(batch[3], dict):
        return batch[3]
    return None


_META_UNKNOWN_KEYS = (
    "subject_id",
    "cohort_id",
    "sample_rate_group_id",
    "age_bucket_id",
    "segment_bucket_id",
)


def _init_unknown_counter() -> Dict[str, Dict[str, int]]:
    return {k: {"unknown": 0, "total": 0} for k in _META_UNKNOWN_KEYS}


def _update_unknown_counter(counter: Dict[str, Dict[str, int]], batch_meta: Optional[Dict[str, torch.Tensor]]) -> None:
    if not isinstance(batch_meta, dict):
        return
    for k in _META_UNKNOWN_KEYS:
        v = batch_meta.get(k)
        if not torch.is_tensor(v):
            continue
        vv = v.detach()
        if vv.numel() == 0:
            continue
        total = int(vv.numel())
        unknown = int((vv == 0).sum().item())
        counter[k]["total"] += total
        counter[k]["unknown"] += unknown


def _unknown_ratio_dict(counter: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, d in counter.items():
        tot = int(d.get("total", 0))
        unk = int(d.get("unknown", 0))
        out[k] = float(unk / tot) if tot > 0 else 0.0
    return out


def _is_router_param_name(name: str) -> bool:
    if "moe_ffn." not in name:
        return False
    router_keys = (
        "spatial_router",
        "spectral_router",
        "router_input_norm",
        "domain_emb_",
        "domain_proj_",
        "adapter_cond_proj_",
        "subject_summary_router_proj_",
        "eeg_summary_router_proj_",
    )
    return any(k in name for k in router_keys)


class ReplayBuffer:
    """Small replay memory storing past samples and optional snapshot logits."""

    def __init__(self, max_samples: int):
        self.max_samples = max(0, int(max_samples))
        self.entries = deque()
        self.num_samples = 0

    def __len__(self):
        return self.num_samples

    def add(self, x, y, batch_meta=None, logits=None):
        if self.max_samples <= 0:
            return
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        meta_cpu = None
        if isinstance(batch_meta, dict):
            meta_cpu = {k: v.detach().cpu() for k, v in batch_meta.items() if torch.is_tensor(v)}
        logits_cpu = logits.detach().cpu() if torch.is_tensor(logits) else None
        n = int(x_cpu.shape[0])
        if n <= 0:
            return
        self.entries.append({
            "x": x_cpu,
            "y": y_cpu,
            "meta": meta_cpu,
            "logits": logits_cpu,
            "n": n,
        })
        self.num_samples += n
        while self.num_samples > self.max_samples and self.entries:
            popped = self.entries.popleft()
            self.num_samples -= int(popped.get("n", 0))

    def sample(self, batch_size: int):
        if not self.entries or batch_size <= 0:
            return None
        ent = random.choice(list(self.entries))
        x, y, meta, logits = ent["x"], ent["y"], ent["meta"], ent["logits"]
        n = int(x.shape[0])
        if n <= batch_size:
            idx = torch.arange(n)
        else:
            idx = torch.randperm(n)[:batch_size]
        out_meta = None
        if isinstance(meta, dict):
            out_meta = {k: v.index_select(0, idx) for k, v in meta.items()}
        out_logits = logits.index_select(0, idx) if torch.is_tensor(logits) else None
        return (
            x.index_select(0, idx),
            y.index_select(0, idx),
            out_meta,
            out_logits,
        )


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None
        self._router_params = []
        self._router_param_names = []
        self._router_warmup_mode = str(getattr(self.params, "moe_router_warmup_mode", "off"))
        self._router_warmup_epochs = int(max(0, getattr(self.params, "moe_router_warmup_epochs", 0)))
        self._router_warmup_lr_scale = float(max(1e-8, getattr(self.params, "moe_router_warmup_lr_scale", 0.1)))
        self._router_warmup_last_state = None

        backbone_params = []
        other_params = []
        n_backbone_trainable = 0
        n_other_trainable = 0
        for name, param in self.model.named_parameters():
            if getattr(self.params, "adapter_only_update", False):
                allow = ("subject_adapter" in name) or ("channel_context_encoder" in name) or ("classifier" in name)
                param.requires_grad = bool(allow)
            if "backbone" in name:
                if params.frozen and not getattr(self.params, "adapter_only_update", False):
                    param.requires_grad = False
                elif not getattr(self.params, "adapter_only_update", False):
                    param.requires_grad = True
                if param.requires_grad:
                    backbone_params.append(param)
                    n_backbone_trainable += int(param.numel())
            else:
                if param.requires_grad:
                    other_params.append(param)
                    n_other_trainable += int(param.numel())

            if param.requires_grad and _is_router_param_name(name):
                self._router_params.append(param)
                self._router_param_names.append(name)

        self.data_length = len(self.data_loader['train'])
        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': 0.001 * (self.params.batch_size / 256) ** 0.5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.params.lr,
                    weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.params.lr, momentum=0.9,
                    weight_decay=self.params.weight_decay)

        # Original CBraMod finetune: cosine over full run, per optimizer step
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        self.replay_buffer = ReplayBuffer(getattr(self.params, "continual_memory_size", 0))
        self.prototype_bank = None
        self._prototype_active = bool(getattr(self.params, "prototype_alignment", False))
        if self._prototype_active and int(getattr(self.params, "num_of_classes", 0)) > 1:
            feat_dim = 200
            bb = getattr(self.model, "backbone", None)
            pe = getattr(bb, "patch_embedding", None)
            if pe is not None and hasattr(pe, "d_model"):
                feat_dim = int(pe.d_model)
            self.prototype_bank = PrototypeBank(
                num_classes=int(self.params.num_of_classes),
                feat_dim=feat_dim,
                momentum=float(getattr(self.params, "prototype_momentum", 0.95)),
            ).cuda()
        print(
            "[mechanism-summary] "
            f"eeg_conditioned_attnres={getattr(self.params, 'attnres_eeg_cond_gates', False)} "
            f"subject_conditioned_attnres={getattr(self.params, 'attnres_subject_gates', False)} "
            f"prototype_alignment={self._prototype_active} "
            f"moe_active={getattr(self.params, 'moe', False)} "
            f"eeg_context_active={getattr(self.params, 'eeg_channel_context', False)} "
            f"subject_summary_active={getattr(self.params, 'use_subject_summary', False)}",
            flush=True,
        )

        print(
            "[optim] "
            f"adapter_only_update={getattr(self.params, 'adapter_only_update', False)} "
            f"trainable_backbone_params={n_backbone_trainable} "
            f"trainable_other_params={n_other_trainable}",
            flush=True,
        )
        if len(self.optimizer.param_groups) >= 2:
            print(
                "[optim] "
                f"lr_backbone={self.optimizer.param_groups[0].get('lr', 0.0):.6g} "
                f"lr_other={self.optimizer.param_groups[1].get('lr', 0.0):.6g} "
                f"weight_decay={self.params.weight_decay}",
                flush=True,
            )

        print(self.model)

    def _apply_router_warmup_state(self, epoch_one_based: int) -> None:
        if not getattr(self.params, 'moe', False) or not self._router_params:
            return
        mode = self._router_warmup_mode
        warm_active = (self._router_warmup_epochs > 0 and epoch_one_based <= self._router_warmup_epochs)
        if mode == 'freeze':
            req_grad = not warm_active
            for p in self._router_params:
                p.requires_grad = req_grad
        elif mode == 'off':
            for p in self._router_params:
                p.requires_grad = True

        state = (mode, warm_active)
        if state != self._router_warmup_last_state:
            print(
                "[router-warmup] "
                f"mode={mode} epoch={epoch_one_based} active={warm_active} "
                f"router_params={len(self._router_params)} grad_scale={self._router_warmup_lr_scale}",
                flush=True,
            )
            self._router_warmup_last_state = state

    def _scale_router_grads_if_needed(self, epoch_one_based: int) -> None:
        if not getattr(self.params, 'moe', False) or not self._router_params:
            return
        if self._router_warmup_mode != 'low_lr':
            return
        if self._router_warmup_epochs <= 0 or epoch_one_based > self._router_warmup_epochs:
            return
        s = self._router_warmup_lr_scale
        for p in self._router_params:
            if p.grad is not None:
                p.grad.mul_(s)

    def _add_moe_auxiliary_loss(self, loss):
        if not getattr(self.params, 'moe', False):
            return loss
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'moe_auxiliary_loss'):
            return loss
        aux = bb.moe_auxiliary_loss()
        return loss + aux

    def _prototype_loss(self, shared_feat, y, epoch_one_based: int):
        device = next(self.model.parameters()).device
        z = torch.zeros((), device=device)
        if self.prototype_bank is None or shared_feat is None:
            return z, {}
        if epoch_one_based < int(getattr(self.params, "prototype_start_epoch", 1)):
            return z, {}
        losses = self.prototype_bank.losses(
            shared_feat,
            y,
            pull_weight=float(getattr(self.params, "prototype_pull_weight", 0.05)),
            push_weight=float(getattr(self.params, "prototype_push_weight", 0.02)),
            margin=float(getattr(self.params, "prototype_margin", 0.1)),
        )
        return losses["total"], {
            "pull": float(losses["pull"].detach().item()),
            "push": float(losses["push"].detach().item()),
            "total": float(losses["total"].detach().item()),
        }

    @torch.no_grad()
    def _prototype_update(self, shared_feat, y):
        if self.prototype_bank is None or shared_feat is None:
            return
        self.prototype_bank.update(shared_feat, y)

    def _print_attnres_diagnostics(self, epoch_one_based: int) -> None:
        interval = int(getattr(self.params, "attnres_diag_interval", 1))
        if interval <= 0 or (epoch_one_based % interval) != 0:
            return
        bb = getattr(self.model, "backbone", None)
        if bb is None or not hasattr(bb, "attnres_gate_diagnostics"):
            return
        diag = bb.attnres_gate_diagnostics()
        if not diag:
            return
        keys = sorted(diag.keys())
        preview = " ".join([f"{k}={diag[k]:.4f}" for k in keys[:14]])
        print(f"[attnres][diag] ep={epoch_one_based} {preview}", flush=True)

    def _accumulate_moe_train_diag(self, agg: Optional[Dict[str, Any]]) -> None:
        if not isinstance(agg, dict):
            return
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'encoder'):
            return
        for layer in bb.encoder.layers:
            m = getattr(layer, 'moe_ffn', None)
            diag = getattr(m, 'last_diagnostics', None) if m is not None else None
            if not isinstance(diag, dict):
                continue
            sp = diag.get('spatial', {})
            sc = diag.get('spectral', {})
            if isinstance(sp, dict):
                agg['max_frac_vals'].append(float(sp.get('pre_max_expert_fraction', 0.0)))
                agg['reroute_vals'].append(float(sp.get('reroute_rate', 0.0)))
                agg['margin_vals'].append(float(sp.get('pre_margin_logit', 0.0)))
            if isinstance(sc, dict):
                agg['max_frac_vals'].append(float(sc.get('pre_max_expert_fraction', 0.0)))
                agg['reroute_vals'].append(float(sc.get('reroute_rate', 0.0)))
                agg['margin_vals'].append(float(sc.get('pre_margin_logit', 0.0)))
            rs = str(diag.get('route_strategy', 'hard_capacity'))
            agg['route_strategies'][rs] = agg['route_strategies'].get(rs, 0) + 1

    def _print_moe_train_summary(self, epoch_one_based: int, agg: Optional[Dict[str, Any]]) -> None:
        if not isinstance(agg, dict) or not agg.get('max_frac_vals'):
            return
        n_layers = len(agg['max_frac_vals']) // 2 if len(agg['max_frac_vals']) >= 2 else len(agg['max_frac_vals'])
        print(
            "[MoE train summary] "
            f"ep={epoch_one_based} "
            f"layers={n_layers} "
            f"avg_pre_max_frac={float(np.mean(agg['max_frac_vals'])):.4f} "
            f"avg_reroute_rate={float(np.mean(agg['reroute_vals'])):.4f} "
            f"avg_pre_margin_logit={float(np.mean(agg['margin_vals'])):.4f} "
            f"route_strategies={agg['route_strategies']}",
            flush=True,
        )

    def _collect_moe_snapshot(self) -> Optional[Dict[str, Any]]:
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'encoder'):
            return None
        max_frac_vals = []
        reroute_vals = []
        overflow_vals = []
        pre_entropy_vals = []
        route_counts: Dict[str, int] = {}
        for layer in bb.encoder.layers:
            m = getattr(layer, 'moe_ffn', None)
            diag = getattr(m, 'last_diagnostics', None) if m is not None else None
            if not isinstance(diag, dict):
                continue
            sp = diag.get('spatial', {})
            sc = diag.get('spectral', {})
            if isinstance(sp, dict):
                max_frac_vals.append(float(sp.get('pre_max_expert_fraction', 0.0)))
                reroute_vals.append(float(sp.get('reroute_rate', 0.0)))
                overflow_vals.append(float(sp.get('overflow_count', 0.0)))
                pre_entropy_vals.append(float(sp.get('pre_entropy', 0.0)))
            if isinstance(sc, dict):
                max_frac_vals.append(float(sc.get('pre_max_expert_fraction', 0.0)))
                reroute_vals.append(float(sc.get('reroute_rate', 0.0)))
                overflow_vals.append(float(sc.get('overflow_count', 0.0)))
                pre_entropy_vals.append(float(sc.get('pre_entropy', 0.0)))
            rs = str(diag.get('route_strategy', 'hard_capacity'))
            route_counts[rs] = route_counts.get(rs, 0) + 1
        if not max_frac_vals:
            return None
        n_layers = len(max_frac_vals) // 2 if len(max_frac_vals) >= 2 else len(max_frac_vals)
        return {
            'layers': n_layers,
            'avg_pre_max_frac': float(np.mean(max_frac_vals)),
            'avg_pre_entropy': float(np.mean(pre_entropy_vals)) if pre_entropy_vals else 0.0,
            'avg_reroute_rate': float(np.mean(reroute_vals)) if reroute_vals else 0.0,
            'avg_overflow_count': float(np.mean(overflow_vals)) if overflow_vals else 0.0,
            'route_strategies': route_counts,
        }

    def _print_moe_snapshot(self, tag: str, epoch_one_based: int, snap: Optional[Dict[str, Any]]) -> None:
        if not isinstance(snap, dict):
            return
        print(
            f"[MoE compare][{tag}] "
            f"ep={epoch_one_based} "
            f"layers={snap.get('layers', 0)} "
            f"avg_pre_max_frac={snap.get('avg_pre_max_frac', 0.0):.4f} "
            f"avg_pre_entropy={snap.get('avg_pre_entropy', 0.0):.4f} "
            f"avg_reroute_rate={snap.get('avg_reroute_rate', 0.0):.4f} "
            f"avg_overflow_count={snap.get('avg_overflow_count', 0.0):.4f} "
            f"route_strategies={snap.get('route_strategies', {})}",
            flush=True,
        )

    def _log_moe_diagnostics(self, epoch_one_based: Optional[int] = None):
        if not getattr(self.params, 'moe_diagnostics', False) or not getattr(self.params, 'moe', False):
            return
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'encoder'):
            return
        was_training = self.model.training
        self.model.eval()
        try:
            batch = next(iter(self.data_loader['val']))
        except StopIteration:
            if was_training:
                self.model.train()
            return
        x = batch[0].cuda()
        batch_meta = _move_meta_to_cuda(_extract_batch_meta(batch))
        label_tok = None
        if len(batch) > 1:
            label_tok = set_moe_diagnostic_labels(batch[1].cuda())
        try:
            with torch.no_grad():
                _ = _forward_with_optional_meta(self.model, x, batch_meta)
        finally:
            if label_tok is not None:
                reset_moe_diagnostic_labels(label_tok)
        val_snap = self._collect_moe_snapshot()
        if epoch_one_based is not None:
            self._print_moe_snapshot('val-diag', epoch_one_based, val_snap)
        print(
            '[MoE diagnostics] one val batch, eval (no router noise; train-time warmup/jitter may differ)  '
            f"route_mode={getattr(self.params, 'moe_route_mode', '?')}  "
            f"psd_feats={getattr(self.params, 'moe_use_psd_router_features', False)}  "
            f"domain_bias={getattr(self.params, 'moe_domain_bias', False)}"
        )
        max_frac_vals = []
        reroute_vals = []
        margin_vals = []
        strat_counts: Dict[str, int] = {}
        for i, layer in enumerate(bb.encoder.layers):
            m = getattr(layer, 'moe_ffn', None)
            diag = getattr(m, 'last_diagnostics', None) if m is not None else None
            if diag is None:
                continue
            sp = diag.get('spatial', {})
            sc = diag.get('spectral', {})
            if sp:
                max_frac_vals.append(float(sp.get('pre_max_expert_fraction', 0.0)))
                reroute_vals.append(float(sp.get('reroute_rate', 0.0)))
                margin_vals.append(float(sp.get('pre_margin_logit', 0.0)))
            if sc:
                max_frac_vals.append(float(sc.get('pre_max_expert_fraction', 0.0)))
                reroute_vals.append(float(sc.get('reroute_rate', 0.0)))
                margin_vals.append(float(sc.get('pre_margin_logit', 0.0)))
            rs = str(diag.get('route_strategy', 'hard_capacity'))
            strat_counts[rs] = strat_counts.get(rs, 0) + 1
            for line in format_moe_diagnostics_lines(i, diag):
                print(line)
        if max_frac_vals:
            print(
                "[MoE summary] "
                f"layers={len(max_frac_vals)//2 if len(max_frac_vals) >= 2 else len(max_frac_vals)} "
                f"avg_pre_max_frac={float(np.mean(max_frac_vals)):.4f} "
                f"avg_reroute_rate={float(np.mean(reroute_vals)):.4f} "
                f"avg_pre_margin_logit={float(np.mean(margin_vals)):.4f} "
                f"route_strategies={strat_counts}",
                flush=True,
            )
        if was_training:
            self.model.train()

    def _model_dir(self) -> str:
        return getattr(self.params, "model_dir", ".") or "."

    def _epoch_end_gc(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _print_slurm_oom_hints() -> None:
        print(
            "[hints] SLURM/host OOM triage: sacct -j <JOBID> "
            "--format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem | "
            "check stderr for: Killed, OOM-kill, cgroup, Out Of Memory",
            flush=True,
        )

    def _on_val_epoch_exception(self, md: str, epoch: int, exc: BaseException) -> None:
        print(traceback.format_exc(), flush=True)
        _mem_report(f"val_exception ep={epoch} {type(exc).__name__}", md)
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary(), flush=True)

    def _continual_on(self) -> bool:
        return getattr(self.params, "continual_mode", "off") != "off"

    def _continual_add_current_batch(self, x, y, batch_meta, pred):
        if not self._continual_on():
            return
        use_distill = getattr(self.params, "continual_mode", "off") == "replay_distill"
        self.replay_buffer.add(x, y, batch_meta=batch_meta, logits=(pred if use_distill else None))

    def _distill_loss(self, cur_logits, tgt_logits, task_type: str):
        if not (torch.is_tensor(cur_logits) and torch.is_tensor(tgt_logits)):
            return torch.zeros((), device=next(self.model.parameters()).device)
        if task_type == "multiclass":
            t = float(getattr(self.params, "continual_distill_temp", 2.0))
            t = max(1e-6, t)
            p = torch.log_softmax(cur_logits / t, dim=-1)
            q = torch.softmax(tgt_logits / t, dim=-1)
            return torch.nn.functional.kl_div(p, q, reduction="batchmean") * (t * t)
        return torch.nn.functional.mse_loss(cur_logits, tgt_logits)

    def _continual_replay_penalty(self, task_type: str):
        if not self._continual_on() or len(self.replay_buffer) <= 0:
            return torch.zeros((), device="cuda"), {}
        rb = int(getattr(self.params, "continual_replay_batch_size", 0))
        replay = self.replay_buffer.sample(rb)
        if replay is None:
            return torch.zeros((), device="cuda"), {}

        rx, ry, rmeta, rlogits = replay
        rx = rx.cuda(non_blocking=True)
        ry = ry.cuda(non_blocking=True)
        rmeta = _move_meta_to_cuda(rmeta)
        rpred = _forward_with_optional_meta(self.model, rx, rmeta)

        rep_w = float(getattr(self.params, "continual_replay_weight", 0.5))
        dist_w = float(getattr(self.params, "continual_distill_weight", 0.2))
        out = torch.zeros((), device=rx.device)
        stats = {}
        if rep_w > 0:
            if task_type == "multiclass" and self.params.downstream_dataset == "ISRUC":
                rep_loss = self.criterion(rpred.transpose(1, 2), ry)
            else:
                rep_loss = self.criterion(rpred, ry)
            out = out + rep_w * rep_loss
            stats["replay"] = float(rep_loss.detach().item())
        if getattr(self.params, "continual_mode", "off") == "replay_distill" and dist_w > 0 and torch.is_tensor(rlogits):
            d = self._distill_loss(rpred, rlogits.cuda(non_blocking=True), task_type=task_type)
            out = out + dist_w * d
            stats["distill"] = float(d.detach().item())
        return out, stats

    def train_for_multiclass(self):
        """Same optimizer/schedule for baseline (--attnres_variant none) and AttnRes variants.

        AttnRes uses strict=False partial load in the model; new modules stay initialized here.
        No staged freeze, no separate pretrained/new LR groups.
        """
        md = self._model_dir()
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_f1_epoch = 0
        train_steps = 0

        self._print_slurm_oom_hints()

        try:
            for epoch in range(self.params.epochs):
                if getattr(self.params, 'moe', False):
                    set_moe_train_epoch(epoch + 1)
                self._apply_router_warmup_state(epoch + 1)
                _mem_report(f"epoch_start ep={epoch + 1}/{self.params.epochs}", md)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                num_classes = int(getattr(self.params, "num_of_classes", 0))
                train_pred_hist = torch.zeros(max(1, num_classes), dtype=torch.long)
                train_true_hist = torch.zeros(max(1, num_classes), dtype=torch.long)
                train_unknown_counter = _init_unknown_counter()
                train_correct = 0
                train_seen = 0

                self.model.train()
                start_time = timer()
                losses = []
                moe_train_agg = None
                if getattr(self.params, 'moe', False):
                    moe_train_agg = {
                        'max_frac_vals': [],
                        'reroute_vals': [],
                        'margin_vals': [],
                        'route_strategies': {},
                    }
                train_first_snap = None
                for batch_idx, batch in enumerate(tqdm_auto(self.data_loader['train'], self.params, mininterval=10)):
                    try:
                        x, y = batch[0], batch[1]
                        self.optimizer.zero_grad(set_to_none=True)
                        x = x.cuda()
                        y = y.cuda()
                        batch_meta = _move_meta_to_cuda(_extract_batch_meta(batch))
                        pred, shared_feat = _forward_with_shared_feature(self.model, x, batch_meta)
                        self._accumulate_moe_train_diag(moe_train_agg)
                        if train_first_snap is None and getattr(self.params, 'moe', False):
                            train_first_snap = self._collect_moe_snapshot()
                        with torch.no_grad():
                            pred_cls = torch.argmax(pred.detach(), dim=-1).reshape(-1).to(dtype=torch.long)
                            true_cls = y.detach().reshape(-1).to(dtype=torch.long)
                            if pred_cls.numel() > 0:
                                binc = torch.bincount(pred_cls.cpu(), minlength=train_pred_hist.numel())
                                train_pred_hist += binc[:train_pred_hist.numel()]
                                tbinc = torch.bincount(true_cls.cpu(), minlength=train_true_hist.numel())
                                train_true_hist += tbinc[:train_true_hist.numel()]
                                train_correct += int((pred_cls == true_cls).sum().item())
                                train_seen += int(true_cls.numel())
                            _update_unknown_counter(train_unknown_counter, batch_meta)
                        if self.params.downstream_dataset == 'ISRUC':
                            loss = self.criterion(pred.transpose(1, 2), y)
                        else:
                            loss = self.criterion(pred, y)
                        ploss, pstats = self._prototype_loss(shared_feat, y, epoch + 1)
                        loss = loss + ploss
                        loss = self._add_moe_auxiliary_loss(loss)
                        continual_penalty, continual_stats = self._continual_replay_penalty("multiclass")
                        loss = loss + continual_penalty

                        if not torch.isfinite(loss).all():
                            lv = float(loss.detach().item()) if loss.numel() == 1 else "non_scalar"
                            print(
                                f"[train] non-finite loss ep={epoch + 1} batch={batch_idx} loss={lv}"
                            )
                            raise RuntimeError("non-finite training loss")

                        loss.backward()
                        self._scale_router_grads_if_needed(epoch + 1)
                        losses.append(float(loss.item()))
                        if self.params.clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                        self.optimizer.step()
                        self.optimizer_scheduler.step()
                        self._prototype_update(shared_feat, y)
                        self._continual_add_current_batch(x, y, batch_meta, pred)
                        train_steps += 1
                        if train_steps % 50 == 0:
                            _mem_report(f"train_step ep={epoch + 1} batch={batch_idx} step={train_steps}", md)
                        if train_steps % 100 == 0 and continual_stats:
                            print(f"[continual] step={train_steps} {continual_stats}", flush=True)
                        if train_steps % 100 == 0 and pstats:
                            print(f"[prototype][train] step={train_steps} {pstats}", flush=True)
                    except RuntimeError as e:
                        err = str(e).lower()
                        if "out of memory" in err:
                            _mem_report(
                                f"OOM_train ep={epoch + 1} batch={batch_idx} step={train_steps}",
                                md,
                            )
                            print(traceback.format_exc(), flush=True)
                        raise

                print(f"finished train loop for epoch {epoch + 1}", flush=True)
                _mem_report(f"finished_train_epoch_{epoch + 1}", md)
                tr_hist = train_pred_hist.tolist()
                tr_true_hist = train_true_hist.tolist()
                tr_total = max(1, int(sum(tr_hist)))
                tr_ratio = [round(float(v) / float(tr_total), 4) for v in tr_hist]
                tr_true_total = max(1, int(sum(tr_true_hist)))
                tr_true_ratio = [round(float(v) / float(tr_true_total), 4) for v in tr_true_hist]
                tr_acc = float(train_correct) / float(max(1, train_seen))
                tr_unknown_ratio = {
                    k: round(v, 4) for k, v in _unknown_ratio_dict(train_unknown_counter).items()
                }
                print(
                    f"[diag][train] ep={epoch + 1} acc={tr_acc:.4f}",
                    flush=True,
                )
                print(
                    f"[diag][train] ep={epoch + 1} pred_hist={tr_hist} pred_ratio={tr_ratio}",
                    flush=True,
                )
                print(
                    f"[diag][train] ep={epoch + 1} true_hist={tr_true_hist} true_ratio={tr_true_ratio}",
                    flush=True,
                )
                print(
                    f"[diag][train] ep={epoch + 1} meta_unknown_ratio={tr_unknown_ratio}",
                    flush=True,
                )
                self._print_moe_train_summary(epoch + 1, moe_train_agg)
                self._print_attnres_diagnostics(epoch + 1)
                if self.prototype_bank is not None:
                    pdiag = self.prototype_bank.diagnostics()
                    print(f"[prototype][bank] ep={epoch + 1} {pdiag}", flush=True)

                lr_cur = self.optimizer.param_groups[0]["lr"]

                print(f"starting validation for epoch {epoch + 1}", flush=True)
                _mem_report(f"starting_val_epoch_{epoch + 1}", md)

                try:
                    with torch.no_grad():
                        acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(
                            self.model, epoch_for_log=epoch + 1
                        )
                    _mem_report(f"after_val ep={epoch + 1}", md)
                    print(
                        "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                            epoch + 1,
                            np.mean(losses) if losses else float("nan"),
                            acc,
                            kappa,
                            f1,
                            lr_cur,
                            (timer() - start_time) / 60
                        )
                    )
                    if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "encoder"):
                        gate_vals = []
                        st = self.params.attnres_start_layer
                        for i, layer in enumerate(self.model.backbone.encoder.layers):
                            if i < st:
                                continue
                            parts = []
                            if hasattr(layer, "pre_attn_gate"):
                                parts.append(f"attn={torch.sigmoid(layer.pre_attn_gate).item():.4f}")
                            if hasattr(layer, "pre_mlp_gate"):
                                parts.append(f"mlp={torch.sigmoid(layer.pre_mlp_gate).item():.4f}")
                            gs = getattr(layer, "last_gate_stats", None)
                            if isinstance(gs, dict):
                                if "pre_attn_std" in gs:
                                    parts.append(
                                        "attn_batch_std="
                                        f"{float(gs.get('pre_attn_std', 0.0)):.4f}"
                                    )
                                if "pre_mlp_std" in gs:
                                    parts.append(
                                        "mlp_batch_std="
                                        f"{float(gs.get('pre_mlp_std', 0.0)):.4f}"
                                    )
                            if parts:
                                gate_vals.append(f"L{i}:" + ",".join(parts))
                        if len(gate_vals) > 0:
                            print("[Gate values] " + " | ".join(gate_vals))
                    if self.prototype_bank is not None:
                        val_diag = getattr(self.val_eval, "last_multiclass_diag", None)
                        sid = None
                        if isinstance(val_diag, dict):
                            sid_raw = val_diag.get("subject_ids")
                            if torch.is_tensor(sid_raw):
                                sid = sid_raw
                        pdiag = self.prototype_bank.diagnostics(subject_ids=sid)
                        print(f"[prototype][val] ep={epoch + 1} {pdiag}", flush=True)
                    print(cm)
                    val_hist = np.asarray(cm).sum(axis=0).astype(int).tolist()
                    val_true_hist = np.asarray(cm).sum(axis=1).astype(int).tolist()
                    val_total = max(1, int(sum(val_hist)))
                    val_ratio = [round(float(v) / float(val_total), 4) for v in val_hist]
                    val_true_total = max(1, int(sum(val_true_hist)))
                    val_true_ratio = [round(float(v) / float(val_true_total), 4) for v in val_true_hist]
                    print(
                        f"[diag][val] ep={epoch + 1} pred_hist={val_hist} pred_ratio={val_ratio}",
                        flush=True,
                    )
                    print(
                        f"[diag][val] ep={epoch + 1} true_hist={val_true_hist} true_ratio={val_true_ratio}",
                        flush=True,
                    )
                    val_diag = getattr(self.val_eval, "last_multiclass_diag", None)
                    if isinstance(val_diag, dict) and "meta_unknown_ratio" in val_diag:
                        vu = {k: round(float(v), 4) for k, v in val_diag["meta_unknown_ratio"].items()}
                        print(f"[diag][val] ep={epoch + 1} meta_unknown_ratio={vu}", flush=True)
                    self._print_moe_snapshot('train-first', epoch + 1, train_first_snap)
                    print("starting MoE diagnostics", flush=True)
                    self._log_moe_diagnostics(epoch_one_based=epoch + 1)
                    if kappa > kappa_best:
                        print("kappa increasing....saving weights !! ")
                        print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                            acc,
                            kappa,
                            f1,
                        ))
                        best_f1_epoch = epoch + 1
                        acc_best = acc
                        kappa_best = kappa
                        f1_best = f1
                        cm_best = cm
                        self.best_model_states = _state_dict_to_cpu(self.model)
                        est_b = _estimate_state_dict_cpu_bytes(self.best_model_states)
                        print(
                            f"[checkpoint] best val kappa improved -> CPU state_dict ~{est_b / (1024 ** 2):.1f} MiB",
                            flush=True,
                        )
                        _mem_report(f"after_best_snapshot ep={epoch + 1}", md)
                except Exception as e:
                    self._on_val_epoch_exception(md, epoch + 1, e)
                    raise

                print(f"epoch {epoch + 1} fully complete", flush=True)
                self._epoch_end_gc()

            if self.best_model_states is None:
                print('Warning: val kappa never improved; using last epoch weights for test/save.')
                self.best_model_states = _state_dict_to_cpu(self.model)

            _mem_report("train_multiclass_done_pre_test", md)

            self.model.load_state_dict(self.best_model_states)
            self.model.cuda()
            if getattr(self.params, 'moe', False):
                set_moe_train_epoch(self.params.epochs + 1)
            with torch.no_grad():
                print("***************************Test************************")
                acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
                print("***************************Test results************************")
                print(
                    "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ),
                    flush=True,
                )
                print(cm, flush=True)
                test_pred_hist = np.asarray(cm).sum(axis=0).astype(int).tolist()
                test_true_hist = np.asarray(cm).sum(axis=1).astype(int).tolist()
                print(f"[diag][test] pred_hist={test_pred_hist}", flush=True)
                print(f"[diag][test] true_hist={test_true_hist}", flush=True)
                print("[post_test] after test confusion matrix", flush=True)

                rd = getattr(self.params, "routing_export_dir", None) or ""
                model_path = ""
                try:
                    print("[post_test] before checkpoint save", flush=True)
                    if not os.path.isdir(self.params.model_dir):
                        os.makedirs(self.params.model_dir)
                    model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(
                        best_f1_epoch, acc, kappa, f1
                    )
                    ck_tag = os.path.basename(model_path).replace(".pth", "")
                    epoch_tag = f"best_ep{best_f1_epoch}"
                    raw_splits = getattr(self.params, "routing_export_splits", "test") or "test"
                    split_list = [s.strip() for s in raw_splits.split(",") if s.strip()]
                    print(
                        "[post_test] expected routing per-sample pattern: "
                        f"faced_routing_<split>_e{epoch_tag}_<checkpoint_tag>_per_sample.csv "
                        f"(checkpoint_tag={ck_tag!r})",
                        flush=True,
                    )
                    for sp in split_list:
                        print(
                            f"[post_test]   example: faced_routing_{sp}_e{epoch_tag}_{ck_tag}_per_sample.csv",
                            flush=True,
                        )

                    torch.save(self.model.state_dict(), model_path)
                    print("[post_test] after checkpoint save", flush=True)
                    exists = os.path.isfile(model_path)
                    sz = os.path.getsize(model_path) if exists else -1
                    print(
                        f"[post_test] checkpoint exists on disk: {exists} path={model_path!r} size_bytes={sz}",
                        flush=True,
                    )
                    print("model save in " + model_path, flush=True)

                    print(
                        f"[post_test] before routing export routing_export_dir={rd!r} downstream={self.params.downstream_dataset!r}",
                        flush=True,
                    )
                    if self.params.downstream_dataset == "FACED" and rd:
                        from utils.faced_routing_export import export_facced_routing_split

                        for sp in split_list:
                            if sp not in self.data_loader:
                                print(f"[routing_export] skip unknown split {sp!r}", flush=True)
                                continue
                            export_facced_routing_split(
                                self.model,
                                self.data_loader[sp],
                                self.params,
                                sp,
                                epoch_tag,
                                ck_tag,
                            )
                    print("[post_test] after routing export", flush=True)
                except Exception:
                    print("[post_test] EXCEPTION in checkpoint save / routing export block", flush=True)
                    traceback.print_exc()
                    print(f"[post_test] model_path={model_path!r}", flush=True)
                    print(f"[post_test] routing_export_dir={rd!r}", flush=True)
                    raise
        except Exception as e:
            _cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
            if _cuda_oom is not None and isinstance(e, _cuda_oom):
                _mem_report("cuda_OOM_exception", md)
                print(f"[train] CUDA OOM: {e!r}", flush=True)
                traceback.print_exc()
            elif isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                _mem_report("runtime_OOM_string", md)
                traceback.print_exc()
            raise

    def train_for_binaryclass(self):
        md = self._model_dir()
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        best_f1_epoch = 0
        train_steps = 0

        self._print_slurm_oom_hints()

        try:
            for epoch in range(self.params.epochs):
                if getattr(self.params, 'moe', False):
                    set_moe_train_epoch(epoch + 1)
                self._apply_router_warmup_state(epoch + 1)
                _mem_report(f"epoch_start_binary ep={epoch + 1}/{self.params.epochs}", md)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                self.model.train()
                start_time = timer()
                losses = []
                moe_train_agg = None
                if getattr(self.params, 'moe', False):
                    moe_train_agg = {
                        'max_frac_vals': [],
                        'reroute_vals': [],
                        'margin_vals': [],
                        'route_strategies': {},
                    }
                train_first_snap = None
                for batch_idx, batch in enumerate(tqdm_auto(self.data_loader['train'], self.params, mininterval=10)):
                    x, y = batch[0], batch[1]
                    self.optimizer.zero_grad(set_to_none=True)
                    x = x.cuda()
                    y = y.cuda()
                    batch_meta = _move_meta_to_cuda(_extract_batch_meta(batch))
                    pred = _forward_with_optional_meta(self.model, x, batch_meta)
                    self._accumulate_moe_train_diag(moe_train_agg)
                    if train_first_snap is None and getattr(self.params, 'moe', False):
                        train_first_snap = self._collect_moe_snapshot()

                    loss = self.criterion(pred, y)
                    loss = self._add_moe_auxiliary_loss(loss)
                    continual_penalty, continual_stats = self._continual_replay_penalty("binary")
                    loss = loss + continual_penalty
                    if not torch.isfinite(loss).all():
                        print(f"[train] non-finite loss ep={epoch + 1} batch={batch_idx}")
                        raise RuntimeError("non-finite training loss")

                    loss.backward()
                    self._scale_router_grads_if_needed(epoch + 1)
                    losses.append(float(loss.item()))
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    self.optimizer_scheduler.step()
                    self._continual_add_current_batch(x, y, batch_meta, pred)
                    train_steps += 1
                    if train_steps % 50 == 0:
                        _mem_report(f"train_step_binary ep={epoch + 1} batch={batch_idx}", md)
                    if train_steps % 100 == 0 and continual_stats:
                        print(f"[continual] step={train_steps} {continual_stats}", flush=True)

                print(f"finished train loop for epoch {epoch + 1}", flush=True)
                _mem_report(f"finished_train_epoch_{epoch + 1}_binary", md)
                self._print_moe_train_summary(epoch + 1, moe_train_agg)

                lr_cur = self.optimizer.param_groups[0]["lr"]

                print(f"starting validation for epoch {epoch + 1}", flush=True)
                _mem_report(f"starting_val_epoch_{epoch + 1}_binary", md)

                try:
                    with torch.no_grad():
                        acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(
                            self.model, epoch_for_log=epoch + 1
                        )
                    _mem_report(f"after_val_binary ep={epoch + 1}", md)
                    print(
                        "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                            epoch + 1,
                            np.mean(losses) if losses else float("nan"),
                            acc,
                            pr_auc,
                            roc_auc,
                            lr_cur,
                            (timer() - start_time) / 60
                        )
                    )
                    print(cm)
                    self._print_moe_snapshot('train-first', epoch + 1, train_first_snap)
                    print("starting MoE diagnostics", flush=True)
                    self._log_moe_diagnostics(epoch_one_based=epoch + 1)
                    if roc_auc > roc_auc_best:
                        print("roc_auc increasing....saving weights !! ")
                        print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                            acc,
                            pr_auc,
                            roc_auc,
                        ))
                        best_f1_epoch = epoch + 1
                        acc_best = acc
                        pr_auc_best = pr_auc
                        roc_auc_best = roc_auc
                        cm_best = cm
                        self.best_model_states = _state_dict_to_cpu(self.model)
                        est_b = _estimate_state_dict_cpu_bytes(self.best_model_states)
                        print(
                            f"[checkpoint] best val roc_auc improved -> CPU state_dict ~{est_b / (1024 ** 2):.1f} MiB",
                            flush=True,
                        )
                        _mem_report(f"after_best_snapshot_binary ep={epoch + 1}", md)
                except Exception as e:
                    self._on_val_epoch_exception(md, epoch + 1, e)
                    raise

                print(f"epoch {epoch + 1} fully complete", flush=True)
                self._epoch_end_gc()

            if self.best_model_states is None:
                print('Warning: val roc_auc never improved; using last epoch weights.')
                self.best_model_states = _state_dict_to_cpu(self.model)
            _mem_report("train_binary_done_pre_test", md)

            self.model.load_state_dict(self.best_model_states)
            if getattr(self.params, 'moe', False):
                set_moe_train_epoch(self.params.epochs + 1)
            with torch.no_grad():
                print("***************************Test************************")
                acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
                print("***************************Test results************************")
                print(
                    "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    )
                )
                print(cm)
                if not os.path.isdir(self.params.model_dir):
                    os.makedirs(self.params.model_dir)
                model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
        except Exception as e:
            _cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
            if _cuda_oom is not None and isinstance(e, _cuda_oom):
                _mem_report("cuda_OOM_exception_binary", md)
                traceback.print_exc()
            elif isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                _mem_report("runtime_OOM_binary", md)
                traceback.print_exc()
            raise

    def train_for_regression(self):
        md = self._model_dir()
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        best_r2_epoch = 0
        train_steps = 0

        self._print_slurm_oom_hints()

        try:
            for epoch in range(self.params.epochs):
                if getattr(self.params, 'moe', False):
                    set_moe_train_epoch(epoch + 1)
                self._apply_router_warmup_state(epoch + 1)
                _mem_report(f"epoch_start_regr ep={epoch + 1}/{self.params.epochs}", md)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                self.model.train()
                start_time = timer()
                losses = []
                moe_train_agg = None
                if getattr(self.params, 'moe', False):
                    moe_train_agg = {
                        'max_frac_vals': [],
                        'reroute_vals': [],
                        'margin_vals': [],
                        'route_strategies': {},
                    }
                train_first_snap = None
                for batch_idx, batch in enumerate(tqdm_auto(self.data_loader['train'], self.params, mininterval=10)):
                    x, y = batch[0], batch[1]
                    self.optimizer.zero_grad(set_to_none=True)
                    x = x.cuda()
                    y = y.cuda()
                    batch_meta = _move_meta_to_cuda(_extract_batch_meta(batch))
                    pred = _forward_with_optional_meta(self.model, x, batch_meta)
                    self._accumulate_moe_train_diag(moe_train_agg)
                    if train_first_snap is None and getattr(self.params, 'moe', False):
                        train_first_snap = self._collect_moe_snapshot()
                    loss = self.criterion(pred, y)
                    loss = self._add_moe_auxiliary_loss(loss)
                    continual_penalty, continual_stats = self._continual_replay_penalty("regression")
                    loss = loss + continual_penalty
                    if not torch.isfinite(loss).all():
                        print(f"[train] non-finite loss ep={epoch + 1} batch={batch_idx}")
                        raise RuntimeError("non-finite training loss")

                    loss.backward()
                    self._scale_router_grads_if_needed(epoch + 1)
                    losses.append(float(loss.item()))
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    self.optimizer_scheduler.step()
                    self._continual_add_current_batch(x, y, batch_meta, pred)
                    train_steps += 1
                    if train_steps % 50 == 0:
                        _mem_report(f"train_step_regr ep={epoch + 1} batch={batch_idx}", md)
                    if train_steps % 100 == 0 and continual_stats:
                        print(f"[continual] step={train_steps} {continual_stats}", flush=True)

                print(f"finished train loop for epoch {epoch + 1}", flush=True)
                _mem_report(f"finished_train_epoch_{epoch + 1}_regr", md)
                self._print_moe_train_summary(epoch + 1, moe_train_agg)

                lr_cur = self.optimizer.param_groups[0]["lr"]

                print(f"starting validation for epoch {epoch + 1}", flush=True)
                _mem_report(f"starting_val_epoch_{epoch + 1}_regr", md)

                try:
                    with torch.no_grad():
                        corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(
                            self.model, epoch_for_log=epoch + 1
                        )
                    _mem_report(f"after_val_regr ep={epoch + 1}", md)
                    print(
                        "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                            epoch + 1,
                            np.mean(losses) if losses else float("nan"),
                            corrcoef,
                            r2,
                            rmse,
                            lr_cur,
                            (timer() - start_time) / 60
                        )
                    )
                    self._print_moe_snapshot('train-first', epoch + 1, train_first_snap)
                    print("starting MoE diagnostics", flush=True)
                    self._log_moe_diagnostics(epoch_one_based=epoch + 1)
                    if r2 > r2_best:
                        print("r2 increasing....saving weights !! ")
                        print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                            corrcoef,
                            r2,
                            rmse,
                        ))
                        best_r2_epoch = epoch + 1
                        corrcoef_best = corrcoef
                        r2_best = r2
                        rmse_best = rmse
                        self.best_model_states = _state_dict_to_cpu(self.model)
                        est_b = _estimate_state_dict_cpu_bytes(self.best_model_states)
                        print(
                            f"[checkpoint] best val r2 improved -> CPU state_dict ~{est_b / (1024 ** 2):.1f} MiB",
                            flush=True,
                        )
                        _mem_report(f"after_best_snapshot_regr ep={epoch + 1}", md)
                except Exception as e:
                    self._on_val_epoch_exception(md, epoch + 1, e)
                    raise

                print(f"epoch {epoch + 1} fully complete", flush=True)
                self._epoch_end_gc()

            if self.best_model_states is None:
                print('Warning: val r2 never improved; using last epoch weights.')
                self.best_model_states = _state_dict_to_cpu(self.model)
            _mem_report("train_regression_done_pre_test", md)

            self.model.load_state_dict(self.best_model_states)
            if getattr(self.params, 'moe', False):
                set_moe_train_epoch(self.params.epochs + 1)
            with torch.no_grad():
                print("***************************Test************************")
                corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
                print("***************************Test results************************")
                print(
                    "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    )
                )

                if not os.path.isdir(self.params.model_dir):
                    os.makedirs(self.params.model_dir)
                model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
        except Exception as e:
            _cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
            if _cuda_oom is not None and isinstance(e, _cuda_oom):
                _mem_report("cuda_OOM_exception_regr", md)
                traceback.print_exc()
            elif isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                _mem_report("runtime_OOM_regr", md)
                traceback.print_exc()
            raise
