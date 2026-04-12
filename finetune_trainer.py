import gc
import json
import os
import re
import shutil
import traceback
from datetime import datetime
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from finetune_evaluator import Evaluator
from utils.tqdm_auto import tqdm_auto
from models.moe import (
    format_moe_diagnostics_lines,
    get_moe_train_epoch,
    reset_moe_diagnostic_labels,
    set_moe_diagnostic_labels,
    set_moe_train_epoch,
)

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


def _safe_tag(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_") or "run"


def _to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        else:
            raise ValueError(
                f"Unsupported downstream_dataset={self.params.downstream_dataset}. "
                "This refactored branch supports FACED and SEED-V only."
            )

        self.best_model_states = None

        self._named_trainable_params: List = []
        grouped = {
            'backbone': [],
            'router': [],
            'experts': [],
            'classifier': [],
            'other': [],
        }
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            if not param.requires_grad:
                continue
            self._named_trainable_params.append((name, param))
            grouped[self._component_name_for_param(name)].append(param)

        self.data_length = len(self.data_loader['train'])
        if getattr(self.params, 'use_component_lr', False) and getattr(self.params, 'multi_lr', False):
            print('[opt] use_component_lr=True overrides multi_lr=True.', flush=True)

        if self.params.optimizer == 'AdamW':
            if getattr(self.params, 'use_component_lr', False):
                self.optimizer = self._build_component_optimizer(grouped, kind='adamw')
            elif self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': grouped['backbone'], 'lr': self.params.lr},
                    {'params': grouped['other'] + grouped['router'] + grouped['experts'] + grouped['classifier'],
                     'lr': 0.001 * (self.params.batch_size / 256) ** 0.5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.params.lr,
                    weight_decay=self.params.weight_decay)
        else:
            if getattr(self.params, 'use_component_lr', False):
                self.optimizer = self._build_component_optimizer(grouped, kind='sgd')
            elif self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': grouped['backbone'], 'lr': self.params.lr},
                    {'params': grouped['other'] + grouped['router'] + grouped['experts'] + grouped['classifier'],
                     'lr': self.params.lr * 5}
                ], momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.params.lr, momentum=0.9,
                    weight_decay=self.params.weight_decay)

        # Original CBraMod finetune: cosine over full run, per optimizer step
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )

        if getattr(self.params, 'use_component_lr', False):
            for i, g in enumerate(self.optimizer.param_groups):
                print(
                    f"[opt] group={i} name={g.get('name', 'unnamed')} lr={g.get('lr', 0.0):.6g} "
                    f"num_params={len(g.get('params', []))}",
                    flush=True,
                )

        print(self.model)

    @staticmethod
    def _component_name_for_param(name: str) -> str:
        if name.startswith('classifier') or '.classifier.' in name:
            return 'classifier'
        if 'backbone' not in name:
            return 'other'
        if 'moe_ffn.' not in name:
            return 'backbone'
        router_keys = (
            'spatial_router',
            'spectral_router',
            'router_input_norm',
            'domain_embeddings',
            'domain_bias_mlp',
            'adapter_cond_',
            'subject_summary_proj',
            'eeg_summary_proj',
            'depth_summary_proj',
        )
        expert_keys = (
            'shared.',
            'spatial_specialists',
            'spectral_specialists',
        )
        if any(k in name for k in router_keys):
            return 'router'
        if any(k in name for k in expert_keys):
            return 'experts'
        return 'backbone'

    def _build_component_optimizer(self, grouped: Dict[str, List[torch.nn.Parameter]], kind: str):
        base_lr = float(self.params.lr)
        groups = []

        def _add(name: str, params: List[torch.nn.Parameter], mult: float):
            if not params:
                return
            groups.append({
                'params': params,
                'lr': base_lr * float(mult),
                'name': name,
            })

        _add('backbone', grouped['backbone'], getattr(self.params, 'lr_backbone_mult', 1.0))
        _add('router', grouped['router'], getattr(self.params, 'lr_router_mult', 1.0))
        _add('experts', grouped['experts'], getattr(self.params, 'lr_expert_mult', 1.0))
        _add('classifier', grouped['classifier'], getattr(self.params, 'lr_classifier_mult', 1.0))
        _add('other', grouped['other'], getattr(self.params, 'lr_other_mult', 1.0))

        if not groups:
            raise RuntimeError('No trainable parameters found for component-wise optimizer.')
        if kind == 'adamw':
            return torch.optim.AdamW(groups, weight_decay=self.params.weight_decay)
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=self.params.weight_decay)

    def _current_lr_by_group(self) -> Dict[str, float]:
        out = {}
        for i, g in enumerate(self.optimizer.param_groups):
            name = str(g.get('name', f'group_{i}'))
            out[name] = float(g.get('lr', 0.0))
        return out

    def _add_moe_auxiliary_loss(self, loss):
        if not getattr(self.params, 'moe', False):
            return loss
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'moe_auxiliary_loss'):
            return loss
        aux = bb.moe_auxiliary_loss()
        return loss + aux

    def _log_moe_diagnostics(self):
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
        batch_meta = _move_meta_to_cuda(batch[3]) if len(batch) >= 4 and isinstance(batch[3], dict) else None
        label_tok = None
        if len(batch) > 1:
            label_tok = set_moe_diagnostic_labels(batch[1].cuda())
        try:
            with torch.no_grad():
                _ = _forward_with_optional_meta(self.model, x, batch_meta)
        finally:
            if label_tok is not None:
                reset_moe_diagnostic_labels(label_tok)
        print(
            '[MoE diagnostics] one val batch, eval (no router noise)  '
            f"route_mode={getattr(self.params, 'moe_route_mode', '?')}  "
            f"psd_feats={getattr(self.params, 'moe_use_psd_router_features', False)}  "
            f"domain_bias={getattr(self.params, 'moe_domain_bias', False)}"
        )
        for i, layer in enumerate(bb.encoder.layers):
            m = getattr(layer, 'moe_ffn', None)
            diag = getattr(m, 'last_diagnostics', None) if m is not None else None
            if diag is None:
                continue
            for line in format_moe_diagnostics_lines(i, diag):
                print(line)
        if was_training:
            self.model.train()

    def _collect_grad_norms(self) -> Dict[str, float]:
        accum = {
            'backbone': 0.0,
            'router': 0.0,
            'experts': 0.0,
            'classifier': 0.0,
            'other': 0.0,
            'depth_summary_path': 0.0,
            'depth_router_proj_spatial': 0.0,
            'depth_router_proj_spectral': 0.0,
        }
        for name, p in self._named_trainable_params:
            if p.grad is None:
                continue
            g2 = float(p.grad.detach().pow(2).sum().item())
            accum[self._component_name_for_param(name)] += g2
            if (
                ('pre_attn_res' in name)
                or ('pre_mlp_res' in name)
                or ('depth_summary_proj' in name)
                or ('attnres_depth_router_proj_spatial' in name)
                or ('attnres_depth_router_proj_spectral' in name)
            ):
                accum['depth_summary_path'] += g2
            if 'attnres_depth_router_proj_spatial' in name:
                accum['depth_router_proj_spatial'] += g2
            if 'attnres_depth_router_proj_spectral' in name:
                accum['depth_router_proj_spectral'] += g2
        return {k: float(v ** 0.5) for k, v in accum.items()}

    @staticmethod
    def _classwise_recall_from_cm(cm: np.ndarray) -> List[float]:
        if cm is None:
            return []
        row_sum = cm.sum(axis=1)
        out = []
        for i in range(cm.shape[0]):
            denom = float(row_sum[i])
            out.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
        return out

    def _collect_layer_moe_diagnostics(self) -> List[Dict[str, Any]]:
        out = []
        if not getattr(self.params, 'moe', False):
            return out
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'encoder'):
            return out
        for i, layer in enumerate(bb.encoder.layers):
            m = getattr(layer, 'moe_ffn', None)
            diag = getattr(m, 'last_diagnostics', None) if m is not None else None
            if diag is None:
                continue
            out.append({'layer': int(i), 'diag': _to_jsonable(diag)})
        return out

    def _warn_depth_summary_flow(self, epoch_one_based: int, grad_norms: Dict[str, float]) -> None:
        if not getattr(self.params, 'moe', False):
            return
        if not getattr(self.params, 'moe_use_attnres_depth_router_features', False):
            return
        grad_mode = str(getattr(self.params, 'moe_attnres_depth_summary_grad_mode', 'detached'))
        if grad_mode != 'delayed_unfreeze':
            return
        unfreeze_epoch = int(getattr(self.params, 'moe_attnres_depth_summary_unfreeze_epoch', 1))
        moe_epoch = int(get_moe_train_epoch())
        print(
            f"[diag][depth_unfreeze] epoch={epoch_one_based} moe_train_epoch={moe_epoch} "
            f"unfreeze_epoch={unfreeze_epoch} grad_mode={grad_mode} "
            f"depth_summary_path_grad={float(grad_norms.get('depth_summary_path', 0.0)):.6g} "
            f"router_proj_spatial_grad={float(grad_norms.get('depth_router_proj_spatial', 0.0)):.6g} "
            f"router_proj_spectral_grad={float(grad_norms.get('depth_router_proj_spectral', 0.0)):.6g}",
            flush=True,
        )
        if epoch_one_based < unfreeze_epoch:
            return

        layers = self._collect_layer_moe_diagnostics()
        detached_layers = []
        inactive_layers = []
        for entry in layers:
            d = entry.get('diag', {}) or {}
            if bool(d.get('attnres_depth_summary_detached', False)):
                detached_layers.append(int(entry['layer']))
            if not bool(d.get('attnres_depth_summary_grad_active', False)):
                inactive_layers.append(int(entry['layer']))

        if detached_layers:
            print(
                f"[warn][depth_unfreeze] epoch={epoch_one_based} detached depth-summary still present in layers={detached_layers}",
                flush=True,
            )
        if inactive_layers:
            print(
                f"[warn][depth_unfreeze] epoch={epoch_one_based} grad_active=False for depth-summary in layers={inactive_layers}",
                flush=True,
            )
        if float(grad_norms.get('depth_summary_path', 0.0)) <= 0.0:
            print(
                f"[warn][depth_unfreeze] epoch={epoch_one_based} depth-summary path grad norm is zero after unfreeze",
                flush=True,
            )

    def _append_machine_readable_epoch_diag(
        self,
        epoch_one_based: int,
        split: str,
        metrics: Dict[str, float],
        grad_norms: Dict[str, float],
        cm: Optional[np.ndarray] = None,
    ) -> None:
        md = self._model_dir()
        os.makedirs(md, exist_ok=True)
        path = os.path.join(md, 'epoch_diagnostics.jsonl')
        payload = {
            'epoch': int(epoch_one_based),
            'split': str(split),
            'timestamp_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'metrics': _to_jsonable(metrics),
            'grad_norms': _to_jsonable(grad_norms),
            'moe_layers': self._collect_layer_moe_diagnostics(),
        }
        if cm is not None:
            payload['confusion_matrix'] = np.asarray(cm).tolist()
            payload['classwise_recall'] = self._classwise_recall_from_cm(np.asarray(cm))
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=True) + '\n')

      @staticmethod
      def _append_json_record(path: str, payload: Dict[str, Any]) -> None:
          rows: List[Dict[str, Any]] = []
          if os.path.isfile(path):
              try:
                  with open(path, 'r', encoding='utf-8') as f:
                      loaded = json.load(f)
                  if isinstance(loaded, list):
                      rows = loaded
              except Exception:
                  rows = []
          rows.append(payload)
          with open(path, 'w', encoding='utf-8') as f:
              json.dump(rows, f, indent=2, ensure_ascii=True)

      def _export_depth_context_diagnostics(self, epoch_one_based: int, split: str) -> None:
          layers = self._collect_layer_moe_diagnostics()
          if not layers:
              return
          md = self._model_dir()
          os.makedirs(md, exist_ok=True)
          block_path = os.path.join(md, 'block_summary_stats.json')
          router_path = os.path.join(md, 'router_context_stats.json')
          routing_path = os.path.join(md, 'routing_diagnostics.json')

          for entry in layers:
              layer = int(entry.get('layer', -1))
              d = entry.get('diag', {}) or {}
              spatial_diag = d.get('spatial', {}) or {}
              spectral_diag = d.get('spectral', {}) or {}

              block_payload = {
                  'epoch': int(epoch_one_based),
                  'split': str(split),
                  'layer': layer,
                  'depth_context_mode': d.get('attnres_depth_context_mode', 'compact_shared'),
                  'block_count': int(d.get('attnres_depth_block_count', 0) or 0),
                  'block_attn_mass_mean': d.get('attnres_depth_block_attn_mass_mean'),
                  'block_mlp_mass_mean': d.get('attnres_depth_block_mlp_mass_mean'),
                  'block_delta_mass_mean': d.get('attnres_depth_block_delta_mass_mean'),
                  'block_summary_norms': d.get('attnres_depth_block_summary_norms'),
              }
              self._append_json_record(block_path, _to_jsonable(block_payload))

              router_payload = {
                  'epoch': int(epoch_one_based),
                  'split': str(split),
                  'layer': layer,
                  'shared_context_norm': d.get('attnres_depth_shared_context_norm'),
                  'spatial_projected_context_norm': d.get('attnres_depth_proj_spatial_norm'),
                  'spectral_projected_context_norm': d.get('attnres_depth_proj_spectral_norm'),
                  'spatial_spectral_proj_cosine': d.get('attnres_depth_proj_cosine'),
                  'spatial_spectral_proj_l2': d.get('attnres_depth_proj_l2'),
              }
              self._append_json_record(router_path, _to_jsonable(router_payload))

              sp_hist = spatial_diag.get('assigned_count_per_expert') or []
              sc_hist = spectral_diag.get('assigned_count_per_expert') or []
              routing_payload = {
                  'epoch': int(epoch_one_based),
                  'split': str(split),
                  'layer': layer,
                  'spatial_assigned_count_per_expert': sp_hist,
                  'spectral_assigned_count_per_expert': sc_hist,
                  'spatial_collapsed_experts': int(sum(1 for v in sp_hist if int(v) == 0)),
                  'spectral_collapsed_experts': int(sum(1 for v in sc_hist if int(v) == 0)),
                  'spatial_routing_entropy_pre_capacity': spatial_diag.get('routing_entropy_pre_capacity'),
                  'spectral_routing_entropy_pre_capacity': spectral_diag.get('routing_entropy_pre_capacity'),
                  'spatial_routing_entropy_post_assignment': spatial_diag.get('routing_entropy_post_assignment'),
                  'spectral_routing_entropy_post_assignment': spectral_diag.get('routing_entropy_post_assignment'),
                  'spatial_pre_top1_histogram': spatial_diag.get('pre_top1_histogram'),
                  'spectral_pre_top1_histogram': spectral_diag.get('pre_top1_histogram'),
                  'spatial_reroute_rate': spatial_diag.get('reroute_rate'),
                  'spectral_reroute_rate': spectral_diag.get('reroute_rate'),
                  'spatial_overflow_count': spatial_diag.get('overflow_count'),
                  'spectral_overflow_count': spectral_diag.get('overflow_count'),
              }
              self._append_json_record(routing_path, _to_jsonable(routing_payload))

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

    def _write_run_summary(
        self,
        task_type: str,
        best_epoch: int,
        best_val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
        model_path: str,
    ) -> None:
        md = self._model_dir()
        os.makedirs(md, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dataset = str(getattr(self.params, 'downstream_dataset', 'unknown'))
        dataset_tag = _safe_tag(dataset).lower()
        run_name = str(getattr(self.params, 'routing_run_name', '') or '')

        summary_payload = {
            'timestamp_utc': ts,
            'dataset': dataset,
            'task_type': task_type,
            'run_name': run_name,
            'best_epoch': int(best_epoch),
            'model_path': model_path,
            'best_val_metrics': _to_jsonable(best_val_metrics),
            'test_metrics': _to_jsonable(test_metrics),
            'config': _to_jsonable(vars(self.params)),
        }
        json_path = os.path.join(md, f"run_summary_{dataset_tag}_{ts}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=True, sort_keys=True)

        csv_path = os.path.join(md, 'experiment_summary.csv')
        row = {
            'timestamp_utc': ts,
            'dataset': dataset,
            'task_type': task_type,
            'run_name': run_name,
            'model_dir': md,
            'model_path': model_path,
            'best_epoch': int(best_epoch),
            'val_balanced_accuracy': best_val_metrics.get('balanced_accuracy', ''),
            'val_kappa': best_val_metrics.get('kappa', ''),
            'val_weighted_f1': best_val_metrics.get('weighted_f1', ''),
            'test_balanced_accuracy': test_metrics.get('balanced_accuracy', ''),
            'test_kappa': test_metrics.get('kappa', ''),
            'test_weighted_f1': test_metrics.get('weighted_f1', ''),
            'seed': getattr(self.params, 'seed', ''),
            'epochs': getattr(self.params, 'epochs', ''),
            'batch_size': getattr(self.params, 'batch_size', ''),
            'lr': getattr(self.params, 'lr', ''),
            'weight_decay': getattr(self.params, 'weight_decay', ''),
            'classifier': getattr(self.params, 'classifier', ''),
            'attnres_variant': getattr(self.params, 'attnres_variant', ''),
            'moe': bool(getattr(self.params, 'moe', False)),
            'moe_num_layers': getattr(self.params, 'moe_num_layers', ''),
            'moe_router_arch': getattr(self.params, 'moe_router_arch', ''),
            'moe_use_attnres_depth_router_features': bool(getattr(self.params, 'moe_use_attnres_depth_router_features', False)),
            'moe_attnres_depth_router_dim': getattr(self.params, 'moe_attnres_depth_router_dim', ''),
              'moe_attnres_depth_context_mode': getattr(self.params, 'moe_attnres_depth_context_mode', ''),
              'moe_attnres_depth_block_count': getattr(self.params, 'moe_attnres_depth_block_count', ''),
            'moe_attnres_depth_summary_mode': getattr(self.params, 'moe_attnres_depth_summary_mode', ''),
            'moe_attnres_depth_probe_mlp_for_router': bool(getattr(self.params, 'moe_attnres_depth_probe_mlp_for_router', False)),
            'moe_attnres_depth_summary_grad_mode': getattr(self.params, 'moe_attnres_depth_summary_grad_mode', ''),
            'moe_attnres_depth_summary_unfreeze_epoch': getattr(self.params, 'moe_attnres_depth_summary_unfreeze_epoch', ''),
            'moe_router_dispatch_mode': getattr(self.params, 'moe_router_dispatch_mode', ''),
            'moe_router_temperature': getattr(self.params, 'moe_router_temperature', ''),
            'moe_router_entropy_coef': getattr(self.params, 'moe_router_entropy_coef', ''),
            'moe_router_entropy_coef_spatial': getattr(self.params, 'moe_router_entropy_coef_spatial', ''),
            'moe_router_entropy_coef_spectral': getattr(self.params, 'moe_router_entropy_coef_spectral', ''),
            'moe_router_balance_kl_coef': getattr(self.params, 'moe_router_balance_kl_coef', ''),
            'moe_router_balance_kl_coef_spatial': getattr(self.params, 'moe_router_balance_kl_coef_spatial', ''),
            'moe_router_balance_kl_coef_spectral': getattr(self.params, 'moe_router_balance_kl_coef_spectral', ''),
            'moe_router_jitter_std': getattr(self.params, 'moe_router_jitter_std', ''),
            'moe_router_jitter_final_std': getattr(self.params, 'moe_router_jitter_final_std', ''),
            'moe_router_jitter_anneal_epochs': getattr(self.params, 'moe_router_jitter_anneal_epochs', ''),
            'moe_router_soft_warmup_epochs': getattr(self.params, 'moe_router_soft_warmup_epochs', ''),
            'moe_uniform_dispatch_warmup_epochs': getattr(self.params, 'moe_uniform_dispatch_warmup_epochs', ''),
            'moe_shared_blend_warmup_epochs': getattr(self.params, 'moe_shared_blend_warmup_epochs', ''),
            'moe_shared_blend_start': getattr(self.params, 'moe_shared_blend_start', ''),
            'moe_shared_blend_end': getattr(self.params, 'moe_shared_blend_end', ''),
            'moe_specialist_branch_mode': getattr(self.params, 'moe_specialist_branch_mode', ''),
            'moe_router_compact_feature_mode': getattr(self.params, 'moe_router_compact_feature_mode', ''),
            'moe_router_compact_feature_dim': getattr(self.params, 'moe_router_compact_feature_dim', ''),
            'moe_expert_init_noise_std': getattr(self.params, 'moe_expert_init_noise_std', ''),
            'moe_domain_bias': bool(getattr(self.params, 'moe_domain_bias', False)),
            'moe_use_psd_router_features': bool(getattr(self.params, 'moe_use_psd_router_features', False)),
            'use_component_lr': bool(getattr(self.params, 'use_component_lr', False)),
            'lr_backbone_mult': getattr(self.params, 'lr_backbone_mult', ''),
            'lr_router_mult': getattr(self.params, 'lr_router_mult', ''),
            'lr_expert_mult': getattr(self.params, 'lr_expert_mult', ''),
            'lr_classifier_mult': getattr(self.params, 'lr_classifier_mult', ''),
            'lr_other_mult': getattr(self.params, 'lr_other_mult', ''),
        }

        write_header = not os.path.isfile(csv_path)
        import csv
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

        print(f"[summary] wrote {json_path}", flush=True)
        print(f"[summary] appended {csv_path}", flush=True)

    def train_for_multiclass(self):
        """Same optimizer/schedule for baseline (--attnres_variant none) and AttnRes variants.

        AttnRes uses strict=False partial load in the model; new modules stay initialized here.
        No staged freeze, no separate pretrained/new LR groups.
        """
        md = self._model_dir()
        f1_best = float('-inf')
        kappa_best = float('-inf')
        acc_best = float('-inf')
        cm_best = None
        best_f1_epoch = 0
        best_val_metrics = {}
        train_steps = 0

        self._print_slurm_oom_hints()

        try:
            for epoch in range(self.params.epochs):
                if getattr(self.params, 'moe', False):
                    set_moe_train_epoch(epoch + 1)
                _mem_report(f"epoch_start ep={epoch + 1}/{self.params.epochs}", md)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                self.model.train()
                start_time = timer()
                losses = []
                for batch_idx, batch in enumerate(tqdm_auto(self.data_loader['train'], self.params, mininterval=10)):
                    try:
                        x, y = batch[0], batch[1]
                        self.optimizer.zero_grad(set_to_none=True)
                        x = x.cuda()
                        y = y.cuda()
                        batch_meta = _move_meta_to_cuda(batch[3]) if len(batch) >= 4 and isinstance(batch[3], dict) else None
                        pred = _forward_with_optional_meta(self.model, x, batch_meta)
                        if self.params.downstream_dataset == 'ISRUC':
                            loss = self.criterion(pred.transpose(1, 2), y)
                        else:
                            loss = self.criterion(pred, y)
                        loss = self._add_moe_auxiliary_loss(loss)

                        if not torch.isfinite(loss).all():
                            lv = float(loss.detach().item()) if loss.numel() == 1 else "non_scalar"
                            print(
                                f"[train] non-finite loss ep={epoch + 1} batch={batch_idx} loss={lv}"
                            )
                            raise RuntimeError("non-finite training loss")

                        loss.backward()
                        losses.append(float(loss.item()))
                        if self.params.clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                        self.optimizer.step()
                        self.optimizer_scheduler.step()
                        train_steps += 1
                        if train_steps % 50 == 0:
                            _mem_report(f"train_step ep={epoch + 1} batch={batch_idx} step={train_steps}", md)
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

                grad_norms = self._collect_grad_norms()
                lr_by_group = self._current_lr_by_group()
                lr_cur = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[diag] ep={epoch + 1} grad_norms={json.dumps(_to_jsonable(grad_norms), ensure_ascii=True)}",
                    flush=True,
                )
                if getattr(self.params, 'use_component_lr', False):
                    print(
                        f"[diag] ep={epoch + 1} lr_groups={json.dumps(_to_jsonable(lr_by_group), ensure_ascii=True)}",
                        flush=True,
                    )

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
                            if parts:
                                gate_vals.append(f"L{i}:" + ",".join(parts))
                        if len(gate_vals) > 0:
                            print("[Gate values] " + " | ".join(gate_vals))
                    print(cm)
                    classwise_recall = self._classwise_recall_from_cm(np.asarray(cm))
                    print(
                        f"[diag] ep={epoch + 1} classwise_recall={json.dumps(classwise_recall)}",
                        flush=True,
                    )
                    self._append_machine_readable_epoch_diag(
                        epoch_one_based=epoch + 1,
                        split='val',
                        metrics={
                            'balanced_accuracy': float(acc),
                            'kappa': float(kappa),
                            'weighted_f1': float(f1),
                            'lr': float(lr_cur),
                            'loss_mean': float(np.mean(losses) if losses else float('nan')),
                            'lr_groups': lr_by_group,
                        },
                        grad_norms=grad_norms,
                        cm=np.asarray(cm),
                    )
                      self._export_depth_context_diagnostics(epoch + 1, 'val')
                    self._warn_depth_summary_flow(epoch + 1, grad_norms)
                    print("starting MoE diagnostics", flush=True)
                    self._log_moe_diagnostics()
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
                        best_val_metrics = {
                            'balanced_accuracy': float(acc),
                            'kappa': float(kappa),
                            'weighted_f1': float(f1),
                        }
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
                    if self.params.downstream_dataset == "FACED" and rd:
                        print(
                            "[post_test] expected FACED routing per-sample pattern: "
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
                    elif rd:
                        print(
                            f"[routing_export] skip: downstream dataset {self.params.downstream_dataset!r} "
                            "has no routing export implementation yet.",
                            flush=True,
                        )
                    print("[post_test] after routing export", flush=True)

                    if not best_val_metrics:
                        best_val_metrics = {
                            'balanced_accuracy': float(acc_best),
                            'kappa': float(kappa_best),
                            'weighted_f1': float(f1_best),
                        }
                    self._write_run_summary(
                        task_type='multiclass',
                        best_epoch=best_f1_epoch,
                        best_val_metrics=best_val_metrics,
                        test_metrics={
                            'balanced_accuracy': float(acc),
                            'kappa': float(kappa),
                            'weighted_f1': float(f1),
                        },
                        model_path=model_path,
                    )
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
                _mem_report(f"epoch_start_binary ep={epoch + 1}/{self.params.epochs}", md)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                self.model.train()
                start_time = timer()
                losses = []
                for batch_idx, batch in enumerate(tqdm_auto(self.data_loader['train'], self.params, mininterval=10)):
                    x, y = batch[0], batch[1]
                    self.optimizer.zero_grad(set_to_none=True)
                    x = x.cuda()
                    y = y.cuda()
                    batch_meta = _move_meta_to_cuda(batch[3]) if len(batch) >= 4 and isinstance(batch[3], dict) else None
                    pred = _forward_with_optional_meta(self.model, x, batch_meta)

                    loss = self.criterion(pred, y)
                    loss = self._add_moe_auxiliary_loss(loss)
                    if not torch.isfinite(loss).all():
                        print(f"[train] non-finite loss ep={epoch + 1} batch={batch_idx}")
                        raise RuntimeError("non-finite training loss")

                    loss.backward()
                    losses.append(float(loss.item()))
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    self.optimizer_scheduler.step()
                    train_steps += 1
                    if train_steps % 50 == 0:
                        _mem_report(f"train_step_binary ep={epoch + 1} batch={batch_idx}", md)

                print(f"finished train loop for epoch {epoch + 1}", flush=True)
                _mem_report(f"finished_train_epoch_{epoch + 1}_binary", md)

                grad_norms = self._collect_grad_norms()
                lr_by_group = self._current_lr_by_group()
                lr_cur = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[diag] ep={epoch + 1} grad_norms={json.dumps(_to_jsonable(grad_norms), ensure_ascii=True)}",
                    flush=True,
                )

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
                    self._append_machine_readable_epoch_diag(
                        epoch_one_based=epoch + 1,
                        split='val',
                        metrics={
                            'balanced_accuracy': float(acc),
                            'pr_auc': float(pr_auc),
                            'roc_auc': float(roc_auc),
                            'lr': float(lr_cur),
                            'loss_mean': float(np.mean(losses) if losses else float('nan')),
                            'lr_groups': lr_by_group,
                        },
                        grad_norms=grad_norms,
                        cm=np.asarray(cm),
                    )
                      self._export_depth_context_diagnostics(epoch + 1, 'val')
                    print("starting MoE diagnostics", flush=True)
                    self._log_moe_diagnostics()
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
                _mem_report(f"epoch_start_regr ep={epoch + 1}/{self.params.epochs}", md)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                self.model.train()
                start_time = timer()
                losses = []
                for batch_idx, batch in enumerate(tqdm_auto(self.data_loader['train'], self.params, mininterval=10)):
                    x, y = batch[0], batch[1]
                    self.optimizer.zero_grad(set_to_none=True)
                    x = x.cuda()
                    y = y.cuda()
                    batch_meta = _move_meta_to_cuda(batch[3]) if len(batch) >= 4 and isinstance(batch[3], dict) else None
                    pred = _forward_with_optional_meta(self.model, x, batch_meta)
                    loss = self.criterion(pred, y)
                    loss = self._add_moe_auxiliary_loss(loss)
                    if not torch.isfinite(loss).all():
                        print(f"[train] non-finite loss ep={epoch + 1} batch={batch_idx}")
                        raise RuntimeError("non-finite training loss")

                    loss.backward()
                    losses.append(float(loss.item()))
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    self.optimizer.step()
                    self.optimizer_scheduler.step()
                    train_steps += 1
                    if train_steps % 50 == 0:
                        _mem_report(f"train_step_regr ep={epoch + 1} batch={batch_idx}", md)

                print(f"finished train loop for epoch {epoch + 1}", flush=True)
                _mem_report(f"finished_train_epoch_{epoch + 1}_regr", md)

                grad_norms = self._collect_grad_norms()
                lr_by_group = self._current_lr_by_group()
                lr_cur = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[diag] ep={epoch + 1} grad_norms={json.dumps(_to_jsonable(grad_norms), ensure_ascii=True)}",
                    flush=True,
                )

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
                    self._append_machine_readable_epoch_diag(
                        epoch_one_based=epoch + 1,
                        split='val',
                        metrics={
                            'corrcoef': float(corrcoef),
                            'r2': float(r2),
                            'rmse': float(rmse),
                            'lr': float(lr_cur),
                            'loss_mean': float(np.mean(losses) if losses else float('nan')),
                            'lr_groups': lr_by_group,
                        },
                        grad_norms=grad_norms,
                        cm=None,
                    )
                      self._export_depth_context_diagnostics(epoch + 1, 'val')
                    print("starting MoE diagnostics", flush=True)
                    self._log_moe_diagnostics()
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
