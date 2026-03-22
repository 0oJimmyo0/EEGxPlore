import os
from timeit import default_timer as timer
from typing import Any, Dict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator
from models.moe import (
    format_moe_diagnostics_lines,
    reset_moe_diagnostic_labels,
    set_moe_diagnostic_labels,
)


def _state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, Any]:
    """Checkpoint snapshot without deepcopy: avoids duplicating GPU weights (OOM on large models)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

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

        print(self.model)

    def _add_moe_load_balance_loss(self, loss):
        coef = float(getattr(self.params, 'moe_load_balance', 0.0) or 0.0)
        if coef <= 0 or not getattr(self.params, 'moe', False):
            return loss
        bb = getattr(self.model, 'backbone', None)
        if bb is None or not hasattr(bb, 'moe_auxiliary_loss'):
            return loss
        aux = bb.moe_auxiliary_loss()
        return loss + coef * aux

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
        label_tok = None
        if len(batch) > 1:
            label_tok = set_moe_diagnostic_labels(batch[1].cuda())
        try:
            with torch.no_grad():
                _ = self.model(x)
        finally:
            if label_tok is not None:
                reset_moe_diagnostic_labels(label_tok)
        print(
            '[MoE diagnostics] one val batch, eval (no router noise)  '
            f"router_mode={getattr(self.params, 'moe_router_mode', '?')}  "
            f"router_arch={getattr(self.params, 'moe_router_arch', '?')}  "
            f"psd_feats={getattr(self.params, 'moe_use_psd_router_features', False)}  "
            f"moe_expert_type={getattr(self.params, 'moe_expert_type', 'generic')}"
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

    def train_for_multiclass(self):
        """Same optimizer/schedule for baseline (--attnres_variant none) and AttnRes variants.

        AttnRes uses strict=False partial load in the model; new modules stay initialized here.
        No staged freeze, no separate pretrained/new LR groups.
        """
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_f1_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)
                loss = self._add_moe_load_balance_loss(loss)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
            print(
                "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                    epoch + 1,
                    np.mean(losses),
                    acc,
                    kappa,
                    f1,
                    optim_state['param_groups'][0]['lr'],
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
            self._log_moe_diagnostics()
            print(cm)
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

        if self.best_model_states is None:
            print('Warning: val kappa never improved; using last epoch weights for test/save.')
            self.best_model_states = _state_dict_to_cpu(self.model)

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
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(
                best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        best_f1_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)
                loss = self._add_moe_load_balance_loss(loss)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
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
        if self.best_model_states is None:
            print('Warning: val roc_auc never improved; using last epoch weights.')
            self.best_model_states = _state_dict_to_cpu(self.model)
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

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        best_r2_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss = self._add_moe_load_balance_loss(loss)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
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

        if self.best_model_states is None:
            print('Warning: val r2 never improved; using last epoch weights.')
            self.best_model_states = _state_dict_to_cpu(self.model)
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
