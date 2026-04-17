from typing import Optional

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
from utils.tqdm_auto import tqdm_auto


def _forward_with_optional_meta(model, x, batch_meta):
    if batch_meta is None:
        return model(x)
    try:
        return model(x, batch_meta=batch_meta)
    except TypeError:
        return model(x)


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model, epoch_for_log: Optional[int] = None):
        model.eval()

        truths = []
        preds = []
        for batch_idx, batch in enumerate(tqdm_auto(self.data_loader, self.params, mininterval=1)):
            if batch_idx == 0 and epoch_for_log is not None:
                print(f"entered first val batch for epoch {epoch_for_log}", flush=True)
            x, y = batch[0], batch[1]
            x = x.cuda()
            y = y.cuda()

            batch_meta = None
            if len(batch) >= 4 and isinstance(batch[3], dict):
                batch_meta = {k: v.cuda(non_blocking=True) for k, v in batch[3].items() if torch.is_tensor(v)}
            pred = _forward_with_optional_meta(model, x, batch_meta)
            pred_y = torch.max(pred, dim=-1)[1]

            # Flatten across all non-class dimensions so sequence tasks (e.g., ISRUC)
            # are evaluated on per-epoch labels rather than nested arrays.
            truths.extend(np.asarray(y.detach().cpu().numpy()).reshape(-1).tolist())
            preds.extend(np.asarray(pred_y.detach().cpu().numpy()).reshape(-1).tolist())

        if epoch_for_log is not None:
            print(f"finished validation loop for epoch {epoch_for_log}", flush=True)
        print("starting confusion matrix / metrics aggregation", flush=True)

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model, epoch_for_log: Optional[int] = None):
        model.eval()

        truths = []
        preds = []
        scores = []
        for batch_idx, batch in enumerate(tqdm_auto(self.data_loader, self.params, mininterval=1)):
            if batch_idx == 0 and epoch_for_log is not None:
                print(f"entered first val batch for epoch {epoch_for_log}", flush=True)
            x, y = batch[0], batch[1]
            x = x.cuda()
            y = y.cuda()
            batch_meta = None
            if len(batch) >= 4 and isinstance(batch[3], dict):
                batch_meta = {k: v.cuda(non_blocking=True) for k, v in batch[3].items() if torch.is_tensor(v)}
            pred = _forward_with_optional_meta(model, x, batch_meta)

            # Support both single-logit (BCE-style) and 2-logit (CE-style) binary heads.
            if pred.ndim == 1:
                pos_scores = torch.sigmoid(pred)
                pred_y = torch.ge(pos_scores, 0.5).long()
            elif pred.ndim >= 2 and pred.shape[-1] == 1:
                logits = pred.squeeze(-1)
                pos_scores = torch.sigmoid(logits)
                pred_y = torch.ge(pos_scores, 0.5).long()
            elif pred.ndim >= 2 and pred.shape[-1] == 2:
                probs = torch.softmax(pred, dim=-1)
                pos_scores = probs[..., 1]
                pred_y = torch.argmax(pred, dim=-1).long()
            else:
                raise ValueError(
                    f"Binary evaluator expects model outputs with last dim 1 or 2; got shape={tuple(pred.shape)}"
                )

            truths.extend(np.asarray(y.long().detach().cpu().numpy()).reshape(-1).tolist())
            preds.extend(np.asarray(pred_y.detach().cpu().numpy()).reshape(-1).tolist())
            scores.extend(np.asarray(pos_scores.detach().cpu().numpy()).reshape(-1).tolist())

        if epoch_for_log is not None:
            print(f"finished validation loop for epoch {epoch_for_log}", flush=True)
        print("starting confusion matrix / metrics aggregation", flush=True)

        truths = np.asarray(truths, dtype=np.int64)
        preds = np.asarray(preds, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        acc = balanced_accuracy_score(truths, preds)

        if np.unique(truths).size < 2:
            print('[warn] Binary evaluation has a single class in labels; AUROC/PR-AUC set to NaN.')
            roc_auc = float('nan')
            pr_auc = float('nan')
        else:
            roc_auc = roc_auc_score(truths, scores)
            precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
            pr_auc = auc(recall, precision)

        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model, epoch_for_log: Optional[int] = None):
        model.eval()

        truths = []
        preds = []
        for batch_idx, batch in enumerate(tqdm_auto(self.data_loader, self.params, mininterval=1)):
            if batch_idx == 0 and epoch_for_log is not None:
                print(f"entered first val batch for epoch {epoch_for_log}", flush=True)
            x, y = batch[0], batch[1]
            x = x.cuda()
            y = y.cuda()
            batch_meta = None
            if len(batch) >= 4 and isinstance(batch[3], dict):
                batch_meta = {k: v.cuda(non_blocking=True) for k, v in batch[3].items() if torch.is_tensor(v)}
            pred = _forward_with_optional_meta(model, x, batch_meta)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        if epoch_for_log is not None:
            print(f"finished validation loop for epoch {epoch_for_log}", flush=True)
        print("starting confusion matrix / metrics aggregation", flush=True)

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse