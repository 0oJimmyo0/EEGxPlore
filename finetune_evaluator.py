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


def _extract_batch_meta(batch):
    if len(batch) >= 3 and isinstance(batch[2], dict):
        return batch[2]
    if len(batch) >= 4 and isinstance(batch[3], dict):
        return batch[3]
    return None


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
        self.last_multiclass_diag = None

    def get_metrics_for_multiclass(self, model, epoch_for_log: Optional[int] = None):
        model.eval()

        truths = []
        preds = []
        unknown_keys = ["subject_id", "cohort_id", "sample_rate_group_id", "age_bucket_id", "segment_bucket_id"]
        unknown_counter = {k: {"unknown": 0, "total": 0} for k in unknown_keys}
        for batch_idx, batch in enumerate(tqdm_auto(self.data_loader, self.params, mininterval=1)):
            if batch_idx == 0 and epoch_for_log is not None:
                print(f"entered first val batch for epoch {epoch_for_log}", flush=True)
            x, y = batch[0], batch[1]
            x = x.cuda()
            y = y.cuda()

            meta = _extract_batch_meta(batch)
            batch_meta = {k: v.cuda(non_blocking=True) for k, v in meta.items() if torch.is_tensor(v)} if isinstance(meta, dict) else None
            pred = _forward_with_optional_meta(model, x, batch_meta)
            pred_y = torch.max(pred, dim=-1)[1]

            if isinstance(batch_meta, dict):
                for k in unknown_keys:
                    v = batch_meta.get(k)
                    if not torch.is_tensor(v) or v.numel() == 0:
                        continue
                    unknown_counter[k]["total"] += int(v.numel())
                    unknown_counter[k]["unknown"] += int((v == 0).sum().item())

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        if epoch_for_log is not None:
            print(f"finished validation loop for epoch {epoch_for_log}", flush=True)
        print("starting confusion matrix / metrics aggregation", flush=True)

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        meta_unknown_ratio = {}
        for k, d in unknown_counter.items():
            tot = int(d["total"])
            unk = int(d["unknown"])
            meta_unknown_ratio[k] = float(unk / tot) if tot > 0 else 0.0
        self.last_multiclass_diag = {
            "meta_unknown_ratio": meta_unknown_ratio,
            "n": int(len(preds)),
        }
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
            meta = _extract_batch_meta(batch)
            batch_meta = {k: v.cuda(non_blocking=True) for k, v in meta.items() if torch.is_tensor(v)} if isinstance(meta, dict) else None
            pred = _forward_with_optional_meta(model, x, batch_meta)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        if epoch_for_log is not None:
            print(f"finished validation loop for epoch {epoch_for_log}", flush=True)
        print("starting confusion matrix / metrics aggregation", flush=True)

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
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
            meta = _extract_batch_meta(batch)
            batch_meta = {k: v.cuda(non_blocking=True) for k, v in meta.items() if torch.is_tensor(v)} if isinstance(meta, dict) else None
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
