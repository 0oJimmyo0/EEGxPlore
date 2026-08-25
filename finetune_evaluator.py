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
        detailed = self.get_detailed_metrics_for_multiclass(model, epoch_for_log=epoch_for_log)
        self.last_macro_f1 = float(detailed['macro_f1'])
        return (
            float(detailed['balanced_accuracy']),
            float(detailed['kappa']),
            float(detailed['weighted_f1']),
            np.asarray(detailed['confusion_matrix']),
        )

    @staticmethod
    def _subject_from_sample_key(sample_key: str) -> str:
        """Extract the subject component from current or legacy SEED-V keys."""
        key = str(sample_key)
        if '_t' in key and '_g' in key:
            return key.split('_', 1)[0]
        prefix = key.rsplit('-', 2)[0]
        return prefix.split('_', 1)[0]

    @staticmethod
    def _metrics_from_arrays(truths: np.ndarray, preds: np.ndarray, num_classes: int):
        if truths.size == 0:
            raise ValueError('Cannot evaluate an empty multiclass prediction set.')
        cm = confusion_matrix(truths, preds, labels=list(range(int(num_classes))))
        prediction_histogram = np.bincount(preds.astype(np.int64), minlength=int(num_classes))
        return {
            'n': int(truths.size),
            'balanced_accuracy': float(balanced_accuracy_score(truths, preds)),
            'kappa': float(cohen_kappa_score(truths, preds)),
            'weighted_f1': float(f1_score(truths, preds, average='weighted', zero_division=0)),
            'macro_f1': float(f1_score(truths, preds, average='macro', zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'classwise_recall': [
                float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
                for i in range(cm.shape[0])
            ],
            'prediction_histogram': prediction_histogram.tolist(),
        }

    def get_detailed_metrics_for_multiclass(self, model, epoch_for_log: Optional[int] = None):
        """Return metrics, histograms, and optional key-derived subject metrics."""
        model.eval()

        truths = []
        preds = []
        sample_keys = []
        keys_are_aligned = True
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
            batch_truths = np.asarray(y.detach().cpu().numpy()).reshape(-1)
            batch_preds = np.asarray(pred_y.detach().cpu().numpy()).reshape(-1)
            truths.extend(batch_truths.tolist())
            preds.extend(batch_preds.tolist())

            batch_keys = None
            if len(batch) >= 3 and isinstance(batch[2], (list, tuple)):
                if all(isinstance(key, (str, bytes)) for key in batch[2]):
                    batch_keys = [key.decode() if isinstance(key, bytes) else str(key) for key in batch[2]]
            if batch_keys is not None and len(batch_keys) == len(batch_truths):
                sample_keys.extend(batch_keys)
            else:
                keys_are_aligned = False

        if epoch_for_log is not None:
            print(f"finished validation loop for epoch {epoch_for_log}", flush=True)
        print("starting confusion matrix / metrics aggregation", flush=True)

        truths = np.asarray(truths, dtype=np.int64)
        preds = np.asarray(preds, dtype=np.int64)
        detailed = self._metrics_from_arrays(
            truths,
            preds,
            int(getattr(self.params, 'num_of_classes', max(int(preds.max()) + 1, 1))),
        )
        if keys_are_aligned and len(sample_keys) == len(truths):
            by_subject = {}
            grouped = {}
            for key, truth, pred in zip(sample_keys, truths.tolist(), preds.tolist()):
                subject = self._subject_from_sample_key(key)
                grouped.setdefault(subject, [[], []])
                grouped[subject][0].append(truth)
                grouped[subject][1].append(pred)
            for subject in sorted(grouped):
                subject_truths = np.asarray(grouped[subject][0], dtype=np.int64)
                subject_preds = np.asarray(grouped[subject][1], dtype=np.int64)
                by_subject[subject] = self._metrics_from_arrays(
                    subject_truths,
                    subject_preds,
                    int(getattr(self.params, 'num_of_classes', 1)),
                )
            detailed['subject_metrics'] = by_subject
        else:
            detailed['subject_metrics'] = {}
        return detailed

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
