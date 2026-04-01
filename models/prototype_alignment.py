from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Running class prototypes for lightweight class-structure alignment."""

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.95, eps: float = 1e-6):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1 for prototype alignment")
        if feat_dim <= 0:
            raise ValueError("feat_dim must be > 0")
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("seen_counts", torch.zeros(self.num_classes, dtype=torch.long))

    def _safe_norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=self.eps)

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be [B,D], got {tuple(features.shape)}")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features/labels batch mismatch")

        feats = features.detach()
        y = labels.detach().long()
        for cls in y.unique():
            c = int(cls.item())
            if c < 0 or c >= self.num_classes:
                continue
            mask = (y == c)
            if int(mask.sum().item()) <= 0:
                continue
            cls_mean = self._safe_norm(feats[mask].mean(dim=0, keepdim=True)).squeeze(0)
            if int(self.seen_counts[c].item()) == 0:
                self.prototypes[c] = cls_mean
            else:
                mixed = self.prototypes[c] * self.momentum + (1.0 - self.momentum) * cls_mean
                self.prototypes[c] = self._safe_norm(mixed.unsqueeze(0)).squeeze(0)
            self.seen_counts[c] += int(mask.sum().item())

    def losses(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        pull_weight: float,
        push_weight: float,
        margin: float,
    ) -> Dict[str, torch.Tensor]:
        if labels.ndim != 1:
            labels = labels.view(-1)
        y = labels.long()
        seen_mask = self.seen_counts > 0
        valid = (y >= 0) & (y < self.num_classes)
        valid = valid & seen_mask.index_select(0, y.clamp(min=0, max=self.num_classes - 1))
        if int(valid.sum().item()) == 0:
            z = features.new_zeros(())
            return {"total": z, "pull": z, "push": z}

        feats = features[valid]
        y = y[valid]

        proto = self._safe_norm(self.prototypes)
        feat_n = self._safe_norm(feats)
        sims = torch.matmul(feat_n, proto.t())

        pos = sims.gather(1, y.view(-1, 1)).squeeze(1)
        pull_loss = (1.0 - pos).mean()

        neg_mask = seen_mask.view(1, -1).expand_as(sims)
        neg_mask.scatter_(1, y.view(-1, 1), False)
        neg = sims.masked_fill(~neg_mask, -1e9).max(dim=1).values
        push_loss = F.relu(neg - pos + float(margin)).mean()

        total = float(pull_weight) * pull_loss + float(push_weight) * push_loss
        return {"total": total, "pull": pull_loss, "push": push_loss}

    def diagnostics(
        self,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        proto_norm = self.prototypes.norm(dim=-1)
        out["proto_norm_mean"] = float(proto_norm.mean().item())
        out["proto_norm_std"] = float(proto_norm.std(unbiased=False).item())

        seen = self.seen_counts > 0
        if int(seen.sum().item()) > 1:
            pn = self._safe_norm(self.prototypes.index_select(0, seen.nonzero(as_tuple=False).view(-1)))
            sim = torch.matmul(pn, pn.t())
            off = ~torch.eye(int(pn.shape[0]), device=sim.device, dtype=torch.bool)
            if int(off.sum().item()) > 0:
                out["between_class_cos"] = float(sim[off].mean().item())

        if features is None or labels is None:
            return out

        if labels.ndim != 1:
            labels = labels.view(-1)
        y = labels.long()
        valid = (y >= 0) & (y < self.num_classes)
        if int(valid.sum().item()) <= 1:
            return out

        feats = features[valid]
        yv = y[valid]
        protos_y = self.prototypes.index_select(0, yv)
        dists = (feats - protos_y).pow(2).sum(dim=-1).sqrt()
        out["within_class_spread"] = float(dists.mean().item())

        if subject_ids is not None:
            sid = subject_ids.view(-1).long()[valid]
            subj_means = []
            for s in sid.unique():
                m = (sid == s)
                if int(m.sum().item()) > 0:
                    subj_means.append(float(dists[m].mean().item()))
            if subj_means:
                t = torch.tensor(subj_means, device=dists.device)
                out["subject_spread_std"] = float(t.std(unbiased=False).item())
        return out
