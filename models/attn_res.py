# models/attn_res.py
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., d]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class FullAttnRes(nn.Module):
    """
    Full Attention Residuals over depth.

    sources: list of tensors, each shaped [B, C, S, D]
             source[0] should be the token embedding / patch embedding.
    """
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        self.query = nn.Parameter(torch.zeros(d_model))  # zero-init per paper

    def forward(self, sources, return_alpha=False):
        # [N, B, C, S, D]
        v = torch.stack(sources, dim=0)
        k = self.norm(v)

        # logits over depth dimension N
        # result: [N, B, C, S]
        logits = torch.einsum('d,n...d->n...', self.query, k)

        alpha = torch.softmax(logits, dim=0)

        # weighted sum over source dimension N -> [B, C, S, D]
        h = torch.sum(alpha.unsqueeze(-1) * v, dim=0)

        if return_alpha:
            return h, alpha
        return h