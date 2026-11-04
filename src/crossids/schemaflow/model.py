"""SchemaFlow: a schema-agnostic in-context flow classifier.

A row is a set of (feature text, value) tokens. The token of feature j with rank-normalised value v is
``P(e_j) + Q(e_j) * v + R(v)`` where ``e_j`` is the frozen text embedding of the feature. A transformer over the
tokens of a row gives the row representation. A query is classified by attention over the labelled support rows.
Nothing in the model depends on which or how many features a dataset has.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SchemaFlow(nn.Module):
    def __init__(self, d_text: int = 384, d: int = 64, heads: int = 4, layers: int = 2, ff: int = 128):
        super().__init__()
        self.p = nn.Linear(d_text, d)
        self.q = nn.Linear(d_text, d, bias=False)
        self.r = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool_q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.d = d

    def encode(self, v: torch.Tensor, e: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
        """v (B, N, F) values, e (B, F, d_text) feature embeddings, fmask (B, F) True where the feature exists."""
        B, N, F = v.shape
        base = self.p(e)  # (B, F, d)
        qe = self.q(e)
        tok = base[:, None] + qe[:, None] * v[..., None] + self.r(v[..., None])  # (B, N, F, d)
        tok = tok.reshape(B * N, F, self.d)
        pad = (~fmask)[:, None, :].expand(B, N, F).reshape(B * N, F)
        h = self.enc(tok, src_key_padding_mask=pad)
        att = (h * self.pool_q).sum(-1) / math.sqrt(self.d)
        att = att.masked_fill(pad, -1e9).softmax(-1)
        row = (att[..., None] * h).sum(1)
        return self.out(row).reshape(B, N, self.d)

    def forward(self, vs, ys, smask, vq, e, fmask):
        """vs (B, S, F), ys (B, S) in {0,1}, smask (B, S) valid support, vq (B, Q, F) -> log-probs (B, Q, 2)."""
        hs = self.encode(vs, e, fmask)
        hq = self.encode(vq, e, fmask)
        score = torch.einsum("bqd,bsd->bqs", self.wq(hq), self.wk(hs)) / math.sqrt(self.d) * self.log_tau.exp()
        score = score.masked_fill(~smask[:, None, :], -1e9)
        lp = score.log_softmax(-1)
        out = []
        for c in (0, 1):
            m = (ys == c) & smask
            out.append(torch.logsumexp(lp.masked_fill(~m[:, None, :], -1e4), dim=-1))
        return torch.stack(out, -1)
