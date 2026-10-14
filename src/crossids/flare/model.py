"""FLARE episode model (DESIGN.md §13) and the ProtoNet baseline (§16).

Input of :meth:`EpisodeModel.forward_episode` is a padded batch of episodes (see ``data.collate``):
``x`` (B, N, D) normalised features, ``feat_pad`` (B, D), ``row_valid`` (B, N), ``is_query`` (B, N) and
``label`` (B, N) in {benign 0, attack 1, unknown 2}. Output is one logit per row (the head reads the query rows).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LABEL_UNKNOWN, gauss_rank

MIXINGS = ("attention", "rowmlp")


@dataclass(frozen=True)
class FlareConfig:
    """Architecture and test-time settings. ``feature_mixing="rowmlp"`` is the fixed-order ablation (§17)."""

    layers: int = 6
    width: int = 128
    heads: int = 4
    mlp_mult: int = 4
    value_hidden: int = 64
    feature_mixing: str = "attention"
    max_features: int = 13
    ensemble: int = 8

    def __post_init__(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.feature_mixing not in MIXINGS:
            raise ValueError(f"feature_mixing must be one of {MIXINGS}")


DEFAULT_CONFIG = FlareConfig()
TINY_CONFIG = FlareConfig(layers=2, width=32, heads=2, value_hidden=16, ensemble=4)
NAMED_CONFIGS = {"default": DEFAULT_CONFIG, "tiny": TINY_CONFIG}


class _RowAttention(nn.Module):
    """Row attention under the §13 mask, computed by block instead of with a dense N x N mask.

    Rows are laid out as [support block | query block]. Support rows attend to valid support rows (and
    themselves). Query rows attend to valid support rows and to themselves, never to other queries. This is the
    same attention as a masked N x N softmax, at O(N * k) cost.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, h: torch.Tensor, n_sup: int, sup_valid: torch.Tensor) -> torch.Tensor:
        B, N, T, C = h.shape
        H, hd = self.heads, self.head_dim
        qkv = self.qkv(h).reshape(B, N, T, 3, H, hd)
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]  # (B, N, T, H, hd)
        scale = hd ** -0.5
        parts = []
        if n_sup > 0:
            sv = sup_valid[:, :n_sup]
            qs = q[:, :n_sup].permute(0, 2, 3, 1, 4).reshape(B * T, H, n_sup, hd)
            ks = k[:, :n_sup].permute(0, 2, 3, 1, 4).reshape(B * T, H, n_sup, hd)
            vs = v[:, :n_sup].permute(0, 2, 3, 1, 4).reshape(B * T, H, n_sup, hd)
            eye = torch.eye(n_sup, dtype=torch.bool, device=h.device)
            allow = (sv[:, :, None] & sv[:, None, :]) | eye
            mask = allow[:, None, None].expand(B, T, 1, n_sup, n_sup).reshape(B * T, 1, n_sup, n_sup)
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=mask)
            parts.append(o.reshape(B, T, H, n_sup, hd).permute(0, 3, 1, 2, 4))  # (B, K, T, H, hd)
        qq, kq, vq = q[:, n_sup:], k[:, n_sup:], v[:, n_sup:]
        self_score = (qq * kq).sum(-1, keepdim=True) * scale  # (B, Q, T, H, 1)
        if n_sup > 0:
            ks_b, vs_b = k[:, :n_sup], v[:, :n_sup]
            s_sup = torch.einsum("bqthd,bkthd->bqthk", qq, ks_b) * scale
            s_sup = s_sup.masked_fill(~sv[:, None, None, None, :], float("-inf"))
            scores = torch.cat([s_sup, self_score], dim=-1)
        else:
            scores = self_score
        p = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out_q = p[..., -1:] * vq
        if n_sup > 0:
            out_q = out_q + torch.einsum("bqthk,bkthd->bqthd", p[..., :n_sup], vs_b)
        parts.append(out_q)
        out = torch.cat(parts, dim=1).reshape(B, N, T, C)
        return self.proj(out)


class _FeatureAttention(nn.Module):
    """Attention over the tokens of one row (all d + 1 tokens attend to each other, padded features masked)."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, h: torch.Tensor, feat_mask: torch.Tensor) -> torch.Tensor:
        B, N, T, C = h.shape
        x = h.reshape(B * N, T, C)
        q, k, v = self.qkv(x).reshape(B * N, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        key_mask = feat_mask[:, None, None, :].expand(B, N, 1, T).reshape(B * N, 1, 1, T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        return self.proj(out.transpose(1, 2).reshape(B * N, T, C)).reshape(B, N, T, C)


class _RowMLP(nn.Module):
    """Fixed-order ablation of feature attention: an MLP over the flattened token axis of each row."""

    def __init__(self, width: int, tokens: int, mult: int):
        super().__init__()
        self.tokens = tokens
        self.net = nn.Sequential(
            nn.Linear(tokens * width, mult * width), nn.GELU(), nn.Linear(mult * width, tokens * width)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, T, C = h.shape
        pad = self.tokens - T
        if pad < 0:
            raise ValueError(f"{T - 1} features exceed max_features={self.tokens - 1}")
        h = F.pad(h, (0, 0, 0, pad))
        out = self.net(h.reshape(B, N, self.tokens * C)).reshape(B, N, self.tokens, C)
        return out[:, :, :T]


class _Layer(nn.Module):
    """Pre-LN block: feature mixing within each row, row attention per token position, GELU MLP."""

    def __init__(self, cfg: FlareConfig):
        super().__init__()
        C = cfg.width
        self.ln_f = nn.LayerNorm(C)
        if cfg.feature_mixing == "attention":
            self.feat = _FeatureAttention(C, cfg.heads)
        else:
            self.feat = _RowMLP(C, cfg.max_features + 1, cfg.mlp_mult)
        self.ln_r = nn.LayerNorm(C)
        self.row = _RowAttention(C, cfg.heads)
        self.ln_m = nn.LayerNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, cfg.mlp_mult * C), nn.GELU(), nn.Linear(cfg.mlp_mult * C, C))
        self.mixing = cfg.feature_mixing

    def forward(self, h: torch.Tensor, feat_mask: torch.Tensor, n_sup: int, sup_valid: torch.Tensor) -> torch.Tensor:
        if self.mixing == "attention":
            h = h + self.feat(self.ln_f(h), feat_mask)
        else:
            h = h + self.feat(self.ln_f(h))
            h = h * feat_mask[:, None, :, None].to(h.dtype)
        h = h + self.row(self.ln_r(h), n_sup, sup_valid)
        return h + self.mlp(self.ln_m(h))


class EpisodeModel(nn.Module):
    """Shared inference: normalisation, test-time ensemble over feature permutations and sign flips, query chunks."""

    kind = "base"

    def forward_episode(self, x, feat_pad, row_valid, is_query, label, return_embedding: bool = False):
        raise NotImplementedError

    def _members(self, d: int, ensemble: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Feature permutations (E, d) and sign vectors (E, d). Member 0 is the identity when E == 1."""
        if ensemble == 1:
            return np.arange(d)[None], np.ones((1, d))
        rng = np.random.default_rng(seed)
        perms = np.stack([rng.permutation(d) for _ in range(ensemble)])
        signs = rng.choice([-1.0, 1.0], size=(ensemble, d))
        return perms, signs

    @torch.no_grad()
    def _run(self, support_X, support_y, query_X, *, chunk: int, ensemble: int | None, seed: int, embed: bool):
        Q = np.asarray(query_X, dtype=np.float64)
        d = Q.shape[1]
        S = np.asarray(support_X, dtype=np.float64)
        S = np.zeros((0, d)) if S.size == 0 else S.reshape(-1, d)
        ns, nq = S.shape[0], Q.shape[0]
        if len(support_y) != ns:
            raise ValueError("support_X and support_y disagree")
        Z = gauss_rank(np.vstack([S, Q]))
        E = 1 if embed else (ensemble or getattr(self, "ensemble", 1))
        perms, signs = self._members(d, E, seed)
        Zs = np.stack([Z[:, perms[e]] * signs[e] for e in range(E)]).astype(np.float32)
        Zs = torch.from_numpy(Zs)
        sup_lab = np.asarray(support_y, dtype=np.int64)
        out = []
        for start in range(0, nq, chunk):
            stop = min(start + chunk, nq)
            c = stop - start
            idx = np.concatenate([np.arange(ns), ns + np.arange(start, stop)])
            n = ns + c
            x = Zs[:, idx]
            is_query = torch.zeros(E, n, dtype=torch.bool)
            is_query[:, ns:] = True
            label = torch.from_numpy(np.concatenate([sup_lab, np.full(c, LABEL_UNKNOWN)]))[None].expand(E, n)
            row_valid = torch.ones(E, n, dtype=torch.bool)
            feat_pad = torch.ones(E, d, dtype=torch.bool)
            if embed:
                res = self.forward_episode(x, feat_pad, row_valid, is_query, label.contiguous(), return_embedding=True)
                out.append(res[0, ns:].float().cpu().numpy())
            else:
                logits = self.forward_episode(x, feat_pad, row_valid, is_query, label.contiguous())
                out.append(logits[:, ns:].float().mean(0).cpu().numpy())
        if not out:
            return np.zeros((0, self.width) if embed else (0,), dtype=np.float64)
        return np.concatenate(out).astype(np.float64)

    def predict_logits(self, support_X, support_y, query_X, chunk: int = 1024, ensemble: int | None = None,
                       seed: int = 0) -> np.ndarray:
        """Ensemble-averaged logits of attack for the query rows. Support may be empty."""
        self.eval()
        return self._run(support_X, support_y, query_X, chunk=chunk, ensemble=ensemble, seed=seed, embed=False)

    def predict_proba(self, support_X, support_y, query_X, chunk: int = 1024, ensemble: int | None = None,
                      seed: int = 0) -> np.ndarray:
        """P(attack) for the query rows given k labelled support rows (both classes present in practice)."""
        logits = self.predict_logits(support_X, support_y, query_X, chunk=chunk, ensemble=ensemble, seed=seed)
        return 1.0 / (1.0 + np.exp(-logits))

    def embed(self, support_X, support_y, query_X, chunk: int = 1024) -> np.ndarray:
        """Label-aware row embeddings (mean of feature tokens after the last layer), identity transform."""
        self.eval()
        return self._run(support_X, support_y, query_X, chunk=chunk, ensemble=1, seed=0, embed=True)


class FlareModel(EpisodeModel):
    kind = "flare"

    def __init__(self, cfg: FlareConfig):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.width
        self.ensemble = cfg.ensemble
        C = cfg.width
        self.value = nn.Sequential(nn.Linear(1, cfg.value_hidden), nn.GELU(), nn.Linear(cfg.value_hidden, C))
        self.absent = nn.Parameter(torch.randn(C) * 0.02)
        self.label_emb = nn.Embedding(3, C)
        self.layers = nn.ModuleList([_Layer(cfg) for _ in range(cfg.layers)])
        self.ln_out = nn.LayerNorm(C)
        self.head = nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, 1))

    def forward_episode(self, x, feat_pad, row_valid, is_query, label, return_embedding: bool = False):
        """Rows must be laid out as [support block | query block] (see ``data.collate``)."""
        B, N, D = x.shape
        n_sup = support_block_size(is_query)
        finite = torch.isfinite(x)
        vals = self.value(torch.where(finite, x, torch.zeros_like(x)).unsqueeze(-1))
        vals = torch.where(finite.unsqueeze(-1), vals, self.absent)
        lab = self.label_emb(label).unsqueeze(2)
        h = torch.cat([lab, vals], dim=2)  # (B, N, T = D + 1, C); token 0 is the label token
        feat_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), feat_pad], dim=1)
        sup_valid = row_valid[:, :n_sup]
        for layer in self.layers:
            h = layer(h, feat_mask, n_sup, sup_valid)
        if return_embedding:
            w = feat_pad.to(h.dtype)[:, None, :, None]
            feats = h[:, :, 1:]
            return (feats * w).sum(2) / w.sum(2).clamp(min=1.0)
        return self.head(self.ln_out(h[:, :, 0])).squeeze(-1)


def support_block_size(is_query: torch.Tensor) -> int:
    """Number of leading support columns. Columns from here on are query columns for every episode."""
    N = is_query.shape[1]
    if not bool(is_query.any()):
        return N
    n_sup = int(torch.argmax(is_query.any(0).to(torch.int8)).item())
    if bool(is_query[:, :n_sup].any()):
        raise ValueError("rows must be laid out as [support block | query block]")
    return n_sup


class ProtoNet(EpisodeModel):
    """Prototype classifier on a row encoder meta-trained on the same episodes. Features enter by position with a mask."""

    kind = "protonet"

    def __init__(self, cfg: FlareConfig):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.width
        self.ensemble = 1
        M = cfg.max_features
        self.encoder = nn.Sequential(
            nn.Linear(2 * M, 2 * cfg.width), nn.GELU(), nn.Linear(2 * cfg.width, cfg.width)
        )

    def forward_episode(self, x, feat_pad, row_valid, is_query, label, return_embedding: bool = False):
        B, N, D = x.shape
        M = self.cfg.max_features
        if D > M:
            raise ValueError(f"{D} features exceed max_features={M}")
        finite = torch.isfinite(x) & feat_pad[:, None, :]
        vals = torch.where(finite, x, torch.zeros_like(x))
        vals = F.pad(vals, (0, M - D))
        mask = F.pad(finite.to(x.dtype), (0, M - D))
        emb = self.encoder(torch.cat([vals, mask], dim=-1))
        if return_embedding:
            return emb
        sup = row_valid & ~is_query
        w1 = (sup & (label == 1)).to(emb.dtype)
        w0 = (sup & (label == 0)).to(emb.dtype)
        p1 = (emb * w1[..., None]).sum(1) / w1.sum(1).clamp(min=1.0)[:, None]
        p0 = (emb * w0[..., None]).sum(1) / w0.sum(1).clamp(min=1.0)[:, None]
        d1 = ((emb - p1[:, None]) ** 2).sum(-1)
        d0 = ((emb - p0[:, None]) ** 2).sum(-1)
        return d0 - d1


def build_model(kind: str, cfg: FlareConfig) -> EpisodeModel:
    if kind == "flare":
        return FlareModel(cfg)
    if kind == "protonet":
        return ProtoNet(cfg)
    raise ValueError(f"unknown model kind {kind!r}")


def save_checkpoint(path: str | Path, model: EpisodeModel, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.kind, "config": asdict(model.cfg), "state_dict": model.state_dict(), **extra}, path)


def load_model(path: str | Path, device: str = "cpu", ensemble: int | None = None) -> EpisodeModel:
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = FlareConfig(**blob["config"])
    if ensemble is not None:
        cfg = replace(cfg, ensemble=ensemble)
    model = build_model(blob["kind"], cfg)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


def batch_to_torch(batch, device: str | torch.device = "cpu"):
    """Convert a ``data.Batch`` to the tensors ``forward_episode`` expects."""
    return dict(
        x=torch.from_numpy(batch.x).to(device),
        feat_pad=torch.from_numpy(batch.feat_pad).to(device),
        row_valid=torch.from_numpy(batch.row_valid).to(device),
        is_query=torch.from_numpy(batch.is_query).to(device),
        label=torch.from_numpy(batch.label).to(device),
        truth=torch.from_numpy(batch.truth).to(device),
    )
