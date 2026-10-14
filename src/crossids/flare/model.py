"""FLARE episode model (DESIGN.md §13) and the ProtoNet baseline (§16), with the context extension of §20.1.

Input of :meth:`EpisodeModel.forward_episode` is a padded batch of episodes (see ``data.collate``):
``x`` (B, N, D) normalised features, ``feat_pad`` (B, D), ``row_valid`` (B, N), ``is_query`` (B, N) and
``label`` (B, N) in {benign 0, attack 1, unknown 2}. Output is one logit per row (the head reads the query rows).

Rows form two blocks: the support-and-context block (labelled support rows, then unlabelled context rows) and the
query block. Support and context rows attend to the valid rows of the first block; query rows attend to the same
rows and to themselves. A model without context (U = 0) is the §13 model.

Inference (``predict_*``) for :class:`FlareModel` caches the per-layer keys and values of the support-and-context
block, which do not depend on the queries, and then runs only the query rows. The result equals the dense forward
pass to float rounding (tested in ``tests/test_flare.py``).
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
SCORE_BUDGET = 2 ** 24  # elements of one attention-score tensor per query chunk (about 64 MB in fp32)
QUERY_CHUNK = 32  # query rows per forward pass of the cached path; smaller chunks keep activations in cache on CPU


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
    """Row attention under the §13 / §20.1 mask, computed by block instead of with a dense N x N mask.

    Rows are laid out as [support + context block | query block]. Non-query rows attend to valid non-query rows
    (and themselves). Query rows attend to valid non-query rows and to themselves, never to other queries. This is
    the same attention as a masked N x N softmax, at O(N * k) cost.
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

    def project(self, h: torch.Tensor):
        """Query, key and value of every token: three tensors (B, N, T, H, hd)."""
        B, N, T, _ = h.shape
        qkv = self.qkv(h).reshape(B, N, T, 3, self.heads, self.head_dim)
        return qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]

    def attend_queries(self, q, k, v, k_sup, v_sup, sup_bias):
        """Query rows attend to cached support-and-context keys and to themselves (§13, §20.1).

        ``q, k, v`` are (E, c, T, H, hd) for a chunk of query rows. ``k_sup, v_sup`` are (E, T, H, K, hd) from the
        support cache and ``sup_bias`` is an additive (K,) mask (0 for valid rows, -inf for padded ones) or None.
        Returns (E, c, T, C) after the output projection.
        """
        E, c, T, H, hd = q.shape
        scale = hd ** -0.5
        K = k_sup.shape[3]
        qp = q.permute(0, 2, 3, 1, 4)  # (E, T, H, c, hd)
        vq = v.permute(0, 2, 3, 1, 4)
        self_score = (q * k).sum(-1).permute(0, 2, 3, 1).unsqueeze(-1) * scale  # (E, T, H, c, 1)
        if K > 0:
            s_sup = torch.matmul(qp, k_sup.transpose(-1, -2)) * scale  # (E, T, H, c, K)
            if sup_bias is not None:
                s_sup = s_sup + sup_bias
            scores = torch.cat([s_sup, self_score], dim=-1)
        else:
            scores = self_score
        p = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        out = p[..., -1:] * vq  # (E, T, H, c, hd)
        if K > 0:
            out = out + torch.matmul(p[..., :K], v_sup)
        out = out.permute(0, 3, 1, 2, 4).reshape(E, c, T, H * hd)
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
        H, hd = self.heads, C // self.heads
        qkv = self.qkv(h.reshape(B * N, T, C)).reshape(B * N, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B*N, H, T, hd)
        # explicit softmax over the T tokens: for T = d + 1 ~ 14 this is faster on CPU than the fused kernel
        scores = torch.matmul(q, k.transpose(-1, -2)) * hd ** -0.5
        key_mask = feat_mask[:, None, None, :].expand(B, N, 1, T).reshape(B * N, 1, 1, T)
        p = torch.softmax(scores.masked_fill(~key_mask, float("-inf")), dim=-1)
        out = torch.matmul(p, v).transpose(1, 2).reshape(B, N, T, C)
        return self.proj(out)


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

    def feat_step(self, h: torch.Tensor, feat_mask: torch.Tensor) -> torch.Tensor:
        """Feature mixing within each row (the first sub-block of a layer)."""
        if self.mixing == "attention":
            return h + self.feat(self.ln_f(h), feat_mask)
        h = h + self.feat(self.ln_f(h))
        return h * feat_mask[:, None, :, None].to(h.dtype)

    def forward(self, h: torch.Tensor, feat_mask: torch.Tensor, n_sup: int, sup_valid: torch.Tensor) -> torch.Tensor:
        h = self.feat_step(h, feat_mask)
        h = h + self.row(self.ln_r(h), n_sup, sup_valid)
        return h + self.mlp(self.ln_m(h))


@dataclass
class SupportCache:
    """Per-layer keys and values of the support-and-context block for one ensemble of E members.

    ``layers[i]`` holds (k, v) of layer i, each (E, T, H, K, hd). ``bias`` is the additive (K,) key mask.
    """

    layers: list
    feat_mask: torch.Tensor  # (E, T) bool
    bias: torch.Tensor | None
    n_sup: int


class EpisodeModel(nn.Module):
    """Shared inference: rank normalisation, test-time ensemble over sign flips (E members), query chunks."""

    kind = "base"
    cached = False  # True when predictions reuse a support cache (FlareModel)
    context_max = 0  # context rows the model was meta-trained with (0 = no context, §13)

    def forward_episode(self, x, feat_pad, row_valid, is_query, label, return_embedding: bool = False):
        raise NotImplementedError

    def _sign_members(self, d: int, ensemble: int, seed: int) -> np.ndarray:
        """Per-feature sign vectors (E, d) for the test-time ensemble. All ones (identity) when E == 1.

        Feature order does not enter: the model is permutation-invariant by construction, so the ensemble
        varies only the sign flips, the same transform as augmentation A1.
        """
        if ensemble == 1:
            return np.ones((1, d))
        rng = np.random.default_rng(seed)
        return rng.choice([-1.0, 1.0], size=(ensemble, d))

    def _query_step(self, E: int, d: int, nb: int, chunk: int) -> int:
        """Query rows per forward pass of the dense path (ProtoNet): the chunk size as given."""
        return chunk

    @torch.inference_mode()
    def _run(self, support_X, support_y, query_X, *, context_X, chunk: int, ensemble: int | None, seed: int,
             embed: bool):
        """Per query row: ensemble-mean logit (or the label-aware embedding with ``embed``).

        Rank normalisation runs once over support, context and query rows together. Each ensemble member
        multiplies the normalised features by its own sign vector.
        """
        Q = np.asarray(query_X, dtype=np.float64)
        d = Q.shape[1]
        S = np.asarray(support_X, dtype=np.float64)
        S = np.zeros((0, d)) if S.size == 0 else S.reshape(-1, d)
        C = np.zeros((0, d)) if context_X is None else np.asarray(context_X, dtype=np.float64).reshape(-1, d)
        ns, nc, nq = S.shape[0], C.shape[0], Q.shape[0]
        if len(support_y) != ns:
            raise ValueError("support_X and support_y disagree")
        Z = gauss_rank(np.vstack([S, C, Q]))
        E = 1 if embed else (ensemble or getattr(self, "ensemble", 1))
        signs = self._sign_members(d, E, seed)
        dev = next(self.parameters()).device
        Zs = torch.from_numpy(np.stack([Z * signs[e] for e in range(E)]).astype(np.float32)).to(dev)
        nb = ns + nc
        block_label = torch.from_numpy(
            np.concatenate([np.asarray(support_y, dtype=np.int64), np.full(nc, LABEL_UNKNOWN)])).to(dev)
        step = self._query_step(E, d, nb, chunk)
        outs = []
        if self.cached:
            cache = self.build_support_cache(Zs[:, :nb], block_label)
            for start in range(0, nq, step):
                stop = min(start + step, nq)
                outs.append(self.query_forward(cache, Zs[:, nb + start: nb + stop], return_embedding=embed))
        else:
            for start in range(0, nq, step):
                stop = min(start + step, nq)
                c = stop - start
                idx = torch.as_tensor(np.concatenate([np.arange(nb), nb + np.arange(start, stop)]), device=dev)
                n = nb + c
                is_query = torch.zeros(E, n, dtype=torch.bool, device=dev)
                is_query[:, nb:] = True
                label = torch.cat([block_label, torch.full((c,), LABEL_UNKNOWN, device=dev)])[None].expand(E, n)
                row_valid = torch.ones(E, n, dtype=torch.bool, device=dev)
                feat_pad = torch.ones(E, d, dtype=torch.bool, device=dev)
                res = self.forward_episode(Zs[:, idx], feat_pad, row_valid, is_query, label.contiguous(),
                                           return_embedding=embed)
                outs.append(res[:, nb:])
        if nq == 0:
            return np.zeros((0, self.width) if embed else (0,), dtype=np.float64)
        if embed:
            return torch.cat(outs, dim=1)[0].float().cpu().numpy().astype(np.float64)
        return torch.cat(outs, dim=1).float().mean(0).cpu().numpy().astype(np.float64)

    def predict_logits(self, support_X, support_y, query_X, chunk: int = QUERY_CHUNK, ensemble: int | None = None,
                       seed: int = 0, context_X=None) -> np.ndarray:
        """Ensemble-averaged logits of attack for the query rows. Support and context may be empty.

        ``context_X`` are unlabelled rows of the same network (§20.1). They are not scored.
        """
        self.eval()
        return self._run(support_X, support_y, query_X, context_X=context_X, chunk=chunk, ensemble=ensemble,
                         seed=seed, embed=False)

    def predict_proba(self, support_X, support_y, query_X, chunk: int = QUERY_CHUNK, ensemble: int | None = None,
                      seed: int = 0, context_X=None) -> np.ndarray:
        """P(attack) for the query rows given k labelled support rows (both classes present in practice)."""
        logits = self.predict_logits(support_X, support_y, query_X, chunk=chunk, ensemble=ensemble, seed=seed,
                                     context_X=context_X)
        return 1.0 / (1.0 + np.exp(-logits))

    def embed(self, support_X, support_y, query_X, chunk: int = QUERY_CHUNK) -> np.ndarray:
        """Label-aware row embeddings (mean of feature tokens after the last layer), identity transform."""
        self.eval()
        return self._run(support_X, support_y, query_X, context_X=None, chunk=chunk, ensemble=1, seed=0, embed=True)


class FlareModel(EpisodeModel):
    kind = "flare"
    cached = True

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

    def _tokens(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Feature tokens and the label token of every row: (B, N, T = D + 1, C). Token 0 is the label token."""
        finite = torch.isfinite(x)
        vals = self.value(torch.where(finite, x, torch.zeros_like(x)).unsqueeze(-1))
        vals = torch.where(finite.unsqueeze(-1), vals, self.absent)
        lab = self.label_emb(label).unsqueeze(2)
        return torch.cat([lab, vals], dim=2)

    @staticmethod
    def _feat_mask(feat_pad: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(feat_pad.shape[0], 1, dtype=torch.bool, device=feat_pad.device)
        return torch.cat([ones, feat_pad], dim=1)

    def forward_episode(self, x, feat_pad, row_valid, is_query, label, return_embedding: bool = False):
        """Rows must be laid out as [support + context block | query block] (see ``data.collate``)."""
        n_sup = support_block_size(is_query)
        h = self._tokens(x, label)
        feat_mask = self._feat_mask(feat_pad)
        sup_valid = row_valid[:, :n_sup]
        for layer in self.layers:
            h = layer(h, feat_mask, n_sup, sup_valid)
        if return_embedding:
            w = feat_pad.to(h.dtype)[:, None, :, None]
            feats = h[:, :, 1:]
            return (feats * w).sum(2) / w.sum(2).clamp(min=1.0)
        return self.head(self.ln_out(h[:, :, 0])).squeeze(-1)

    def _query_step(self, E: int, d: int, nb: int, chunk: int) -> int:
        """Query rows per chunk, capped so that the (E, T, H, c, nb + 1) attention scores stay within SCORE_BUDGET."""
        return max(1, min(chunk, SCORE_BUDGET // max(1, E * (d + 1) * self.cfg.heads * (nb + 1))))

    def build_support_cache(self, x: torch.Tensor, label: torch.Tensor, feat_pad: torch.Tensor | None = None,
                            sup_valid: torch.Tensor | None = None) -> SupportCache:
        """Support-and-context block of every ensemble member, run once and cached per layer.

        ``x`` is (E, K, D), ``label`` (K,). Rows of this block never attend to queries, so the states computed here
        are the same for every query chunk.
        """
        E, K, D = x.shape
        if feat_pad is None:
            feat_pad = torch.ones(E, D, dtype=torch.bool, device=x.device)
        feat_mask = self._feat_mask(feat_pad)
        sv = torch.ones(E, K, dtype=torch.bool, device=x.device) if sup_valid is None else \
            sup_valid[None].expand(E, K)
        h = self._tokens(x, label[None].expand(E, K))
        layers = []
        for layer in self.layers:
            h = layer.feat_step(h, feat_mask)
            hn = layer.ln_r(h)
            _, k, v = layer.row.project(hn)
            layers.append((k.permute(0, 2, 3, 1, 4).contiguous(), v.permute(0, 2, 3, 1, 4).contiguous()))
            h = h + layer.row(hn, K, sv)
            h = h + layer.mlp(layer.ln_m(h))
        bias = None if sup_valid is None else torch.where(sup_valid, 0.0, float("-inf")).to(x.dtype)
        return SupportCache(layers=layers, feat_mask=feat_mask, bias=bias, n_sup=K)

    def query_forward(self, cache: SupportCache, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """Logits (E, c) of a chunk of query rows x (E, c, D), or their embeddings (E, c, C).

        Query rows attend to the cached block and to themselves only, so the result does not depend on the
        other queries in the chunk.
        """
        E, c, _ = x.shape
        label = torch.full((E, c), LABEL_UNKNOWN, dtype=torch.long, device=x.device)
        h = self._tokens(x, label)
        for li, layer in enumerate(self.layers):
            h = layer.feat_step(h, cache.feat_mask)
            hn = layer.ln_r(h)
            q, k, v = layer.row.project(hn)
            ks, vs = cache.layers[li]
            h = h + layer.row.attend_queries(q, k, v, ks, vs, cache.bias)
            h = h + layer.mlp(layer.ln_m(h))
        if return_embedding:
            w = cache.feat_mask.to(h.dtype)[:, None, 1:, None]
            feats = h[:, :, 1:]
            return (feats * w).sum(2) / w.sum(2).clamp(min=1.0)
        return self.head(self.ln_out(h[:, :, 0])).squeeze(-1)


def support_block_size(is_query: torch.Tensor) -> int:
    """Number of leading support-and-context columns. Columns from here on are query columns for every episode."""
    N = is_query.shape[1]
    if not bool(is_query.any()):
        return N
    n_sup = int(torch.argmax(is_query.any(0).to(torch.int8)).item())
    if bool(is_query[:, :n_sup].any()):
        raise ValueError("rows must be laid out as [support + context block | query block]")
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
    """Checkpoint with the config and state dict. ``context_max`` (§20.1) is stored at the top level when given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.kind, "config": asdict(model.cfg), "state_dict": model.state_dict(), **extra}, path)


def load_model(path: str | Path, device: str = "cpu", ensemble: int | None = None) -> EpisodeModel:
    """Load a checkpoint. Checkpoints written before §20.1 have no ``context_max`` and load as context_max = 0."""
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = FlareConfig(**blob["config"])
    if ensemble is not None:
        cfg = replace(cfg, ensemble=ensemble)
    model = build_model(blob["kind"], cfg)
    model.load_state_dict(blob["state_dict"])
    model.context_max = int(blob.get("context_max", 0))
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
