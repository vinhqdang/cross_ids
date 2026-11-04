"""Rank normalisation and meta-training episodes over native schemas."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

CLIP = 4.0


def rank_ref(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Gaussian rank of each column of X against the reference rows ``ref`` (NaN -> 0, constant column -> 0)."""
    out = np.zeros(X.shape, np.float32)
    n = len(ref)
    for j in range(X.shape[1]):
        col = ref[:, j]
        col = np.sort(col[~np.isnan(col)])
        if len(col) < 2 or col[0] == col[-1]:
            continue
        x = X[:, j]
        u = (np.searchsorted(col, x, "left") + np.searchsorted(col, x, "right")) / 2.0
        u = np.clip((u + 0.5) / (len(col) + 1.0), 1e-4, 1 - 1e-4)
        z = np.clip(ndtri(u), -CLIP, CLIP)
        out[:, j] = np.where(np.isnan(x), 0.0, z)
    return out


class Domain:
    def __init__(self, key, X, y, names, emb):
        self.key, self.X, self.y, self.names, self.emb = key, X, y, names, emb
        self.idx = [np.flatnonzero(y == 0), np.flatnonzero(y == 1)]


def draw_episode(rng, dom: Domain, fmax=32, smax=64, q=64, pool_n=512, fdrop=True):
    """One episode: a feature subset, an unlabelled pool for normalisation, a labelled support and queries."""
    F = dom.X.shape[1]
    if fdrop:
        keep = rng.random(F) < rng.uniform(0.4, 1.0)
        if keep.sum() < min(4, F):
            keep[rng.choice(F, min(4, F), replace=False)] = True
        cols = np.flatnonzero(keep)
    else:
        cols = np.arange(F)
    if len(cols) > fmax:
        cols = rng.choice(cols, fmax, replace=False)
    pool = rng.choice(len(dom.X), min(pool_n, len(dom.X)), replace=False)
    ks = int(rng.integers(4, smax + 1))
    n_att = int(np.clip(rng.binomial(ks, rng.uniform(0.1, 0.9)), 1, ks - 1))
    sa = rng.choice(dom.idx[1], n_att, replace=False)
    sb = rng.choice(dom.idx[0], ks - n_att, replace=False)
    s = np.concatenate([sb, sa])
    ys = np.concatenate([np.zeros(len(sb)), np.ones(len(sa))]).astype(np.int64)
    avoid = set(s.tolist())
    qa = rng.choice([i for i in dom.idx[1] if i not in avoid][:50000] or dom.idx[1], q // 2, replace=False)
    qb = rng.choice([i for i in dom.idx[0] if i not in avoid][:50000] or dom.idx[0], q - q // 2, replace=False)
    qi = np.concatenate([qb, qa])
    yq = np.concatenate([np.zeros(len(qb)), np.ones(len(qa))]).astype(np.int64)
    ref = dom.X[pool][:, cols]
    return dict(
        vs=rank_ref(ref, dom.X[s][:, cols]), ys=ys, vq=rank_ref(ref, dom.X[qi][:, cols]), yq=yq,
        e=dom.emb[cols],
    )


def collate(eps, fmax=32, smax=64):
    import torch

    B = len(eps)
    q = eps[0]["vq"].shape[0]
    vs = np.zeros((B, smax, fmax), np.float32)
    vq = np.zeros((B, q, fmax), np.float32)
    ys = np.zeros((B, smax), np.int64)
    sm = np.zeros((B, smax), bool)
    fm = np.zeros((B, fmax), bool)
    e = np.zeros((B, fmax, eps[0]["e"].shape[1]), np.float32)
    yq = np.zeros((B, q), np.int64)
    for i, ep in enumerate(eps):
        s, f = ep["vs"].shape
        vs[i, :s, :f] = ep["vs"]
        vq[i, :, :f] = ep["vq"]
        ys[i, :s] = ep["ys"]
        sm[i, :s] = True
        fm[i, :f] = True
        e[i, :f] = ep["e"]
        yq[i] = ep["yq"]
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    return t(vs), t(ys), t(sm), t(vq), t(e), t(fm), t(yq)
