"""Label-selection strategies for a target dataset with an unlabelled candidate pool.

Every strategy gets the rank-normalised pool ``Z`` (n, F), a budget ``k``, a numpy Generator and an ``oracle``
callable that returns the labels of the requested pool rows (this is what the analyst provides). It returns the
selected pool indices. Strategies that query in rounds call the oracle once per round.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def _medoids(Z: np.ndarray, cand: np.ndarray, m: int, rng) -> np.ndarray:
    """m medoids (pool indices) of the candidate rows ``cand`` under k-means."""
    m = min(m, len(cand))
    km = KMeans(m, n_init=1, random_state=int(rng.integers(1 << 30))).fit(Z[cand])
    out = []
    for c in range(m):
        idx = np.flatnonzero(km.labels_ == c)
        if len(idx):
            out.append(cand[idx[np.argmin(((Z[cand[idx]] - km.cluster_centers_[c]) ** 2).sum(1))]])
    return np.asarray(out, dtype=int)


def _top_up(sel, n, k, rng):
    sel = np.unique(sel)
    if len(sel) < k:
        rest = np.setdiff1d(np.arange(n), sel)
        sel = np.concatenate([sel, rng.choice(rest, k - len(sel), replace=False)])
    return sel[:k]


def random_sel(Z, k, rng, oracle, ctx=None):
    return rng.choice(len(Z), k, replace=False)


def kmedoid(Z, k, rng, oracle, ctx=None):
    return _top_up(_medoids(Z, np.arange(len(Z)), k, rng), len(Z), k, rng)


def anomaly(Z, k, rng, oracle, ctx=None):
    """Attack-seeking: the k most anomalous rows (isolation forest), no diversity."""
    iso = IsolationForest(n_estimators=100, random_state=int(rng.integers(1 << 30))).fit(Z)
    return np.argsort(iso.score_samples(Z))[:k]


def _uncertainty(Z, lab, y, rng, trees=100):
    rf = RandomForestClassifier(trees, random_state=int(rng.integers(1 << 30)), n_jobs=1).fit(Z[lab], y)
    p = rf.predict_proba(Z)[:, 1]
    return -np.abs(p - 0.5)


def uncertainty(Z, k, rng, oracle, ctx=None, rounds=4):
    """Plain batch active learning: random seed batch, then uncertainty-ranked batches."""
    n = len(Z)
    b = max(1, k // rounds)
    lab = rng.choice(n, b, replace=False)
    y = oracle(lab)
    while len(lab) < k:
        take = min(b, k - len(lab))
        if len(np.unique(y)) < 2:
            iso = IsolationForest(n_estimators=100, random_state=int(rng.integers(1 << 30))).fit(Z)
            score = -iso.score_samples(Z)
        else:
            score = _uncertainty(Z, lab, y, rng)
        score[lab] = -np.inf
        new = np.argsort(-score)[:take]
        lab = np.concatenate([lab, new])
        y = np.concatenate([y, oracle(new)])
    return lab


def coverfocus(Z, k, rng, oracle, ctx=None, frac_cover=0.5, top=0.25):
    """Cover, then focus: half the budget on k-means medoids (coverage of traffic modes), the other half on medoids
    among the most uncertain rows under a forest fit to the labels so far (anomaly score if one class only)."""
    n = len(Z)
    k1 = max(2, int(round(k * frac_cover)))
    lab = _top_up(_medoids(Z, np.arange(n), k1, rng), n, k1, rng)
    y = oracle(lab)
    k2 = k - len(lab)
    if k2 <= 0:
        return lab
    if len(np.unique(y)) < 2:
        iso = IsolationForest(n_estimators=100, random_state=int(rng.integers(1 << 30))).fit(Z)
        score = -iso.score_samples(Z)
    else:
        score = _uncertainty(Z, lab, y, rng)
    score[lab] = -np.inf
    order = np.argsort(-score)
    cand = order[: max(k2 * 4, int(top * n))]
    cand = cand[np.isfinite(score[cand])]
    new = _medoids(Z, cand, k2, rng)
    return _top_up(np.concatenate([lab, new]), n, k, rng)


def _src_z(ctx):
    """Source score turned into a Gaussian rank over the pool (robust to the score's scale)."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    u = (rankdata(ctx["src"]) - 0.5) / len(ctx["src"])
    return ndtri(u).astype(np.float32)


def srcaug(Z, k, rng, oracle, ctx=None, lam=1.0):
    """K-medoids in the feature space augmented with the source model's attack score.

    The score enters as one extra coordinate scaled by ``lam * sqrt(F)`` so that it carries a share of the distance
    comparable to the whole feature vector. No target labels are used to build the score."""
    F = Z.shape[1]
    Za = np.concatenate([Z, (lam * np.sqrt(F) * _src_z(ctx))[:, None]], axis=1)
    return _top_up(_medoids(Za, np.arange(len(Z)), k, rng), len(Z), k, rng)


def srcstrat(Z, k, rng, oracle, ctx=None):
    """K-medoids inside two strata of the source score (upper and lower half), half of the budget each."""
    order = np.argsort(ctx["src"])
    lo, hi = order[: len(Z) // 2], order[len(Z) // 2 :]
    k_hi = k // 2
    sel = np.concatenate([_medoids(Z, hi, k_hi, rng), _medoids(Z, lo, k - k_hi, rng)])
    return _top_up(sel, len(Z), k, rng)


STRATEGIES = {
    "random": random_sel,
    "kmedoid": kmedoid,
    "anomaly": anomaly,
    "uncertainty": uncertainty,
    "coverfocus": coverfocus,
    "srcaug": srcaug,
    "srcaug-l05": lambda Z, k, rng, oracle, ctx=None: srcaug(Z, k, rng, oracle, ctx, lam=0.5),
    "srcaug-l2": lambda Z, k, rng, oracle, ctx=None: srcaug(Z, k, rng, oracle, ctx, lam=2.0),
    "srcstrat": srcstrat,
}
