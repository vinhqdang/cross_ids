"""Active cold-start selection of which target flows to label (DESIGN.md §15).

Half of the budget is chosen label-free: greedy k-center on the embeddings the encoder produces for the pool with an
empty support. The other half is chosen with the first half's labels as support, scoring each remaining pool row by
predictive entropy times its distance to the selected set (batch-mode uncertainty-diversity). Pool labels are only
read for the selected rows, which is what the analyst would label.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .model import EpisodeModel


def kcenter_greedy(points: np.ndarray, n_select: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point selection with a seeded random start. Returns indices in selection order."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    n_select = min(int(n_select), n)
    if n_select <= 0:
        return np.zeros(0, dtype=np.int64)
    first = int(rng.integers(n))
    chosen = [first]
    dist = np.linalg.norm(points - points[first], axis=1)
    while len(chosen) < n_select:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def empty_support_embeddings(model: EpisodeModel, pool_X: np.ndarray) -> np.ndarray:
    """Label-free row embeddings: every pool row is a query and no label is revealed."""
    d = pool_X.shape[1]
    return model.embed(np.zeros((0, d)), np.zeros(0, dtype=np.int64), pool_X)


def select_flare_al(model: EpisodeModel, pool_X: np.ndarray, pool_y: np.ndarray, k: int, *,
                    rng: np.random.Generator, emb0: np.ndarray, ensemble: int | None = None) -> np.ndarray:
    """Pool indices of the k flows to label. ``pool_y`` is read only at the selected indices."""
    k1 = k // 2
    k2 = k - k1
    first = kcenter_greedy(emb0, k1, rng)
    selected = list(first)
    if k2 == 0:
        return np.asarray(selected, dtype=np.int64)
    probs = model.predict_proba(pool_X[first], pool_y[first], pool_X, ensemble=ensemble)
    score_unc = binary_entropy(probs)
    dist = np.min(np.linalg.norm(emb0[:, None, :] - emb0[first][None, :, :], axis=2), axis=1)
    for _ in range(k2):
        score = score_unc * dist
        score[selected] = -np.inf
        nxt = int(np.argmax(score))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(emb0 - emb0[nxt], axis=1))
    return np.asarray(selected, dtype=np.int64)


def select_kcenter_raw(pool_X: np.ndarray, k: int, *, rng: np.random.Generator) -> np.ndarray:
    """Baseline: greedy k-center on standardised raw features of the pool (no model, no labels)."""
    Z = StandardScaler().fit_transform(np.asarray(pool_X, dtype=np.float64))
    return kcenter_greedy(Z, k, rng)
