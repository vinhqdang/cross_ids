"""Invariance-driven feature selection (IFS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, rankdata


def signed_strength(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature ROC-AUC minus 0.5, computed from ranks."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.zeros(Z.shape[1])
    ranks = rankdata(Z, axis=0)
    auc = (ranks[y].sum(axis=0) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc - 0.5


@dataclass
class Selection:
    """Result of feature selection: kept indices and diagnostics."""

    keep: np.ndarray
    weights: np.ndarray
    strength: np.ndarray


def select_multi_source(
    domain_Z: list[np.ndarray],
    domain_y: list[np.ndarray],
    tau: float = 0.05,
    min_features: int = 6,
) -> Selection:
    """Keep features whose signed AUC strength has one sign and magnitude >= tau in every source."""
    S = np.stack([signed_strength(Z, y) for Z, y in zip(domain_Z, domain_y)])
    abs_s = np.abs(S)
    same_sign = np.all(S > 0, axis=0) | np.all(S < 0, axis=0)
    min_abs = abs_s.min(axis=0)
    max_abs = np.maximum(abs_s.max(axis=0), 1e-12)
    keep_mask = same_sign & (min_abs >= tau)
    keep = np.flatnonzero(keep_mask)
    if keep.size < min(min_features, S.shape[1]):
        order = np.argsort(-min_abs)
        keep = np.sort(order[: min(min_features, S.shape[1])])
    return Selection(keep=keep, weights=min_abs / max_abs, strength=S)


def select_ks(source_Z: np.ndarray, target_Z: np.ndarray, drop_frac: float = 0.25, min_features: int = 6) -> Selection:
    """Drop the features with the largest source-target KS statistic (never below min_features)."""
    d = source_Z.shape[1]
    ks = np.array([ks_2samp(source_Z[:, j], target_Z[:, j], method="asymp").statistic for j in range(d)])
    n_drop = int(np.floor(drop_frac * d))
    n_keep = max(d - n_drop, min(min_features, d))
    keep = np.sort(np.argsort(ks, kind="stable")[:n_keep])
    return Selection(keep=keep, weights=1.0 - ks, strength=ks)


def keep_all(d: int) -> Selection:
    """Selection that keeps every feature."""
    return Selection(keep=np.arange(d), weights=np.ones(d), strength=np.zeros(d))
