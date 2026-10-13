"""CORAL: second-order alignment of source features to the unlabelled target pool, then HGB."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import Domain, Method, pool_sources
from .baselines import make_hgb


def _sym_power(C: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(C)
    values = np.maximum(values, 1e-12)
    return (vectors * values**power) @ vectors.T


def coral_align(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """CORAL (Sun et al., 2016): whiten with C_s + I, re-colour with C_t + I, no mean shift."""
    d = Xs.shape[1]
    Cs = np.atleast_2d(np.cov(Xs, rowvar=False)) + np.eye(d)
    Ct = np.atleast_2d(np.cov(Xt, rowvar=False)) + np.eye(d)
    transform = _sym_power(Cs, -0.5) @ _sym_power(Ct, 0.5)
    return Xs @ transform


class Coral(Method):
    """Standardised features, CORAL-aligned sources (when a target pool is given), then HGB."""

    name = "coral"
    transductive = True
    supports_inductive = False

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.scaler: StandardScaler | None = None
        self.model = None

    def fit(self, sources: Sequence[Domain], target_pool: np.ndarray | None = None) -> "Coral":
        X, y = pool_sources(sources)
        self.scaler = StandardScaler().fit(X)
        Zs = self.scaler.transform(X)
        aligned = False
        if target_pool is not None:
            Zt = self.scaler.transform(np.asarray(target_pool, dtype=np.float64))
            Zs = coral_align(Zs, Zt)
            aligned = True
        self.info["aligned"] = float(aligned)
        self.model = make_hgb(self.seed).fit(Zs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("method is not fitted")
        return self.model.predict_proba(self.scaler.transform(np.asarray(X, dtype=np.float64)))[:, 1]
