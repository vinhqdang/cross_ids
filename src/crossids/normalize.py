"""Per-domain feature normalisation used by RIFT and its ablations."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

KINDS = ("rank", "robust", "standard")


class DomainNormalizer:
    """Label-free normaliser fitted on one domain's feature matrix.

    ``rank`` maps each feature through its empirical CDF and the standard normal
    quantile function, evaluated by interpolation over stored quantiles.
    ``robust`` uses median and IQR, ``standard`` uses mean and standard deviation.
    """

    def __init__(self, kind: str = "rank", n_quantiles: int = 1000, eps: float = 1e-12):
        if kind not in KINDS:
            raise ValueError(f"unknown normaliser kind {kind!r}")
        self.kind = kind
        self.n_quantiles = n_quantiles
        self.eps = eps
        self._grids: list[np.ndarray] = []
        self._mids: list[np.ndarray] = []
        self._tail_scale: np.ndarray | None = None
        self._loc: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, X: np.ndarray, mask: np.ndarray | None = None) -> "DomainNormalizer":
        """Fit on the rows of ``X`` selected by ``mask`` (all rows when None)."""
        Xf = np.asarray(X, dtype=np.float64)
        if mask is not None and np.any(mask):
            Xf = Xf[np.asarray(mask, dtype=bool)]
        if Xf.shape[0] == 0:
            raise ValueError("cannot fit a normaliser on zero rows")
        if self.kind == "rank":
            self._fit_rank(Xf)
        elif self.kind == "robust":
            q25, med, q75 = np.percentile(Xf, [25, 50, 75], axis=0)
            self._loc = med
            self._scale = np.maximum(q75 - q25, 1e-6)
        else:
            self._loc = Xf.mean(axis=0)
            self._scale = np.maximum(Xf.std(axis=0), 1e-6)
        return self

    def _fit_rank(self, Xf: np.ndarray) -> None:
        n = Xf.shape[0]
        probs = np.linspace(0.0, 1.0, self.n_quantiles)
        self._grids = []
        self._mids = []
        self._tail_scale = np.maximum(Xf.std(axis=0), 1e-6)
        for j in range(Xf.shape[1]):
            col = np.sort(Xf[:, j])
            grid = np.unique(np.quantile(col, probs))
            left = np.searchsorted(col, grid, side="left") / n
            right = np.searchsorted(col, grid, side="right") / n
            self._grids.append(grid)
            self._mids.append((left + right) / 2.0)

    def _rank_gauss(self, x: np.ndarray, j: int) -> np.ndarray:
        """Gaussian rank score of ``x`` for feature ``j``, extended linearly beyond the fitted range."""
        grid, mids = self._grids[j], self._mids[j]
        scale = self._tail_scale[j]
        z = ndtri(np.clip(np.interp(x, grid, mids), self.eps, 1.0 - self.eps))
        z_lo = ndtri(np.clip(mids[0], self.eps, 1.0 - self.eps))
        z_hi = ndtri(np.clip(mids[-1], self.eps, 1.0 - self.eps))
        z = np.where(x > grid[-1], z_hi + (x - grid[-1]) / scale, z)
        z = np.where(x < grid[0], z_lo + (x - grid[0]) / scale, z)
        return z

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalise rows of ``X`` with the statistics fitted by :meth:`fit`."""
        Xf = np.asarray(X, dtype=np.float64)
        if self.kind == "rank":
            if not self._grids:
                raise RuntimeError("normaliser is not fitted")
            out = np.empty_like(Xf)
            for j in range(Xf.shape[1]):
                out[:, j] = self._rank_gauss(Xf[:, j], j)
            return out
        if self._loc is None or self._scale is None:
            raise RuntimeError("normaliser is not fitted")
        return (Xf - self._loc) / self._scale
