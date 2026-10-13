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


def _piecewise_linear(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Interpolate on the grid ``xs``; beyond either end, extend linearly with that end's segment slope."""
    y = np.interp(x, xs, ys)
    if xs.size < 2:
        return y
    s_lo = (ys[1] - ys[0]) / (xs[1] - xs[0])
    s_hi = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + (x - xs[0]) * s_lo, y)
    return np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * s_hi, y)


class QuantileTransport:
    """Per-feature monotone map ``x -> F_R^{-1}(F_A(x))`` from a target anchor sample to a reference profile.

    ``fit(A, R)``: ``A`` holds raw target features (the anchor set), ``R`` holds reference rows already in the
    shared normalised space. Both empirical CDFs use ``n_quantiles`` grid points with midpoint plotting
    positions; outside the fitted range each side extends linearly with its end slope.
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self._a: list[tuple[np.ndarray, np.ndarray]] = []
        self._r: list[tuple[np.ndarray, np.ndarray]] = []

    def _grid(self, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = col.shape[0]
        probs = np.linspace(0.0, 1.0, self.n_quantiles)
        sorted_col = np.sort(col)
        grid = np.unique(np.quantile(sorted_col, probs))
        left = np.searchsorted(sorted_col, grid, side="left") / n
        right = np.searchsorted(sorted_col, grid, side="right") / n
        return grid, (left + right) / 2.0

    def fit(self, A: np.ndarray, R: np.ndarray) -> "QuantileTransport":
        A = np.asarray(A, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        if A.shape[0] == 0 or R.shape[0] == 0:
            raise ValueError("cannot fit a quantile transport on zero rows")
        if A.shape[1] != R.shape[1]:
            raise ValueError("anchor and reference must have the same number of features")
        self._a = [self._grid(A[:, j]) for j in range(A.shape[1])]
        self._r = [self._grid(R[:, j]) for j in range(R.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._a:
            raise RuntimeError("quantile transport is not fitted")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            gA, mA = self._a[j]
            gR, mR = self._r[j]
            u = _piecewise_linear(X[:, j], gA, mA)
            out[:, j] = _piecewise_linear(u, mR, gR)
        return out
