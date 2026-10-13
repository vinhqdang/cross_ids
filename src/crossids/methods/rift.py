"""RIFT: rank-invariant flow transfer with ablation switches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..labelshift import adjust_posterior, estimate_prior
from ..normalize import DomainNormalizer
from ..selection import keep_all, select_ks, select_multi_source
from .base import Domain, Method, pool_sources
from .baselines import make_hgb


@dataclass(frozen=True)
class RiftConfig:
    """Component switches. Defaults give the full RIFT method."""

    dn: str = "rank"
    anchor: str = "benign"
    ifs: bool = True
    lsc: bool = True
    tau: float = 0.05
    drop_frac: float = 0.25
    min_features: int = 6
    n_anchor_iter: int = 2
    cal_frac: float = 0.2


class Rift(Method):
    """Domain-wise normalisation, invariance-driven selection, HGB, calibration and label-shift correction."""

    name = "rift"
    transductive = True

    def __init__(self, config: RiftConfig | None = None, seed: int = 0, name: str = "rift"):
        super().__init__(seed)
        self.config = config or RiftConfig()
        self.name = name
        self.keep: np.ndarray = np.array([], dtype=int)
        self.model = None
        self.calibrator: IsotonicRegression | None = None
        self.global_scaler: StandardScaler | None = None
        self.target_norm: DomainNormalizer | None = None
        self.pi_s = float("nan")
        self.pi_t: float | None = None

    def fit(self, sources: Sequence[Domain], target_pool: np.ndarray | None = None) -> "Rift":
        cfg = self.config
        srcs = list(sources)
        X_all, y_all = pool_sources(srcs)
        d = X_all.shape[1]
        transductive = target_pool is not None
        pool = np.asarray(target_pool, dtype=np.float64) if transductive else None

        Z_list, anchor_rows = self._normalise_sources(srcs)
        if cfg.dn == "global":
            Zp = self.global_scaler.transform(pool) if transductive else None
        else:
            Zp = None
            if transductive:
                self.target_norm = DomainNormalizer(cfg.dn).fit(pool)
                Zp = self.target_norm.transform(pool)
            else:
                pooled = np.vstack(anchor_rows)
                self.target_norm = DomainNormalizer(cfg.dn).fit(pooled)

        self.keep = self._select(srcs, Z_list, Zp, d)
        self.info["n_features"] = float(len(self.keep))
        self._fit_classifier(Z_list, y_all)

        if transductive and cfg.dn != "global" and cfg.anchor == "benign":
            self._refine_target_anchor(pool)

        if transductive and cfg.lsc:
            p_pool = self._score(self._target_Z(pool))
            self.pi_t = estimate_prior(p_pool, self.pi_s)
            self.info["pi_t"] = self.pi_t
        self.info["pi_s"] = self.pi_s
        self.info["transductive"] = float(transductive)
        return self

    def _normalise_sources(self, srcs: list[Domain]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-domain normalisation of sources; returns normalised matrices and the anchor rows."""
        cfg = self.config
        if cfg.dn == "global":
            X_all = np.vstack([s.X for s in srcs])
            self.global_scaler = StandardScaler().fit(X_all)
            Z_list = [self.global_scaler.transform(s.X) for s in srcs]
            return Z_list, [s.X for s in srcs]
        Z_list: list[np.ndarray] = []
        anchor_rows: list[np.ndarray] = []
        for s in srcs:
            mask = (s.y == 0) if cfg.anchor == "benign" else None
            if mask is not None and mask.sum() < 2:
                self.info["anchor_fallback"] = self.info.get("anchor_fallback", 0.0) + 1.0
                mask = None
            norm = DomainNormalizer(cfg.dn).fit(s.X, mask)
            Z_list.append(norm.transform(s.X))
            anchor_rows.append(s.X[mask] if mask is not None else s.X)
        return Z_list, anchor_rows

    def _select(self, srcs: list[Domain], Z_list: list[np.ndarray], Zp: np.ndarray | None, d: int) -> np.ndarray:
        cfg = self.config
        if not cfg.ifs:
            return keep_all(d).keep
        if len(srcs) >= 2:
            sel = select_multi_source(Z_list, [s.y for s in srcs], cfg.tau, cfg.min_features)
            self.info["ifs_min_weight"] = float(sel.weights[sel.keep].min()) if sel.keep.size else 0.0
            return sel.keep
        if Zp is not None:
            sel = select_ks(Z_list[0], Zp, cfg.drop_frac, cfg.min_features)
            return sel.keep
        return keep_all(d).keep

    def _fit_classifier(self, Z_list: list[np.ndarray], y_all: np.ndarray) -> None:
        Zs = np.vstack([Z[:, self.keep] for Z in Z_list])
        Z_fit, Z_cal, y_fit, y_cal = train_test_split(
            Zs, y_all, test_size=self.config.cal_frac, stratify=y_all, random_state=self.seed
        )
        self.model = make_hgb(self.seed).fit(Z_fit, y_fit)
        raw = self.model.predict_proba(Z_cal)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y_cal)
        self.pi_s = float(np.mean(y_cal))

    def _refine_target_anchor(self, pool: np.ndarray) -> None:
        """Re-fit the target normaliser on pseudo-benign pool rows (score below the median)."""
        norm = self.target_norm
        for _ in range(self.config.n_anchor_iter):
            scores = self._score(norm.transform(pool)[:, self.keep])
            mask = scores < np.median(scores)
            if mask.sum() < 2:
                break
            norm = DomainNormalizer(self.config.dn).fit(pool, mask)
        self.target_norm = norm

    def _target_Z(self, X: np.ndarray) -> np.ndarray:
        if self.config.dn == "global":
            return self.global_scaler.transform(X)[:, self.keep]
        return self.target_norm.transform(X)[:, self.keep]

    def _score(self, Z: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(Z)[:, 1]
        return self.calibrator.predict(raw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.calibrator is None:
            raise RuntimeError("method is not fitted")
        p = self._score(self._target_Z(np.asarray(X, dtype=np.float64)))
        if self.pi_t is not None:
            p = adjust_posterior(p, self.pi_s, self.pi_t)
        return p
