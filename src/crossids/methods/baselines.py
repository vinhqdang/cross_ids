"""Supervised baselines on globally standardised Common Flow Schema features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import Domain, Method, pool_sources


def make_hgb(seed: int) -> HistGradientBoostingClassifier:
    """Histogram gradient boosting used by the baselines and by RIFT."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced",
        random_state=seed,
    )


class ScaledBaseline(Method):
    """Global StandardScaler fitted on the sources, then a sklearn-style estimator."""

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.scaler: StandardScaler | None = None
        self.model = None

    def _build(self):
        raise NotImplementedError

    def fit(self, sources: Sequence[Domain], target_pool: np.ndarray | None = None) -> "ScaledBaseline":
        X, y = pool_sources(sources)
        self.scaler = StandardScaler().fit(X)
        self.model = self._build().fit(self.scaler.transform(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("method is not fitted")
        return self.model.predict_proba(self.scaler.transform(np.asarray(X, dtype=np.float64)))[:, 1]


class LogisticBaseline(ScaledBaseline):
    name = "lr"

    def _build(self):
        return LogisticRegression(class_weight="balanced", max_iter=2000)


class RandomForestBaseline(ScaledBaseline):
    name = "rf"

    def _build(self):
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.seed,
        )


class HgbBaseline(ScaledBaseline):
    name = "hgb"

    def _build(self):
        return make_hgb(self.seed)


class MlpBaseline(ScaledBaseline):
    name = "mlp"

    def _build(self):
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            early_stopping=True,
            max_iter=200,
            random_state=self.seed,
        )


class CatBoostBaseline(ScaledBaseline):
    """CatBoost with balanced class weights and early stopping on a 10% source split."""

    name = "catboost"

    def fit(self, sources: Sequence[Domain], target_pool: np.ndarray | None = None) -> "CatBoostBaseline":
        from catboost import CatBoostClassifier

        X, y = pool_sources(sources)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        Z_tr, Z_va, y_tr, y_va = train_test_split(Z, y, test_size=0.1, stratify=y, random_state=self.seed)
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            auto_class_weights="Balanced",
            thread_count=-1,
            verbose=0,
            random_seed=self.seed,
            od_type="Iter",
            od_wait=50,
        )
        self.model.fit(Z_tr, y_tr, eval_set=(Z_va, y_va), use_best_model=True)
        return self


BASELINE_CLASSES = {
    "lr": LogisticBaseline,
    "rf": RandomForestBaseline,
    "hgb": HgbBaseline,
    "mlp": MlpBaseline,
    "catboost": CatBoostBaseline,
}
