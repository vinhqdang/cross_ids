"""Method interface shared by baselines, RIFT and its ablations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class MethodUnavailable(RuntimeError):
    """Raised when a method cannot run in this environment (for example torch is missing)."""


@dataclass
class Domain:
    """One labelled source domain."""

    name: str
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)


def pool_sources(sources: Sequence[Domain]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the feature matrices and labels of the source domains."""
    if not sources:
        raise ValueError("at least one source domain is required")
    X = np.vstack([np.asarray(s.X, dtype=np.float64) for s in sources])
    y = np.concatenate([np.asarray(s.y).astype(int) for s in sources])
    return X, y


class Method(ABC):
    """A cross-domain detector. Target labels are never passed to ``fit``."""

    name: str = "method"
    transductive: bool = False

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.info: dict[str, float] = {}

    @abstractmethod
    def fit(self, sources: Sequence[Domain], target_pool: np.ndarray | None = None) -> "Method":
        """Fit on labelled sources and, for transductive methods, unlabelled target features."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for the rows of ``X``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard decision at threshold 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)
