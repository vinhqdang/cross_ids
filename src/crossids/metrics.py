"""Detection metrics on the target test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

METRIC_NAMES = ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc", "detection_rate", "fpr")


def evaluate(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Threshold-free and threshold-based detection metrics."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    has_both = 0 < y.sum() < y.size
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return {
        "roc_auc": float(roc_auc_score(y, proba)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if has_both else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if has_both else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if has_both else 0.0,
        "detection_rate": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
    }


def per_attack_recall(y: np.ndarray, pred: np.ndarray, attack_type: np.ndarray) -> pd.DataFrame:
    """Recall per raw attack type over the attack rows of the target test split."""
    frame = pd.DataFrame(
        {
            "attack_type": np.asarray(attack_type, dtype=object),
            "detected": np.asarray(pred).astype(int),
            "is_attack": np.asarray(y).astype(int),
        }
    )
    attacks = frame[frame["is_attack"] == 1]
    grouped = attacks.groupby("attack_type", sort=True)["detected"]
    out = pd.DataFrame({"n": grouped.size(), "detected": grouped.sum()}).reset_index()
    out["recall"] = out["detected"] / out["n"]
    return out
