"""Label-shift estimation and posterior adjustment (Saerens, Latinne and Decaestecker EM)."""

from __future__ import annotations

import numpy as np

PRIOR_MIN = 0.001
PRIOR_MAX = 0.999


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def adjust_posterior(p: np.ndarray, pi_s: float, pi_t: float) -> np.ndarray:
    """Re-weight source posteriors to the target attack fraction ``pi_t``."""
    p = _clip_prob(p)
    a = pi_t / pi_s
    b = (1.0 - pi_t) / (1.0 - pi_s)
    return (a * p) / (a * p + b * (1.0 - p))


def estimate_prior(p: np.ndarray, pi_s: float, max_iter: int = 100, tol: float = 1e-6) -> float:
    """Estimate the target attack fraction from unlabelled calibrated posteriors ``p``."""
    p = _clip_prob(p)
    pi_t = float(np.clip(pi_s, PRIOR_MIN, PRIOR_MAX))
    for _ in range(max_iter):
        adjusted = adjust_posterior(p, pi_s, pi_t)
        new = float(np.clip(adjusted.mean(), PRIOR_MIN, PRIOR_MAX))
        done = abs(new - pi_t) < tol
        pi_t = new
        if done:
            break
    return pi_t
