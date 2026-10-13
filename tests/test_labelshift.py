"""Saerens EM prior estimation and posterior adjustment."""

import numpy as np
from scipy.stats import norm

from crossids.labelshift import adjust_posterior, estimate_prior


def _posterior(x: np.ndarray, pi: float, mu1: float = 2.0) -> np.ndarray:
    l1 = pi * norm.pdf(x, loc=mu1)
    l0 = (1.0 - pi) * norm.pdf(x, loc=0.0)
    return l1 / (l1 + l0)


def test_em_recovers_known_target_prior_within_tolerance():
    rng = np.random.default_rng(0)
    pi_s, pi_t = 0.3, 0.7
    n = 20000
    y = (rng.random(n) < pi_t).astype(int)
    x = rng.normal(loc=2.0 * y, scale=1.0)
    p = _posterior(x, pi_s)
    estimate = estimate_prior(p, pi_s)
    assert abs(estimate - pi_t) < 0.05


def test_em_is_identity_when_target_matches_source():
    rng = np.random.default_rng(1)
    pi = 0.4
    y = (rng.random(5000) < pi).astype(int)
    x = rng.normal(loc=2.0 * y, scale=1.0)
    p = _posterior(x, pi)
    assert abs(estimate_prior(p, pi) - pi) < 0.03


def test_adjust_posterior_matches_the_formula_and_identity_case():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(adjust_posterior(p, 0.3, 0.3), p)
    out = adjust_posterior(p, 0.3, 0.6)
    a = 0.6 / 0.3
    b = 0.4 / 0.7
    assert np.allclose(out, a * p / (a * p + b * (1 - p)))
    assert np.all(out > p)


def test_adjust_posterior_extremes_stay_fixed_or_saturate():
    p = np.array([0.0, 1.0])
    # Same prior: the posterior is returned unchanged even at the extremes.
    assert np.allclose(adjust_posterior(p, 0.5, 0.5), [1e-6, 1 - 1e-6], atol=1e-9)
    # Any target prior keeps the output in [0, 1] and orders the two extremes.
    for pi_t in (0.001, 0.2, 0.9, 0.999):
        out = adjust_posterior(p, 0.5, pi_t)
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert out[0] < out[1]
    # Raising the target prior raises the adjusted posterior of an uncertain row, and lowering it lowers it.
    mid = np.array([0.5])
    assert adjust_posterior(mid, 0.5, 0.001)[0] < 0.01
    assert adjust_posterior(mid, 0.5, 0.999)[0] > 0.99


def test_estimate_prior_is_clipped_for_degenerate_posteriors():
    assert estimate_prior(np.ones(100), 0.5) <= 0.999 + 1e-12
    assert estimate_prior(np.zeros(100), 0.5) >= 0.001 - 1e-12
