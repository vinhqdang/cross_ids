"""Domain normalisers: monotone invariance of the rank transform and anchored fitting."""

import numpy as np
from scipy.stats import spearmanr

from crossids.normalize import DomainNormalizer


def test_rank_transform_is_invariant_to_monotone_transforms():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=3000), rng.gamma(2.0, size=3000)])
    base = DomainNormalizer("rank").fit(X).transform(X)
    for g in (np.exp, lambda v: v**3, lambda v: np.arcsinh(5.0 * v)):
        Y = g(X)
        Z = DomainNormalizer("rank").fit(Y).transform(Y)
        assert np.max(np.abs(Z - base)) < 0.05
        for j in range(X.shape[1]):
            assert spearmanr(Z[:, j], base[:, j]).statistic > 0.999


def test_rank_keeps_order_outside_the_fitted_range():
    rng = np.random.default_rng(1)
    benign = rng.normal(size=(2000, 1))
    norm = DomainNormalizer("rank").fit(benign)
    far = np.linspace(4.0, 50.0, 200).reshape(-1, 1)
    z = norm.transform(far)[:, 0]
    assert np.all(np.diff(z) > 0)


def test_rank_is_label_free_and_mask_selects_rows():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 2))
    X[500:, 0] += 10.0
    mask = np.arange(1000) < 500
    z_masked = DomainNormalizer("rank").fit(X, mask).transform(X[:500])
    z_full = DomainNormalizer("rank").fit(X).transform(X[:500])
    assert np.mean(z_masked[:, 0]) > np.mean(z_full[:, 0])


def test_robust_and_standard_centre_and_scale():
    rng = np.random.default_rng(3)
    X = rng.normal(loc=5.0, scale=2.0, size=(5000, 3))
    Zs = DomainNormalizer("standard").fit(X).transform(X)
    Zr = DomainNormalizer("robust").fit(X).transform(X)
    assert np.allclose(Zs.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(Zs.std(axis=0), 1.0, atol=1e-6)
    assert np.allclose(np.median(Zr, axis=0), 0.0, atol=0.05)
