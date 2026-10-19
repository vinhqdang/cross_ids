"""Rank normalisation against a limited reference set (control for transductive normalisation)."""
import numpy as np
import pytest

from crossids.flare.data import gauss_rank, gauss_rank_ref


def test_reference_equal_to_data_reproduces_gauss_rank():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    np.testing.assert_allclose(gauss_rank_ref(X, X), gauss_rank(X))


def test_reference_rank_is_monotone_and_clipped():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(20, 2))
    X = np.vstack([np.full((1, 2), -1e9), np.sort(rng.normal(size=(30, 2)), axis=0), np.full((1, 2), 1e9)])
    Z = gauss_rank_ref(ref, X)
    assert np.all(np.diff(Z, axis=0) >= 0)
    assert Z.min() >= -4 and Z.max() <= 4


def test_nan_stays_nan_and_empty_reference_column_is_nan():
    ref = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    X = np.array([[2.0, 5.0], [np.nan, 1.0]])
    Z = gauss_rank_ref(ref, X)
    assert np.isfinite(Z[0, 0]) and np.isnan(Z[1, 0])
    assert np.isnan(Z[:, 1]).all()


def test_query_values_do_not_change_other_rows_when_reference_is_fixed():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(40, 3))
    a = rng.normal(size=(5, 3))
    b = np.vstack([a, rng.normal(size=(100, 3))])
    np.testing.assert_allclose(gauss_rank_ref(ref, b)[:5], gauss_rank_ref(ref, a))


def test_support_only_reference_makes_a_query_independent_of_the_other_queries():
    import torch
    from test_flare2 import _tiny_flare2

    torch.manual_seed(0)
    model = _tiny_flare2(0).eval()
    model.rank_query_cap = 0
    rng = np.random.default_rng(3)
    d = 8
    X = rng.normal(size=(12, d))
    y = np.repeat([0, 1], 3)
    Q = rng.normal(size=(10, d))
    alone = model.predict_proba(X[:6], y, Q[:1], seed=0)
    batch = model.predict_proba(X[:6], y, Q, seed=0)
    np.testing.assert_allclose(alone, batch[:1], rtol=1e-4, atol=1e-5)
