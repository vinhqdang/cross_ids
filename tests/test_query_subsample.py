"""Stratified, deterministic subsampling of the query rows (TabPFN evaluations need not score every test row)."""
import numpy as np

from crossids.flare.evaluate import stratified_subsample


def test_returns_all_rows_when_under_the_cap():
    y = np.array([0, 1, 0, 1, 1])
    np.testing.assert_array_equal(stratified_subsample(y, 10, seed=0), np.arange(5))


def test_keeps_class_proportions_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.1).astype(int)
    a = stratified_subsample(y, 2000, seed=3)
    b = stratified_subsample(y, 2000, seed=3)
    c = stratified_subsample(y, 2000, seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert abs(len(a) - 2000) <= 2 and len(np.unique(a)) == len(a)
    assert abs(y[a].mean() - y.mean()) < 0.005
    assert np.all(np.diff(a) > 0)  # sorted, so row order is preserved


def test_both_classes_survive_even_when_one_is_rare():
    y = np.zeros(10000, dtype=int)
    y[:20] = 1
    s = stratified_subsample(y, 500, seed=0)
    assert y[s].sum() >= 1
