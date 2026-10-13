"""Invariance-driven feature selection: sign consistency across sources and KS dropping."""

import numpy as np

from crossids.selection import keep_all, select_ks, select_multi_source, signed_strength


def _domain(rng, n, informative_sign, weak=False):
    y = rng.integers(0, 2, size=n)
    f0 = rng.normal(size=n) + informative_sign[0] * 2.0 * y
    f1 = rng.normal(size=n) + informative_sign[1] * 2.0 * y
    f2 = rng.normal(size=n) + (0.05 if weak else 0.0) * y
    return np.column_stack([f0, f1, f2]), y


def test_signed_strength_matches_auc_direction():
    rng = np.random.default_rng(0)
    X, y = _domain(rng, 4000, (1.0, -1.0))
    s = signed_strength(X, y)
    assert s[0] > 0.3
    assert s[1] < -0.3
    assert abs(s[2]) < 0.05


def test_multi_source_keeps_only_sign_consistent_strong_features():
    rng = np.random.default_rng(1)
    X1, y1 = _domain(rng, 4000, (1.0, 1.0), weak=True)
    X2, y2 = _domain(rng, 4000, (1.0, -1.0), weak=True)
    sel = select_multi_source([X1, X2], [y1, y2], tau=0.05, min_features=1)
    assert list(sel.keep) == [0]
    assert 0.0 < sel.weights[0] <= 1.0


def test_multi_source_tops_up_to_min_features():
    rng = np.random.default_rng(2)
    X1, y1 = _domain(rng, 2000, (1.0, -1.0))
    X2, y2 = _domain(rng, 2000, (1.0, 1.0))
    sel = select_multi_source([X1, X2], [y1, y2], tau=0.05, min_features=2)
    assert sel.keep.size == 2


def test_ks_drops_most_shifted_features_and_respects_minimum():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(3000, 8))
    target = source.copy()
    target[:, 0] += 3.0
    target[:, 1] += 2.0
    sel = select_ks(source, target, drop_frac=0.25, min_features=6)
    assert 0 not in sel.keep and 1 not in sel.keep
    assert sel.keep.size == 6
    sel_small = select_ks(source, target, drop_frac=0.9, min_features=6)
    assert sel_small.keep.size == 6


def test_keep_all_returns_every_index():
    assert list(keep_all(4).keep) == [0, 1, 2, 3]
