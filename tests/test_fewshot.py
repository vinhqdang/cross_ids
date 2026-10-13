"""Few-shot reference helpers: stratified labelled budgets and the Isolation Forest shift score."""

import numpy as np

from crossids.fewshot import labelled_budget, shift_score


def test_budget_is_seeded_stratified_and_keeps_both_classes():
    y = np.array([0] * 190 + [1] * 20)
    for k in (0.1, 1, 5):
        idx = labelled_budget(y, k, seed=0)
        counts = np.bincount(y[idx], minlength=2)
        assert counts.min() >= 1
        assert len(idx) == max(2, int(round(len(y) * k / 100.0)))
        assert np.array_equal(idx, labelled_budget(y, k, seed=0))
    big = labelled_budget(y, 50, seed=1)
    assert abs(np.mean(y[big]) - y.mean()) < 0.05


def test_budget_returns_everything_when_k_covers_the_split():
    y = np.array([0, 1, 0, 1])
    assert list(labelled_budget(y, 100, seed=0)) == [0, 1, 2, 3]


def test_shift_score_is_higher_for_a_displaced_target():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(3000, 4))
    same = rng.normal(size=(1000, 4))
    moved = rng.normal(size=(1000, 4)) + 4.0
    assert shift_score(source, moved, seed=0) > shift_score(source, same, seed=0) + 0.05
