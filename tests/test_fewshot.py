"""Few-shot reference helpers: stratified labelled budgets and the Isolation Forest shift score."""

import numpy as np

from crossids.fewshot import labelled_budget, shift_score, targetonly_iterations, targetonly_scores


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


def test_targetonly_control_fits_tiny_and_small_budgets_without_crashing():
    rng = np.random.default_rng(1)
    y = np.array([0] * 180 + [1] * 20)
    X = rng.normal(size=(200, 5)) + y[:, None] * 1.5
    test_X = rng.normal(size=(60, 5)) + 1.5
    # k=0.1 gives 2 rows (one per class), k=5 gives 10 rows, k=50 gives 100 rows
    scores = targetonly_scores((X, y), test_X, [0.1, 5, 50], seed=0, threads=1, device="cpu")
    assert set(scores) == {0.1, 5, 50}
    for vals in scores.values():
        assert vals.shape == (60,)
        assert np.all((vals >= 0) & (vals <= 1))
    again = targetonly_scores((X, y), test_X, [5], seed=0, threads=1, device="cpu")
    assert np.allclose(again[5], scores[5])


def test_targetonly_iterations_use_fixed_budget_for_tiny_splits():
    assert targetonly_iterations(2) == 300
    assert targetonly_iterations(19) == 300
    assert targetonly_iterations(20) == 100
