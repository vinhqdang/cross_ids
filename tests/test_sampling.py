"""Distinct-row sampling: duplicate-insensitive priors, bottom-k sketches and short-class shrinkage."""

import numpy as np
import pandas as pd

from crossids.sampling import DistinctSampler, plan_counts, row_hashes

COLS = ("a", "b", "c")


def _frame(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=list(COLS))
    frame["label"] = labels.astype(np.int8)
    frame["attack_type"] = np.where(labels == 1, "attack", "benign")
    frame["dataset"] = "syn"
    return frame


def _stats(frame: pd.DataFrame) -> dict:
    return {"raw": len(frame), "unlabelled": 0, "dropped_nan_by_label": {0: 0, 1: 0}}


def test_duplicated_attacks_do_not_inflate_the_prior():
    rng = np.random.default_rng(0)
    benign = rng.normal(size=(1000, 3))
    attacks = rng.normal(size=(300, 3)) + 3.0
    repeats = attacks[rng.integers(0, 300, size=8100)]
    features = np.vstack([benign, attacks, repeats])
    labels = np.r_[np.zeros(1000), np.ones(300 + 8100)]
    frame = _frame(features, labels)

    sampler = DistinctSampler(max_rows=5000, features=COLS, seed=0)
    sampler.add(frame, _stats(frame))
    table, meta = sampler.finalize()

    assert np.isclose(meta["prior_raw"], 0.9, atol=0.01)
    assert np.isclose(meta["prior_distinct"], 300 / 1300)
    assert meta["distinct_est_per_label"]["1"] == 300
    assert meta["distinct_est_per_label"]["0"] == 1000
    n = len(table)
    assert np.isclose(meta["prior_cached"], meta["prior_distinct"], atol=1.0 / n)
    assert table.duplicated(subset=[*COLS, "label"]).sum() == 0
    assert int(table["label"].sum()) == 300


def test_bottom_k_sketch_is_capped_and_estimates_distinct_rows():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(5000, 3))
    labels = np.zeros(5000)
    frame = _frame(features, labels)
    sampler = DistinctSampler(max_rows=200, features=COLS, seed=3)
    sampler.add(frame, _stats(frame))
    assert len(sampler.sketches[0]) == 200
    estimate = sampler.distinct()[0]
    assert abs(estimate - 5000) / 5000 < 0.25
    table, meta = sampler.finalize()
    assert len(table) == 200 and meta["rows_cached"] == 200


def test_short_attack_class_shrinks_n_instead_of_back_filling():
    # 5 benign and 90 attack distinct rows, 100 requested: 10 attacks cannot be drawn, so n shrinks.
    assert plan_counts(5, 90, 5, 90, 100) == (5, 90)
    n0, n1 = plan_counts(5, 90, 5, 90, 100)
    assert n0 <= 5 and n1 <= 90
    assert n0 + n1 < 100


def test_same_features_with_different_labels_are_kept_as_distinct_rows():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    keys = row_hashes(x, np.array([0, 1]), seed=0)
    assert keys[0] != keys[1]
    assert row_hashes(x[:1], np.array([0]), seed=0)[0] == keys[0]
    assert row_hashes(x[:1], np.array([0]), seed=1)[0] != keys[0]


def test_dropped_nan_counts_reach_the_metadata():
    frame = _frame(np.arange(12, dtype=float).reshape(4, 3), np.array([0, 0, 1, 1]))
    sampler = DistinctSampler(max_rows=10, features=COLS, seed=0)
    sampler.add(frame, {"raw": 7, "unlabelled": 1, "dropped_nan_by_label": {0: 2, 1: 0}})
    _, meta = sampler.finalize()
    assert meta["rows_raw"] == 7
    assert meta["unlabelled_dropped"] == 1
    assert meta["dropped_nan_per_label"] == {"0": 2, "1": 0}
    assert meta["valid_per_label"] == {"0": 2, "1": 2}
