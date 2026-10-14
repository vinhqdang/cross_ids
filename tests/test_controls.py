"""Section 10 controls on synthetic data: concept-shift kappa, separability, oracle selection, resume, tree invariance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score

from crossids.controls import (
    CORE_FEATURES,
    _cohen_kappa,
    concept_row,
    oracle_summary,
    run_concept,
    run_oracle,
    separability_scores,
)


def _rule_data(n, seed, invert=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, (1 - y if invert else y)


def _frame(n, seed, shift=0.0, dataset="x"):
    """Core-feature table with a learnable label, shaped like the processed parquet files."""
    rng = np.random.default_rng(seed)
    X = np.abs(rng.normal(size=(n, len(CORE_FEATURES)))) + shift
    y = (X[:, 0] > np.median(X[:, 0])).astype(np.int8)
    frame = pd.DataFrame(X, columns=list(CORE_FEATURES))
    frame["label"] = y
    frame["attack_type"] = np.where(y == 1, "attack", "benign")
    frame["dataset"] = dataset
    return frame


def test_kappa_matches_sklearn_and_is_nan_without_variation():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 500)
    b = np.where(rng.random(500) < 0.8, a, 1 - a)
    assert abs(_cohen_kappa(a, b) - cohen_kappa_score(a, b)) < 1e-9
    assert np.isnan(_cohen_kappa(np.zeros(10, int), np.zeros(10, int)))


def test_concept_kappa_is_near_one_when_the_rule_is_shared():
    Xs, ys = _rule_data(4000, 0)
    Xt, yt = _rule_data(4000, 1)
    Xs_test, ys_test = _rule_data(2000, 2)
    row = concept_row(Xs, ys, Xs_test, ys_test, Xt, yt, seed=0, threads=1)
    assert row["cross_overlap_fraction"] > 0.8
    assert row["cross_kappa"] > 0.8
    assert row["indomain_kappa"] > 0.8


def test_concept_kappa_is_negative_when_the_rule_is_inverted():
    Xs, ys = _rule_data(4000, 0)
    Xs_test, ys_test = _rule_data(2000, 2)
    Xt, yt = _rule_data(4000, 1, invert=True)
    row = concept_row(Xs, ys, Xs_test, ys_test, Xt, yt, seed=0, threads=1)
    assert row["cross_kappa"] < 0
    assert row["indomain_kappa"] > 0.8


def test_separability_is_high_for_shifted_domains_and_chance_for_identical_ones():
    shifted = {"a": _frame(600, 0, 0.0, "a"), "b": _frame(600, 1, 5.0, "b"), "c": _frame(600, 2, 10.0, "c")}
    summary, conf = separability_scores(shifted, "all rows", seed=0, threads=1, max_rows=600, folds=3, iterations=60)
    assert summary["pooled_accuracy"] > 0.95
    assert conf.groupby("true_dataset")["row_fraction"].sum().round(6).eq(1.0).all()

    same = {k: _frame(600, seed, 0.0, k) for k, seed in (("a", 3), ("b", 4), ("c", 5))}
    summary, _ = separability_scores(same, "all rows", seed=0, threads=1, max_rows=600, folds=3, iterations=60)
    assert abs(summary["pooled_accuracy"] - 1 / 3) < 0.12


def test_oracle_summary_picks_the_maximum_and_ignores_nan():
    base = dict(pair="a->b", source="a", target="b", transfer_type="same-family", seed=0, n_train=10, n_test=10,
                pr_auc=0.5, balanced_accuracy=0.5)
    grid = pd.DataFrame([
        {**base, "family": "hgb", "config": "c1", "norm": "none", "subset": "all", "roc_auc": 0.61},
        {**base, "family": "lr", "config": "c2", "norm": "rank", "subset": "all", "roc_auc": 0.83},
        {**base, "family": "rf", "config": "c3", "norm": "none", "subset": "-rate", "roc_auc": float("nan")},
        {**base, "family": "catboost", "config": "c4", "norm": "rank", "subset": "-duration", "roc_auc": 0.7},
    ])
    out = oracle_summary(grid, runs=None)
    assert len(out) == 1
    assert out.loc[0, "oracle_roc_auc"] == 0.83
    assert out.loc[0, "oracle_config"].startswith("lr | c2 | norm=rank | subset=all")
    assert out.loc[0, "median_roc_auc"] == np.median([0.61, 0.83, 0.7])


def test_tree_splits_do_not_depend_on_global_standardisation():
    rng = np.random.default_rng(0)
    X = rng.lognormal(size=(1500, 4))
    y = (X[:, 0] + X[:, 1] > 3).astype(int)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    a = HistGradientBoostingClassifier(max_iter=30, early_stopping=False, random_state=0).fit(X, y)
    b = HistGradientBoostingClassifier(max_iter=30, early_stopping=False, random_state=0).fit(Z, y)
    assert np.allclose(a.predict_proba(X)[:, 1], b.predict_proba(Z)[:, 1], atol=1e-8)


def _core_frames(keys, n=500):
    return {key: _frame(n, i, 0.0, key) for i, key in enumerate(keys)}


def test_oracle_resume_skips_done_rows_and_restores_missing_ones(tmp_path):
    frames = _core_frames(["cicids2017", "cicids2018"])
    pairs = [("cicids2017", "cicids2018")]
    path = run_oracle(pairs, [0], tmp_path, threads=1, frames=frames, families=("lr",))
    first = pd.read_csv(path, dtype=str)
    assert len(first) == 4 * 3 * 6  # 4 configs x 3 norms x 6 subsets for lr
    assert first["roc_auc"].astype(float).notna().all()

    run_oracle(pairs, [0], tmp_path, threads=1, frames=frames, families=("lr",))
    assert len(pd.read_csv(path, dtype=str)) == len(first)

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    run_oracle(pairs, [0], tmp_path, threads=1, frames=frames, families=("lr",))
    again = pd.read_csv(path, dtype=str)
    assert len(again) == len(first)
    assert again.duplicated(["pair", "seed", "family", "config", "norm", "subset"]).sum() == 0


def test_concept_resume_does_not_duplicate_pairs(tmp_path):
    frames = _core_frames(["cicids2017", "cicids2018"], n=800)
    pairs = [("cicids2017", "cicids2018")]
    path = run_concept(pairs, [0], tmp_path, threads=1, frames=frames)
    first = pd.read_csv(path)
    assert len(first) == 1
    run_concept(pairs, [0], tmp_path, threads=1, frames=frames)
    assert len(pd.read_csv(path)) == 1
