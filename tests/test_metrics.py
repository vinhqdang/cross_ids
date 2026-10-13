"""Metric evaluation, including the single-class target case."""

import numpy as np

from crossids.metrics import evaluate, per_attack_recall


def test_single_class_target_gives_nan_for_rank_and_chance_corrected_metrics():
    y = np.zeros(50, dtype=int)
    pred = np.r_[np.zeros(45), np.ones(5)].astype(int)
    scores = np.linspace(0, 1, 50)
    out = evaluate(y, scores, pred)
    for name in ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "detection_rate"):
        assert np.isnan(out[name]), name
    assert np.isclose(out["fpr"], 5 / 50)
    assert out["f1"] == 0.0


def test_two_class_metrics_match_hand_computed_values():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.7, 0.9])
    out = evaluate(y, scores, pred)
    assert np.isclose(out["roc_auc"], 1.0)
    assert np.isclose(out["detection_rate"], 1.0)
    assert np.isclose(out["fpr"], 0.5)
    assert np.isclose(out["balanced_accuracy"], 0.75)
    assert np.isclose(out["f1"], 2 * 2 / (2 * 2 + 1 + 0))
    assert np.isfinite(out["mcc"])


def test_per_attack_recall_counts_attack_rows_only():
    y = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 1, 0, 1, 1])
    attack = np.array(["benign", "botnet-s42", "botnet-s42", "dos", "benign"], dtype=object)
    frame = per_attack_recall(y, pred, attack).set_index("attack_type")
    assert list(frame.index) == ["botnet-s42", "dos"]
    assert np.isclose(frame.loc["botnet-s42", "recall"], 0.5)
    assert frame.loc["dos", "n"] == 1
