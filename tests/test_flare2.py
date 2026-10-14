"""FLARE-2 (DESIGN.md section 21): task sampler, held-out-source split, label-slot model, class evaluation, compatibility.

Everything runs on CPU with the tiny configuration and small synthetic pools, except the held-out-source check that
reads the processed iot23 and cicids2017 tables once.
"""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from crossids.flare import data, evaluate_class as ec, train
from crossids.flare.data import (
    AugConfig,
    ClassPools,
    Episode,
    EpisodeConfig,
    TaskConfig,
    TaskSampler,
    batch_arrays,
    collate,
    heldout_sources,
    validation_source,
)
from crossids.flare.model import TINY_CONFIG, FlareConfig, Flare2Model, FlareModel, batch_to_torch, load_model, save_checkpoint

torch.set_num_threads(2)

D = 5
NO_AUG = AugConfig(sign_flip=False, warp=False, subset=False, composite=False, label_noise=False)
TINY_F2 = replace(TINY_CONFIG, slots=16)


def _class_pools(seed: int = 0, n_datasets: int = 2, n_classes: int = 5) -> list[ClassPools]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_datasets):
        benign = rng.normal(0, 1, (400 + 100 * i, D)).astype(np.float32)
        attack = [rng.normal(c + i, 1.0, (60 + 10 * c, D)).astype(np.float32) for c in range(n_classes)]
        out.append(ClassPools(f"ds{i}", benign, [f"att{c}" for c in range(n_classes)], attack))
    return out


def _cfg(**over) -> EpisodeConfig:
    base = EpisodeConfig(k_min=1, k_max=32, query_size=24, aug=NO_AUG)
    return replace(base, **over)


def _manual_episode(X: np.ndarray, label: np.ndarray, truth: np.ndarray, n_support: int, slot_of: np.ndarray,
                    n_slots: int, binary: bool = False) -> Episode:
    return Episode(x=X.astype(np.float32), label=label.astype(np.int64), truth=truth.astype(np.int64),
                   n_support=n_support, n_context=0, meta={"binary_task": binary}, n_classes=int(len(slot_of)),
                   slot_of=np.asarray(slot_of, dtype=np.int64), n_slots=n_slots)


def _tiny_flare2(seed: int = 0, **over) -> Flare2Model:
    torch.manual_seed(seed)
    return Flare2Model(replace(TINY_F2, **over))


# ---------- task sampler ----------


def test_task_mix_counts_and_class_numbers_follow_section_21_3():
    sampler = TaskSampler(_class_pools(), _cfg(query_size=16), seed=0, tasks=TaskConfig())
    eps = sampler.batch(800)
    kinds = pd.Series([e.meta["task"] for e in eps]).value_counts(normalize=True)
    assert abs(kinds.get("binary", 0) - 0.25) < 0.06
    assert abs(kinds.get("one_vs_benign", 0) - 0.25) < 0.06
    assert abs(kinds.get("class_vs_class", 0) - 0.15) < 0.05
    assert abs(kinds.get("nway", 0) - 0.35) < 0.06
    for e in eps:
        if e.meta["task"] == "nway":
            assert 3 <= e.n_classes <= 6
        else:
            assert e.n_classes == 2
        assert e.n_classes * 1 <= e.n_support  # at least one support row per class


def test_support_is_stratified_per_class_and_capped():
    sampler = TaskSampler(_class_pools(seed=3), _cfg(query_size=24), seed=1, tasks=TaskConfig(mix=(0, 0, 0, 1)))
    for e in sampler.batch(60):
        counts = np.bincount(e.label[: e.n_support], minlength=e.n_classes)
        assert np.all(counts == counts[0]), "k per class must be equal"
        assert 1 <= counts[0] <= 32
        assert e.n_support <= 160
        q = e.truth[e.n_support:]
        assert set(np.unique(q)) == set(range(e.n_classes)), "every class appears among the queries"
        assert len(q) == 24


def test_task_sampler_is_deterministic_per_seed():
    a = TaskSampler(_class_pools(), _cfg(), seed=7).batch(5)
    b = TaskSampler(_class_pools(), _cfg(), seed=7).batch(5)
    for e1, e2 in zip(a, b):
        np.testing.assert_array_equal(e1.x, e2.x)
        np.testing.assert_array_equal(e1.label, e2.label)
        np.testing.assert_array_equal(e1.slot_of, e2.slot_of)
        assert e1.meta == e2.meta


def test_label_noise_and_composite_follow_the_augmentation_switches():
    cfg = replace(_cfg(), aug=AugConfig(composite=True, composite_p=1.0, label_noise=True, noise_rate=0.5))
    eps = TaskSampler(_class_pools(), cfg, seed=2, tasks=TaskConfig(mix=(0, 1, 0, 0))).batch(30)
    assert all(e.meta["composite"] for e in eps)
    assert all(e.meta["source_attack"] not in (None, e.meta["source"]) for e in eps)
    noisy = sum(int(np.sum(e.label[: e.n_support] != e.truth[: e.n_support])) for e in eps)
    assert noisy > 0


def test_task_config_rejects_a_bad_mix():
    with pytest.raises(ValueError):
        TaskConfig(mix=(0.5, 0.5, 0.5, 0.5))


def test_validation_source_is_the_next_dataset_and_targets_are_excluded():
    assert validation_source("cicids2017") == "cicids2018"
    assert validation_source("iot23") == "cicids2017"  # wraps around
    assert validation_source("ctu13") == "cicids2017"  # not in the order: first dataset
    for target in data.target_keys():
        train_keys, v = heldout_sources(target)
        assert v not in train_keys and target not in train_keys
        assert set(train_keys) | {v} == set(data.meta_train_keys(target))
        assert len(train_keys) == len(data.meta_train_keys(target)) - 1


def test_heldout_validation_uses_only_v_and_never_the_target():
    pytest.importorskip("pyarrow")
    pools, synthetic, val, keys = train.build_training_data(
        "iot23", 0, None, "data", replace(EpisodeConfig(query_size=16), aug=NO_AUG), n_val=6,
        model_kind="flare2", val_mode="heldout", tasks=TaskConfig(), n_slots=16)
    assert not synthetic
    names = {p.name for p in pools}
    assert names == set(heldout_sources("iot23")[0])
    assert "iot23" not in names and "cicids2017" not in names
    assert {e.meta["source"] for e in val} == {"cicids2017"}


# ---------- model ----------


def test_slot_relabelling_permutes_the_class_outputs():
    """Permuting the class names and the slot assignment together leaves every slot logit unchanged, so the class
    columns permute accordingly."""
    torch.manual_seed(0)
    model = _tiny_flare2(0).eval()
    rng = np.random.default_rng(0)
    N, M, k, q = 4, 8, 3, 10
    X = rng.normal(size=(N * k + q, D))
    sup_y = np.repeat(np.arange(N), k)
    q_y = rng.integers(0, N, q)
    truth = np.concatenate([sup_y, q_y])
    slot_of = np.array([5, 0, 3, 7])
    label = np.concatenate([sup_y, -np.ones(q, np.int64)])
    A = _manual_episode(X, label, truth, N * k, slot_of, M)
    sigma = np.array([2, 0, 3, 1])  # class c is renamed sigma[c]
    slot_B = np.empty(N, np.int64)
    slot_B[sigma] = slot_of  # the renamed class keeps the slot of the original class
    label_B = np.concatenate([sigma[sup_y], -np.ones(q, np.int64)])
    truth_B = np.concatenate([sigma[sup_y], sigma[q_y]])
    B = _manual_episode(X, label_B, truth_B, N * k, slot_B, M)
    outs = []
    for ep in (A, B):
        b = batch_to_torch(collate([ep]))
        with torch.no_grad():
            outs.append(model.forward_episode(b["x"], b["feat_pad"], b["row_valid"], b["is_query"], b["label"])[0])
    la, lb = outs[0].numpy(), outs[1].numpy()
    qa = la[N * k:]
    qb = lb[N * k:]
    np.testing.assert_allclose(qa, qb, atol=1e-5)  # same slot logits: the slots carry the same support tokens
    for c in range(N):
        np.testing.assert_allclose(qb[:, slot_B[sigma[c]]], qa[:, slot_of[c]], atol=1e-5)


def test_query_predictions_do_not_depend_on_other_queries_or_chunking():
    """Attention part of section 13: a query attends to the cached support and to itself only. Rank normalisation
    runs over the whole episode (so it is excluded here by feeding normalised values directly)."""
    model = _tiny_flare2(1).eval()
    g = torch.Generator().manual_seed(1)
    Zs = torch.randn(1, 12, D, generator=g)
    lab = torch.tensor([[0, 1, 2] * 4]).long()
    cache = model.build_support_cache(Zs, lab)
    Zq = torch.randn(1, 20, D, generator=g)
    with torch.no_grad():
        full = model.query_forward(cache, Zq)
        one = model.query_forward(cache, Zq[:, :1])
        chunks = torch.cat([model.query_forward(cache, Zq[:, i:i + 3]) for i in range(0, 20, 3)], dim=1)
    torch.testing.assert_close(full[:, :1], one, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(full, chunks, atol=1e-5, rtol=1e-5)


def test_multiclass_predictions_are_distributions_over_the_support_classes():
    model = _tiny_flare2(2)
    rng = np.random.default_rng(2)
    Xs = rng.normal(size=(20, D))
    ys = np.tile(np.arange(4), 5)
    Xq = rng.normal(size=(15, D))
    P = model.predict_classes(Xs, ys, Xq, seed=0)
    assert P.shape == (15, 4)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(P >= 0)
    with pytest.raises(ValueError):
        model.predict_classes(np.zeros((17, D)), np.arange(17), Xq)  # 17 classes exceed the 16 slots


def test_binary_api_agrees_with_the_two_class_prediction():
    model = _tiny_flare2(3)
    rng = np.random.default_rng(3)
    Xs = rng.normal(size=(10, D))
    ys = np.tile([0, 1], 5)
    Xq = rng.normal(size=(12, D))
    p_bin = model.predict_proba(Xs, ys, Xq, seed=4)
    p_cls = model.predict_classes(Xs, ys, Xq, seed=4)
    np.testing.assert_allclose(p_bin, p_cls[:, 1], atol=1e-6)


def test_slot_assignments_are_random_per_member_and_identity_without_permutation():
    model = _tiny_flare2(4)
    slots = model._slot_assignments(3, 6, seed=0)
    assert slots.shape == (6, 3) and len({tuple(r) for r in slots}) > 1
    assert all(len(set(r)) == 3 for r in slots)
    model.slot_perm = False
    np.testing.assert_array_equal(model._slot_assignments(3, 6, seed=0), np.tile(np.arange(3), (6, 1)))


def test_collated_class_batch_has_slot_targets_and_active_slots():
    e = _manual_episode(np.zeros((6, D)), np.array([0, 1, -1, -1, -1, -1]), np.array([0, 1, 0, 1, 0, 1]),
                        2, np.array([6, 1]), 8, binary=True)
    arr = batch_arrays([e])
    assert arr["slot_target"][0, 2:].tolist() == [6, 1, 6, 1]
    assert arr["slot_active"][0].tolist() == [False, True, False, False, False, False, True, False]
    assert arr["label"][0, :2].tolist() == [6, 1] and arr["label"][0, 2:].tolist() == [8] * 4
    assert bool(arr["binary"][0])


class RuleSource:
    """Episodes whose rule is drawn fresh each episode: 3 classes are the three equal-width rank bands of one randomly
    chosen feature (random sign), among 6 features (1 informative, 5 noise). Only the support reveals the rule."""

    def __init__(self, seed: int, d: int = 6, shots: int = 16, query: int = 48):
        self.rng = np.random.default_rng(seed)
        self.d, self.shots, self.query = d, shots, query

    def sample(self) -> Episode:
        rng, N, k, q, d = self.rng, 3, self.shots, self.query, self.d
        n = N * k * 2 + q
        X = rng.normal(size=(n, d))
        j = rng.integers(d)
        sign = rng.choice([-1.0, 1.0])
        ranks = np.argsort(np.argsort(sign * X[:, j])) / n  # rank in [0, 1)
        cls = np.minimum((ranks * N).astype(np.int64), N - 1)
        sup = np.concatenate([rng.choice(np.flatnonzero(cls == c), k, replace=False) for c in range(N)])
        qry = np.setdiff1d(np.arange(n), sup)[:q]
        order = np.concatenate([sup, qry])
        X, cls = X[order], cls[order]
        label = np.concatenate([cls[: N * k], -np.ones(q, np.int64)])
        return Episode(x=data.gauss_rank(X).astype(np.float32), label=label, truth=cls.astype(np.int64),
                       n_support=N * k, n_context=0, meta={"binary_task": False}, n_classes=N,
                       slot_of=rng.permutation(16)[:N], n_slots=16)

    def batch(self, n: int) -> list:
        return [self.sample() for _ in range(n)]


def _rule_macro_f1(model, source: RuleSource, episodes: int = 60) -> float:
    model.eval()
    scores = []
    for _ in range(episodes):
        ep = source.sample()
        P = model.predict_classes(ep.x[: ep.n_support], ep.label[: ep.n_support], ep.x[ep.n_support:], seed=0)
        scores.append(_macro_f1(ep.truth[ep.n_support:], P.argmax(1), 3))
    return float(np.mean(scores))


def test_flare2_training_reads_the_rule_from_the_support_set(tmp_path):
    """Measured on CPU: macro-F1 must exceed chance (0.33) clearly on fresh episodes after meta-training."""
    model = _tiny_flare2(5)
    train.fit(model, RuleSource(0), RuleSource(99).batch(16), steps=600, batch_size=8, device="cpu",
              val_every=10 ** 9, log_every=10 ** 9, warmup=20, lr=1e-3)
    f1 = _rule_macro_f1(model, RuleSource(12345))
    assert f1 > 0.55, f1


def test_flare2_episode_stream_is_deterministic_with_two_workers():
    """DataLoader with two worker processes yields the same class-episode batches on every run."""
    from torch.utils.data import DataLoader

    from crossids.flare.data import EpisodeStream

    pools = _class_pools(seed=4)
    cfg = _cfg(query_size=12)

    def take(n: int) -> list[dict]:
        stream = EpisodeStream(pools, cfg, seed=3, batch_size=4, tasks=TaskConfig(), n_slots=16)
        return [b for _, b in zip(range(n), DataLoader(stream, batch_size=None, num_workers=2))]

    a, b = take(3), take(3)
    assert {"slot_target", "slot_active", "class_slots", "binary"} <= set(a[0])
    for x, y in zip(a, b):
        assert set(x) == set(y)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])


def test_flare2_loss_decreases_on_the_rule_task(tmp_path):
    """Fast check: the training cross-entropy falls by at least 10% over a short run."""
    model = _tiny_flare2(5)
    train.fit(model, RuleSource(0), RuleSource(99).batch(4), steps=400, batch_size=8, device="cpu", out=tmp_path,
              val_every=10 ** 9, log_every=10, warmup=5, lr=1e-3)
    recs = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in recs if "loss" in r and "val_bce" not in r]
    first, last = np.mean(losses[:3]), np.mean(losses[-3:])
    assert last <= 0.9 * first, (first, last)


def _macro_f1(y, p, n):
    from sklearn.metrics import f1_score

    return f1_score(y, p, labels=list(range(n)), average="macro", zero_division=0)


def test_flare2_checkpoint_round_trip_keeps_predictions(tmp_path):
    model = _tiny_flare2(6)
    model.slot_perm = False
    rng = np.random.default_rng(6)
    Xs, ys = rng.normal(size=(8, D)), np.tile(np.arange(4), 2)
    Xq = rng.normal(size=(9, D))
    before = model.predict_classes(Xs, ys, Xq, seed=2)
    save_checkpoint(tmp_path / "best.pt", model, step=1, context_max=0, slot_perm=False)
    loaded = load_model(tmp_path / "best.pt")
    assert loaded.kind == "flare2" and loaded.slot_perm is False
    np.testing.assert_allclose(loaded.predict_classes(Xs, ys, Xq, seed=2), before, atol=1e-6)


def test_pre_flare2_checkpoints_still_load_and_predict_identically(tmp_path):
    """A FLARE checkpoint written without the ``slots`` config key (before section 21) loads and predicts the same."""
    torch.manual_seed(8)
    model = FlareModel(TINY_CONFIG).eval()
    rng = np.random.default_rng(8)
    Xs, ys = rng.normal(size=(10, D)), np.tile([0, 1], 5)
    Xq = rng.normal(size=(11, D))
    ref = model.predict_proba(Xs, ys, Xq, seed=1)
    save_checkpoint(tmp_path / "old.pt", model, step=0)
    blob = torch.load(tmp_path / "old.pt", weights_only=False)
    blob["config"].pop("slots")
    torch.save(blob, tmp_path / "old.pt")
    loaded = load_model(tmp_path / "old.pt")
    assert loaded.kind == "flare" and loaded.slot_perm is True
    np.testing.assert_allclose(loaded.predict_proba(Xs, ys, Xq, seed=1), ref, atol=1e-7)


def test_repository_flare_checkpoint_loads_as_flare():
    from pathlib import Path

    path = Path("runs/flare/iot23_s0/best.pt")
    if not path.exists():
        pytest.skip("no local FLARE checkpoint")
    model = load_model(path)
    assert model.kind == "flare"
    p = model.predict_proba(np.zeros((2, 4)), np.array([0, 1]), np.ones((3, 4)), seed=0)
    assert p.shape == (3,) and np.all(np.isfinite(p))


def test_run_name_and_default_suffixes():
    base = dict(target="iot23", seed=0, model="flare2", val_mode=None, tasks="mixed", no_slot_perm=False,
                context_max=0)
    assert train.run_name(Namespace(**base)) == "iot23_s0_f2"
    assert train.run_name(Namespace(**{**base, "tasks": "binary"})) == "iot23_s0_f2_binary"
    assert train.run_name(Namespace(**{**base, "no_slot_perm": True})) == "iot23_s0_f2_noperm"
    assert train.run_name(Namespace(**{**base, "val_mode": "insource"})) == "iot23_s0_f2_insrc"
    assert train.run_name(Namespace(**{**base, "context_max": 512})) == "iot23_s0_f2_ctx"
    assert train.run_name(Namespace(**{**base, "model": "flare", "tasks": "mixed"})) == "iot23_s0"


def test_fit_logs_insource_validation_next_to_heldout(tmp_path):
    model = _tiny_flare2(9)
    src = TaskSampler(_class_pools(0), _cfg(), seed=0)
    val = TaskSampler(_class_pools(1), _cfg(), seed=1).batch(4)
    aux = {"insrc": TaskSampler(_class_pools(2), _cfg(), seed=2).batch(4)}
    summary = train.fit(model, src, val, steps=2, out=tmp_path, batch_size=2, device="cpu", val_every=2,
                        log_every=1, warmup=1, val_aux=aux, meta={"slot_perm": True})
    rec = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines() if "val_bce" in line][-1]
    assert {"val_bce", "val_auc", "insrc_val_bce", "insrc_val_auc"} <= set(rec)
    assert summary["best_step"] == 2
    assert load_model(tmp_path / "best.pt").kind == "flare2"


# ---------- attack-class evaluation ----------


def _toy_class_split(n_classes: int = 3, seed: int = 0, rows_per_class: int = 220):
    from crossids.flare.data import Split

    rng = np.random.default_rng(seed)
    names = [f"c{c}" for c in range(n_classes)]
    X, y, a = [], [], []
    for c, name in enumerate(["benign"] + names):
        n = rows_per_class
        X.append(rng.normal(c, 1.0, (n, 4)))
        y.append(np.zeros(n, int) if c == 0 else np.ones(n, int))
        a.append(np.array([name if c else "BENIGN"] * n, dtype=object))
    X, y, a = np.vstack(X), np.concatenate(y), np.concatenate(a)
    idx = rng.permutation(len(y))
    X, y, a = X[idx], y[idx], a[idx]
    cut = int(0.7 * len(y))
    return Split(key="iot23", seed=seed, features=tuple(f"f{i}" for i in range(4)), X_train=X[:cut],
                 y_train=y[:cut], X_test=X[cut:], y_test=y[cut:], attack_test=a[cut:], attack_train=a[:cut])


def test_evaluate_class_end_to_end_on_a_tiny_split(tmp_path, monkeypatch):
    split = _toy_class_split()
    monkeypatch.setattr(ec, "load_split", lambda *a, **k: split)
    monkeypatch.setattr(ec, "target_features", lambda key: split.features)
    runs = tmp_path / "runs"
    model = _tiny_flare2(10)
    save_checkpoint(runs / "iot23_s0_f2" / "best.pt", model, step=1, context_max=0, slot_perm=True)
    out = runs / "class_iot23.csv"
    methods = ["flare2", "catboost", "protonet"]  # protonet has no checkpoint: skipped
    ec.evaluate_class_seed("iot23", 0, out, runs, "data", methods, ("perclass", "nway"), ks=(2,), n_draws=2)
    frame = pd.read_csv(out, keep_default_na=False, dtype={"cls": str, "pred": str})
    assert list(frame.columns) == ec.RESULT_COLUMNS
    assert set(frame["method"]) == {"flare2", "catboost"}
    per = frame[(frame.task == "perclass") & (frame.method == "flare2")]
    assert set(per.metric) == set(ec.PERCLASS_METRICS) and set(per.cls) == {"c0", "c1", "c2"}
    nway = frame[(frame.task == "nway") & (frame.method == "catboost")]
    conf = nway[nway.metric == "confusion"]
    assert len(conf) == 2 * 4 * 4  # 2 draws x 4 classes x 4 predictions
    f1 = nway[nway.metric == "macro_f1"]
    assert len(f1) == 2 and np.all((f1.value >= 0) & (f1.value <= 1))
    sup = json.loads((runs / "class_supports_iot23_s0.json").read_text())
    assert sup["supports"]["nway"]["2"][0]["cls"].count("benign") == 2
    # resuming does not duplicate rows
    before = len(frame)
    ec.evaluate_class_seed("iot23", 0, out, runs, "data", methods, ("perclass", "nway"), ks=(2,), n_draws=2)
    assert len(pd.read_csv(out)) == before


def test_flare2_nway_runs_beyond_eight_classes_with_sixteen_slots(tmp_path, monkeypatch):
    split = _toy_class_split(n_classes=9, rows_per_class=260)  # benign + 9 attack classes = 10 N-way classes
    monkeypatch.setattr(ec, "load_split", lambda *a, **k: split)
    monkeypatch.setattr(ec, "target_features", lambda key: split.features)
    runs = tmp_path / "runs"
    save_checkpoint(runs / "iot23_s0_f2" / "best.pt", _tiny_flare2(11), step=1, context_max=0, slot_perm=True)
    out = runs / "class_iot23.csv"
    ec.evaluate_class_seed("iot23", 0, out, runs, "data", ["flare2"], ("nway",), ks=(2,), n_draws=1)
    frame = pd.read_csv(out, keep_default_na=False, dtype={"cls": str, "pred": str})
    f1 = frame[(frame.metric == "macro_f1")]
    assert len(f1) == 1 and np.isfinite(f1.value.iloc[0])
    assert frame[frame.metric == "confusion"]["cls"].nunique() == 10


def test_perclass_and_nway_class_lists_use_the_50_row_rule():
    cls_test = np.array(["benign"] * 60 + ["a"] * 50 + ["b"] * 49 + ["c"] * 70, dtype=object)
    per, nway = ec.eval_classes(cls_test)
    assert per == ["a", "c"]
    assert nway == ["benign", "a", "c"]


def test_class_supports_are_reused_only_for_the_same_grid(tmp_path):
    cls_train = np.array(["benign"] * 300 + ["a"] * 200 + ["b"] * 200, dtype=object)
    path = tmp_path / "s.json"
    first = ec.load_or_draw_class_supports(path, "iot23", 0, cls_train, ["a", "b"], ["benign", "a", "b"], ks=(2, 4),
                                           n_draws=3)
    again = ec.load_or_draw_class_supports(path, "iot23", 0, cls_train, ["a", "b"], ["benign", "a", "b"], ks=(2,),
                                           n_draws=2)
    assert again["nway"][2] == first["nway"][2][:2]
    more = ec.load_or_draw_class_supports(path, "iot23", 0, cls_train, ["a", "b"], ["benign", "a", "b"], ks=(8,),
                                          n_draws=3)
    assert 8 in more["nway"] and len(more["nway"][8]) == 3


def _class_csv_rows() -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    base = dict(seed=0, target="iot23")
    for method, bump in [("flare2", 0.2), ("catboost", 0.0)]:
        for k in (2, 8):
            for draw in range(3):
                for cls in ("benign", "c0", "c1"):
                    names = ["benign", "c0", "c1"]
                    for metric, v in [("macro_f1", 0.5 + bump + 0.01 * draw), ("balanced_accuracy", 0.6 + bump)]:
                        rows.append({**base, "task": "nway", "method": method, "k": k, "draw": draw, "metric": metric,
                                     "cls": "", "pred": "", "value": v, "n": 90})
                    for pred in names:
                        share = (0.8 + bump / 4) if pred == cls else 0.1
                        rows.append({**base, "task": "nway", "method": method, "k": k, "draw": draw,
                                     "metric": "confusion", "cls": cls, "pred": pred, "value": 30 * share,
                                     "n": 30})
                    rows.append({**base, "task": "nway", "method": method, "k": k, "draw": draw, "metric": "recall",
                                 "cls": cls, "pred": "", "value": 0.7 + bump, "n": 30})
                    for metric, v in [("roc_auc", 0.6 + bump + 0.02 * rng.random()), ("recall", 0.5 + bump),
                                      ("fpr", 0.1)]:
                        rows.append({**base, "task": "perclass", "method": method, "k": k, "draw": draw,
                                     "metric": metric, "cls": cls, "pred": "", "value": v, "n": 100})
    return rows


def test_class_report_writes_curves_wilcoxon_and_k8_heatmaps(tmp_path):
    from crossids.flare import report_class

    runs = tmp_path / "runs"
    runs.mkdir()
    pd.DataFrame(_class_csv_rows()).to_csv(runs / "class_iot23.csv", index=False)
    out = tmp_path / "out"
    report_class.run(runs, out)
    wil = pd.read_csv(out / "wilcoxon_vs_catboost.csv")
    assert {"nway", "perclass"} <= set(wil["task"]) and set(wil["method"]) == {"flare2"}
    conf = pd.read_csv(out / "confusion_k8.csv")
    assert set(conf["method"]) == {"flare2", "catboost"} and conf["share"].between(0, 1).all()
    assert (out / "confusion_k8_flare2.png").exists() and (out / "confusion_k8_catboost.png").exists()
    assert (out / "nway_macro_f1_vs_k.png").exists() and (out / "perclass_roc_auc_vs_k.png").exists()
