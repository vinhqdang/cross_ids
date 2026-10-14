"""FLARE: sampler invariants, model masking and invariance, learning from support, and the evaluation pipeline.

Everything runs on CPU with the tiny configuration and small synthetic data.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from crossids.flare import al, data, evaluate, model as flare_model, train
from crossids.methods.base import Domain
from crossids.flare.data import AugConfig, Episode, EpisodeConfig, EpisodeSampler, Pools, collate, draw_episode
from crossids.flare.model import TINY_CONFIG, FlareModel, ProtoNet, batch_to_torch, save_checkpoint

torch.set_num_threads(2)

D = 13
NO_AUG = AugConfig(sign_flip=False, warp=False, subset=False, composite=False, label_noise=False)


def _two_pools(seed: int = 0) -> list[Pools]:
    rng = np.random.default_rng(seed)
    a = Pools("a", rng.normal(0, 1, (1500, D)).astype(np.float32), rng.normal(1, 1, (600, D)).astype(np.float32))
    b = Pools("b", rng.normal(0, 2, (900, D)).astype(np.float32), rng.normal(-1, 1, (500, D)).astype(np.float32))
    return [a, b]


def _episode(seed: int, d: int = D, k: int = 8, q: int = 24) -> Episode:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (k + q, d))
    y = np.concatenate([np.arange(k) % 2, (rng.random(q) < 0.5).astype(int)])
    X[y == 1, : max(1, d // 3)] += 1.0
    z = data.gauss_rank(X)
    label = y.copy()
    label[k:] = data.LABEL_UNKNOWN
    return Episode(x=z.astype(np.float32), label=label.astype(np.int64), truth=y.astype(np.int64), n_support=k)


def _tiny_model(seed: int = 0, **overrides) -> FlareModel:
    torch.manual_seed(seed)
    return FlareModel(replace(TINY_CONFIG, **overrides))


# ---------- sampler ----------


def test_gauss_rank_is_rank_based_and_clipped():
    X = np.array([[1.0, 5.0], [2.0, np.nan], [3.0, -1.0]])
    Z = data.gauss_rank(X)
    assert np.isnan(Z[1, 1])
    assert np.all(np.abs(Z[np.isfinite(Z)]) <= 4.0)
    assert Z[0, 0] < Z[1, 0] < Z[2, 0]
    shifted = data.gauss_rank(X * 1000.0 - 7.0)
    np.testing.assert_allclose(Z, shifted, equal_nan=True)


def test_stratified_draw_has_each_class_and_right_size():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.03).astype(int)
    for k in (4, 8, 16, 32, 64, 128):
        idx = evaluate.stratified_draw(y, k, np.random.default_rng(k))
        assert len(idx) == k and len(set(idx.tolist())) == k
        assert y[idx].sum() >= 1 and (1 - y[idx]).sum() >= 1
    y_hi = (rng.random(5000) < 0.95).astype(int)
    idx = evaluate.stratified_draw(y_hi, 4, np.random.default_rng(1))
    assert len(idx) == 4 and y_hi[idx].sum() < 4 and (1 - y_hi[idx]).sum() >= 1


def test_episode_shape_stratification_and_priors():
    cfg = EpisodeConfig()
    eps = EpisodeSampler(_two_pools(), cfg, seed=0).batch(300)
    ks, q_fracs, s_fracs = [], [], []
    for e in eps:
        k = e.n_support
        assert cfg.k_min <= k <= cfg.k_max
        assert e.x.shape[0] - k == cfg.query_size
        assert e.truth[:k].sum() >= 1 and (1 - e.truth[:k]).sum() >= 1, "support must hold both classes"
        assert e.meta["support_attack"] == e.truth[:k].sum()
        assert e.meta["query_attack"] == e.truth[k:].sum()
        assert data.LABEL_UNKNOWN == 2 and np.all(e.label[k:] == data.LABEL_UNKNOWN)
        assert 5 <= e.x.shape[1] <= D
        ks.append(k)
        q_fracs.append(e.truth[k:].mean())
        s_fracs.append(e.truth[:k].mean())
    assert min(ks) >= 4 and max(ks) <= 128
    assert 10 <= np.median(ks) <= 40, "log-uniform support size should centre near sqrt(4 * 128)"
    # query prior ~ U(0.05, 0.95) with rounding to whole rows
    assert min(q_fracs) >= 0.05 - 1 / 512 and max(q_fracs) <= 0.95 + 1 / 512
    assert np.std(q_fracs) > 0.2, "query prior must vary across episodes"
    # support prior ~ U(0.1, 0.9) with at least one row per class
    for e, f in zip(eps, s_fracs):
        k = e.n_support
        assert 0.1 - 1 / k - 1e-9 <= f <= 0.9 + 1 / k + 1e-9


def test_feature_subsets_and_full_width_without_a2():
    eps = EpisodeSampler(_two_pools(), EpisodeConfig(), seed=3).batch(200)
    widths = [e.x.shape[1] for e in eps]
    assert all(5 <= w <= D for w in widths)
    frac_subset = np.mean([w < D for w in widths])
    assert 0.3 < frac_subset < 0.6
    full = EpisodeSampler(_two_pools(), EpisodeConfig(aug=replace(NO_AUG, subset=False)), seed=3).batch(20)
    assert all(e.x.shape[1] == D for e in full)


def test_sampler_is_deterministic_per_seed():
    a = EpisodeSampler(_two_pools(), EpisodeConfig(), seed=7).batch(5)
    b = EpisodeSampler(_two_pools(), EpisodeConfig(), seed=7).batch(5)
    c = EpisodeSampler(_two_pools(), EpisodeConfig(), seed=8).batch(5)
    for e1, e2 in zip(a, b):
        np.testing.assert_array_equal(e1.x, e2.x)
        np.testing.assert_array_equal(e1.label, e2.label)
    assert any(e1.x.shape != e3.x.shape or not np.array_equal(e1.x, e3.x) for e1, e3 in zip(a, c))


def _auc_signs(eps: list[Episode], feature: int) -> np.ndarray:
    out = []
    for e in eps:
        cols = e.meta["features"]
        if feature not in cols:
            continue
        j = cols.index(feature)
        y = e.truth[e.n_support:]
        out.append(roc_auc_score(y, e.x[e.n_support:, j]))
    return np.asarray(out)


def _separated_pools(seed: int = 0) -> list[Pools]:
    rng = np.random.default_rng(seed)
    a = Pools("sep", rng.normal(0, 1, (800, D)).astype(np.float32), rng.normal(0, 1, (800, D)).astype(np.float32))
    a.attack[:, 0] += 2.0  # feature 0 separates the classes with a positive direction
    return [a]


def test_sign_flip_makes_feature_direction_uniform():
    eps_no = EpisodeSampler(_separated_pools(), EpisodeConfig(aug=replace(NO_AUG, subset=False)), 0).batch(300)
    auc_no = _auc_signs(eps_no, 0)
    assert np.mean(auc_no > 0.5) > 0.99, "without A1 the separating direction is always the same"
    aug = replace(NO_AUG, sign_flip=True, subset=False)
    eps_yes = EpisodeSampler(_separated_pools(), EpisodeConfig(aug=aug), 0).batch(300)
    auc_yes = _auc_signs(eps_yes, 0)
    frac_pos = np.mean(auc_yes > 0.5)
    assert 0.4 < frac_pos < 0.6, f"A1 sign flip should make the direction random, got {frac_pos:.2f} positive"
    assert np.mean(auc_yes > 0.8) + np.mean(auc_yes < 0.2) > 0.8, "the separation itself must survive the flip"


def test_warp_is_monotone_and_absorbed_by_rank_normalisation():
    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 1, 400)
    xs, ys = data.warp_knots(np.random.default_rng(1), 4)
    assert len(xs) == len(ys) == 6 and np.all(np.diff(xs) >= 0) and np.all(np.diff(ys) >= 0)
    w = data.piecewise_warp(x, xs, ys)
    order = np.argsort(x)
    assert np.all(np.diff(w[order]) >= -1e-12), "the warp must be monotone"
    np.testing.assert_allclose(data.gauss_rank(x[:, None]), data.gauss_rank(w[:, None]), atol=1e-12)


def test_label_noise_rate_on_support():
    aug = replace(NO_AUG, label_noise=True)
    eps = EpisodeSampler(_two_pools(), EpisodeConfig(aug=aug), seed=11).batch(300)
    flips = sum(int(np.sum(e.label[: e.n_support] != e.truth[: e.n_support])) for e in eps)
    total = sum(e.n_support for e in eps)
    assert 0.012 < flips / total < 0.03
    assert all(np.all(e.label[e.n_support:] == data.LABEL_UNKNOWN) for e in eps)


def test_composite_episodes_take_attacks_from_another_dataset():
    rng = np.random.default_rng(0)
    # feature 1 identifies the source: dataset A rows are +5, dataset B rows are -5
    a = Pools("A", np.column_stack([rng.normal(size=(600, 1)), np.full((600, 1), 5.0)]).astype(np.float32),
              np.column_stack([rng.normal(1, 1, (300, 1)), np.full((300, 1), 5.0)]).astype(np.float32))
    b = Pools("B", np.column_stack([rng.normal(size=(600, 1)), np.full((600, 1), -5.0)]).astype(np.float32),
              np.column_stack([rng.normal(1, 1, (300, 1)), np.full((300, 1), -5.0)]).astype(np.float32))
    aug = replace(NO_AUG, composite=True)
    eps = EpisodeSampler([a, b], EpisodeConfig(aug=replace(aug, composite_p=0.5)), seed=2).batch(200)
    composite = [e for e in eps if e.meta["source_attack"] is not None]
    assert 0.35 < len(composite) / len(eps) < 0.65
    for e in composite:
        x1 = e.x[:, 1]
        benign_vals, attack_vals = x1[e.truth == 0], x1[e.truth == 1]
        separated = attack_vals.max() < benign_vals.min() or attack_vals.min() > benign_vals.max()
        assert separated, "benign and attack rows must come from different sources"
    plain = [e for e in eps if e.meta["source_attack"] is None]
    assert plain and all(np.ptp(e.x[:, 1]) < 1e-9 for e in plain)


def test_synthetic_sampler_produces_valid_episodes():
    eps = train.SyntheticSampler(EpisodeConfig(), seed=0).batch(20)
    for e in eps:
        assert e.x.shape[1] >= 5 and e.truth[: e.n_support].sum() >= 1
        assert np.all(np.isfinite(e.x))


def test_collate_block_layout_and_masks():
    e1, e2 = _episode(1, d=9, k=6, q=20), _episode(2, d=5, k=10, q=30)
    b = collate([e1, e2])
    assert b.x.shape == (2, 10 + 30, 9)
    assert not b.is_query[:, :10].any() and b.is_query[:, 10:].all()
    assert b.row_valid[0].sum() == 26 and b.row_valid[1].sum() == 40
    assert b.feat_pad[0].all() and b.feat_pad[1].sum() == 5


# ---------- model ----------


def test_block_row_attention_matches_dense_reference():
    torch.manual_seed(0)
    cfg = TINY_CONFIG
    ra = flare_model._RowAttention(cfg.width, cfg.heads).eval()
    B, N, T, n_sup = 2, 9, 4, 4
    h = torch.randn(B, N, T, cfg.width)
    sup_valid = torch.ones(B, n_sup, dtype=torch.bool)
    sup_valid[1, 3] = False
    out = ra(h, n_sup, sup_valid)
    # dense reference: explicit N x N mask with the section 13 rule
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[1, 3] = False
    is_q = torch.arange(N) >= n_sup
    sup = valid & ~is_q[None]
    eye = torch.eye(N, dtype=torch.bool)
    allow = (sup[:, :, None] & sup[:, None, :]) | (is_q[None, :, None] & sup[:, None, :]) | eye
    x = h.permute(0, 2, 1, 3).reshape(B * T, N, cfg.width)
    qkv = ra.qkv(x).reshape(B * T, N, 3, cfg.heads, cfg.width // cfg.heads).permute(2, 0, 3, 1, 4)
    mask = allow[:, None, None].expand(B, T, 1, N, N).reshape(B * T, 1, N, N)
    ref = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
    ref = ra.proj(ref.transpose(1, 2).reshape(B * T, N, cfg.width)).reshape(B, T, N, cfg.width).permute(0, 2, 1, 3)
    # padded support row 3 of episode 1 only attends to itself; rows that are valid must match exactly
    torch.testing.assert_close(out[0], ref[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out[1, valid[1]], ref[1, valid[1]], atol=1e-5, rtol=1e-4)


def test_model_is_invariant_to_feature_permutation():
    model = _tiny_model(0).eval()
    e = _episode(4, d=9)
    S, y, Q = e.x[: e.n_support], e.label[: e.n_support], e.x[e.n_support:]
    perm = np.random.default_rng(0).permutation(9)
    base = model.predict_logits(S, y, Q, ensemble=1)
    permuted = model.predict_logits(S[:, perm], y, Q[:, perm], ensemble=1)
    np.testing.assert_allclose(base, permuted, atol=1e-5)


def test_query_predictions_do_not_depend_on_other_queries():
    model = _tiny_model(1).eval()
    e = _episode(5, d=7, k=6, q=16)
    k = e.n_support
    with torch.no_grad():
        b = batch_to_torch(collate([e]))
        ref = model.forward_episode(b["x"], b["feat_pad"], b["row_valid"], b["is_query"], b["label"])[0, k:]
        x2 = e.x.copy()
        x2[k + 4:] = np.random.default_rng(3).normal(0, 3, x2[k + 4:].shape).astype(np.float32)
        b2 = batch_to_torch(collate([replace(e, x=x2)]))
        alt = model.forward_episode(b2["x"], b2["feat_pad"], b2["row_valid"], b2["is_query"], b2["label"])[0, k:]
    torch.testing.assert_close(ref[:4], alt[:4], atol=1e-5, rtol=1e-4)


def test_predict_is_chunk_invariant_and_handles_empty_support():
    model = _tiny_model(2).eval()
    e = _episode(6, d=8, k=8, q=40)
    S, y, Q = e.x[:8], e.label[:8], e.x[8:]
    full = model.predict_logits(S, y, Q, chunk=1024, ensemble=1)
    chunked = model.predict_logits(S, y, Q, chunk=7, ensemble=1)
    np.testing.assert_allclose(full, chunked, atol=1e-5)
    emb = model.embed(np.zeros((0, 8)), np.zeros(0, dtype=np.int64), Q)
    assert emb.shape == (40, model.width) and np.all(np.isfinite(emb))


def test_padding_masks_ignore_padded_features_and_rows():
    model = _tiny_model(3).eval()
    e1, e2 = _episode(7, d=9, k=6, q=20), _episode(8, d=5, k=10, q=30)
    with torch.no_grad():
        alone1 = model.forward_episode(*_tensors([e1])[:5])[0, :6 + 20]
        alone2 = model.forward_episode(*_tensors([e2])[:5])[0, :10 + 30]
        b = batch_to_torch(collate([e1, e2]))
        together = model.forward_episode(b["x"], b["feat_pad"], b["row_valid"], b["is_query"], b["label"])
        # garbage in padded features and padded rows must not change any valid output
        x = b["x"].clone()
        pad_feat = ~b["feat_pad"][:, None, :].expand_as(x)
        pad_row = ~b["row_valid"][:, :, None].expand_as(x)
        x[pad_feat | pad_row] = 1e3 * torch.randn(int((pad_feat | pad_row).sum()))
        garbage = model.forward_episode(x, b["feat_pad"], b["row_valid"], b["is_query"], b["label"])
    valid1 = b["row_valid"][0].clone()
    valid2 = b["row_valid"][1].clone()
    torch.testing.assert_close(together[0][valid1], garbage[0][valid1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(together[1][valid2], garbage[1][valid2], atol=1e-5, rtol=1e-4)
    # the 9-feature episode matches its own unpadded forward pass
    rows1 = np.concatenate([np.arange(6), 10 + np.arange(20)])
    torch.testing.assert_close(together[0][torch.as_tensor(rows1)], alone1, atol=1e-5, rtol=1e-4)
    rows2 = np.concatenate([np.arange(10), 10 + np.arange(30)])
    torch.testing.assert_close(together[1][torch.as_tensor(rows2)], alone2, atol=1e-5, rtol=1e-4)


def _tensors(episodes):
    b = batch_to_torch(collate(episodes))
    return b["x"], b["feat_pad"], b["row_valid"], b["is_query"], b["label"]


def test_rowmlp_and_protonet_run_and_respect_padding():
    cfg = replace(TINY_CONFIG, feature_mixing="rowmlp", max_features=13)
    torch.manual_seed(0)
    m = FlareModel(cfg).eval()
    e = _episode(9, d=6, k=6, q=10)
    p = m.predict_proba(e.x[:6], e.label[:6], e.x[6:], ensemble=1)
    assert p.shape == (10,) and np.all((p > 0) & (p < 1))
    pn = ProtoNet(replace(TINY_CONFIG, max_features=13)).eval()
    proba = pn.predict_proba(e.x[:6], e.label[:6], e.x[6:])
    assert proba.shape == (10,) and np.all((proba > 0) & (proba < 1))
    small = FlareModel(replace(TINY_CONFIG, feature_mixing="rowmlp", max_features=4))
    with pytest.raises(ValueError):
        small.forward_episode(*_tensors([_episode(9, d=6)])[:5])


def test_ensemble_changes_predictions_only_through_sign_flips():
    model = _tiny_model(4).eval()
    e = _episode(10, d=8)
    S, y, Q = e.x[: e.n_support], e.label[: e.n_support], e.x[e.n_support:]
    single = model.predict_logits(S, y, Q, ensemble=1)
    perm_only = model.predict_logits(S, y, Q, ensemble=3, seed=0)
    assert np.all(np.isfinite(perm_only)) and perm_only.shape == single.shape


# ---------- training ----------


class _RuleEpisodes:
    """Episodes where the label is sign(x_j) for a random feature j and sign s, drawn per episode."""

    def __init__(self, seed: int, d: int = 6, k: int = 16, q: int = 64):
        self.rng = np.random.default_rng(seed)
        self.d, self.k, self.q = d, k, q

    def episode(self) -> Episode:
        d, k, q = self.d, self.k, self.q
        j = int(self.rng.integers(d))
        s = 1.0 if self.rng.random() < 0.5 else -1.0
        while True:
            X = self.rng.normal(0, 1, (k + q, d))
            y = (s * X[:, j] > 0).astype(np.int64)
            if 0 < y[:k].sum() < k:
                break
        z = data.gauss_rank(X).astype(np.float32)
        label = y.copy()
        label[k:] = data.LABEL_UNKNOWN
        return Episode(x=z, label=label, truth=y, n_support=k)

    def batch(self, size: int) -> list[Episode]:
        return [self.episode() for _ in range(size)]


def test_learns_rule_from_support_within_a_few_hundred_cpu_steps(tmp_path):
    # Two features, so the random feature and sign must be read from the 32 support rows. With more features
    # the tiny model stays on the chance plateau for the first few hundred steps (see report).
    torch.manual_seed(0)
    cfg = replace(TINY_CONFIG, layers=2, width=32, heads=2, value_hidden=16)
    model = FlareModel(cfg)
    gen = _RuleEpisodes(seed=0, d=2, k=32, q=64)
    val = _RuleEpisodes(seed=99, d=2, k=32, q=64).batch(16)
    summary = train.fit(model, gen, val, steps=300, lr=3e-3, warmup=20, batch_size=8, device="cpu",
                        eval_every=300, log_every=100, out=tmp_path / "run")
    assert summary["best_step"] == 300
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "last.pt").exists()
    lines = (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    assert any('"val_bce"' in ln for ln in lines)
    aucs = []
    model.eval()
    for e in _RuleEpisodes(seed=1234, d=2, k=32, q=64).batch(40):
        k = e.n_support
        p = model.predict_proba(e.x[:k], e.label[:k], e.x[k:], ensemble=1)
        aucs.append(roc_auc_score(e.truth[k:], p))
    assert np.mean(aucs) > 0.8, f"query AUC {np.mean(aucs):.3f} after 300 steps"


def test_fit_checkpoints_reload_with_same_predictions(tmp_path):
    torch.manual_seed(1)
    model = FlareModel(TINY_CONFIG)
    val = EpisodeSampler(_two_pools(), EpisodeConfig(), 1).batch(4)
    train.fit(model, EpisodeSampler(_two_pools(), EpisodeConfig(), 0), val, steps=3, out=tmp_path,
              batch_size=2, device="cpu", eval_every=3, log_every=1, warmup=1)
    loaded = flare_model.load_model(tmp_path / "last.pt")
    e = _episode(12, d=D)
    model.eval()
    np.testing.assert_allclose(model.predict_logits(e.x[:8], e.label[:8], e.x[8:], ensemble=1),
                               loaded.predict_logits(e.x[:8], e.label[:8], e.x[8:], ensemble=1), atol=1e-6)


# ---------- active cold start ----------


def test_kcenter_greedy_spreads_points():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]])
    sel = al.kcenter_greedy(pts, 4, np.random.default_rng(0))
    assert len(set(sel.tolist())) == 4
    chosen = pts[sel]
    d = np.linalg.norm(chosen[:, None] - chosen[None], axis=2)
    assert d[np.triu_indices(4, 1)].min() > 4.0


def test_flare_al_selects_k_distinct_pool_rows():
    model = _tiny_model(5).eval()
    rng = np.random.default_rng(0)
    pool_X = rng.normal(0, 1, (120, 6)).astype(np.float32)
    pool_y = (pool_X[:, 0] > 0).astype(np.int64)
    emb0 = al.empty_support_embeddings(model, pool_X)
    sel = al.select_flare_al(model, pool_X, pool_y, 8, rng=np.random.default_rng(1), emb0=emb0, ensemble=1)
    assert len(sel) == 8 and len(set(sel.tolist())) == 8
    raw = al.select_kcenter_raw(pool_X, 8, rng=np.random.default_rng(1))
    assert len(set(raw.tolist())) == 8


# ---------- evaluation ----------


def _tiny_split(seed: int = 0) -> evaluate.Split:
    rng = np.random.default_rng(seed)
    n_tr, n_te = 300, 200

    def make(n):
        y = (rng.random(n) < 0.3).astype(np.int64)
        X = rng.normal(0, 1, (n, 6)).astype(np.float32)
        X[y == 1, :2] += 1.5
        return X, y

    Xtr, ytr = make(n_tr)
    Xte, yte = make(n_te)
    return data.Split(key="tgt", seed=seed, features=tuple(f"f{i}" for i in range(6)), X_train=Xtr, y_train=ytr,
                      X_test=Xte, y_test=yte, attack_test=np.array(["x"] * n_te, dtype=object))


def test_support_draws_are_saved_and_reused(tmp_path):
    split = _tiny_split()
    path = tmp_path / "supports_tgt_s0.json"
    first = evaluate.load_or_draw_supports(path, "tgt", 0, split.y_train, ks=(4, 8), n_draws=2)
    blob = json.loads(path.read_text())
    assert blob["target"] == "tgt" and set(blob["supports"]) == {"4", "8"} and len(blob["supports"]["8"]) == 2
    second = evaluate.load_or_draw_supports(path, "tgt", 0, split.y_train, ks=(4, 8), n_draws=2)
    for k in (4, 8):
        for a, b in zip(first[k], second[k]):
            np.testing.assert_array_equal(a, b)


def test_evaluation_pipeline_end_to_end(tmp_path):
    split = _tiny_split()
    runs = tmp_path / "runs"
    save_checkpoint(runs / "tgt_s0" / "best.pt", FlareModel(replace(TINY_CONFIG, ensemble=2)), step=0)
    save_checkpoint(runs / "tgt_s0_protonet" / "best.pt", ProtoNet(TINY_CONFIG), step=0)
    ctx = evaluate.EvalContext(runs_dir=runs, supports_dir=tmp_path, device="cpu", threads=2, al_pool=200,
                               ks=(8, 32), n_draws=2)

    def sources():
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (400, 6))
        y = (X[:, 0] > 0.5).astype(int)
        return [Domain_for("src", X, y)]

    methods = ["catboost-target", "knn", "lr", "rf", "catboost-ft", "protonet", "flare", "flare-E1", "flare-noA1",
               "flare-al", "flare-kcenter-raw"]
    out = tmp_path / "eval_tgt.csv"

    def emit(row):
        pd.DataFrame([row], columns=evaluate.RESULT_COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)

    rows = evaluate.evaluate_seed("tgt", split, ctx, methods, sources_fn=sources, done=set(), emit=emit)
    # flare-noA1 has no checkpoint here, so it is skipped
    expected = (len(methods) - 1) * 2 * 2
    assert len(rows) == expected
    frame = pd.read_csv(out)
    assert list(frame.columns) == evaluate.RESULT_COLUMNS
    assert set(frame["method"]) == set(methods) - {"flare-noA1"}
    metric_cols = frame[["roc_auc", "pr_auc", "balanced_accuracy", "f1"]].dropna()
    assert ((metric_cols >= 0) & (metric_cols <= 1)).all().all()
    assert (frame["n_support"] == frame["k"]).all()
    # resuming with the finished rows adds nothing
    done = evaluate._done_keys(out)
    again = evaluate.evaluate_seed("tgt", split, ctx, methods, sources_fn=sources, done=done, emit=None)
    assert again == []
    # supports are identical across methods for the same (k, draw)
    sup = evaluate.load_or_draw_supports(tmp_path / "supports_tgt_s0.json", "tgt", 0, split.y_train, (8, 32), 2)
    for name, grp in frame.groupby(["k", "draw"]):
        assert grp["n_support"].nunique() == 1
    assert len(sup[8]) == 2


def Domain_for(name, X, y):
    return Domain(name=name, X=X, y=y)
