"""Method registry: every method fits and predicts; fit takes no target labels; RIFT vs HGB."""

import inspect
from scipy.stats import norm

import numpy as np
import pytest

from crossids.datasets import make_synthetic_domains
from crossids.methods import METHOD_NAMES, make_method
from crossids.methods.base import Domain, Method
from crossids.methods.coral import coral_align
from crossids.methods.rift import Rift, RiftConfig
from crossids.metrics import evaluate


def _split(domains, frac=0.7):
    """Per domain: a source/training part and a held-out test part."""
    train, test, pool = [], [], None
    for i, d in enumerate(domains):
        k = int(frac * len(d.X))
        if i < len(domains) - 1:
            train.append(Domain(d.name, d.X[:k], d.y[:k], d.meta.iloc[:k]))
        else:
            pool = d.X[:k]
        test.append((d.X[k:], d.y[k:]))
    return train, pool, test[-1]


@pytest.fixture(scope="module")
def synthetic():
    return _split(make_synthetic_domains(n_domains=3, n=700, seed=0))


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_method_fits_and_predicts(name, synthetic):
    train, pool, (Xt, yt) = synthetic
    method = make_method(name, seed=0)
    method.fit(train, pool if method.transductive else None)
    proba = method.predict_proba(Xt)
    assert proba.shape == (len(Xt),)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert set(np.unique(method.predict(Xt))) <= {0, 1}


def test_fit_signature_has_no_label_argument():
    params = list(inspect.signature(Method.fit).parameters)
    assert params == ["self", "sources", "target_pool"]


@pytest.mark.parametrize("name", ["coral", "dann", "rift", "rift-noDN", "rift-noLSC"])
def test_transductive_methods_are_deterministic_given_pool(name):
    train, pool, (Xt, _) = _split(make_synthetic_domains(n_domains=3, n=600, seed=3))
    first = make_method(name, 0).fit(train, pool).predict_proba(Xt)
    second = make_method(name, 0).fit(train, pool.copy()).predict_proba(Xt)
    assert np.allclose(first, second)


def test_rift_reports_estimated_prior_and_selection(synthetic):
    train, pool, _ = synthetic
    method = make_method("rift", 0).fit(train, pool)
    assert 0.001 <= method.info["pi_t"] <= 0.999
    assert 0 < method.info["n_features"] <= train[0].X.shape[1]


def test_rift_beats_hgb_on_synthetic_cross_domain_over_three_seeds():
    rift_auc, hgb_auc = [], []
    for seed in range(3):
        train, pool, (Xt, yt) = _split(make_synthetic_domains(n_domains=3, n=600, seed=seed))
        for name, out in (("rift", rift_auc), ("hgb", hgb_auc)):
            method = make_method(name, seed)
            method.fit(train, pool if method.transductive else None)
            out.append(evaluate(yt, method.decision_scores(Xt), method.predict(Xt))["roc_auc"])
    assert np.mean(rift_auc) > np.mean(hgb_auc)


def test_rift_anchor_is_the_lowest_scoring_pool_fraction_with_a_floor():
    train, pool, _ = _split(make_synthetic_domains(n_domains=3, n=800, seed=5))
    method = Rift(seed=0).fit(train, pool)
    n = len(pool)
    frac = max(0.05, 0.8 * (1.0 - method.info["pi_anchor"]))
    assert len(method.anchor_idx_) == max(2, int(round(frac * n)))
    assert len(method.anchor_idx_) >= int(0.05 * n)
    assert len(set(method.anchor_idx_.tolist())) == len(method.anchor_idx_)


def test_rift_st_is_transductive_only_and_adds_pseudo_labels():
    train, pool, (Xt, yt) = _split(make_synthetic_domains(n_domains=3, n=600, seed=6))
    method = make_method("rift-st", 0)
    assert method.transductive and not method.supports_inductive
    method.fit(train, pool)
    assert method.info["st_pseudo_attack"] + method.info["st_pseudo_benign"] > 0
    proba = method.predict_proba(Xt)
    assert np.all((proba >= 0) & (proba <= 1))


def test_coral_aligned_covariance_matches_target_covariance_plus_identity():
    rng = np.random.default_rng(0)
    M1 = np.array([[3, 0.5, 0, 0], [0, 2, 0.3, 0], [0, 0, 1.5, 0.2], [0, 0, 0, 1.0]])
    M2 = np.array([[1, 0, 0, 0], [0.4, 2, 0, 0], [0, 0.5, 1, 0], [0.2, 0, 0, 0.7]])
    # The +I regulariser makes the identity approximate; it tightens as source variance grows.
    Xs = (rng.normal(size=(20000, 4)) @ M1) * 10.0 + 5.0
    Xt = (rng.normal(size=(20000, 4)) @ M2) * 10.0 - 2.0
    aligned = coral_align(Xs, Xt)
    target_cov = np.cov(Xt, rowvar=False) + np.eye(4)
    assert np.allclose(np.cov(aligned, rowvar=False), target_cov, rtol=0.02, atol=0.02 * np.abs(target_cov).max())


def test_coral_has_no_mean_shift():
    rng = np.random.default_rng(1)
    Xs = rng.normal(size=(3000, 3)) + 4.0
    Xt = rng.normal(size=(3000, 3)) * 2.0 - 7.0
    aligned = coral_align(Xs, Xt)
    # A pure linear re-colouring: Xs @ T with no intercept, so the sources are not moved to the target mean.
    T = np.linalg.lstsq(Xs, aligned, rcond=None)[0]
    assert np.allclose(aligned, Xs @ T, atol=1e-8)
    assert np.linalg.norm(aligned.mean(axis=0) - Xt.mean(axis=0)) > 5.0


def test_dann_gradient_reversal_negates_and_scales_the_gradient():
    torch = pytest.importorskip("torch")
    from crossids.methods.dann import _GradientReversal

    x = torch.randn(4, 3, requires_grad=True)
    w = torch.tensor([[1.0, -2.0, 0.5]] * 4)
    out = _GradientReversal.apply(x, 0.7)
    assert torch.equal(out.detach(), x.detach())
    (out * w).sum().backward()
    assert torch.allclose(x.grad, -0.7 * w)


def test_xgboost_and_catboost_learn_and_write_no_files_in_cwd(tmp_path, monkeypatch):
    train, pool, (Xt, yt) = _split(make_synthetic_domains(n_domains=3, n=600, seed=7))
    monkeypatch.chdir(tmp_path)
    for name in ("xgboost", "catboost"):
        method = make_method(name, 0, threads=1, device="cpu").fit(train)
        assert evaluate(yt, method.predict_proba(Xt), method.predict(Xt))["roc_auc"] > 0.7
    assert not (tmp_path / "catboost_info").exists()


def test_em_recovers_prior_through_rift_under_pure_label_shift():
    """Same class-conditionals in sources and target; only the attack fraction differs."""

    def draw(n, pi, rng):
        y = (rng.random(n) < pi).astype(int)
        X = rng.normal(size=(n, 5)) + y[:, None] * np.array([1.2, 1.0, 0.8, 0.0, 0.0])
        X[:, 3] = rng.normal(size=n) + 1.0 * y
        return X, y

    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        sources = [Domain("a", *draw(4000, 0.5, rng)), Domain("b", *draw(4000, 0.5, rng))]
        pool, _ = draw(6000, 0.2, rng)
        _, y_test = draw(6000, 0.2, rng)
        # Global scaler: per-domain normalisers fitted on the pool change the marginals under label shift,
        # which biases EM (see README); this test isolates the label-shift estimate.
        method = Rift(RiftConfig(dn="global"), seed=seed).fit(sources, pool)
        assert abs(method.info["pi_t"] - y_test.mean()) < 0.07


def _pure_label_shift(pi_t, seed):
    """Same class-conditionals in both sources and target; only the attack fraction differs."""

    def draw(n, pi, rng):
        y = (rng.random(n) < pi).astype(int)
        X = rng.normal(size=(n, 5)) + y[:, None] * np.array([1.2, 1.0, 0.8, 0.0, 0.0])
        X[:, 3] = rng.normal(size=n) + 1.0 * y
        return X, y

    rng = np.random.default_rng(seed)
    sources = [Domain("a", *draw(4000, 0.5, rng)), Domain("b", *draw(4000, 0.5, rng))]
    pool, y_pool = draw(6000, pi_t, rng)
    return sources, pool, float(y_pool.mean())


def test_prior_init_is_validated_and_registered_with_the_intended_defaults():
    from crossids.methods import METHOD_NAMES, RIFT_VARIANTS

    assert RiftConfig().prior_init == "source"
    assert RIFT_VARIANTS["rift-st"].prior_init == "source"
    assert RIFT_VARIANTS["rift-poolInit"].prior_init == "pool"
    assert "rift-poolInit" in METHOD_NAMES
    with pytest.raises(ValueError):
        Rift(RiftConfig(prior_init="other"))


@pytest.mark.parametrize("pi_t", [0.2, 0.8])
def test_source_initial_prior_is_accurate_under_pure_label_shift(pi_t):
    """The first EM estimate (pool mapped through the pooled source benign-anchored normaliser)."""
    for seed in (0, 1, 2):
        sources, pool, truth = _pure_label_shift(pi_t, seed)
        method = Rift(RiftConfig(prior_init="source"), seed=seed).fit(sources, pool)
        assert abs(method.info["pi_init"] - truth) < 0.07


@pytest.mark.xfail(
    strict=True,
    reason="anchor-matched transport still biased under pure label shift: anchor purity 0.97 (true 0.2) and "
    "0.80 (true 0.8) and the cut point depend on the estimated prior; measured 0.23-0.32 at true 0.2, "
    "0.86-0.95 at true 0.8; see README",
)
@pytest.mark.parametrize("pi_t", [0.2, 0.8])
def test_default_rift_prior_under_pure_label_shift_within_tolerance(pi_t):
    for seed in (0, 1, 2):
        sources, pool, truth = _pure_label_shift(pi_t, seed)
        method = Rift(seed=seed).fit(sources, pool)
        assert abs(method.info["pi_t"] - truth) < 0.07


def test_method_prior_estimate_has_no_target_labels_in_the_pool_argument():
    params = inspect.signature(Rift.fit).parameters
    assert list(params) == ["self", "sources", "target_pool"]


def test_rift_ranking_score_is_posterior_plus_a_tiny_raw_term(synthetic):
    train, pool, (Xt, _) = synthetic
    method = make_method("rift", 0).fit(train, pool)
    posterior = method.predict_proba(Xt)
    ranking = method.decision_scores(Xt)
    assert np.all(np.abs(ranking - posterior) <= 1e-6 + 1e-12)
    assert len(np.unique(ranking)) >= len(np.unique(posterior))
