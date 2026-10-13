"""Method registry: every method fits and predicts; fit takes no target labels; RIFT vs HGB."""

import inspect

import numpy as np
import pytest

from crossids.datasets import make_synthetic_domains
from crossids.methods import METHOD_NAMES, make_method
from crossids.methods.base import Domain, Method
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


def test_rift_beats_hgb_on_synthetic_cross_domain():
    train, pool, (Xt, yt) = _split(make_synthetic_domains(n_domains=3, n=2000, seed=1))
    scores = {}
    for name in ("rift", "hgb"):
        method = make_method(name, 0)
        method.fit(train, pool if method.transductive else None)
        p = method.predict_proba(Xt)
        scores[name] = evaluate(yt, p, (p >= 0.5).astype(int))
    assert scores["rift"]["roc_auc"] > scores["hgb"]["roc_auc"]
    assert scores["rift"]["balanced_accuracy"] > scores["hgb"]["balanced_accuracy"]


def test_rift_reports_estimated_prior_and_selection(synthetic):
    train, pool, _ = synthetic
    method = make_method("rift", 0).fit(train, pool)
    assert 0.001 <= method.info["pi_t"] <= 0.999
    assert 0 < method.info["n_features"] <= train[0].X.shape[1]
