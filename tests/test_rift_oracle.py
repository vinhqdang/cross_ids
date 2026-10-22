"""Oracle diagnostics of RIFT (true target prior, truly benign anchor): kept apart from the label-free methods."""
import numpy as np
import pytest

from crossids.datasets import make_synthetic_domains
from crossids.methods import ALL_METHOD_NAMES, METHOD_NAMES, make_method
from crossids.methods.base import Domain


def _setup():
    doms = make_synthetic_domains(n_domains=3, n=700, seed=0)
    train = [Domain(d.name, d.X[:490], d.y[:490], d.meta.iloc[:490]) for d in doms[:-1]]
    pool_X, pool_y = doms[-1].X[:490], doms[-1].y[:490]
    return train, pool_X, pool_y


def test_oracle_variants_are_registered_but_not_label_free_methods():
    for name in ("rift-oraclePrior", "rift-oracleAnchor"):
        assert name in ALL_METHOD_NAMES and name not in METHOD_NAMES
        assert make_method(name, seed=0) is not None


def test_oracle_prior_uses_the_true_pool_prior():
    train, X, y = _setup()
    m = make_method("rift-oraclePrior", seed=0)
    m.oracle_pool_labels = y
    m.fit(train, X)
    assert m.pi_t == pytest.approx(float(y.mean()))


def test_oracle_anchor_is_truly_benign():
    train, X, y = _setup()
    m = make_method("rift-oracleAnchor", seed=0)
    m.oracle_pool_labels = y
    m.fit(train, X)
    assert np.all(y[m.anchor_idx_] == 0)
    assert m.pi_t == pytest.approx(float(y.mean()))


def test_oracle_without_labels_is_an_error():
    train, X, _ = _setup()
    with pytest.raises(ValueError):
        make_method("rift-oraclePrior", seed=0).fit(train, X)
