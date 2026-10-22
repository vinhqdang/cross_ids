"""TabPFN control variants: ranked features and source rows as extra context."""
import numpy as np
import pytest

from crossids.flare import evaluate as ev
from crossids.methods.base import Domain


def _sources():
    rng = np.random.default_rng(0)
    return [Domain(f"d{i}", rng.normal(size=(500, 4)) + i, (rng.random(500) < 0.3).astype(int), None) for i in range(3)]


def test_variants_are_registered_with_their_base_model():
    assert ev.TABPFN_VARIANTS == {"tabpfn-v3.5-rank": ("tabpfn-v3.5", "rank"), "tabpfn-v3.5-src": ("tabpfn-v3.5", "src")}
    for name in ev.TABPFN_VARIANTS:
        assert name in ev.ALL_METHODS


def test_source_context_is_balanced_over_datasets_and_deterministic():
    st = ev.SeedState("t", 0, None, None, sources_fn=_sources)
    X, y = st.source_context(n=300, draw=1)
    assert X.shape == (300, 4) and y.shape == (300,)
    means = np.round([X[i * 100:(i + 1) * 100, 0].mean() for i in range(3)])
    assert list(means) == [0.0, 1.0, 2.0]  # 100 rows from each of the three datasets, in dataset order
    X2, _ = st.source_context(n=300, draw=1)
    X3, _ = st.source_context(n=300, draw=2)
    np.testing.assert_array_equal(X, X2)
    assert not np.array_equal(X, X3)


def test_rank_features_use_support_and_query_only():
    rng = np.random.default_rng(1)
    Xs, Xq = rng.normal(size=(8, 3)), rng.normal(size=(50, 3))
    Zs, Zq = ev.rank_support_query(Xs, Xq)
    assert Zs.shape == Xs.shape and Zq.shape == Xq.shape
    assert np.abs(np.vstack([Zs, Zq])).max() <= 4.0
