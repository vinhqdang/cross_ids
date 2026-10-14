"""FLARE data: Part I target splits, Gaussian-rank normalisation and the meta-training episode sampler (DESIGN.md §12, §14).

Episodes are plain numpy objects. Batching and padding happen in :func:`collate`, the model consumes the
resulting :class:`Batch`. Every random draw goes through one ``numpy.random.Generator`` owned by the sampler,
so a seed fixes the whole episode stream. :class:`EpisodeStream` runs the sampler in DataLoader workers, each with
its own seeded generator.

The model sees per-episode ranks of each feature, so it is invariant to any strictly monotone per-feature transform
by construction. For that reason augmentation A1 is the sign flip only. The monotone warp is kept behind
``AugConfig.warp`` (default off) because rank normalisation erases it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from torch.utils.data import IterableDataset

from ..datasets import load_processed
from ..experiment import split_tables
from ..schema import CORE_FEATURES, MINIMAL_FEATURES, SOURCES, supports

TEST_SIZE = 0.3
LABEL_UNKNOWN = 2  # label-table index of query rows (benign 0, attack 1, unknown 2)
CLIP = 4.0


def target_keys() -> tuple[str, ...]:
    """All eight datasets, each usable as a LODO target (CTU-13 is the extra minimal-tier target)."""
    return tuple(SOURCES)


def target_features(key: str) -> tuple[str, ...]:
    """Core-tier datasets use the core tier, the rest (CTU-13) the minimal tier."""
    return CORE_FEATURES if supports(key, "core") else MINIMAL_FEATURES


def meta_train_keys(target: str) -> list[str]:
    """Core-tier datasets other than the target. CTU-13 is never a meta-training dataset."""
    return [k for k in SOURCES if supports(k, "core") and k != target]


@dataclass
class Split:
    """One dataset's Part I split (seeded, stratified 70/30) on a fixed feature list."""

    key: str
    seed: int
    features: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    attack_test: np.ndarray = field(repr=False)


def load_split(key: str, seed: int, data_dir: str | Path = "data", features: tuple[str, ...] | None = None) -> Split:
    """Part I split of one dataset. Uses the same ``split_tables`` call as the benchmark, so test rows are identical."""
    feats = tuple(features or target_features(key))
    frame = load_processed(key, Path(data_dir))
    train, test = split_tables({key: frame}, TEST_SIZE, seed)[key]
    return Split(
        key=key,
        seed=seed,
        features=feats,
        X_train=train[list(feats)].to_numpy(dtype=np.float32),
        y_train=train["label"].to_numpy(dtype=np.int64),
        X_test=test[list(feats)].to_numpy(dtype=np.float32),
        y_test=test["label"].to_numpy(dtype=np.int64),
        attack_test=test["attack_type"].to_numpy(dtype=object),
    )


def gauss_rank(X: np.ndarray, clip: float = CLIP) -> np.ndarray:
    """Per-column Gaussian rank over the rows given, clipped to [-clip, clip]. NaN entries stay NaN (absent)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("gauss_rank expects a 2-D array")
    out = np.full(X.shape, np.nan)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        c = int(m.sum())
        if c == 0:
            continue
        ranks = rankdata(col[m])
        out[m, j] = np.clip(norm.ppf(ranks / (c + 1.0)), -clip, clip)
    return out


def piecewise_warp(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Monotone piecewise-linear warp of one column, mapped through its own min/max (CDF-style knots on [0, 1])."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if not hi > lo:
        return x
    u = (x - lo) / (hi - lo)
    return lo + np.interp(u, xs, ys) * (hi - lo)


def warp_knots(rng: np.random.Generator, n_knots: int) -> tuple[np.ndarray, np.ndarray]:
    """Random knots of a monotone map on [0, 1] with fixed end points: n_knots interior knots per axis."""
    a = np.sort(rng.uniform(0.0, 1.0, n_knots))
    b = np.sort(rng.uniform(0.0, 1.0, n_knots))
    return np.concatenate([[0.0], a, [1.0]]), np.concatenate([[0.0], b, [1.0]])


@dataclass(frozen=True)
class AugConfig:
    """Switches for the §14 augmentations. Ablations turn individual switches off."""

    sign_flip: bool = True  # A1: sign flip per feature (p = sign_p)
    warp: bool = False  # A1 monotone warp (4 knots). Off by default: rank normalisation makes it a no-op.
    subset: bool = True  # A2 (p = subset_p)
    composite: bool = True  # A3 (p = composite_p)
    label_noise: bool = True  # A4 (rate = noise_rate on support labels)
    sign_p: float = 0.5
    subset_p: float = 0.5
    composite_p: float = 0.3
    noise_rate: float = 0.02
    warp_knots: int = 4
    min_features: int = 5


@dataclass(frozen=True)
class EpisodeConfig:
    """Episode shape (§14): support size log-uniform in [k_min, k_max], query size and priors."""

    k_min: int = 4
    k_max: int = 128
    query_size: int = 256
    query_prior: tuple[float, float] = (0.05, 0.95)
    support_prior: tuple[float, float] = (0.1, 0.9)
    clip: float = CLIP
    aug: AugConfig = field(default_factory=AugConfig)


@dataclass
class Episode:
    """Normalised episode. Rows are support first, then query.

    ``label`` holds the (possibly noisy) support label, and ``LABEL_UNKNOWN`` for query rows.
    ``truth`` holds the true label of every row; only query truth is used for the loss.
    """

    x: np.ndarray
    label: np.ndarray
    truth: np.ndarray
    n_support: int
    meta: dict = field(default_factory=dict)


@dataclass
class Pools:
    """Benign and attack rows of one meta-training dataset (its training split)."""

    name: str
    benign: np.ndarray
    attack: np.ndarray

    @property
    def n_rows(self) -> int:
        return len(self.benign) + len(self.attack)


def pools_from_split(split: Split, which: str = "train") -> Pools:
    """Class pools of a split's training part (default) or its held-out part (validation pools)."""
    if which == "train":
        X, y = split.X_train, split.y_train
    elif which == "test":
        X, y = split.X_test, split.y_test
    else:
        raise ValueError(f"unknown part {which!r}")
    return Pools(split.key, X[y == 0], X[y == 1])


def _take(rng: np.random.Generator, pool: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return pool[:0]
    if len(pool) == 0:
        raise ValueError("empty class pool")
    replace = len(pool) < n
    return pool[rng.choice(len(pool), size=n, replace=replace)]


def draw_episode(
    rng: np.random.Generator,
    benign: np.ndarray,
    attack: np.ndarray,
    cfg: EpisodeConfig,
    composite_attack: np.ndarray | None = None,
    meta: dict | None = None,
) -> Episode:
    """Build one episode from class pools.

    ``benign`` and ``attack`` are (n, d) arrays of the same features. ``composite_attack`` replaces the attack
    pool (A3, benign from one dataset and attack from another); the caller decides when to use it.
    """
    aug = cfg.aug
    d = benign.shape[1]
    k = int(np.floor(np.exp(rng.uniform(np.log(cfg.k_min), np.log(cfg.k_max + 1)))))
    k = int(np.clip(k, cfg.k_min, cfg.k_max))
    support_prior = rng.uniform(*cfg.support_prior)
    n_s_att = int(np.clip(np.round(k * support_prior), 1, k - 1))
    query_prior = rng.uniform(*cfg.query_prior)
    q = cfg.query_size
    q_att = int(np.round(q * query_prior))
    q_att = int(np.clip(q_att, 1, q - 1))

    attack_pool = composite_attack if composite_attack is not None else attack
    s_rows = np.concatenate([_take(rng, attack_pool, n_s_att), _take(rng, benign, k - n_s_att)])
    s_lab = np.concatenate([np.ones(n_s_att, np.int64), np.zeros(k - n_s_att, np.int64)])
    q_rows = np.concatenate([_take(rng, attack_pool, q_att), _take(rng, benign, q - q_att)])
    q_lab = np.concatenate([np.ones(q_att, np.int64), np.zeros(q - q_att, np.int64)])
    s_order = rng.permutation(k)
    q_order = rng.permutation(q)
    s_rows, s_lab = s_rows[s_order], s_lab[s_order]
    q_rows, q_lab = q_rows[q_order], q_lab[q_order]

    if aug.subset and d > aug.min_features and rng.random() < aug.subset_p:
        m = int(rng.integers(min(aug.min_features, d), d + 1))
        cols = np.sort(rng.choice(d, size=m, replace=False))
    else:
        cols = np.arange(d)

    X = np.vstack([s_rows, q_rows])[:, cols].astype(np.float64)
    truth = np.concatenate([s_lab, q_lab])
    n = X.shape[0]

    flipped = np.zeros(len(cols), dtype=bool)
    if aug.sign_flip:
        flipped = rng.random(len(cols)) < aug.sign_p
        X[:, flipped] *= -1.0
    if aug.warp:
        for j in range(X.shape[1]):
            xs, ys = warp_knots(rng, aug.warp_knots)
            X[:, j] = piecewise_warp(X[:, j], xs, ys)

    label = truth.copy()
    label[k:] = LABEL_UNKNOWN
    noisy = np.zeros(k, dtype=bool)
    if aug.label_noise:
        noisy = rng.random(k) < aug.noise_rate
        label[:k][noisy] = 1 - label[:k][noisy]

    info = {"k": k, "support_attack": n_s_att, "query_attack": q_att, "features": cols.tolist(),
            "flipped": flipped.tolist(), "label_noise_rows": int(noisy.sum()),
            "composite": composite_attack is not None}
    info.update(meta or {})
    return Episode(x=gauss_rank(X, cfg.clip).astype(np.float32), label=label, truth=truth, n_support=k, meta=info)


class EpisodeSampler:
    """Episode stream over real meta-training datasets. Dataset drawn with probability proportional to sqrt(rows)."""

    def __init__(self, pools: list[Pools], cfg: EpisodeConfig, seed: int):
        if not pools:
            raise ValueError("at least one dataset is required")
        self.pools = list(pools)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        w = np.sqrt(np.array([p.n_rows for p in self.pools], dtype=np.float64))
        self.weights = w / w.sum()

    def sample(self) -> Episode:
        a = int(self.rng.choice(len(self.pools), p=self.weights))
        pa = self.pools[a]
        comp_attack, b_name = None, None
        if self.cfg.aug.composite and len(self.pools) > 1 and self.rng.random() < self.cfg.aug.composite_p:
            others = [i for i in range(len(self.pools)) if i != a]
            b = int(others[self.rng.integers(len(others))])
            comp_attack = self.pools[b].attack
            b_name = self.pools[b].name
        return draw_episode(self.rng, pa.benign, pa.attack, self.cfg, comp_attack,
                            meta={"source": pa.name, "source_attack": b_name})

    def batch(self, size: int) -> list[Episode]:
        return [self.sample() for _ in range(size)]


def synthetic_pools(rng: np.random.Generator, n_rows: int = 2048, max_features: int = 13) -> Pools:
    """Random-function domain: correlated, partly skewed features and a labelling by a random 1-hidden-layer teacher.

    Used for the ``synthetic`` ablation (no IDS data) and for tests. The prior of each domain is random.
    """
    d = int(rng.integers(5, max_features + 1))
    z = rng.standard_normal((n_rows, d))
    x = z @ (rng.standard_normal((d, d)) / np.sqrt(d))
    for j in range(d):
        if rng.random() < 0.5:
            x[:, j] = np.exp(x[:, j] * rng.uniform(0.5, 1.5))
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    n_inf = int(rng.integers(2, min(d, 4) + 1))
    W = np.zeros((d, int(rng.integers(4, 9))))
    W[rng.choice(d, size=n_inf, replace=False)] = rng.standard_normal((n_inf, W.shape[1]))
    f = np.tanh(x @ W) @ rng.standard_normal(W.shape[1])
    pi = rng.uniform(0.05, 0.95)
    y = (f > np.quantile(f, 1.0 - pi)).astype(int)
    if y.sum() == 0 or y.sum() == n_rows:
        raise RuntimeError("degenerate synthetic domain")
    return Pools("synthetic", x[y == 0], x[y == 1])


class SyntheticSampler:
    """Episode stream of random-function domains (``synthetic`` ablation). Every episode has its own domain."""

    def __init__(self, cfg: EpisodeConfig, seed: int, max_features: int = 13):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.max_features = max_features

    def sample(self) -> Episode:
        pools = synthetic_pools(self.rng, max_features=self.max_features)
        return draw_episode(self.rng, pools.benign, pools.attack, self.cfg, meta={"source": "synthetic"})

    def batch(self, size: int) -> list[Episode]:
        return [self.sample() for _ in range(size)]


@dataclass
class Batch:
    """Padded episodes. Row and feature masks mark the real entries."""

    x: np.ndarray  # (B, N, D) float32, padded values are 0
    feat_pad: np.ndarray  # (B, D) bool, True for real features
    row_valid: np.ndarray  # (B, N) bool
    is_query: np.ndarray  # (B, N) bool
    label: np.ndarray  # (B, N) int64 in {0, 1, LABEL_UNKNOWN}
    truth: np.ndarray  # (B, N) float32, true label of query rows (0 elsewhere)


def collate(episodes: list[Episode]) -> Batch:
    """Pad episodes into the block layout [support block | query block].

    The support block has ``max n_support`` columns and the query block ``max n_query`` columns. Padding sits inside
    each block, so every episode in a batch shares the same block boundary (``model.support_block_size``).
    """
    if not episodes:
        raise ValueError("empty batch")
    B = len(episodes)
    K = max(e.n_support for e in episodes)
    Qm = max(e.x.shape[0] - e.n_support for e in episodes)
    N = K + Qm
    D = max(e.x.shape[1] for e in episodes)
    x = np.zeros((B, N, D), np.float32)
    feat_pad = np.zeros((B, D), bool)
    row_valid = np.zeros((B, N), bool)
    is_query = np.zeros((B, N), bool)
    is_query[:, K:] = True
    label = np.full((B, N), LABEL_UNKNOWN, np.int64)
    truth = np.zeros((B, N), np.float32)
    for b, e in enumerate(episodes):
        n_s = e.n_support
        n_q = e.x.shape[0] - n_s
        d = e.x.shape[1]
        feat_pad[b, :d] = True
        rows = np.concatenate([np.arange(n_s), K + np.arange(n_q)])
        x[b, rows, :d] = e.x
        row_valid[b, rows] = True
        label[b, rows] = e.label
        truth[b, rows] = e.truth
    return Batch(x, feat_pad, row_valid, is_query, label, truth)


def _stream_seed(base_seed: int, stream: int, worker: int, n_workers: int) -> int:
    """Seed of one worker's episode generator. Depends only on (seed, stream offset, worker id, worker count)."""
    return int(np.random.SeedSequence([base_seed, stream, worker, n_workers]).generate_state(1)[0])


def batch_arrays(episodes: list[Episode]) -> dict[str, np.ndarray]:
    """Collated batch as a dict of numpy arrays (what the DataLoader moves to the main process)."""
    b = collate(episodes)
    return {"x": b.x, "feat_pad": b.feat_pad, "row_valid": b.row_valid, "is_query": b.is_query,
            "label": b.label, "truth": b.truth}


class EpisodeStream(IterableDataset):
    """Infinite stream of collated episode batches for a torch DataLoader (use ``batch_size=None``).

    Each worker draws from its own generator seeded by (seed, stream, worker id, worker count). With a fixed
    worker count the sequence of batches is deterministic. ``stream`` offsets the seed, which is how a resumed
    run starts a fresh stream.
    """

    def __init__(self, pools: list[Pools] | None, cfg: EpisodeConfig, seed: int, batch_size: int = 8,
                 stream: int = 0, synthetic: bool = False, max_features: int = 13):
        super().__init__()
        self.pools = pools
        self.cfg = cfg
        self.seed = seed
        self.batch_size = batch_size
        self.stream = stream
        self.synthetic = synthetic
        self.max_features = max_features

    def __iter__(self):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        seed = _stream_seed(self.seed, self.stream, worker, n_workers)
        if self.synthetic:
            sampler = SyntheticSampler(self.cfg, seed, max_features=self.max_features)
        else:
            sampler = EpisodeSampler(self.pools, self.cfg, seed)
        while True:
            yield batch_arrays(sampler.batch(self.batch_size))
