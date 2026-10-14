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
    attack_train: np.ndarray | None = field(default=None, repr=False)  # raw attack_type of the train part (FLARE-2)


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
        attack_train=train["attack_type"].to_numpy(dtype=object),
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
    """Episode shape (§14): support size log-uniform in [k_min, k_max], query size and priors.

    ``context_max`` > 0 turns on the unlabelled context rows of §20.1: U ~ U{0..context_max} rows from the same
    dataset, with their own attack prior ``context_prior``. The default 0 leaves episodes exactly as in §14.
    """

    k_min: int = 4
    k_max: int = 128
    query_size: int = 256
    query_prior: tuple[float, float] = (0.05, 0.95)
    support_prior: tuple[float, float] = (0.1, 0.9)
    clip: float = CLIP
    aug: AugConfig = field(default_factory=AugConfig)
    context_max: int = 0
    context_prior: tuple[float, float] = (0.05, 0.95)


@dataclass
class Episode:
    """Normalised episode. Rows are support first, then context (``n_context`` unlabelled rows), then query.

    ``label`` holds the (possibly noisy) support label, and ``LABEL_UNKNOWN`` for context and query rows.
    ``truth`` holds the true label of every row; only query truth is used for the loss.
    """

    x: np.ndarray
    label: np.ndarray
    truth: np.ndarray
    n_support: int
    n_context: int = 0
    meta: dict = field(default_factory=dict)
    # FLARE-2 class episodes (section 21): label holds class ids (-1 = unknown), truth class ids for every row
    n_classes: int = 0
    slot_of: np.ndarray | None = None  # (n_classes,) label slot of each class; None for the binary episodes
    n_slots: int = 0


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


def _partition(rng: np.random.Generator, pool: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    """Disjoint row groups of the given sizes from one pool (with replacement only if the pool is too small)."""
    total = int(sum(sizes))
    if total == 0:
        return [pool[:0] for _ in sizes]
    if len(pool) == 0:
        raise ValueError("empty class pool")
    idx = rng.choice(len(pool), size=total, replace=total > len(pool))
    out, o = [], 0
    for s in sizes:
        out.append(pool[idx[o:o + s]])
        o += s
    return out


def _disjoint_rows(rng, benign, attack, attack_pool, k, n_s_att, u, c_att, q, q_att):
    """Support, context and query rows of one episode. Rows of one dataset never repeat across the three parts.

    Attacks come from ``attack_pool`` (another dataset for composite episodes); context attacks always come from the
    episode's own dataset, so they can never coincide with a composite row.
    """
    if attack_pool is attack:
        s_att, q_att_rows, c_att_rows = _partition(rng, attack, [n_s_att, q_att, c_att])
    else:
        s_att, q_att_rows = _partition(rng, attack_pool, [n_s_att, q_att])
        (c_att_rows,) = _partition(rng, attack, [c_att])
    s_ben, q_ben, c_ben = _partition(rng, benign, [k - n_s_att, q - q_att, u - c_att])
    s_rows = np.concatenate([s_att, s_ben])
    s_lab = np.concatenate([np.ones(len(s_att), np.int64), np.zeros(len(s_ben), np.int64)])
    c_rows = np.concatenate([c_att_rows, c_ben])
    c_lab = np.concatenate([np.ones(len(c_att_rows), np.int64), np.zeros(len(c_ben), np.int64)])
    q_rows = np.concatenate([q_att_rows, q_ben])
    q_lab = np.concatenate([np.ones(len(q_att_rows), np.int64), np.zeros(len(q_ben), np.int64)])
    return s_rows, s_lab, c_rows, c_lab, q_rows, q_lab


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
    if cfg.context_max > 0:
        # §20.1: context rows, drawn without replacement so that support, context and query are disjoint rows
        u = int(rng.integers(0, cfg.context_max + 1))
        c_att = int(np.clip(np.round(u * rng.uniform(*cfg.context_prior)), 0, u))
        s_rows, s_lab, c_rows, c_lab, q_rows, q_lab = _disjoint_rows(
            rng, benign, attack, attack_pool, k, n_s_att, u, c_att, q, q_att)
        s_order = rng.permutation(k)
        c_order = rng.permutation(u)
        q_order = rng.permutation(q)
        s_rows, s_lab = s_rows[s_order], s_lab[s_order]
        c_rows, c_lab = c_rows[c_order], c_lab[c_order]
        q_rows, q_lab = q_rows[q_order], q_lab[q_order]
    else:
        u, c_att = 0, 0
        s_rows = np.concatenate([_take(rng, attack_pool, n_s_att), _take(rng, benign, k - n_s_att)])
        s_lab = np.concatenate([np.ones(n_s_att, np.int64), np.zeros(k - n_s_att, np.int64)])
        q_rows = np.concatenate([_take(rng, attack_pool, q_att), _take(rng, benign, q - q_att)])
        q_lab = np.concatenate([np.ones(q_att, np.int64), np.zeros(q - q_att, np.int64)])
        s_order = rng.permutation(k)
        q_order = rng.permutation(q)
        s_rows, s_lab = s_rows[s_order], s_lab[s_order]
        q_rows, q_lab = q_rows[q_order], q_lab[q_order]
        c_rows = np.zeros((0, d), dtype=s_rows.dtype)
        c_lab = np.zeros(0, np.int64)

    if aug.subset and d > aug.min_features and rng.random() < aug.subset_p:
        m = int(rng.integers(min(aug.min_features, d), d + 1))
        cols = np.sort(rng.choice(d, size=m, replace=False))
    else:
        cols = np.arange(d)

    X = np.vstack([s_rows, c_rows, q_rows])[:, cols].astype(np.float64)
    truth = np.concatenate([s_lab, c_lab, q_lab])
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
            "composite": composite_attack is not None, "context": u, "context_attack": c_att}
    info.update(meta or {})
    return Episode(x=gauss_rank(X, cfg.clip).astype(np.float32), label=label, truth=truth, n_support=k,
                   n_context=u, meta=info)


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


# ---------- FLARE-2: held-out-source validation and label-space tasks (DESIGN.md section 21) ----------

VALIDATION_ORDER = ("cicids2017", "cicids2018", "usbids1", "insdn", "unswnb15", "toniot", "iot23")
MIN_CLASS_ROWS = 30
TASK_BINARY, TASK_OVB, TASK_CVC, TASK_NWAY = 0, 1, 2, 3
TASK_NAMES = ("binary", "one_vs_benign", "class_vs_class", "nway")
DEFAULT_TASK_MIX = (0.25, 0.25, 0.15, 0.35)
SUPPORT_CAP = 160  # total support rows of one episode
CLASS_K_MAX = 32  # per-class support, k ~ log-uniform on {1..32}
NWAY_RANGE = (3, 6)  # N ~ U{3..6} for N-way tasks


def validation_source(target: str) -> str:
    """V of section 21.2: the next dataset after ``target`` in VALIDATION_ORDER, wrapping and skipping the target.

    A target outside the order (CTU-13) has no position in it and takes the first dataset of the order.
    """
    if target not in VALIDATION_ORDER:
        return VALIDATION_ORDER[0]
    i = VALIDATION_ORDER.index(target)
    return VALIDATION_ORDER[(i + 1) % len(VALIDATION_ORDER)]


def heldout_sources(target: str) -> tuple[list[str], str]:
    """(training datasets, validation dataset V) of held-out-source selection: the meta-training datasets minus V."""
    keys = meta_train_keys(target)
    v = validation_source(target)
    if v not in keys:
        raise ValueError(f"validation dataset {v} is not a meta-training dataset of {target}")
    return [k for k in keys if k != v], v


@dataclass
class ClassPools:
    """Attack-class rows of one dataset. Classes are raw ``attack_type`` values with >= MIN_CLASS_ROWS train rows."""

    name: str
    benign: np.ndarray
    classes: list[str]
    attack: list[np.ndarray]

    @property
    def n_rows(self) -> int:
        return len(self.benign) + sum(len(a) for a in self.attack)

    @property
    def all_attack(self) -> np.ndarray:
        return np.concatenate(self.attack)


def class_pools(split: Split, which: str = "train", min_rows: int = MIN_CLASS_ROWS) -> ClassPools:
    """Class pools of a split. The class list always comes from the train part (so validation uses the same classes)."""
    if split.attack_train is None:
        raise ValueError("split has no attack_type of its train part")
    y_tr, a_tr = split.y_train, split.attack_train
    counts = {}
    for c in a_tr[y_tr == 1]:
        counts[c] = counts.get(c, 0) + 1
    names = sorted(c for c, n in counts.items() if n >= min_rows)
    if which == "train":
        X, y, a = split.X_train, split.y_train, split.attack_train
    elif which == "test":
        X, y, a = split.X_test, split.y_test, split.attack_test
    else:
        raise ValueError(f"unknown part {which!r}")
    attack = [X[(y == 1) & (a == c)] for c in names]
    keep = [i for i, rows in enumerate(attack) if len(rows) > 0]
    return ClassPools(split.key, X[y == 0], [names[i] for i in keep], [attack[i] for i in keep])


@dataclass(frozen=True)
class TaskConfig:
    """Task mix of section 21.3: binary, one class vs benign, class vs class, N-way (probabilities summing to one)."""

    mix: tuple[float, float, float, float] = DEFAULT_TASK_MIX
    nway: tuple[int, int] = NWAY_RANGE
    k_max: int = CLASS_K_MAX
    support_cap: int = SUPPORT_CAP

    def __post_init__(self) -> None:
        if len(self.mix) != 4 or min(self.mix) < 0 or abs(sum(self.mix) - 1.0) > 1e-9:
            raise ValueError("task mix must be four non-negative probabilities summing to one")

    @classmethod
    def binary_only(cls) -> "TaskConfig":
        """Ablation ``--tasks binary``: only the binary attack-vs-benign task."""
        return cls(mix=(1.0, 0.0, 0.0, 0.0))


class TaskSampler:
    """Episode stream over label-space tasks of FLARE-2 (section 21.3). Dataset drawn with probability ~ sqrt(rows).

    Every episode draws a task type, its classes (benign and attack classes; A3 composites take the attack classes
    from another dataset than the benign rows), a per-class support size k (log-uniform on {1..32}, with at most
    ``support_cap`` support rows in total), unlabelled context rows when ``cfg.context_max`` > 0, and a query set whose
    class prior is Dirichlet(1). Augmentations A1, A2 and A4 apply as in :class:`EpisodeSampler`. Class-to-slot maps
    are drawn per episode (a random injection into ``n_slots`` slots) unless ``slot_perm`` is False.
    """

    def __init__(self, pools: list[ClassPools], cfg: EpisodeConfig, seed: int, tasks: TaskConfig | None = None,
                 n_slots: int = 16, slot_perm: bool = True):
        pools = [p for p in pools if p.classes]
        if not pools:
            raise ValueError("no dataset has an attack class with enough rows")
        self.pools = list(pools)
        self.cfg = cfg
        self.tasks = tasks or TaskConfig()
        self.n_slots = int(n_slots)
        self.slot_perm = bool(slot_perm)
        self.rng = np.random.default_rng(seed)
        w = np.sqrt(np.array([p.n_rows for p in self.pools], dtype=np.float64))
        self.weights = w / w.sum()

    def _parts(self, rng, kind: int, pa: ClassPools, src: ClassPools):
        """(names, row pools, benign-first) for a task kind, or None when the kind is infeasible for this draw."""
        if kind == TASK_CVC:
            if len(pa.attack) < 2:
                return None
            i, j = rng.choice(len(pa.attack), size=2, replace=False)
            return [pa.classes[i], pa.classes[j]], [pa.attack[i], pa.attack[j]]
        if kind == TASK_BINARY:
            return ["benign", "attack"], [pa.benign, src.all_attack]
        if kind == TASK_OVB:
            c = int(rng.integers(len(src.attack)))
            return ["benign", src.classes[c]], [pa.benign, src.attack[c]]
        items_n = [("benign", pa.benign)] + [(c, rows) for c, rows in zip(src.classes, src.attack)]
        n = min(int(rng.integers(self.tasks.nway[0], self.tasks.nway[1] + 1)), len(items_n))
        if n < 3:
            return None
        pick = rng.choice(len(items_n), size=n, replace=False)
        return [items_n[i][0] for i in pick], [items_n[i][1] for i in pick]

    def sample(self) -> Episode:
        rng = self.rng
        cfg, aug = self.cfg, self.cfg.aug
        a = int(rng.choice(len(self.pools), p=self.weights))
        pa = self.pools[a]
        kind = int(rng.choice(4, p=np.asarray(self.tasks.mix)))
        src, comp_name = pa, None
        if kind != TASK_CVC and aug.composite and len(self.pools) > 1 and rng.random() < aug.composite_p:
            others = [i for i in range(len(self.pools)) if i != a]
            src = self.pools[others[int(rng.integers(len(others)))]]
            comp_name = src.name
        parts = self._parts(rng, kind, pa, src)
        if parts is None:  # infeasible draw: fall back to one class vs benign, which every dataset supports
            kind = TASK_OVB
            parts = self._parts(rng, kind, pa, src)
        names, rows = parts
        if kind in (TASK_CVC, TASK_NWAY):
            order = rng.permutation(len(rows))
            names = [names[i] for i in order]
            rows = [rows[i] for i in order]
        N = len(rows)
        d = rows[0].shape[1]
        cap = max(1, self.tasks.support_cap // N)
        k = int(np.clip(np.floor(np.exp(rng.uniform(0.0, np.log(self.tasks.k_max + 1)))), 1, self.tasks.k_max))
        k = min(k, cap)
        u = int(rng.integers(0, cfg.context_max + 1)) if cfg.context_max > 0 else 0
        u_counts = rng.multinomial(u, rng.dirichlet(np.ones(N))) if u > 0 else np.zeros(N, np.int64)
        q = cfg.query_size
        q_counts = 1 + rng.multinomial(q - N, rng.dirichlet(np.ones(N)))

        S, Sy, C, Cy, Q, Qy = [], [], [], [], [], []
        for c in range(N):
            s_r, c_r, q_r = _partition(rng, rows[c], [k, int(u_counts[c]), int(q_counts[c])])
            S.append(s_r); Sy.append(np.full(len(s_r), c, np.int64))
            C.append(c_r); Cy.append(np.full(len(c_r), c, np.int64))
            Q.append(q_r); Qy.append(np.full(len(q_r), c, np.int64))
        S, Sy = np.vstack(S), np.concatenate(Sy)
        C, Cy = np.vstack(C), np.concatenate(Cy)
        Q, Qy = np.vstack(Q), np.concatenate(Qy)
        s_order, c_order, q_order = rng.permutation(len(S)), rng.permutation(len(C)), rng.permutation(len(Q))
        S, Sy, C, Cy, Q, Qy = S[s_order], Sy[s_order], C[c_order], Cy[c_order], Q[q_order], Qy[q_order]

        s_lab = Sy.copy()
        if aug.label_noise and N > 1:  # A4: a noisy support label names one of the other classes
            noisy = rng.random(len(s_lab)) < aug.noise_rate
            shift = rng.integers(1, N, size=int(noisy.sum()))
            s_lab[noisy] = (s_lab[noisy] + shift) % N

        X = np.vstack([S, C, Q])
        truth = np.concatenate([Sy, Cy, Qy])
        n_s, n_c = len(S), len(C)
        cols = np.arange(d)
        if aug.subset and d > aug.min_features and rng.random() < aug.subset_p:
            m = int(rng.integers(min(aug.min_features, d), d + 1))
            cols = np.sort(rng.choice(d, size=m, replace=False))
        X = X[:, cols].astype(np.float64)
        if aug.sign_flip:
            flipped = rng.random(len(cols)) < aug.sign_p
            X[:, flipped] *= -1.0
        label = np.concatenate([s_lab, np.full(n_c + len(Q), -1, np.int64)])
        slot_of = rng.permutation(self.n_slots)[:N] if self.slot_perm else np.arange(N)
        binary = kind in (TASK_BINARY, TASK_OVB)
        meta = {"source": pa.name, "source_attack": comp_name, "task": TASK_NAMES[kind], "classes": names,
                "k": k, "binary_task": binary, "composite": comp_name is not None, "context": u}
        return Episode(x=gauss_rank(X, cfg.clip).astype(np.float32), label=label, truth=truth.astype(np.int64),
                       n_support=n_s, n_context=n_c, meta=meta, n_classes=N,
                       slot_of=np.asarray(slot_of, dtype=np.int64), n_slots=self.n_slots)

    def batch(self, size: int) -> list[Episode]:
        return [self.sample() for _ in range(size)]


@dataclass
class Batch:
    """Padded episodes. Row and feature masks mark the real entries."""

    x: np.ndarray  # (B, N, D) float32, padded values are 0
    feat_pad: np.ndarray  # (B, D) bool, True for real features
    row_valid: np.ndarray  # (B, N) bool
    is_query: np.ndarray  # (B, N) bool
    label: np.ndarray  # (B, N) int64 in {0, 1, LABEL_UNKNOWN}; FLARE-2: label slots, unknown = n_slots
    truth: np.ndarray  # (B, N) float32, true label of query rows (0 elsewhere)
    slot_target: np.ndarray | None = None  # FLARE-2: (B, N) int64 slot of the true class of query rows
    slot_active: np.ndarray | None = None  # FLARE-2: (B, M) bool, slots used by the episode's classes
    class_slots: np.ndarray | None = None  # FLARE-2: (B, C_MAX) int64 slot of each class, -1 padded
    binary: np.ndarray | None = None  # FLARE-2: (B,) bool, two-class episodes with class 1 = attack


def collate(episodes: list[Episode]) -> Batch:
    """Pad episodes into the block layout [support + context block | query block].

    The first block has ``max(n_support + n_context)`` columns and the query block ``max n_query`` columns.
    Padding sits inside each block, so every episode in a batch shares the same block boundary
    (``model.support_block_size``). FLARE-2 class episodes go to :func:`_collate_classes`.
    """
    if not episodes:
        raise ValueError("empty batch")
    if episodes[0].slot_of is not None:
        return _collate_classes(episodes)
    B = len(episodes)
    K = max(e.n_support + e.n_context for e in episodes)
    Qm = max(e.x.shape[0] - e.n_support - e.n_context for e in episodes)
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
        n_s = e.n_support + e.n_context  # support and context rows form the non-query block
        n_q = e.x.shape[0] - n_s
        d = e.x.shape[1]
        feat_pad[b, :d] = True
        rows = np.concatenate([np.arange(n_s), K + np.arange(n_q)])
        x[b, rows, :d] = e.x
        row_valid[b, rows] = True
        label[b, rows] = e.label
        truth[b, rows] = e.truth
    return Batch(x, feat_pad, row_valid, is_query, label, truth)


def _collate_classes(episodes: list[Episode]) -> Batch:
    """FLARE-2 batch: labels are slot ids (unknown = n_slots), plus slot targets, active slots and binary flags."""
    if any(e.slot_of is None for e in episodes):
        raise ValueError("cannot mix class episodes with binary episodes in one batch")
    B = len(episodes)
    M = episodes[0].n_slots
    K = max(e.n_support + e.n_context for e in episodes)
    Qm = max(e.x.shape[0] - e.n_support - e.n_context for e in episodes)
    N = K + Qm
    D = max(e.x.shape[1] for e in episodes)
    C_MAX = max(e.n_classes for e in episodes)
    x = np.zeros((B, N, D), np.float32)
    feat_pad = np.zeros((B, D), bool)
    row_valid = np.zeros((B, N), bool)
    is_query = np.zeros((B, N), bool)
    is_query[:, K:] = True
    label = np.full((B, N), M, np.int64)
    truth = np.zeros((B, N), np.float32)
    slot_target = np.zeros((B, N), np.int64)
    slot_active = np.zeros((B, M), bool)
    class_slots = np.full((B, C_MAX), -1, np.int64)
    binary = np.zeros(B, bool)
    for b, e in enumerate(episodes):
        n_s = e.n_support + e.n_context
        n_q = e.x.shape[0] - n_s
        d = e.x.shape[1]
        feat_pad[b, :d] = True
        rows = np.concatenate([np.arange(n_s), K + np.arange(n_q)])
        x[b, rows, :d] = e.x
        row_valid[b, rows] = True
        lab = np.asarray(e.label, dtype=np.int64)
        label[b, rows] = np.where(lab >= 0, e.slot_of[np.clip(lab, 0, None)], M)
        truth[b, rows] = np.asarray(e.truth, dtype=np.float32)
        slot_target[b, K + np.arange(n_q)] = e.slot_of[np.asarray(e.truth[n_s:], dtype=np.int64)]
        slot_active[b, e.slot_of] = True
        class_slots[b, : e.n_classes] = e.slot_of
        binary[b] = bool(e.meta.get("binary_task", False))
    return Batch(x, feat_pad, row_valid, is_query, label, truth, slot_target, slot_active, class_slots, binary)


def _stream_seed(base_seed: int, stream: int, worker: int, n_workers: int) -> int:
    """Seed of one worker's episode generator. Depends only on (seed, stream offset, worker id, worker count)."""
    return int(np.random.SeedSequence([base_seed, stream, worker, n_workers]).generate_state(1)[0])


def batch_arrays(episodes: list[Episode]) -> dict[str, np.ndarray]:
    """Collated batch as a dict of numpy arrays (what the DataLoader moves to the main process)."""
    b = collate(episodes)
    out = {"x": b.x, "feat_pad": b.feat_pad, "row_valid": b.row_valid, "is_query": b.is_query,
           "label": b.label, "truth": b.truth}
    if b.slot_target is not None:
        out.update(slot_target=b.slot_target, slot_active=b.slot_active, class_slots=b.class_slots, binary=b.binary)
    return out


class EpisodeStream(IterableDataset):
    """Infinite stream of collated episode batches for a torch DataLoader (use ``batch_size=None``).

    Each worker draws from its own generator seeded by (seed, stream, worker id, worker count). With a fixed
    worker count the sequence of batches is deterministic. ``stream`` offsets the seed, which is how a resumed
    run starts a fresh stream.
    """

    def __init__(self, pools: list[Pools] | None, cfg: EpisodeConfig, seed: int, batch_size: int = 8,
                 stream: int = 0, synthetic: bool = False, max_features: int = 13, tasks: TaskConfig | None = None,
                 n_slots: int = 16, slot_perm: bool = True):
        super().__init__()
        self.pools = pools
        self.cfg = cfg
        self.seed = seed
        self.batch_size = batch_size
        self.stream = stream
        self.synthetic = synthetic
        self.max_features = max_features
        # FLARE-2: when ``tasks`` is given, ``pools`` are ClassPools and episodes come from the TaskSampler
        self.tasks = tasks
        self.n_slots = n_slots
        self.slot_perm = slot_perm

    def __iter__(self):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        seed = _stream_seed(self.seed, self.stream, worker, n_workers)
        if self.synthetic:
            sampler = SyntheticSampler(self.cfg, seed, max_features=self.max_features)
        elif self.tasks is not None:
            sampler = TaskSampler(self.pools, self.cfg, seed, self.tasks, self.n_slots, self.slot_perm)
        else:
            sampler = EpisodeSampler(self.pools, self.cfg, seed)
        while True:
            yield batch_arrays(sampler.batch(self.batch_size))
