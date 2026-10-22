"""Few-shot evaluation of FLARE, its ablations, ProtoNet, active cold-start and the baselines (DESIGN.md §12, §16, §18).

For every (seed, k, draw) the same stratified support rows, drawn from the Part I target train split, are given to
every method. Metrics are computed on the Part I target test split. Results are appended to a CSV and rows already
present are skipped, so an interrupted run resumes.

Methods of §20: ``flare-u`` (FLARE-U, 512 unlabelled context rows from the target train split), ``flare+cb`` and
``flare-u+cb`` (mean of FLARE's and the target-only CatBoost's P(attack) on the same support), and the k = 0 zero-shot
references ``catboost-lodo`` (CatBoost on all meta-training datasets, no target labels) and
``catboost-pairwise-mean`` / ``catboost-pairwise-best`` (read from the Part I benchmark, ``results/runs.csv``).

``flare2`` is the FLARE-2 checkpoint (section 21) used through its binary API (P(attack) over two random label slots,
ensembled over sign flips and slot permutations). Attack-class evaluation is in ``crossids.flare.evaluate_class``.

Example::

    python -m crossids.flare.evaluate --target usbids1 --seeds 0 1 2 --runs-dir runs/flare --out runs/flare/eval_usbids1.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..methods.base import Domain
from ..methods.baselines import CatBoostBaseline
from ..metrics import evaluate as score_predictions
from ..metrics import per_attack_recall
from .al import empty_support_embeddings, select_flare_al, select_kcenter_raw
from .data import Split, gauss_rank, load_split, meta_train_keys, target_features, target_keys
from .model import EpisodeModel, load_model

log = logging.getLogger("crossids.flare.evaluate")

FEW_SHOT_KS = (4, 8, 16, 32, 64, 128)
N_DRAWS = 10
AL_POOL_CAP = 20_000
METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "f1")
RESULT_COLUMNS = ["method", "target", "k", "draw", "seed", *METRICS, "n_support", "n_attack_support", "seconds"]
ATTACK_COLUMNS = ["method", "target", "k", "draw", "seed", "attack_type", "n", "detected", "recall"]
FLARE_VARIANTS = {
    "flare": ("", None),
    "flare-E1": ("", 1),  # same checkpoint, no test-time ensemble (section 17)
    "flare-noA1": ("_noA1", None),
    "flare-noA3": ("_noA3", None),
    "flare-rowmlp": ("_rowmlp", None),
    "flare-synthetic": ("_synthetic", None),
    "flare-u": ("_ctx", None),  # FLARE-U (section 20.1): needs the context checkpoint and context rows
    "flare2": ("_f2", None),  # FLARE-2 (section 21.5): binary detection through the two-class API
    "flare2-binary": ("_f2_binary", None),  # ablation: binary attack-vs-benign episodes only
    "flare2-noperm": ("_f2_noperm", None),  # ablation: no class-to-slot permutation
    "flare2-insrc": ("_f2_insrc", None),  # ablation: in-source validation
    "flare2-ctx": ("_f2_ctx", None),  # FLARE-2 with context rows (needs the context checkpoint)
    "flare2-m": ("_f2m", None),  # FLARE-2 trained with the unsupported-query loss mask (same model kind as flare2)
    "flare2-m-binary": ("_f2m_binary", None),  # ablation: binary episodes only, with the loss mask
    "flare2-rank256": ("_f2", None),  # same checkpoint, ranks over support + 256 random query rows
    "flare2-rank0": ("_f2", None),  # same checkpoint, ranks over the support only
}
RANK_QUERY_CAP = {"flare2-rank256": 256, "flare2-rank0": 0}  # normalisation controls (transductive vs not)
CONTEXT_ROWS = 512  # unlabelled target rows given to flare-u and flare2-ctx (sections 20.1, 21)
CONTEXT_METHODS = ("flare-u", "flare2-ctx")
COMBOS = {"flare+cb": "flare", "flare-u+cb": "flare-u"}  # section 20.2: mean of the two P(attack)
COMBO_PARTS = ("catboost-target",)  # per-(k, draw) predictions cached within a draw and reused by the combinations
ZERO_SHOT = ("catboost-lodo", "catboost-pairwise-mean", "catboost-pairwise-best")  # section 20.3, k = 0 rows
PAIRWISE_METHOD = "catboost"
PROTONET = "protonet"
PROTONET_SUFFIX = "_protonet"
BASELINES = ("catboost-target", "rf", "lr", "knn", "catboost-ft", "tabpfn-v2", "tabpfn-v3", "tabpfn-v3.5",
             "tabpfn-v3.5-rank", "tabpfn-v3.5-src")
# TabPFN baselines. The default backend runs the weights on this machine (versions other than v2 need the Prior Labs
# license accepted for the token in TABPFN_TOKEN or the git-ignored token file). Setting TABPFN_BACKEND=api sends the
# support and query rows of the (public) benchmark data to the Prior Labs service instead, which has a daily quota
# and whose models can change over time. The token is never logged.
TABPFN_LOCAL = {"tabpfn-v2": "v2", "tabpfn-v3": "v3", "tabpfn-v3.5": "v3.5"}  # method -> tabpfn ModelVersion value
TABPFN_API = {"tabpfn-v2": "v2_default", "tabpfn-v3": "v3_default", "tabpfn-v3.5": "v3.5_default"}  # tabpfn_client
TABPFN_VERSIONS = TABPFN_LOCAL
# Controls: "rank" feeds TabPFN the Gaussian-rank features FLARE-2 sees (ranked over support and query rows);
# "src" adds source rows from the other datasets (labels as in the source) to the support set as context.
TABPFN_VARIANTS = {"tabpfn-v3.5-rank": ("tabpfn-v3.5", "rank"), "tabpfn-v3.5-src": ("tabpfn-v3.5", "src")}
SRC_CONTEXT_ROWS = 1000
SRC_POOL_PER_DOMAIN = 5000
TABPFN_API_RETRIES = 3
TABPFN_TOKEN_FILE = Path("runs/tabpfn_token.txt")  # git-ignored; the token is never logged or committed
TABPFN_CHUNK = int(os.environ.get("TABPFN_CHUNK", 30000))  # query rows per predict call (lower it to save GPU memory)
ACTIVE = ("flare-al", "flare-kcenter-raw")
FULLLABEL = "catboost-fulllabel"  # upper reference: trained on the whole target train split (k = 0 in the CSV)
ALL_METHODS = BASELINES + (PROTONET,) + tuple(FLARE_VARIANTS) + ACTIVE + (FULLLABEL,) + tuple(COMBOS) + ZERO_SHOT
PER_ATTACK_K = 16
DEFAULT_SEEDS = (0, 1, 2)


def stratified_draw(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k row indices with at least one row per class, allocated in proportion to class frequency."""
    y = np.asarray(y).astype(int)
    classes = [int(c) for c in np.unique(y)]
    if k < len(classes):
        raise ValueError(f"k={k} cannot hold {len(classes)} classes")
    members = {c: np.flatnonzero(y == c) for c in classes}
    n = len(y)
    alloc = {c: min(len(members[c]), max(1, int(round(k * len(members[c]) / n)))) for c in classes}
    while sum(alloc.values()) > k:
        c = max((c for c in classes if alloc[c] > 1), key=lambda c: alloc[c], default=None)
        if c is None:
            break
        alloc[c] -= 1
    while sum(alloc.values()) < k:
        room = [c for c in classes if alloc[c] < len(members[c])]
        if not room:
            break
        c = max(room, key=lambda c: len(members[c]) - alloc[c])
        alloc[c] += 1
    rows = [rng.choice(members[c], size=alloc[c], replace=False) for c in classes]
    return np.sort(np.concatenate(rows)).astype(np.int64)


def draw_supports(y_train: np.ndarray, seed: int, ks=FEW_SHOT_KS, n_draws: int = N_DRAWS) -> dict[int, list[np.ndarray]]:
    return {k: [stratified_draw(y_train, k, np.random.default_rng([seed, k, d])) for d in range(n_draws)] for k in ks}


def load_or_draw_supports(path: Path, target: str, seed: int, y_train: np.ndarray, ks=FEW_SHOT_KS,
                          n_draws: int = N_DRAWS) -> dict[int, list[np.ndarray]]:
    """Support draws shared by all methods. Saved as JSON of target-train row indices and reused when present."""
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        cached = {int(k): v for k, v in blob.get("supports", {}).items()}
        if (blob.get("target"), blob.get("seed"), blob.get("n_train")) == (target, seed, len(y_train)) and all(
            k in cached and len(cached[k]) >= n_draws for k in ks
        ):
            return {k: [np.asarray(d, dtype=np.int64) for d in cached[k][:n_draws]] for k in ks}
    drawn = draw_supports(y_train, seed, ks, n_draws)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"target": target, "seed": seed, "n_train": int(len(y_train)),
               "supports": {str(k): [d.tolist() for d in drawn[k]] for k in ks}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return drawn


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted row indices of a random subsample of about ``n`` rows that keeps the label proportions.

    Every label keeps at least one row. All rows are returned when there are at most ``n``. Deterministic in
    (y, n, seed), so every method scored on the subsample sees the same rows.
    """
    y = np.asarray(y)
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    parts = []
    for lab in np.unique(y):
        members = np.flatnonzero(y == lab)
        take = max(1, int(round(n * len(members) / len(y))))
        parts.append(rng.choice(members, size=min(take, len(members)), replace=False))
    return np.sort(np.concatenate(parts))


def rank_support_query(Xs: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-rank features of support and query rows, ranked jointly as FLARE-2 does."""
    Z = gauss_rank(np.vstack([Xs, Xq]))
    return Z[: len(Xs)], Z[len(Xs):]


def load_tabpfn_token(path: Path = TABPFN_TOKEN_FILE) -> bool:
    """Make TABPFN_TOKEN available (needed for TabPFN versions whose weights require a license acceptance).

    An existing environment variable wins. Otherwise the first line of ``path`` is used when the file exists.
    Returns whether a token is set afterwards.
    """
    import os

    if os.environ.get("TABPFN_TOKEN"):
        return True
    path = Path(path)
    if path.exists():
        token = path.read_text(encoding="utf8").strip()
        if token:
            os.environ["TABPFN_TOKEN"] = token.splitlines()[0].strip()
            return True
    return False


def tabpfn_backend() -> str:
    """'local' (default) or 'api', from the TABPFN_BACKEND environment variable."""
    import os

    backend = os.environ.get("TABPFN_BACKEND", "local")
    if backend not in ("local", "api"):
        raise ValueError(f"TABPFN_BACKEND must be 'local' or 'api', got {backend!r}")
    return backend


def _tabpfn(name: str, device: str, seed: int):
    try:  # use the Windows certificate store when available (weights and the API are reached over HTTPS)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    load_tabpfn_token()
    if tabpfn_backend() == "api":
        import os

        import tabpfn_client

        tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
        return tabpfn_client.TabPFNClassifier(model_path=TABPFN_API[name], random_state=seed)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(ModelVersion(TABPFN_LOCAL[name]), device=device,
                                                       random_state=seed)


def _tabpfn_proba(name: str, device: str, seed: int, Xs, ys, Xt) -> np.ndarray:
    """P(attack) of a TabPFN baseline on the query rows. API calls are retried on transient errors."""
    last = None
    for attempt in range(TABPFN_API_RETRIES if tabpfn_backend() == "api" else 1):
        try:
            est = _tabpfn(name, device, seed).fit(Xs, ys)
            return np.concatenate([est.predict_proba(Xt[i:i + TABPFN_CHUNK])[:, 1]
                                   for i in range(0, len(Xt), TABPFN_CHUNK)])
        except Exception as exc:  # network or service error: wait and retry
            last = exc
            log.warning("%s attempt %d failed: %s", name, attempt + 1, type(exc).__name__)
            time.sleep(30 * (attempt + 1))
    raise last


def tabpfn_status(name: str) -> tuple[bool, str]:
    """Whether the TabPFN version of method ``name`` can be used here (package, weights, license). Reason otherwise."""
    TABPFN_VERSIONS[name]  # KeyError for an unknown method
    load_tabpfn_token()
    try:
        X = np.random.default_rng(0).standard_normal((20, 3))
        y = (X[:, 0] > 0).astype(int)
        _tabpfn(name, "cpu", 0).fit(X, y).predict_proba(X[:2])
    except Exception as exc:  # depends on environment: package, weights download or license acceptance
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        return False, f"{type(exc).__name__}: {first[:160]}"
    return True, ""


@dataclass
class EvalContext:
    runs_dir: Path
    supports_dir: Path
    device: str = "cpu"
    threads: int = 2
    al_pool: int = AL_POOL_CAP
    ks: tuple[int, ...] = FEW_SHOT_KS
    n_draws: int = N_DRAWS
    benchmark_dir: Path = Path("results")  # folder with the Part I runs.csv (read only)
    max_query: int | None = None  # score every method on a fixed stratified subsample of this many test rows


@dataclass
class SeedState:
    """Per (target, seed) resources: the split, source domains for catboost-ft and cached models.

    ``cache`` holds the P(attack) of the components of the combinations for the current (k, draw) only.
    """

    target: str
    seed: int
    split: Split
    ctx: EvalContext
    sources_fn: object = None  # callable returning source Domains, or None
    models: dict = field(default_factory=dict)
    missing: set = field(default_factory=set)
    source_base: CatBoostBaseline | None = None
    pool: tuple | None = None
    emb0: np.ndarray | None = None
    cache: dict = field(default_factory=dict)
    source_pool: list | None = None

    def source_context(self, n: int = SRC_CONTEXT_ROWS, draw: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``n`` source rows, an equal share from every source dataset (dataset order), seeded by (seed, draw)."""
        if self.source_pool is None:
            rng = np.random.default_rng([self.seed, 5150])
            self.source_pool = []
            for dom in self.sources_fn():
                take = rng.permutation(len(dom.y))[:SRC_POOL_PER_DOMAIN]
                self.source_pool.append((np.asarray(dom.X)[take], np.asarray(dom.y)[take]))
        rng = np.random.default_rng([self.seed, 5151, int(draw)])
        per = n // len(self.source_pool)
        Xs, ys = [], []
        for X, y in self.source_pool:
            take = rng.permutation(len(y))[:per]
            Xs.append(X[take])
            ys.append(y[take])
        return np.vstack(Xs), np.concatenate(ys)

    def model(self, name: str) -> EpisodeModel | None:
        suffix = PROTONET_SUFFIX if name == PROTONET else FLARE_VARIANTS[name][0]
        if name not in self.models:
            path = self.ctx.runs_dir / f"{self.target}_s{self.seed}{suffix}" / "best.pt"
            if not path.exists():
                if name not in self.missing:
                    log.warning("skipping %s on %s seed %d: no checkpoint at %s", name, self.target, self.seed, path)
                    self.missing.add(name)
                return None
            ens = FLARE_VARIANTS[name][1] if name in FLARE_VARIANTS else None
            self.models[name] = load_model(path, device=self.ctx.device, ensemble=ens)
            if name in RANK_QUERY_CAP:
                self.models[name].rank_query_cap = RANK_QUERY_CAP[name]
        return self.models[name]

    def source_model(self) -> CatBoostBaseline:
        if self.source_base is None:
            if self.sources_fn is None:
                raise RuntimeError("catboost-ft needs source domains")
            self.source_base = CatBoostBaseline(seed=self.seed, threads=self.ctx.threads, device=self.ctx.device)
            self.source_base.fit(self.sources_fn())
        return self.source_base

    def pool_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Target train pool for active selection (seeded subsample, capped for cost)."""
        if self.pool is None:
            n = len(self.split.y_train)
            take = min(self.ctx.al_pool, n)
            idx = np.sort(np.random.default_rng([self.seed, 555]).choice(n, size=take, replace=False))
            self.pool = (self.split.X_train[idx], self.split.y_train[idx])
        return self.pool



def _sklearn_predict(estimator, Xs, ys, Xt) -> np.ndarray:
    return estimator.fit(Xs, ys).predict_proba(Xt)[:, 1]


def context_rows(split: Split, sup_idx: np.ndarray, seed: int, k: int, draw: int,
                 n: int = CONTEXT_ROWS) -> np.ndarray:
    """Unlabelled context rows of FLARE-U: ``n`` seeded rows of the target train split, disjoint from the support.

    Only features are returned. Labels of the context are never read.
    """
    keep = np.ones(len(split.y_train), dtype=bool)
    keep[np.asarray(sup_idx, dtype=np.int64)] = False
    pool = np.flatnonzero(keep)
    rng = np.random.default_rng([int(seed), 9001, int(k), int(draw)])
    take = min(int(n), len(pool))
    return split.X_train[np.sort(rng.choice(pool, size=take, replace=False))]


def _compute(name: str, st: SeedState, idx: np.ndarray, k: int, draw: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Uncached (P(attack) on the target test split, labels of the support actually used) or None if unavailable."""
    Xs, ys = st.split.X_train[idx], st.split.y_train[idx]
    Xt = st.split.X_test
    ctx = st.ctx
    if name in COMBOS:
        parts = []
        for part in (COMBOS[name], "catboost-target"):
            res = _run_method(part, st, idx, k, draw)
            if res is None:
                return None
            parts.append(res[0])
        return np.mean(parts, axis=0), ys
    if name == "catboost-target":
        from catboost import CatBoostClassifier

        est = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                                 verbose=0, random_seed=draw, allow_writing_files=False, thread_count=ctx.threads)
        return _sklearn_predict(est, Xs, ys, Xt), ys
    if name == "rf":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=ctx.threads, random_state=draw)
        return _sklearn_predict(est, Xs, ys, Xt), ys
    if name == "lr":
        est = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        return _sklearn_predict(est, Xs, ys, Xt), ys
    if name == "knn":
        est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, len(ys))))
        return _sklearn_predict(est, Xs, ys, Xt), ys
    if name == "catboost-ft":
        from catboost import CatBoostClassifier

        base = st.source_model()
        est = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                                 verbose=0, random_seed=draw, allow_writing_files=False, thread_count=ctx.threads)
        est.fit(base.scaler.transform(Xs.astype(np.float64)), ys, init_model=base.model)
        return est.predict_proba(base.scaler.transform(Xt.astype(np.float64)))[:, 1], ys
    if name in TABPFN_VARIANTS:
        base, kind = TABPFN_VARIANTS[name]
        device = "cuda" if ctx.device == "cuda" else "cpu"
        if kind == "rank":
            Zs, Zt = rank_support_query(Xs, Xt)
            return _tabpfn_proba(base, device, draw, Zs, ys, Zt), ys
        Xc, yc = st.source_context(SRC_CONTEXT_ROWS, draw)
        return _tabpfn_proba(base, device, draw, np.vstack([Xc, Xs]), np.concatenate([yc, ys]), Xt), ys
    if name in TABPFN_VERSIONS:
        return _tabpfn_proba(name, "cuda" if ctx.device == "cuda" else "cpu", draw, Xs, ys, Xt), ys
    if name in CONTEXT_METHODS:
        model = st.model(name)
        if model is None:
            return None
        ctx_X = context_rows(st.split, idx, st.seed, k, draw)
        return model.predict_proba(Xs, ys, Xt, seed=draw, context_X=ctx_X), ys
    if name in FLARE_VARIANTS or name == PROTONET:
        model = st.model(name)
        if model is None:
            return None
        return model.predict_proba(Xs, ys, Xt, seed=draw), ys
    if name in ACTIVE:
        base = st.model("flare")
        if base is None:
            return None
        pool_X, pool_y = st.pool_rows()
        rng = np.random.default_rng([st.seed, k, draw, 1 if name == "flare-al" else 2])
        if name == "flare-al":
            if st.emb0 is None:
                st.emb0 = empty_support_embeddings(base, pool_X)
            sel = select_flare_al(base, pool_X, pool_y, k, rng=rng, emb0=st.emb0)
        else:
            sel = select_kcenter_raw(pool_X, k, rng=rng)
        Xa, ya = pool_X[sel], pool_y[sel]
        return base.predict_proba(Xa, ya, Xt, seed=draw), ya
    raise KeyError(f"unknown method {name!r}")


def _run_method(name: str, st: SeedState, idx: np.ndarray, k: int, draw: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(P(attack) on the target test split, labels of the support used) for one method, or None if unavailable.

    Components of the combinations (catboost-target, the FLARE models) are kept in ``st.cache`` for the current
    (k, draw), so ``flare+cb`` reuses the CatBoost predictions of ``catboost-target`` instead of refitting.
    """
    cacheable = name in COMBO_PARTS or name in ("flare", "flare-u")
    key = (name, k, draw)
    if cacheable and key in st.cache:
        return st.cache[key]
    res = _compute(name, st, idx, k, draw)
    if cacheable and res is not None:
        st.cache[key] = res
    return res


def zero_shot_catboost_lodo(st: SeedState) -> np.ndarray:
    """CatBoost (catboost-ft base model) trained on all meta-training datasets pooled. Reads no target labels."""
    return st.source_model().predict_proba(st.split.X_test)


def pairwise_reference(target: str, benchmark: Path) -> dict[str, dict]:
    """Zero-shot Part I rows of one target: pairwise CatBoost, transductive, seed 0. Mean and oracle-best source.

    The mean row averages every metric over the sources. The best row is the source with the highest ROC-AUC and
    carries that source's metrics. Returns {} when the benchmark has no such rows.
    """
    path = Path(benchmark) / "runs.csv"
    if not path.exists():
        return {}
    runs = pd.read_csv(path, low_memory=False)
    sel = runs[(runs["experiment"] == "pairwise") & (runs["mode"] == "transductive")
               & (runs["method"] == PAIRWISE_METHOD) & (runs["seed"] == 0) & (runs["target"] == target)]
    if sel.empty:
        return {}
    if sel["sources"].duplicated().any():
        raise ValueError(f"duplicate pairwise sources for {target} in {path}")
    best = sel.loc[sel["roc_auc"].idxmax()]
    log.info("pairwise best source for %s: %s (roc_auc %.4f) of %d sources", target, best["sources"],
             float(best["roc_auc"]), len(sel))
    return {
        "catboost-pairwise-mean": {m: float(sel[m].mean()) for m in METRICS},
        "catboost-pairwise-best": {m: float(best[m]) for m in METRICS},
    }


def score_row(y_test: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    m = score_predictions(y_test, proba, (np.asarray(proba) >= 0.5).astype(int))
    return {name: m[name] for name in METRICS}


def _fulllabel_proba(split: Split, ctx: EvalContext) -> np.ndarray:
    from catboost import CatBoostClassifier

    est = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, auto_class_weights="Balanced", verbose=0,
                             random_seed=split.seed, allow_writing_files=False, thread_count=ctx.threads)
    return _sklearn_predict(est, split.X_train, split.y_train, split.X_test)


def _zero_shot_rows(target: str, st: SeedState, methods: list[str], done: set) -> list[dict]:
    """k = 0 rows (draw 0) of the zero-shot references. CatBoost on the meta-training datasets is fitted once."""
    rows = []
    split = st.split
    wanted = [m for m in ZERO_SHOT if m in methods and (m, target, 0, 0, split.seed) not in done]
    if "catboost-lodo" in wanted:
        t0 = time.perf_counter()
        proba = zero_shot_catboost_lodo(st)
        rows.append({"method": "catboost-lodo", "target": target, "k": 0, "draw": 0, "seed": split.seed,
                     **score_row(split.y_test, proba), "n_support": 0, "n_attack_support": 0,
                     "seconds": round(time.perf_counter() - t0, 3)})
    pairs = {}
    if split.seed == 0 and any(m.startswith("catboost-pairwise") for m in wanted):
        pairs = pairwise_reference(target, st.ctx.benchmark_dir)
        if not pairs:
            log.warning("no pairwise catboost rows for %s in %s", target, st.ctx.benchmark_dir)
    for name, metrics in pairs.items():
        if name in wanted:
            rows.append({"method": name, "target": target, "k": 0, "draw": 0, "seed": split.seed,
                         **metrics, "n_support": 0, "n_attack_support": 0, "seconds": 0.0})
    return rows


def evaluate_seed(target: str, split: Split, ctx: EvalContext, methods: list[str], *, sources_fn=None,
                  done: set | None = None, emit=None, emit_attack=None) -> list[dict]:
    """All requested (method, k, draw) rows for one target and seed. ``emit`` receives each row as it is produced.

    ``emit_attack`` receives per-attack-type recall rows for k = PER_ATTACK_K. The full-label CatBoost reference
    is computed once per seed and stored with k = 0 and draw = 0, as are the zero-shot references (ZERO_SHOT).
    """
    done = done or set()
    rows: list[dict] = []
    if ctx.max_query:
        sel = stratified_subsample(split.y_test, ctx.max_query, split.seed)
        split = replace(split, X_test=split.X_test[sel], y_test=split.y_test[sel],
                        attack_test=np.asarray(split.attack_test)[sel])
    st = SeedState(target, split.seed, split, ctx, sources_fn=sources_fn)
    sup = load_or_draw_supports(ctx.supports_dir / f"supports_{target}_s{split.seed}.json", target, split.seed,
                                split.y_train, ctx.ks, ctx.n_draws)
    y_test = split.y_test

    def _emit_all(batch: list[dict]) -> None:
        for row in batch:
            rows.append(row)
            if emit is not None:
                emit(row)

    if FULLLABEL in methods and (FULLLABEL, target, 0, 0, split.seed) not in done:
        proba = _fulllabel_proba(split, ctx)
        _emit_all([{"method": FULLLABEL, "target": target, "k": 0, "draw": 0, "seed": split.seed,
                    **score_row(y_test, proba), "n_support": int(len(split.y_train)),
                    "n_attack_support": int(np.sum(split.y_train)), "seconds": 0.0}])
    _emit_all(_zero_shot_rows(target, st, methods, done))
    methods = [m for m in methods if m != FULLLABEL and m not in ZERO_SHOT]
    for k in ctx.ks:
        for draw in range(ctx.n_draws):
            idx = sup[k][draw]
            st.cache.clear()
            for name in methods:
                key = (name, target, k, draw, split.seed)
                if key in done:
                    continue
                t0 = time.perf_counter()
                res = _run_method(name, st, idx, k, draw)
                if res is None:
                    continue
                proba, ys_used = res
                row = {"method": name, "target": target, "k": k, "draw": draw, "seed": split.seed,
                       **score_row(y_test, proba), "n_support": int(len(ys_used)),
                       "n_attack_support": int(np.sum(ys_used)), "seconds": round(time.perf_counter() - t0, 3)}
                rows.append(row)
                if emit is not None:
                    emit(row)
                if emit_attack is not None and k == PER_ATTACK_K:
                    per = per_attack_recall(y_test, (np.asarray(proba) >= 0.5).astype(int), split.attack_test)
                    for r in per.itertuples(index=False):
                        emit_attack({"method": name, "target": target, "k": k, "draw": draw, "seed": split.seed,
                                     "attack_type": r.attack_type, "n": int(r.n), "detected": int(r.detected),
                                     "recall": float(r.recall)})
                log.info("%s %s k=%d draw=%d seed=%d roc_auc=%.4f pr_auc=%.4f bal_acc=%.4f f1=%.4f",
                         name, target, k, draw, split.seed, row["roc_auc"], row["pr_auc"],
                         row["balanced_accuracy"], row["f1"])
    return rows


def _done_keys(path: Path) -> set[tuple]:
    """(method, target, k, draw, seed) of rows already in the CSV, so an interrupted run resumes."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=["method", "target", "k", "draw", "seed"])
    return {(str(r.method), str(r.target), int(r.k), int(r.draw), int(r.seed)) for r in frame.itertuples(index=False)}


def evaluate_target(target: str, seeds, out: Path, ctx: EvalContext, methods: list[str], data_dir: str | Path,
                    per_attack: Path | None = None) -> Path:
    """Evaluate one target over the given seeds and append the results to ``out`` (resumable).

    Per-attack recall goes to ``per_attack`` (default: ``per_attack_<target>.csv`` next to ``out``).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)
    feats = target_features(target)

    def emit(row: dict) -> None:
        pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)

    attack_out = Path(per_attack) if per_attack is not None else out.parent / f"per_attack_{target}.csv"
    attack_out.parent.mkdir(parents=True, exist_ok=True)

    def emit_attack(row: dict) -> None:
        pd.DataFrame([row], columns=ATTACK_COLUMNS).to_csv(attack_out, mode="a", header=not attack_out.exists(),
                                                           index=False)

    for seed in seeds:
        split = load_split(target, seed, data_dir, features=feats)

        def sources_fn(seed=seed):
            return [Domain(name=k, X=s.X_train.astype(np.float64), y=s.y_train) for k in meta_train_keys(target)
                    for s in [load_split(k, seed, data_dir, features=feats)]]

        evaluate_seed(target, split, ctx, methods, sources_fn=sources_fn, done=done, emit=emit,
                      emit_attack=emit_attack)
    return out


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m crossids.flare.evaluate", description="Few-shot evaluation of FLARE")
    p.add_argument("--target", required=True, choices=target_keys())
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    p.add_argument("--ks", nargs="+", type=int, default=list(FEW_SHOT_KS))
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--max-query", type=int, default=None,
                   help="score on a fixed stratified subsample of this many test rows (default: all)")
    p.add_argument("--out", default=None, help="CSV to append to (default runs/flare/eval_<target>.csv)")
    p.add_argument("--runs-dir", default="runs/flare")
    p.add_argument("--supports-dir", default=None, help="where supports_<target>_s<seed>.json lives (default: the CSV's folder)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--al-pool", type=int, default=AL_POOL_CAP)
    p.add_argument("--benchmark", default="results", help="folder with the Part I runs.csv (read only)")
    p.add_argument("--per-attack", default=None,
                   help="CSV for per-attack recall (default: per_attack_<target>.csv next to --out)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out or f"{args.runs_dir}/eval_{args.target}.csv")
    methods = list(args.methods)
    for name in [m for m in methods if m in TABPFN_VERSIONS or m in TABPFN_VARIANTS]:
        ok, reason = tabpfn_status(TABPFN_VARIANTS[name][0] if name in TABPFN_VARIANTS else name)
        if not ok:
            log.warning("skipping %s: %s", name, reason)
            methods.remove(name)
    ctx = EvalContext(runs_dir=Path(args.runs_dir), supports_dir=Path(args.supports_dir or out.parent),
                      device=args.device, threads=args.threads, al_pool=args.al_pool,
                      ks=tuple(args.ks), n_draws=args.draws, benchmark_dir=Path(args.benchmark),
                      max_query=args.max_query)
    path = evaluate_target(args.target, args.seeds, out, ctx, methods, args.data_dir, per_attack=args.per_attack)
    log.info("results in %s", path)


if __name__ == "__main__":
    main()
