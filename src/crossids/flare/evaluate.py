"""Few-shot evaluation of FLARE, its ablations, ProtoNet, active cold-start and the baselines (DESIGN.md §12, §16, §18).

For every (seed, k, draw) the same stratified support rows, drawn from the Part I target train split, are given to
every method. Metrics are computed on the Part I target test split. Results are appended to a CSV and rows already
present are skipped, so an interrupted run resumes.

Example::

    python -m crossids.flare.evaluate --target usbids1 --seeds 0 1 2 --runs-dir runs/flare --out runs/flare/eval_usbids1.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
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
from .data import Split, load_split, meta_train_keys, target_features, target_keys
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
}
PROTONET = "protonet"
PROTONET_SUFFIX = "_protonet"
BASELINES = ("catboost-target", "rf", "lr", "knn", "catboost-ft", "tabpfn")
ACTIVE = ("flare-al", "flare-kcenter-raw")
FULLLABEL = "catboost-fulllabel"  # upper reference: trained on the whole target train split (k = 0 in the CSV)
ALL_METHODS = BASELINES + (PROTONET,) + tuple(FLARE_VARIANTS) + ACTIVE + (FULLLABEL,)
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


def tabpfn_status() -> tuple[bool, str]:
    """Whether TabPFN can be used here (package installed and weights downloadable). Reason otherwise."""
    try:
        from tabpfn import TabPFNClassifier
    except Exception as exc:  # pragma: no cover - depends on environment
        return False, f"tabpfn not importable: {type(exc).__name__}"
    try:
        X = np.random.default_rng(0).standard_normal((20, 3))
        y = (X[:, 0] > 0).astype(int)
        TabPFNClassifier(device="cpu", n_estimators=1).fit(X, y).predict_proba(X[:2])
    except Exception as exc:  # pragma: no cover - needs Prior Labs weights or token
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        return False, f"weights unavailable ({type(exc).__name__}: {first[:160]})"
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


@dataclass
class SeedState:
    """Per (target, seed) resources: the split, source domains for catboost-ft and cached models."""

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


def _run_method(name: str, st: SeedState, Xs, ys, k: int, draw: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (P(attack) on the target test split, labels of the support actually used) or None if unavailable."""
    Xt = st.split.X_test
    ctx = st.ctx
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
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier

        est = TabPFNClassifier(device=ctx.device if ctx.device == "cuda" else "cpu", random_state=draw)
        return _sklearn_predict(est, Xs, ys, Xt), ys
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


def score_row(y_test: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    m = score_predictions(y_test, proba, (np.asarray(proba) >= 0.5).astype(int))
    return {name: m[name] for name in METRICS}


def _fulllabel_proba(split: Split, ctx: EvalContext) -> np.ndarray:
    from catboost import CatBoostClassifier

    est = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, auto_class_weights="Balanced", verbose=0,
                             random_seed=split.seed, allow_writing_files=False, thread_count=ctx.threads)
    return _sklearn_predict(est, split.X_train, split.y_train, split.X_test)


def evaluate_seed(target: str, split: Split, ctx: EvalContext, methods: list[str], *, sources_fn=None,
                  done: set | None = None, emit=None, emit_attack=None) -> list[dict]:
    """All requested (method, k, draw) rows for one target and seed. ``emit`` receives each row as it is produced.

    ``emit_attack`` receives per-attack-type recall rows for k = PER_ATTACK_K. The full-label CatBoost reference
    is computed once per seed and stored with k = 0 and draw = 0.
    """
    done = done or set()
    rows: list[dict] = []
    st = SeedState(target, split.seed, split, ctx, sources_fn=sources_fn)
    sup = load_or_draw_supports(ctx.supports_dir / f"supports_{target}_s{split.seed}.json", target, split.seed,
                                split.y_train, ctx.ks, ctx.n_draws)
    y_test = split.y_test
    if FULLLABEL in methods and (FULLLABEL, target, 0, 0, split.seed) not in done:
        proba = _fulllabel_proba(split, ctx)
        row = {"method": FULLLABEL, "target": target, "k": 0, "draw": 0, "seed": split.seed,
               **score_row(y_test, proba), "n_support": int(len(split.y_train)),
               "n_attack_support": int(np.sum(split.y_train)), "seconds": 0.0}
        rows.append(row)
        if emit is not None:
            emit(row)
    methods = [m for m in methods if m != FULLLABEL]
    for k in ctx.ks:
        for draw in range(ctx.n_draws):
            idx = sup[k][draw]
            Xs, ys = split.X_train[idx], split.y_train[idx]
            for name in methods:
                key = (name, target, k, draw, split.seed)
                if key in done:
                    continue
                t0 = time.perf_counter()
                res = _run_method(name, st, Xs, ys, k, draw)
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


def evaluate_target(target: str, seeds, out: Path, ctx: EvalContext, methods: list[str], data_dir: str | Path) -> Path:
    """Evaluate one target over the given seeds and append the results to ``out`` (resumable)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)
    feats = target_features(target)

    def emit(row: dict) -> None:
        pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)

    attack_out = out.parent / f"per_attack_{target}.csv"

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
    p.add_argument("--out", default=None, help="CSV to append to (default runs/flare/eval_<target>.csv)")
    p.add_argument("--runs-dir", default="runs/flare")
    p.add_argument("--supports-dir", default=None, help="where supports_<target>_s<seed>.json lives (default: the CSV's folder)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--al-pool", type=int, default=AL_POOL_CAP)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out or f"{args.runs_dir}/eval_{args.target}.csv")
    methods = list(args.methods)
    if "tabpfn" in methods:
        ok, reason = tabpfn_status()
        if not ok:
            log.warning("skipping tabpfn: %s", reason)
            methods.remove("tabpfn")
    ctx = EvalContext(runs_dir=Path(args.runs_dir), supports_dir=Path(args.supports_dir or out.parent),
                      device=args.device, threads=args.threads, al_pool=args.al_pool,
                      ks=tuple(args.ks), n_draws=args.draws)
    path = evaluate_target(args.target, args.seeds, out, ctx, methods, args.data_dir)
    log.info("results in %s", path)


if __name__ == "__main__":
    main()
