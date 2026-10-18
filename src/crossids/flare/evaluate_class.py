"""Attack-class evaluation of FLARE-2 and the baselines (DESIGN.md section 21.5).

Two tasks on the Part I target split, with supports drawn from the target train split and shared by all methods:

* ``perclass``: for every target attack class c with at least 50 test rows, a k-per-class support of c and benign
  (k in {2, 4, 8, 16, 32}, 10 draws). Queries are the test rows of c and of benign. Metrics: ROC-AUC, recall at 0.5
  and false-positive rate at 0.5.
* ``nway``: N-way attack-type classification over all target classes with at least 50 test rows (benign included).
  Support is k rows per class. Metrics: macro-F1, balanced accuracy, per-class recall and the confusion matrix.

Output is one long-format CSV (``task, method, target, seed, k, draw, metric, cls, pred, value, n``) that is appended
to and resumed on re-runs. The support draws go to ``class_supports_<target>_s<seed>.json`` next to it.

Methods: ``flare2`` and its checkpoint variants (``flare2-binary``, ``flare2-noperm``, ``flare2-insrc``, ``flare2-ctx``, ``flare2-m``),
``catboost`` (multi-class, 300 iterations, balanced weights), ``rf``, ``lr``, ``knn``, ``protonet`` and ``flare``.
FLARE and ProtoNet are binary models: N-way uses one-vs-rest over the per-class episodes (argmax of P(class c)).
A method whose checkpoint is missing is skipped with a log line. FLARE-2 holds up to ``cfg.slots`` (16 by default)
classes; a task with more classes raises an error in the model rather than being silently skipped.

Example::

    python -m crossids.flare.evaluate_class --target usbids1 --seeds 0 --methods flare2 catboost \
        --tasks perclass nway --out runs/flare/class_usbids1_s0.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Split, load_split, target_features, target_keys
from .evaluate import context_rows
from .model import EpisodeModel, load_model

log = logging.getLogger("crossids.flare.evaluate_class")

KS = (2, 4, 8, 16, 32)
N_DRAWS = 10
MIN_TEST_ROWS = 50
CB_ITER = 300
BENIGN = "benign"
TASKS = ("perclass", "nway")
CHUNK = 256  # query rows per forward pass of the neural models (the attention budget caps it further)
PERCLASS_METRICS = ("roc_auc", "recall", "fpr")
NWAY_METRICS = ("macro_f1", "balanced_accuracy")
CHECKPOINT_SUFFIX = {
    "flare2": "_f2",
    "flare2-binary": "_f2_binary",
    "flare2-noperm": "_f2_noperm",
    "flare2-insrc": "_f2_insrc",
    "flare2-ctx": "_f2_ctx",
    "flare2-m": "_f2m",
    "flare2-m-binary": "_f2m_binary",
    "flare": "",
    "protonet": "_protonet",
}
METHODS = ("flare2", "flare2-binary", "flare2-noperm", "flare2-insrc", "flare2-ctx", "flare2-m", "flare2-m-binary",
           "catboost", "rf", "lr", "knn", "protonet", "flare")
NEURAL = tuple(m for m in METHODS if m in CHECKPOINT_SUFFIX)
RESULT_COLUMNS = ["task", "method", "target", "seed", "k", "draw", "metric", "cls", "pred", "value", "n"]


def row_classes(y: np.ndarray, attack: np.ndarray) -> np.ndarray:
    """Class name of each row: ``benign`` for label 0, otherwise the raw attack type."""
    attack = np.asarray(attack, dtype=object)
    return np.where(np.asarray(y) == 0, BENIGN, attack).astype(object)


def eval_classes(cls_test: np.ndarray) -> tuple[list[str], list[str]]:
    """(per-class attack classes, N-way classes) of the target test split: classes with at least 50 test rows.

    The N-way list starts with benign and keeps the attack classes sorted by name, so class indices are stable.
    """
    counts = pd.Series(cls_test).value_counts()
    attacks = sorted(c for c in counts.index if c != BENIGN and counts[c] >= MIN_TEST_ROWS)
    nway = ([BENIGN] if counts.get(BENIGN, 0) >= MIN_TEST_ROWS else []) + attacks
    return attacks, nway


def draw_class_supports(cls_train: np.ndarray, perclass: list[str], nway: list[str], seed: int, ks=KS,
                        n_draws: int = N_DRAWS) -> dict:
    """Support draws of both tasks: {"perclass": {c: {k: [draw...]}}, "nway": {k: [draw...]}}.

    A draw is ``{"idx": [train row indices], "cls": [class name of each row]}``. Streams are seeded by (seed, k, draw,
    class), so draws do not depend on the order of the class list.
    """
    benign_rows = np.flatnonzero(cls_train == BENIGN)
    members = {c: np.flatnonzero(cls_train == c) for c in set(nway) | set(perclass)}
    out = {"perclass": {}, "nway": {}}
    for c in perclass:
        out["perclass"][c] = {}
        for k in ks:
            draws = []
            for d in range(n_draws):
                rng = np.random.default_rng([seed, k, d, zlib.crc32(c.encode())])
                pos = rng.choice(members[c], size=k, replace=False)
                neg = rng.choice(benign_rows, size=k, replace=False)
                draws.append({"idx": np.concatenate([pos, neg]).astype(int).tolist(), "cls": [c] * k + [BENIGN] * k})
            out["perclass"][c][k] = draws
    for k in (ks if nway else ()):  # no N-way draws without the N-way class list
        draws = []
        for d in range(n_draws):
            rng = np.random.default_rng([seed, k, d, 777])
            idx, names = [], []
            for c in nway:
                rows = rng.choice(members[c] if c != BENIGN else benign_rows, size=k, replace=False)
                idx.append(rows)
                names += [c] * k
            draws.append({"idx": np.concatenate(idx).astype(int).tolist(), "cls": names})
        out["nway"][k] = draws
    return out


def load_or_draw_class_supports(path: Path, target: str, seed: int, cls_train: np.ndarray, perclass: list[str],
                                nway: list[str], ks=KS, n_draws: int = N_DRAWS) -> dict:
    """Class-level supports shared by all methods, stored as JSON and reused when the task definition and the
    requested (k, draw) grid are all covered by the file."""
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        sup = blob.get("supports", {})
        same_task = (blob.get("target"), blob.get("seed"), blob.get("n_train"), blob.get("perclass"),
                     blob.get("nway")) == (target, seed, len(cls_train), perclass, nway)

        def covered(per_k: dict) -> bool:
            return all(str(k) in per_k and len(per_k[str(k)]) >= n_draws for k in ks)

        nway_ok = not nway or covered(sup.get("nway", {}))
        if same_task and nway_ok and all(
                covered(sup.get("perclass", {}).get(c, {})) for c in perclass):
            return {
                "perclass": {c: {int(k): sup["perclass"][c][str(k)][:n_draws] for k in ks} for c in perclass},
                "nway": {int(k): sup["nway"][str(k)][:n_draws] for k in ks} if nway else {},
            }
    drawn = draw_class_supports(cls_train, perclass, nway, seed, ks, n_draws)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"target": target, "seed": seed, "n_train": int(len(cls_train)), "perclass": perclass, "nway": nway,
               "supports": {"perclass": {c: {str(k): v for k, v in per.items()} for c, per in drawn["perclass"].items()},
                            "nway": {str(k): v for k, v in drawn["nway"].items()}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return drawn


def _binary_metrics(y: np.ndarray, score: np.ndarray, hard: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    hard = np.asarray(hard).astype(int)
    tp = int(np.sum((y == 1) & (hard == 1)))
    fn = int(np.sum((y == 1) & (hard == 0)))
    fp = int(np.sum((y == 0) & (hard == 1)))
    tn = int(np.sum((y == 0) & (hard == 0)))
    auc = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return {"roc_auc": auc, "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


@dataclass
class Ctx:
    target: str
    seed: int
    split: Split
    runs_dir: Path
    cls_train: np.ndarray
    cls_test: np.ndarray
    perclass: list[str]
    nway: list[str]
    device: str = "cpu"
    threads: int = 2
    models: dict = field(default_factory=dict)
    missing: set = field(default_factory=set)

    def model(self, name: str) -> EpisodeModel | None:
        """Checkpoint of a neural method, or None (logged once) when it is not trained."""
        if name not in self.models:
            path = self.runs_dir / f"{self.target}_s{self.seed}{CHECKPOINT_SUFFIX[name]}" / "best.pt"
            if not path.exists():
                if name not in self.missing:
                    log.warning("skipping %s on %s seed %d: no checkpoint at %s", name, self.target, self.seed, path)
                    self.missing.add(name)
                return None
            self.models[name] = load_model(path, device=self.device)
        return self.models[name]


def _sk_binary(name: str, Xs, ys, Xq, draw: int, threads: int) -> np.ndarray:
    if name == "catboost":
        from catboost import CatBoostClassifier

        est = CatBoostClassifier(iterations=CB_ITER, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                                 verbose=0, random_seed=draw, allow_writing_files=False, thread_count=threads)
    elif name == "rf":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=threads, random_state=draw)
    elif name == "lr":
        est = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    else:
        est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, len(ys))))
    return est.fit(Xs, ys).predict_proba(Xq)[:, 1]


def _sk_multiclass(name: str, Xs, ys, Xq, draw: int, threads: int) -> np.ndarray:
    if name == "catboost":
        from catboost import CatBoostClassifier

        est = CatBoostClassifier(iterations=CB_ITER, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                                 verbose=0, random_seed=draw, allow_writing_files=False, thread_count=threads,
                                 loss_function="MultiClass")
    elif name == "rf":
        est = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=threads, random_state=draw)
    elif name == "lr":
        est = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    else:
        est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, len(np.unique(ys)))))
    est.fit(Xs, ys)
    return est.predict_proba(Xq)


def _neural_binary(ctx: Ctx, name: str, idx: np.ndarray, ys: np.ndarray, Xq: np.ndarray, k: int, draw: int):
    """P(class 1) of a neural method for a two-class support, or None when the checkpoint is missing."""
    model = ctx.model(name)
    if model is None:
        return None
    Xs = ctx.split.X_train[idx]
    if name.startswith("flare2"):
        ctx_X = context_rows(ctx.split, idx, ctx.seed, k, draw) if name == "flare2-ctx" else None
        return model.predict_proba(Xs, ys, Xq, chunk=CHUNK, seed=draw, context_X=ctx_X)
    return model.predict_proba(Xs, ys, Xq, chunk=CHUNK, seed=draw)


def perclass_rows(ctx: Ctx, method: str, supports: dict, k: int, draw: int) -> list[dict] | None:
    """Per-class detection rows of one method for one (k, draw)."""
    if method in NEURAL and ctx.model(method) is None:
        return None
    rows = []
    cls_test = ctx.cls_test
    for c in ctx.perclass:
        draw_rec = supports["perclass"][c][k][draw]
        idx = np.asarray(draw_rec["idx"], dtype=np.int64)
        ys = (np.asarray(draw_rec["cls"], dtype=object) == c).astype(np.int64)
        qmask = (cls_test == c) | (cls_test == BENIGN)
        Xq = ctx.split.X_test[qmask]
        yq = (cls_test[qmask] == c).astype(np.int64)
        Xs = ctx.split.X_train[idx]
        if method in NEURAL:
            score = _neural_binary(ctx, method, idx, ys, Xq, k, draw)
        else:
            score = _sk_binary(method, Xs, ys, Xq, draw, ctx.threads)
        hard = (score >= 0.5).astype(int)
        m = _binary_metrics(yq, score, hard)
        for metric in PERCLASS_METRICS:
            rows.append({"task": "perclass", "method": method, "target": ctx.target, "seed": ctx.seed, "k": k,
                         "draw": draw, "metric": metric, "cls": c, "pred": "", "value": m[metric],
                         "n": int(len(yq))})
    return rows


def nway_rows(ctx: Ctx, method: str, supports: dict, k: int, draw: int) -> list[dict] | None:
    """N-way classification rows of one method for one (k, draw). None when the method cannot run here."""
    classes = ctx.nway
    N = len(classes)
    if method in NEURAL and ctx.model(method) is None:
        return None
    draw_rec = supports["nway"][k][draw]
    idx = np.asarray(draw_rec["idx"], dtype=np.int64)
    index = {c: i for i, c in enumerate(classes)}
    ys = np.asarray([index[c] for c in draw_rec["cls"]], dtype=np.int64)
    qmask = np.isin(ctx.cls_test, classes)
    Xq = ctx.split.X_test[qmask]
    yq = np.asarray([index[c] for c in ctx.cls_test[qmask]], dtype=np.int64)
    Xs = ctx.split.X_train[idx]
    if method.startswith("flare2"):
        model = ctx.model(method)
        ctx_X = context_rows(ctx.split, idx, ctx.seed, k, draw) if method == "flare2-ctx" else None
        proba = model.predict_classes(Xs, ys, Xq, chunk=CHUNK, seed=draw, context_X=ctx_X)
    elif method in ("flare", "protonet"):
        model = ctx.model(method)
        scores = np.stack([model.predict_logits(Xs, (ys == c).astype(np.int64), Xq, chunk=CHUNK, seed=draw)
                           for c in range(N)], axis=1)
        proba = scores
    else:
        proba = _sk_multiclass(method, Xs, ys, Xq, draw, ctx.threads)
    pred = np.argmax(proba, axis=1)
    labels = list(range(N))
    cm = confusion_matrix(yq, pred, labels=labels)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    base = {"task": "nway", "method": method, "target": ctx.target, "seed": ctx.seed, "k": k, "draw": draw}
    rows = [
        {**base, "metric": "macro_f1", "cls": "", "pred": "",
         "value": float(f1_score(yq, pred, labels=labels, average="macro", zero_division=0)), "n": int(len(yq))},
        {**base, "metric": "balanced_accuracy", "cls": "", "pred": "",
         "value": float(balanced_accuracy_score(yq, pred)), "n": int(len(yq))},
    ]
    for i, c in enumerate(classes):
        rows.append({**base, "metric": "recall", "cls": c, "pred": "", "value": float(recall[i]),
                     "n": int(cm[i].sum())})
        for j, p in enumerate(classes):
            rows.append({**base, "metric": "confusion", "cls": c, "pred": p, "value": float(cm[i, j]),
                         "n": int(cm[i].sum())})
    return rows


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=["task", "method", "target", "seed", "k", "draw"])
    return {(str(r.task), str(r.method), str(r.target), int(r.seed), int(r.k), int(r.draw))
            for r in frame.itertuples(index=False)}


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0,
                                                      index=False)


def evaluate_class_seed(target: str, seed: int, out: Path, runs_dir: Path, data_dir: str | Path, methods, tasks,
                        device: str = "cpu", threads: int = 2, ks=KS, n_draws: int = N_DRAWS) -> Path:
    """Run both tasks for one (target, seed) and append rows to ``out`` (resumable)."""
    out = Path(out)
    split = load_split(target, seed, data_dir, features=target_features(target))
    cls_train = row_classes(split.y_train, split.attack_train)
    cls_test = row_classes(split.y_test, split.attack_test)
    perclass, nway = eval_classes(cls_test)
    log.info("%s seed %d: per-class targets %s; N-way classes %s", target, seed, perclass, nway)
    supports = load_or_draw_class_supports(out.parent / f"class_supports_{target}_s{seed}.json", target, seed,
                                           cls_train, perclass, nway, ks, n_draws)
    ctx = Ctx(target, seed, split, Path(runs_dir), cls_train, cls_test, perclass, nway, device=device, threads=threads)
    done = _done_keys(out)
    for k in ks:
        for draw in range(n_draws):
            for method in methods:
                if "perclass" in tasks and perclass and ("perclass", method, target, seed, k, draw) not in done:
                    t0 = time.perf_counter()
                    rows = perclass_rows(ctx, method, supports, k, draw)
                    if rows is not None:
                        _append(out, rows)
                        log.info("perclass %s %s seed %d k=%d draw=%d (%.1fs)", method, target, seed, k, draw,
                                 time.perf_counter() - t0)
                if "nway" in tasks and len(nway) >= 3 and ("nway", method, target, seed, k, draw) not in done:
                    t0 = time.perf_counter()
                    rows = nway_rows(ctx, method, supports, k, draw)
                    if rows is not None:
                        _append(out, rows)
                        log.info("nway %s %s seed %d k=%d draw=%d macro-F1 %.4f (%.1fs)", method, target, seed, k,
                                 draw, next(r["value"] for r in rows if r["metric"] == "macro_f1"),
                                 time.perf_counter() - t0)
    return out


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m crossids.flare.evaluate_class",
                                description="Attack-class evaluation of FLARE-2 (section 21.5)")
    p.add_argument("--target", required=True, choices=target_keys())
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--ks", nargs="+", type=int, default=list(KS))
    p.add_argument("--draws", type=int, default=N_DRAWS)
    p.add_argument("--out", default=None, help="long-format CSV to append to (default runs/flare/class_<target>.csv)")
    p.add_argument("--runs-dir", default="runs/flare")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out or f"{args.runs_dir}/class_{args.target}.csv")
    for seed in args.seeds:
        path = evaluate_class_seed(args.target, seed, out, args.runs_dir, args.data_dir, args.methods, args.tasks,
                                   device=args.device, threads=args.threads, ks=tuple(args.ks), n_draws=args.draws)
        log.info("results in %s", path)


if __name__ == "__main__":
    main()
