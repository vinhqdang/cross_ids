"""Few-shot reference: CatBoost trained on sources, continued on a labelled target budget.

Two rows are produced per budget k from the same labelled rows:
``catboost`` (sources first, then continued with ``init_model`` on the k% target rows) and
``targetonly`` (a control trained only on those target rows, with the same hyperparameters).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .methods.base import Domain
from .methods.baselines import CatBoostBaseline

MAX_SHIFT_FIT_ROWS = 50_000
SMALL_BUDGET_ROWS = 20
CONTINUATION_ITERATIONS = 100
SMALL_BUDGET_ITERATIONS = 300
FEWSHOT_KINDS = ("catboost", "targetonly")


def shift_score(source_X: np.ndarray, target_pool: np.ndarray, seed: int) -> float:
    """Mean anomaly score of the target pool under an Isolation Forest fitted on source features.

    Higher values mean the target lies further from the source distribution.
    """
    scaler = StandardScaler().fit(source_X)
    fit_X = scaler.transform(source_X)
    if fit_X.shape[0] > MAX_SHIFT_FIT_ROWS:
        rng = np.random.default_rng(seed)
        fit_X = fit_X[rng.choice(fit_X.shape[0], MAX_SHIFT_FIT_ROWS, replace=False)]
    forest = IsolationForest(n_estimators=200, random_state=seed).fit(fit_X)
    return float(np.mean(-forest.score_samples(scaler.transform(target_pool))))


def labelled_budget(y: np.ndarray, k_percent: float, seed: int) -> np.ndarray:
    """Indices of a seeded, class-stratified k% subset of the target train split.

    Every class present keeps at least one row, so the continued model always sees both
    classes even for the smallest budgets. Allocation is proportional to class frequency.
    """
    y = np.asarray(y).astype(int)
    n = len(y)
    size = min(n, max(2, int(round(n * k_percent / 100.0))))
    if size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    classes = [c for c in np.unique(y)]
    members = {c: np.flatnonzero(y == c) for c in classes}
    alloc = {c: min(len(members[c]), max(1, int(round(size * len(members[c]) / n)))) for c in classes}
    while sum(alloc.values()) > size:
        c = max((c for c in classes if alloc[c] > 1), key=lambda c: alloc[c], default=None)
        if c is None:
            break
        alloc[c] -= 1
    while sum(alloc.values()) < size:
        room = [c for c in classes if alloc[c] < len(members[c])]
        if not room:
            break
        c = max(room, key=lambda c: len(members[c]) - alloc[c])
        alloc[c] += 1
    chosen = np.concatenate([rng.choice(members[c], size=alloc[c], replace=False) for c in classes])
    return np.sort(chosen)


def _catboost_params(seed: int, threads: int | None, device: str, iterations: int) -> dict:
    params = dict(
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        allow_writing_files=False,
    )
    if device == "cuda":
        params["task_type"] = "GPU"
    else:
        params["thread_count"] = threads or -1
    return params


def targetonly_iterations(n_rows: int) -> int:
    """Fixed iteration count for the target-only control; tiny budgets get more trees, no early stopping."""
    return SMALL_BUDGET_ITERATIONS if n_rows < SMALL_BUDGET_ROWS else CONTINUATION_ITERATIONS


def fewshot_scores(
    sources: list[Domain],
    target_train: tuple[np.ndarray, np.ndarray],
    test_X: np.ndarray,
    ks: list[float],
    seed: int,
    threads: int | None,
    device: str,
) -> dict[float, np.ndarray]:
    """Scores on the target test rows for each labelled budget k (percent of the target train split)."""
    from catboost import CatBoostClassifier

    base = CatBoostBaseline(seed=seed, threads=threads, device=device).fit(sources)
    X_lab, y_lab = target_train
    out: dict[float, np.ndarray] = {}
    for k in ks:
        idx = labelled_budget(y_lab, k, seed)
        Z = base.scaler.transform(X_lab[idx])
        model = CatBoostClassifier(**_catboost_params(seed, threads, device, CONTINUATION_ITERATIONS))
        model.fit(Z, y_lab[idx].astype(int), init_model=base.model)
        out[k] = model.predict_proba(base.scaler.transform(np.asarray(test_X, dtype=np.float64)))[:, 1]
    return out


def targetonly_scores(
    target_train: tuple[np.ndarray, np.ndarray],
    test_X: np.ndarray,
    ks: list[float],
    seed: int,
    threads: int | None,
    device: str,
) -> dict[float, np.ndarray]:
    """Control: CatBoost trained only on the same labelled target rows as ``fewshot_scores``.

    Uses the same row selection (``labelled_budget``, same seed) and hyperparameters, without source
    data or ``init_model``. Tree splits are invariant to per-feature affine rescaling, so the raw
    features are used directly.
    """
    from catboost import CatBoostClassifier

    X_lab, y_lab = target_train
    X_lab = np.asarray(X_lab, dtype=np.float64)
    test = np.asarray(test_X, dtype=np.float64)
    out: dict[float, np.ndarray] = {}
    for k in ks:
        idx = labelled_budget(y_lab, k, seed)
        y_sub = np.asarray(y_lab[idx]).astype(int)
        model = CatBoostClassifier(**_catboost_params(seed, threads, device, targetonly_iterations(len(idx))))
        model.fit(X_lab[idx], y_sub)
        out[k] = model.predict_proba(test)[:, 1]
    return out
