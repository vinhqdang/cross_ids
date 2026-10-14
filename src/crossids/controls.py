"""Evidence controls for DESIGN.md section 10: oracle grid, concept-shift test, dataset separability, fair few-shot.

Every control reads ``data/processed/*.parquet`` and uses ``experiment.split_tables`` (70/30 stratified, seeded), so
the target test rows are the rows the benchmark evaluates. ``results/runs.csv`` and ``results/per_attack.csv`` are only
read (the oracle and few-shot controls are compared against the benchmark's rows). Outputs go to a separate directory.

Resumable: each long-running control appends one row per fit or per pair to its CSV, and rows whose key is already in
the file are skipped on the next run. Summaries and figures are regenerated from the full CSV at the end of each run.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import QuantileTransformer, StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from .analysis import INK, MUTED, PALETTE, SEQUENTIAL_CMAP, STYLE, _annotated_heatmap, _Report, load_runs  # noqa: E402
from .datasets import load_processed, real_info  # noqa: E402
from .experiment import split_tables, transfer_type  # noqa: E402
from .fewshot import _catboost_params, labelled_budget  # noqa: E402
from .methods.base import Domain  # noqa: E402
from .methods.baselines import CatBoostBaseline  # noqa: E402
from .metrics import evaluate  # noqa: E402
from .schema import features_for  # noqa: E402

log = logging.getLogger("crossids.controls")

CONTROL_NAMES = ("oracle", "concept", "separability", "fewshot_fair")
CORE_DATASETS = ("cicids2017", "cicids2018", "usbids1", "insdn", "unswnb15", "toniot", "iot23")
CORE_FEATURES = tuple(features_for("core"))
ALL_PAIRS = tuple((a, b) for a in CORE_DATASETS for b in CORE_DATASETS if a != b)
TEST_SIZE = 0.3

# Oracle grid (see oracle.md for the reasoning behind each reduction)
ORACLE_SOURCE_ROWS = 30_000
BOOST_ITERATIONS = 150
BOOST_LEARNING_RATE = 0.2
RF_TREES = 100
LR_MAX_ITER = 500
FAMILIES = ("hgb", "catboost", "xgboost", "rf", "lr")
FAMILY_GRID = {
    "hgb": ("max_leaf_nodes", (15, 63)),
    "catboost": ("depth", (4, 6)),
    "xgboost": ("max_depth", (4, 8)),
    "rf": ("max_depth", (8, 16)),
    "lr": ("C", (0.1, 10.0)),
}
CLASS_WEIGHTS = ("balanced", "none")
NORMS_TREE = ("none", "rank")  # global standard is an affine map per feature: tree splits are invariant to it
NORMS_LINEAR = ("none", "global", "rank")
FEATURE_GROUPS = {
    "volume": ("fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "total_pkts", "total_bytes"),
    "rate": ("bytes_per_s", "pkts_per_s"),
    "size_per_pkt": ("fwd_bytes_per_pkt", "bwd_bytes_per_pkt"),
    "direction_ratio": ("fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio"),
    "duration": ("duration",),
}
SUBSETS = ("all", "-volume", "-rate", "-size_per_pkt", "-direction_ratio", "-duration")

# Concept-shift test
CONCEPT_K = 10
CONCEPT_REF_ROWS = 50_000
CONCEPT_QUERY_ROWS = 10_000
CONCEPT_PERCENTILE = 90

# Separability
SEP_ROWS_PER_DATASET = 20_000
SEP_FOLDS = 5
SEP_ITERATIONS = 300

# Few-shot
FEWSHOT_KS = (0.1, 1, 5)
FEWSHOT_ITERATIONS = (100, 300, 500)
FEWSHOT_VARIANTS = ("continued", "targetonly")

ORACLE_COLUMNS = [
    "pair", "source", "target", "transfer_type", "seed", "family", "config", "norm", "subset",
    "n_train", "n_test", "roc_auc", "pr_auc", "balanced_accuracy", "fit_seconds",
]
ORACLE_KEY = ("pair", "seed", "family", "config", "norm", "subset")
CONCEPT_COLUMNS = [
    "pair", "source", "target", "transfer_type", "seed", "n_reference", "n_target", "n_source_test", "threshold",
    "cross_overlap_fraction", "cross_agreement", "cross_kappa", "cross_n_overlap",
    "indomain_overlap_fraction", "indomain_agreement", "indomain_kappa", "indomain_n_overlap",
]
CONCEPT_KEY = ("pair", "seed")
SEP_COLUMNS = [
    "condition", "seed", "n_rows", "n_datasets", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
    "pooled_accuracy", "pooled_macro_f1", "majority_baseline",
]
SEP_KEY = ("condition", "seed")
CONFUSION_COLUMNS = ["condition", "seed", "true_dataset", "predicted_dataset", "count", "row_fraction"]
FEWSHOT_COLUMNS = [
    "pair", "source", "target", "seed", "variant", "iterations", "k", "n_labelled", "roc_auc", "balanced_accuracy",
    "fit_seconds", "benchmark_roc_auc", "benchmark_balanced_accuracy",
]
FEWSHOT_KEY = ("pair", "seed", "variant", "iterations", "k")


# ---------------------------------------------------------------------------------------------------------------------
# Data and splits
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """One dataset's benchmark split, as feature matrices and integer labels."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_core_frames(data_dir: Path, datasets: tuple[str, ...] = CORE_DATASETS) -> dict[str, pd.DataFrame]:
    """Cached harmonised tables of the core-tier datasets."""
    return {key: load_processed(key, Path(data_dir)) for key in datasets}


def core_splits(frames: dict[str, pd.DataFrame], seed: int, test_size: float = TEST_SIZE) -> dict[str, Split]:
    """The benchmark split (``experiment.split_tables``) turned into arrays on the core features."""
    parts = split_tables(frames, test_size, seed)
    out: dict[str, Split] = {}
    for key, (train, test) in parts.items():
        out[key] = Split(
            X_train=train[list(CORE_FEATURES)].to_numpy(dtype=np.float64),
            y_train=train["label"].to_numpy().astype(int),
            X_test=test[list(CORE_FEATURES)].to_numpy(dtype=np.float64),
            y_test=test["label"].to_numpy().astype(int),
        )
    return out


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted indices of a seeded, stratified subsample of at most ``n`` rows."""
    y = np.asarray(y).astype(int)
    if len(y) <= n:
        return np.arange(len(y))
    counts = np.bincount(y)
    present = counts[counts > 0]
    strat = y if len(present) == 2 and present.min() >= 2 else None
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=strat, random_state=seed)
    return np.sort(idx)


def pair_name(source: str, target: str) -> str:
    return f"{source}->{target}"


def _parse_pair(text: str) -> tuple[str, str]:
    if "->" not in text:
        raise ValueError(f"pair {text!r} must look like source->target")
    a, b = (part.strip() for part in text.split("->", 1))
    if a == b or a not in CORE_DATASETS or b not in CORE_DATASETS:
        raise ValueError(f"pair {text!r} must name two different core datasets: {', '.join(CORE_DATASETS)}")
    return a, b


def parse_pairs(texts: list[str] | None) -> list[tuple[str, str]]:
    return list(ALL_PAIRS) if not texts else [_parse_pair(t) for t in texts]


# ---------------------------------------------------------------------------------------------------------------------
# CSV resume helpers
# ---------------------------------------------------------------------------------------------------------------------


def _check_header(path: Path, columns: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    if header != columns:
        raise ValueError(f"{path} has a different header than this version of crossids; move or delete it")


def _done_keys(path: Path, key: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=list(key), dtype=str, keep_default_na=False)
    return {tuple(str(v) for v in row) for row in frame.itertuples(index=False)}


def _append(path: Path, rows: list[dict], columns: list[str]) -> None:
    _check_header(path, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


# ---------------------------------------------------------------------------------------------------------------------
# 1. Oracle upper bound
# ---------------------------------------------------------------------------------------------------------------------


def oracle_configs(family: str) -> list[dict]:
    """Four configurations per family: one capacity setting times class weighting balanced / none."""
    name, values = FAMILY_GRID[family]
    return [{name: value, "class_weight": weight} for value in values for weight in CLASS_WEIGHTS]


def config_label(cfg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def oracle_norms(family: str) -> tuple[str, ...]:
    return NORMS_LINEAR if family == "lr" else NORMS_TREE


def subset_columns() -> dict[str, list[int]]:
    """Column indices of each feature subset, in ``CORE_FEATURES`` order."""
    out: dict[str, list[int]] = {}
    for name in SUBSETS:
        dropped = set() if name == "all" else set(FEATURE_GROUPS[name[1:]])
        out[name] = [i for i, f in enumerate(CORE_FEATURES) if f not in dropped]
    return out


def make_oracle_model(family: str, cfg: dict, seed: int, threads: int, y_fit: np.ndarray):
    balanced = cfg["class_weight"] == "balanced"
    if family == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=BOOST_ITERATIONS,
            learning_rate=BOOST_LEARNING_RATE,
            max_leaf_nodes=int(cfg["max_leaf_nodes"]),
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        )
    if family == "catboost":
        from catboost import CatBoostClassifier

        params = dict(
            iterations=BOOST_ITERATIONS,
            depth=int(cfg["depth"]),
            learning_rate=BOOST_LEARNING_RATE,
            verbose=0,
            random_seed=seed,
            allow_writing_files=False,
            thread_count=threads,
        )
        if balanced:
            params["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(**params)
    if family == "xgboost":
        from xgboost import XGBClassifier

        n_pos = max(int(y_fit.sum()), 1)
        n_neg = max(len(y_fit) - n_pos, 1)
        return XGBClassifier(
            n_estimators=BOOST_ITERATIONS,
            max_depth=int(cfg["max_depth"]),
            learning_rate=BOOST_LEARNING_RATE,
            tree_method="hist",
            scale_pos_weight=n_neg / n_pos if balanced else 1.0,
            n_jobs=threads,
            random_state=seed,
            eval_metric="logloss",
        )
    if family == "rf":
        return RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=int(cfg["max_depth"]),
            min_samples_leaf=5,
            class_weight="balanced_subsample" if balanced else None,
            n_jobs=threads,
            random_state=seed,
        )
    if family == "lr":
        return LogisticRegression(C=float(cfg["C"]), class_weight="balanced" if balanced else None, max_iter=LR_MAX_ITER)
    raise ValueError(f"unknown oracle family {family!r}")


def _rank_transformer(seed: int, n_rows: int) -> QuantileTransformer:
    """Uniform rank transform that uses every row (no subsampling) with at most 1000 quantiles."""
    return QuantileTransformer(
        n_quantiles=min(1000, n_rows), output_distribution="uniform", subsample=n_rows, random_state=seed
    )


def _normalised_arrays(source: Split, target: Split, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Source subsample labels, target test labels, and the (source, target) feature matrices per normalisation.

    ``none``: raw features. ``global``: StandardScaler fitted on the source subsample (source statistics).
    ``rank``: QuantileTransformer fitted per domain; the target one only sees target *train* features (no test rows).
    """
    idx = stratified_subsample(source.y_train, ORACLE_SOURCE_ROWS, seed)
    Xs, ys = source.X_train[idx], source.y_train[idx]
    Xt, yt = target.X_test, target.y_test
    scaler = StandardScaler().fit(Xs)
    qs = _rank_transformer(seed, len(Xs)).fit(Xs)
    qt = _rank_transformer(seed, len(target.X_train)).fit(target.X_train)
    arrays = {
        "none": (Xs, Xt),
        "global": (scaler.transform(Xs), scaler.transform(Xt)),
        "rank": (qs.transform(Xs), qt.transform(Xt)),
    }
    return ys, yt, arrays


def run_oracle(
    pairs: list[tuple[str, str]],
    seeds: list[int],
    out_dir: Path,
    threads: int,
    frames: dict[str, pd.DataFrame],
    families: tuple[str, ...] = FAMILIES,
    runs: pd.DataFrame | None = None,
) -> Path:
    """Fit every grid cell per pair on the source subsample and score the target test split (resumable)."""
    path = out_dir / "oracle_grid.csv"
    _check_header(path, ORACLE_COLUMNS)
    done = _done_keys(path, ORACLE_KEY)
    columns = subset_columns()
    seconds_per_pair: list[float] = []
    total_pairs = len(pairs) * len(seeds)
    for seed in seeds:
        splits = core_splits(frames, seed)
        for source_key, target_key in pairs:
            pair = pair_name(source_key, target_key)
            todo = [
                (family, cfg, norm, subset)
                for family in families
                for cfg in oracle_configs(family)
                for norm in oracle_norms(family)
                for subset in SUBSETS
                if (pair, str(seed), family, config_label(cfg), norm, subset) not in done
            ]
            if not todo:
                log.info("oracle %s seed=%d: skip (done)", pair, seed)
                continue
            t0 = time.perf_counter()
            transfer = transfer_type(real_info(source_key), real_info(target_key))
            ys, yt, arrays = _normalised_arrays(splits[source_key], splits[target_key], seed)
            with threadpool_limits(limits=threads), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for family, cfg, norm, subset in todo:
                    cols = columns[subset]
                    Xs, Xt = arrays[norm]
                    model = make_oracle_model(family, cfg, seed, threads, ys)
                    t1 = time.perf_counter()
                    model.fit(Xs[:, cols], ys)
                    proba = model.predict_proba(Xt[:, cols])[:, 1]
                    fit_seconds = time.perf_counter() - t1
                    metrics = evaluate(yt, proba, (proba >= 0.5).astype(int))
                    row = {
                        "pair": pair, "source": source_key, "target": target_key, "transfer_type": transfer,
                        "seed": seed, "family": family, "config": config_label(cfg), "norm": norm, "subset": subset,
                        "n_train": len(ys), "n_test": len(yt), "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"], "balanced_accuracy": metrics["balanced_accuracy"],
                        "fit_seconds": round(fit_seconds, 3),
                    }
                    _append(path, [row], ORACLE_COLUMNS)
                    done.add((pair, str(seed), family, config_label(cfg), norm, subset))
            elapsed = time.perf_counter() - t0
            seconds_per_pair.append(elapsed)
            mean = float(np.mean(seconds_per_pair))
            log.info(
                "oracle %s seed=%d: %d fits in %.1fs (ROC-AUC best so far from file). "
                "Mean %.1fs/pair; estimated total for %d pair-seeds: %.0f min",
                pair, seed, len(todo), elapsed, mean, total_pairs, mean * total_pairs / 60.0,
            )
    return path


def oracle_summary(grid: pd.DataFrame, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per pair and seed: best grid cell by target ROC-AUC, the median cell and the benchmark hgb value."""
    rows = []
    for (pair, seed), sub in grid.groupby(["pair", "seed"], sort=False):
        valid = sub[np.isfinite(sub["roc_auc"].astype(float))]
        if valid.empty:
            continue
        best = valid.loc[valid["roc_auc"].astype(float).idxmax()]
        bench = _benchmark_hgb(runs, pair, int(seed))
        rows.append({
            "pair": pair,
            "source": best["source"],
            "target": best["target"],
            "transfer_type": best["transfer_type"],
            "seed": int(seed),
            "n_cells": int(len(valid)),
            "oracle_roc_auc": float(best["roc_auc"]),
            "oracle_config": f"{best['family']} | {best['config']} | norm={best['norm']} | subset={best['subset']}",
            "oracle_pr_auc": float(best["pr_auc"]),
            "oracle_balanced_accuracy": float(best["balanced_accuracy"]),
            "median_roc_auc": float(valid["roc_auc"].astype(float).median()),
            "benchmark_hgb_roc_auc": bench["roc_auc"],
            "benchmark_hgb_balanced_accuracy": bench["balanced_accuracy"],
            "n_source_rows": int(best["n_train"]),
            "n_test_rows": int(best["n_test"]),
        })
    return pd.DataFrame(rows)


def _benchmark_hgb(runs: pd.DataFrame | None, pair: str, seed: int) -> dict[str, float]:
    nan = {"roc_auc": math.nan, "balanced_accuracy": math.nan}
    if runs is None or runs.empty:
        return nan
    sel = runs[
        (runs["method"] == "hgb") & (runs["experiment"] == "pairwise") & (runs["mode"] == "transductive")
        & (runs["scenario"] == pair) & (runs["seed"].astype(int) == seed)
    ]
    if sel.empty:
        return nan
    row = sel.iloc[-1]
    return {"roc_auc": float(row["roc_auc"]), "balanced_accuracy": float(row["balanced_accuracy"])}


ORACLE_NOTE = (
    "Oracle upper bound over a reduced grid. Each pair is a source subsample (stratified, seeded, at most "
    f"{ORACLE_SOURCE_ROWS:,} rows of the 70k source train split; the benchmark uses all of it) and the full target test "
    "split. Grid per family: hgb max_leaf_nodes {15, 63}; catboost depth {4, 6}; xgboost max_depth {4, 8}; rf max_depth "
    f"{{8, 16}} with {RF_TREES} trees; lr C {{0.1, 10}}; each crossed with class weighting balanced / none. Boosted models use "
    f"{BOOST_ITERATIONS} rounds at learning rate {BOOST_LEARNING_RATE} (fixed, not searched). Normalisation: none, rank "
    "(per-domain quantile transform, target fitted on target train features only) for trees; none, global standard, rank "
    "for lr. Global standard is omitted for trees because a per-feature affine map leaves tree splits unchanged (checked "
    "in tests). Feature subsets: all core features and the five drop-one groups. Because the grid is a subset of all "
    "hyperparameters, the maximum is an upper bound only over this grid, not over all configurations."
)


def _plot_oracle(summary: pd.DataFrame) -> plt.Figure:
    s = summary.sort_values("oracle_roc_auc").reset_index(drop=True)
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.6, 0.21 * len(s) + 1.5))
    ax.scatter(s["oracle_roc_auc"], y, color=PALETTE[0], s=20, zorder=3, label="oracle (best grid cell)")
    ax.scatter(s["median_roc_auc"], y, color=PALETTE[2], marker="s", s=14, zorder=3, label="median grid cell")
    ax.scatter(s["benchmark_hgb_roc_auc"], y, color=PALETTE[1], marker="D", s=14, zorder=3,
               label="benchmark hgb (full source train)")
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(s["pair"], fontsize=7)
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("target-test ROC-AUC", color=MUTED)
    ax.set_title("Oracle upper bound per pairwise transfer (seed 0, transductive hgb from benchmark)", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def write_oracle_outputs(out_dir: Path, grid_path: Path, runs: pd.DataFrame | None) -> None:
    grid = _read(grid_path)
    if grid.empty:
        return
    summary = oracle_summary(grid, runs)
    rep = _Report(out_dir)
    rep.table(
        "oracle_pairs", summary,
        "per pair and seed: oracle (best grid cell) ROC-AUC with its config, median cell, benchmark hgb",
        "Oracle upper bound per pairwise transfer",
        note=ORACLE_NOTE + "\n\n" + _oracle_headline(summary),
    )
    with plt.rc_context(STYLE):
        rep.figure("oracle_roc_auc", _plot_oracle(summary[summary["seed"] == summary["seed"].min()]),
                   "oracle vs median grid cell vs benchmark hgb, per pair")


def _oracle_headline(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    best = summary["oracle_roc_auc"]
    gap = summary["oracle_roc_auc"] - summary["benchmark_hgb_roc_auc"]
    return (
        f"Pairs: {len(summary)}. Oracle ROC-AUC mean {best.mean():.3f}, median {best.median():.3f}, "
        f"max {best.max():.3f}. Pairs with oracle ROC-AUC below 0.6: {int((best < 0.6).sum())}. "
        f"Oracle minus benchmark hgb: mean {gap.mean():+.3f} (negative values are pairs where the benchmark's "
        "full-train hgb beats the reduced grid)."
    )


# ---------------------------------------------------------------------------------------------------------------------
# 2. Concept-shift test
# ---------------------------------------------------------------------------------------------------------------------


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for binary labels; NaN when chance agreement is 1 (both raters constant and equal)."""
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    if a.size == 0:
        return math.nan
    po = float(np.mean(a == b))
    pe = float(np.mean(a == 1) * np.mean(b == 1) + np.mean(a == 0) * np.mean(b == 0))
    return math.nan if pe >= 1.0 else (po - pe) / (1.0 - pe)


class _KnnIndex:
    """Brute-force k-NN index over standardised source rows, with majority-vote and leave-self-out queries."""

    def __init__(self, Z: np.ndarray, y: np.ndarray, k: int, threads: int) -> None:
        self.Z = Z
        self.y = np.asarray(y).astype(int)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=threads).fit(Z)

    def query(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean neighbour distance and fraction of attack neighbours for each query row."""
        dist, ind = self.nn.kneighbors(Q, n_neighbors=self.k)
        return dist.mean(axis=1), self.y[ind].mean(axis=1)

    def leave_self_out_distance(self, rows: np.ndarray) -> np.ndarray:
        """Mean distance to the k nearest *other* reference rows, for reference rows given by index."""
        dist, ind = self.nn.kneighbors(self.Z[rows], n_neighbors=self.k + 1)
        self_hit = ind == rows[:, None]
        no_hit = ~self_hit.any(axis=1)
        self_hit[no_hit, -1] = True  # duplicates may push the row itself out of the list: drop the farthest instead
        return dist[~self_hit].reshape(len(rows), self.k).mean(axis=1)


def _overlap_stats(mean_dist: np.ndarray, frac_attack: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    inside = mean_dist <= threshold
    pred = (frac_attack > 0.5).astype(int)  # majority vote; a 5-5 tie is benign
    n_in = int(inside.sum())
    if n_in == 0:
        return {"overlap_fraction": 0.0, "agreement": math.nan, "kappa": math.nan, "n_overlap": 0}
    yy = np.asarray(y).astype(int)[inside]
    pp = pred[inside]
    return {
        "overlap_fraction": float(inside.mean()),
        "agreement": float(np.mean(yy == pp)),
        "kappa": _cohen_kappa(yy, pp),
        "n_overlap": n_in,
    }


def concept_row(
    src_train_X: np.ndarray,
    src_train_y: np.ndarray,
    src_test_X: np.ndarray,
    src_test_y: np.ndarray,
    tgt_test_X: np.ndarray,
    tgt_test_y: np.ndarray,
    seed: int,
    threads: int,
    k: int = CONCEPT_K,
) -> dict[str, float]:
    """Cross-dataset and in-domain overlap statistics for one source (train split) and target (test split)."""
    scaler = StandardScaler().fit(src_train_X)
    ref = stratified_subsample(src_train_y, CONCEPT_REF_ROWS, seed)
    index = _KnnIndex(scaler.transform(src_train_X[ref]), src_train_y[ref], k, threads)
    rng = np.random.default_rng(seed)
    loo_rows = rng.choice(len(ref), size=min(CONCEPT_QUERY_ROWS, len(ref)), replace=False)
    threshold = float(np.percentile(index.leave_self_out_distance(loo_rows), CONCEPT_PERCENTILE))
    t_sub = stratified_subsample(tgt_test_y, CONCEPT_QUERY_ROWS, seed)
    s_sub = stratified_subsample(src_test_y, CONCEPT_QUERY_ROWS, seed)
    cross = _overlap_stats(*index.query(scaler.transform(tgt_test_X[t_sub])), np.asarray(tgt_test_y)[t_sub], threshold)
    indomain = _overlap_stats(*index.query(scaler.transform(src_test_X[s_sub])), np.asarray(src_test_y)[s_sub], threshold)
    return {
        "n_reference": int(len(ref)),
        "n_target": int(len(t_sub)),
        "n_source_test": int(len(s_sub)),
        "threshold": threshold,
        **{f"cross_{key}": value for key, value in cross.items()},
        **{f"indomain_{key}": value for key, value in indomain.items()},
    }


def run_concept(
    pairs: list[tuple[str, str]],
    seeds: list[int],
    out_dir: Path,
    threads: int,
    frames: dict[str, pd.DataFrame],
) -> Path:
    path = out_dir / "concept_pairs.csv"
    done = _done_keys(path, CONCEPT_KEY)
    for seed in seeds:
        splits = core_splits(frames, seed)
        for source_key, target_key in pairs:
            pair = pair_name(source_key, target_key)
            if (pair, str(seed)) in done:
                log.info("concept %s seed=%d: skip (done)", pair, seed)
                continue
            t0 = time.perf_counter()
            src, tgt = splits[source_key], splits[target_key]
            stats = concept_row(src.X_train, src.y_train, src.X_test, src.y_test, tgt.X_test, tgt.y_test, seed, threads)
            row = {
                "pair": pair, "source": source_key, "target": target_key,
                "transfer_type": transfer_type(real_info(source_key), real_info(target_key)), "seed": seed, **stats,
            }
            _append(path, [row], CONCEPT_COLUMNS)
            log.info("concept %s seed=%d: cross kappa %.3f (overlap %.2f), in-domain kappa %.3f (%.1fs)", pair, seed,
                     stats["cross_kappa"], stats["cross_overlap_fraction"], stats["indomain_kappa"],
                     time.perf_counter() - t0)
    return path


CONCEPT_NOTE = (
    f"Standardised with source-train statistics; k = {CONCEPT_K} nearest source-train neighbours (at most "
    f"{CONCEPT_REF_ROWS:,} reference rows); queries are at most {CONCEPT_QUERY_ROWS:,} rows of the target test split. "
    f"Overlap region: mean neighbour distance at or below the {CONCEPT_PERCENTILE}th percentile of leave-self-out "
    "source-to-source distances. Neighbour label is the majority vote (5-5 ties are benign). The in-domain reference is "
    "source test rows against the same source-train index. Kappa near or below 0 inside the overlap means the same "
    "flow statistics carry different labels across the two datasets."
)


def write_concept_outputs(out_dir: Path, pairs_path: Path) -> None:
    table = _read(pairs_path)
    if table.empty:
        return
    table = table.sort_values("cross_kappa").reset_index(drop=True)
    rep = _Report(out_dir)
    rep.table(
        "concept_pairs", table,
        "per pair: overlap fraction, neighbour agreement and kappa, cross-dataset and in-domain",
        "Concept-shift test (kNN overlap region)", note=CONCEPT_NOTE,
    )
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 0.22 * len(table) + 1.5))
        y = np.arange(len(table))
        ax.scatter(table["cross_kappa"], y, color=PALETTE[0], s=22, zorder=3, label="cross-dataset (source -> target)")
        ax.scatter(table["indomain_kappa"], y, color=PALETTE[2], marker="s", s=16, zorder=3,
                   label="in-domain reference (source test)")
        ax.axvline(0.0, color=MUTED, linestyle=":", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(table["pair"], fontsize=7)
        ax.set_xlabel("Cohen's kappa inside the overlap region", color=MUTED)
        ax.set_title("Neighbour-label kappa in the overlap region (k=10)", loc="left")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        rep.figure("concept_kappa", fig, "kappa inside the overlap, cross-dataset vs in-domain, per pair")


# ---------------------------------------------------------------------------------------------------------------------
# 3. Dataset separability
# ---------------------------------------------------------------------------------------------------------------------


def separability_scores(
    frames: dict[str, pd.DataFrame],
    condition: str,
    seed: int,
    threads: int,
    max_rows: int = SEP_ROWS_PER_DATASET,
    folds: int = SEP_FOLDS,
    iterations: int = SEP_ITERATIONS,
) -> tuple[dict, pd.DataFrame]:
    """CatBoost predicts which dataset a flow came from (k-fold CV). ``condition``: 'all rows' or 'benign only'."""
    from catboost import CatBoostClassifier

    keys = list(frames)
    parts, labels = [], []
    for i, key in enumerate(keys):
        frame = frames[key]
        if condition == "benign only":
            frame = frame[frame["label"].to_numpy().astype(int) == 0]
        if len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=seed)
        parts.append(frame[list(CORE_FEATURES)].to_numpy(dtype=np.float64))
        labels.append(np.full(len(frame), i))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_acc, fold_f1 = [], []
    oof = np.empty_like(y)
    for train_idx, test_idx in skf.split(X, y):
        model = CatBoostClassifier(
            iterations=iterations, depth=6, learning_rate=0.1, verbose=0, random_seed=seed,
            thread_count=threads, allow_writing_files=False,
        )
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx])).ravel().astype(int)
        oof[test_idx] = pred
        fold_acc.append(accuracy_score(y[test_idx], pred))
        fold_f1.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    counts = np.bincount(y, minlength=len(keys))
    summary = {
        "condition": condition,
        "seed": seed,
        "n_rows": int(len(y)),
        "n_datasets": len(keys),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc, ddof=1)) if folds > 1 else math.nan,
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_std": float(np.std(fold_f1, ddof=1)) if folds > 1 else math.nan,
        "pooled_accuracy": float(accuracy_score(y, oof)),
        "pooled_macro_f1": float(f1_score(y, oof, average="macro", zero_division=0)),
        "majority_baseline": float(counts.max() / counts.sum()),
    }
    conf = np.zeros((len(keys), len(keys)), dtype=int)
    np.add.at(conf, (y, oof), 1)
    rows = []
    for i, true_key in enumerate(keys):
        total = conf[i].sum()
        for j, pred_key in enumerate(keys):
            rows.append({
                "condition": condition, "seed": seed, "true_dataset": true_key, "predicted_dataset": pred_key,
                "count": int(conf[i, j]), "row_fraction": float(conf[i, j] / total) if total else math.nan,
            })
    return summary, pd.DataFrame(rows)


def run_separability(frames: dict[str, pd.DataFrame], seeds: list[int], out_dir: Path, threads: int) -> Path:
    path = out_dir / "separability.csv"
    confusion_path = out_dir / "separability_confusion.csv"
    done = _done_keys(path, SEP_KEY)
    for seed in seeds:
        for condition in ("all rows", "benign only"):
            if (condition, str(seed)) in done:
                log.info("separability %s seed=%d: skip (done)", condition, seed)
                continue
            t0 = time.perf_counter()
            with threadpool_limits(limits=threads):
                summary, conf = separability_scores(frames, condition, seed, threads)
            _append(confusion_path, conf.to_dict("records"), CONFUSION_COLUMNS)
            _append(path, [summary], SEP_COLUMNS)
            log.info("separability %s seed=%d: accuracy %.4f, macro-F1 %.4f, n=%d (%.0fs)", condition, seed,
                     summary["pooled_accuracy"], summary["pooled_macro_f1"], summary["n_rows"], time.perf_counter() - t0)
    return path


def write_separability_outputs(out_dir: Path) -> None:
    summary = _read(out_dir / "separability.csv")
    conf = _read(out_dir / "separability_confusion.csv")
    if summary.empty:
        return
    rep = _Report(out_dir)
    rep.table(
        "separability", summary,
        "CatBoost dataset-identity classifier: accuracy and macro-F1 (5-fold CV, per fold and pooled)",
        "Dataset separability",
        note=(
            f"Up to {SEP_ROWS_PER_DATASET:,} rows per dataset, core features, {SEP_FOLDS}-fold stratified CV, CatBoost "
            f"({SEP_ITERATIONS} iterations, depth 6). Chance for {len(CORE_DATASETS)} datasets is 1/{len(CORE_DATASETS)} "
            "= 0.143; the majority baseline column is the accuracy of always predicting the largest dataset."
        ),
    )
    with plt.rc_context(STYLE):
        seeds = sorted(conf["seed"].unique())
        conditions = [c for c in ("all rows", "benign only") if c in set(conf["condition"])]
        fig, axes = plt.subplots(1, len(conditions), figsize=(4.4 * len(conditions) + 1.2, 4.2))
        axes = np.atleast_1d(axes)
        seed0 = seeds[0]
        for ax, condition in zip(axes, conditions):
            sub = conf[(conf["condition"] == condition) & (conf["seed"] == seed0)]
            mat = sub.pivot(index="true_dataset", columns="predicted_dataset", values="row_fraction")
            mat = mat.loc[list(CORE_DATASETS), list(CORE_DATASETS)]
            _annotated_heatmap(ax, mat, SEQUENTIAL_CMAP, plt.Normalize(0, 1))
            ax.set_xlabel("predicted dataset", color=MUTED)
            ax.set_ylabel("true dataset", color=MUTED)
            acc = summary[(summary["condition"] == condition) & (summary["seed"] == seed0)]["pooled_accuracy"]
            ax.set_title(f"{condition}: pooled accuracy {float(acc.iloc[0]):.3f}", loc="left", color=INK)
        fig.tight_layout()
        rep.figure("separability_confusion", fig, "row-normalised confusion matrix of the dataset classifier")


# ---------------------------------------------------------------------------------------------------------------------
# 4. Fair few-shot comparison
# ---------------------------------------------------------------------------------------------------------------------


def fewshot_pair_rows(
    base: CatBoostBaseline,
    target_train: tuple[np.ndarray, np.ndarray],
    test_X: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    threads: int,
    ks: tuple[float, ...] = FEWSHOT_KS,
    iterations: tuple[int, ...] = FEWSHOT_ITERATIONS,
    variants: tuple[str, ...] = FEWSHOT_VARIANTS,
    wanted: set[tuple[str, int, float]] | None = None,
) -> list[dict]:
    """Rows for every (variant, iterations, k) in ``wanted`` (all of them when None), using the benchmark's budgets."""
    from catboost import CatBoostClassifier

    X_lab, y_lab = target_train
    X_lab = np.asarray(X_lab, dtype=np.float64)
    test_X = np.asarray(test_X, dtype=np.float64)
    test_scaled = base.scaler.transform(test_X)
    rows = []
    for k in ks:
        idx = labelled_budget(y_lab, k, seed)
        y_sub = np.asarray(y_lab[idx]).astype(int)
        for variant in variants:
            for it in iterations:
                if wanted is not None and (variant, it, k) not in wanted:
                    continue
                params = _catboost_params(seed, threads, "cpu", it)
                t0 = time.perf_counter()
                model = CatBoostClassifier(**params)
                if variant == "continued":
                    model.fit(base.scaler.transform(X_lab[idx]), y_sub, init_model=base.model)
                    proba = model.predict_proba(test_scaled)[:, 1]
                else:
                    model.fit(X_lab[idx], y_sub)
                    proba = model.predict_proba(test_X)[:, 1]
                metrics = evaluate(test_y, proba, (proba >= 0.5).astype(int))
                rows.append({
                    "variant": variant, "iterations": it, "k": k, "n_labelled": int(len(idx)),
                    "roc_auc": metrics["roc_auc"], "balanced_accuracy": metrics["balanced_accuracy"],
                    "fit_seconds": round(time.perf_counter() - t0, 3),
                })
    return rows


def run_fewshot_fair(
    pairs: list[tuple[str, str]],
    seeds: list[int],
    out_dir: Path,
    threads: int,
    frames: dict[str, pd.DataFrame],
    runs: pd.DataFrame | None = None,
) -> Path:
    path = out_dir / "fewshot_fair_rows.csv"
    done = _done_keys(path, FEWSHOT_KEY)
    for seed in seeds:
        splits = core_splits(frames, seed)
        bases: dict[str, CatBoostBaseline] = {}
        for source_key, target_key in pairs:
            pair = pair_name(source_key, target_key)
            wanted = {
                (variant, it, k)
                for variant in FEWSHOT_VARIANTS for it in FEWSHOT_ITERATIONS for k in FEWSHOT_KS
                if (pair, str(seed), variant, str(it), str(k)) not in done
            }
            if not wanted:
                log.info("fewshot_fair %s seed=%d: skip (done)", pair, seed)
                continue
            if source_key not in bases:
                src = splits[source_key]
                domain = Domain(name=source_key, X=src.X_train, y=src.y_train, meta=pd.DataFrame())
                bases[source_key] = CatBoostBaseline(seed=seed, threads=threads, device="cpu").fit([domain])
            tgt = splits[target_key]
            with threadpool_limits(limits=threads):
                rows = fewshot_pair_rows(bases[source_key], (tgt.X_train, tgt.y_train), tgt.X_test, tgt.y_test,
                                         seed, threads, wanted=wanted)
            bench = _benchmark_fewshot(runs, pair, seed)
            out_rows = []
            for r in rows:
                ref = bench.get((r["variant"], r["k"]), {}) if r["iterations"] == 100 else {}
                out_rows.append({
                    "pair": pair, "source": source_key, "target": target_key, "seed": seed, **r,
                    "benchmark_roc_auc": ref.get("roc_auc", math.nan),
                    "benchmark_balanced_accuracy": ref.get("balanced_accuracy", math.nan),
                })
            _append(path, out_rows, FEWSHOT_COLUMNS)
            done.update((pair, str(seed), r["variant"], str(r["iterations"]), str(r["k"])) for r in rows)
            log.info("fewshot_fair %s seed=%d: %d rows", pair, seed, len(out_rows))
    return path


def _benchmark_fewshot(runs: pd.DataFrame | None, pair: str, seed: int) -> dict:
    """Benchmark few-shot rows for the same pair, seed and budget (100-tree continued and target-only)."""
    out: dict = {}
    if runs is None or runs.empty:
        return out
    sel = runs[(runs["experiment"] == "fewshot") & (runs["scenario"] == pair) & (runs["seed"].astype(int) == seed)]
    for _, row in sel.iterrows():
        method = str(row["method"])
        for variant, prefix in (("continued", "fewshot-catboost@"), ("targetonly", "fewshot-targetonly@")):
            if method.startswith(prefix):
                k = float(method[len(prefix):])
                out[(variant, k if k != int(k) else int(k))] = {
                    "roc_auc": float(row["roc_auc"]), "balanced_accuracy": float(row["balanced_accuracy"])}
    return out


def _fewshot_summary(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["iterations"] = rows["iterations"].astype(int)
    rows["k"] = rows["k"].astype(float)
    out = []
    for (variant, it, k), sub in rows.groupby(["variant", "iterations", "k"], sort=True):
        out.append({
            "variant": variant, "iterations": int(it), "k": float(k), "n_pairs": int(sub["pair"].nunique()),
            "roc_auc_mean": float(sub["roc_auc"].mean()), "roc_auc_std": float(sub["roc_auc"].std(ddof=1)),
            "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(sub["balanced_accuracy"].std(ddof=1)),
            "benchmark_roc_auc_mean": float(sub["benchmark_roc_auc"].mean()),
            "mean_n_labelled": float(sub["n_labelled"].mean()),
        })
    return pd.DataFrame(out)


def _plot_fewshot(summary: pd.DataFrame) -> plt.Figure:
    series = [("continued", it) for it in FEWSHOT_ITERATIONS] + [("targetonly", it) for it in FEWSHOT_ITERATIONS]
    colour_of = {it: PALETTE[i] for i, it in enumerate(FEWSHOT_ITERATIONS)}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for ax, (metric, label) in zip(axes, (("roc_auc", "ROC-AUC"), ("balanced_accuracy", "balanced accuracy"))):
        for variant, it in series:
            sub = summary[(summary["variant"] == variant) & (summary["iterations"] == it)].sort_values("k")
            if sub.empty:
                continue
            ax.errorbar(
                sub["k"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"].fillna(0.0),
                color=colour_of[it], marker="o", markersize=4, capsize=2, elinewidth=0.6,
                linestyle="-" if variant == "continued" else "--",
                label=f"{'continued from source' if variant == 'continued' else 'target-only'}, +{it} trees",
            )
        ax.set_xscale("log")
        ax.set_xlabel("labelled target samples per class, k (%, log scale)", color=MUTED)
        ax.set_ylabel(label, color=MUTED)
        ax.set_ylim(0, 1.02)
    ks = sorted(summary["k"].unique())
    axes[0].set_xticks(ks)
    axes[0].set_xticklabels([f"{k:g}" for k in ks])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.suptitle("Fair few-shot: mean over pairs, same labelled rows and tree budgets per variant", x=0.01, ha="left",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    return fig


def write_fewshot_outputs(out_dir: Path) -> None:
    rows = _read(out_dir / "fewshot_fair_rows.csv")
    if rows.empty:
        return
    summary = _fewshot_summary(rows)
    rep = _Report(out_dir)
    rep.table(
        "fewshot_fair", summary,
        "mean over pairs per (variant, extra trees, k): ROC-AUC and balanced accuracy on target test",
        "Fair few-shot comparison",
        note=(
            "continued = source-pretrained CatBoost (benchmark base model) plus N extra trees fitted on the labelled "
            "target rows; targetonly = CatBoost with N trees on the same labelled rows and no source data. Rows come "
            "from `labelled_budget` with the benchmark seed. benchmark_roc_auc is the benchmark's 100-tree row "
            "(fewshot-catboost / fewshot-targetonly), shown as a reproduction check for the 100-tree variants."
        ),
    )
    with plt.rc_context(STYLE):
        rep.figure("fewshot_fair", _plot_fewshot(summary), "fair few-shot ROC-AUC and balanced accuracy vs k")


# ---------------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------------


def run_controls(
    which: list[str],
    out_dir: Path,
    seeds: list[int],
    pairs: list[tuple[str, str]] | None = None,
    threads: int = 2,
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
) -> dict[str, Path]:
    """Run the requested controls in the given order. Each writes CSV, a markdown summary and figures to ``out_dir``."""
    unknown = [w for w in which if w not in CONTROL_NAMES]
    if unknown:
        raise ValueError(f"unknown controls {unknown}; choose from {', '.join(CONTROL_NAMES)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pairs or list(ALL_PAIRS)
    frames = load_core_frames(Path(data_dir))
    runs = load_runs(Path(results_dir))
    written: dict[str, Path] = {}
    for name in which:
        log.info("control %s: starting (threads=%d, seeds=%s)", name, threads, seeds)
        if name == "separability":
            written[name] = run_separability(frames, seeds, out_dir, threads)
            write_separability_outputs(out_dir)
        elif name == "concept":
            written[name] = run_concept(pairs, seeds, out_dir, threads, frames)
            write_concept_outputs(out_dir, written[name])
        elif name == "fewshot_fair":
            written[name] = run_fewshot_fair(pairs, seeds, out_dir, threads, frames, runs)
            write_fewshot_outputs(out_dir)
        elif name == "oracle":
            written[name] = run_oracle(pairs, seeds, out_dir, threads, frames, runs=runs)
            write_oracle_outputs(out_dir, written[name], runs)
    return written
