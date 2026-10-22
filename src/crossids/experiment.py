"""Experiment protocols (in-domain, pairwise, LODO, few-shot), resumable runs and result files."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .datasets import DatasetInfo, load_processed, real_info, synthetic_info, synthetic_tables
from .fewshot import FEWSHOT_KINDS, fewshot_scores, shift_score, targetonly_scores
from .methods import ALL_METHOD_NAMES, METHOD_NAMES, make_method
from .methods.base import Domain, MethodUnavailable
from .metrics import METRIC_NAMES, evaluate, per_attack_recall
from .schema import ALL_TIERS, SOURCES, features_for, supports

log = logging.getLogger("crossids")

EXPERIMENT_NAMES = ("indomain", "pairwise", "lodo", "fewshot")
MODE_NAMES = ("transductive", "inductive")

DEFAULT_CONFIG: dict = {
    "source": "real",
    "datasets": list(SOURCES),
    "max_rows": 100_000,
    "seeds": [0, 1, 2],
    "schema": "core",
    "test_size": 0.3,
    "modes": list(MODE_NAMES),
    "methods": list(METHOD_NAMES),
    "experiments": list(EXPERIMENT_NAMES),
    "fewshot_k": [0.1, 1, 5],
    "device": "cpu",
    "threads": None,
    "data_dir": "data",
    "results_dir": "results",
    "synthetic": {"n_domains": 3, "n": 4000, "seed": 0},
}

RUN_COLUMNS = [
    "config_hash",
    "schema",
    "mode",
    "method",
    "experiment",
    "scenario",
    "transfer_type",
    "sources",
    "target",
    "seed",
    "pool",
    "n_train",
    "n_test",
    "attack_fraction_test",
    *METRIC_NAMES,
    "true_prior",
    "est_prior",
    "prior_abs_error",
    "n_features",
    "anchor_purity",
    "shift_score",
    "fewshot_k",
    "fit_seconds",
    "predict_seconds",
]
PER_ATTACK_COLUMNS = ["config_hash", "schema", "mode", "method", "experiment", "scenario", "seed", "attack_type", "n", "detected", "recall"]
RESULT_RESUME_COLUMNS = ("schema", "mode", "method", "experiment", "scenario", "seed")


def fewshot_method(kind: str, k: float) -> str:
    """Method name of a few-shot row: ``fewshot-catboost@k`` (continued) or ``fewshot-targetonly@k`` (control)."""
    return f"fewshot-{kind}@{k}"


def config_hash(cfg: dict) -> str:
    """Stable hash of the experiment-relevant configuration."""
    relevant = {k: v for k, v in cfg.items() if k not in ("results_dir", "data_dir")}
    return hashlib.sha1(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Read a YAML config, merge it over the defaults, apply overrides and validate."""
    with open(path, encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(user)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    unknown = [m for m in cfg["methods"] if m not in ALL_METHOD_NAMES]
    if unknown:
        raise ValueError(f"unknown methods in config: {unknown}")
    unknown = [e for e in cfg["experiments"] if e not in EXPERIMENT_NAMES]
    if unknown:
        raise ValueError(f"unknown experiments in config: {unknown}")
    unknown = [m for m in cfg["modes"] if m not in MODE_NAMES]
    if unknown:
        raise ValueError(f"unknown modes in config: {unknown}")
    if cfg["schema"] not in ALL_TIERS:
        raise ValueError(f"unknown schema {cfg['schema']!r}")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {cfg['device']!r}")
    return cfg


@dataclass(frozen=True)
class Scenario:
    """One source/target configuration of an experiment."""

    experiment: str
    name: str
    sources: tuple[str, ...]
    target: str
    transfer_type: str


def transfer_type(a: DatasetInfo, b: DatasetInfo) -> str:
    """Classify a source-target pair by family and extractor."""
    if a.family == b.family:
        return "same-family"
    if a.extractor == b.extractor:
        return "same-extractor"
    return "cross-extractor"


def load_tables(cfg: dict, data_dir: Path, datasets: list[str] | None = None) -> tuple[dict, dict[str, DatasetInfo]]:
    """Harmonised tables and metadata of the datasets that support the configured schema."""
    tier = cfg["schema"]
    if cfg["source"] == "synthetic":
        tables = synthetic_tables(**cfg["synthetic"])
        return tables, synthetic_info(list(tables))
    tables: dict[str, pd.DataFrame] = {}
    infos: dict[str, DatasetInfo] = {}
    for key in datasets or cfg["datasets"]:
        if not supports(key, tier):
            log.info("skipping %s: does not support schema %s", key, tier)
            continue
        tables[key] = load_processed(key, data_dir)
        infos[key] = real_info(key)
    return tables, infos


def build_scenarios(experiments: list[str], keys: list[str], infos: dict[str, DatasetInfo]) -> list[Scenario]:
    """Enumerate in-domain, pairwise and LODO scenarios over the available datasets."""
    scenarios: list[Scenario] = []
    for exp in experiments:
        if exp == "indomain":
            scenarios += [Scenario("indomain", k, (k,), k, "in-domain") for k in keys]
        elif exp in ("pairwise", "fewshot"):
            for a in keys:
                for b in keys:
                    if a != b:
                        tt = transfer_type(infos[a], infos[b])
                        scenarios.append(Scenario(exp, f"{a}->{b}", (a,), b, tt))
        elif exp == "lodo":
            for b in keys:
                others = tuple(k for k in keys if k != b)
                if others:
                    scenarios.append(Scenario("lodo", f"leave-{b}-out", others, b, "lodo"))
        else:
            raise ValueError(f"unknown experiment {exp!r}")
    return scenarios


def split_tables(tables: dict[str, pd.DataFrame], test_size: float, seed: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Stratified train/test split of every table (plain random split when a class is too small)."""
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key, frame in tables.items():
        counts = frame["label"].value_counts()
        stratify = frame["label"] if len(counts) == 2 and counts.min() >= 2 else None
        train, test = train_test_split(frame, test_size=test_size, stratify=stratify, random_state=seed)
        out[key] = (train.reset_index(drop=True), test.reset_index(drop=True))
    return out


def _check_header(path: Path, columns: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    if header != columns:
        raise ValueError(
            f"{path} has a different header than this version of crossids; move or delete it, "
            "or choose another results_dir"
        )


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=list(RESULT_RESUME_COLUMNS), dtype=str)
    return {tuple(str(v) for v in row) for row in frame.itertuples(index=False)}


def _append(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=not path.exists(), index=False)


def _domain(key: str, frame: pd.DataFrame, features: list[str]) -> Domain:
    meta = frame[["attack_type", "dataset"]].reset_index(drop=True)
    return Domain(name=key, X=frame[features].to_numpy(dtype=np.float64), y=frame["label"].to_numpy(), meta=meta)


def run_experiments(cfg: dict, data_dir: Path | None = None, results_dir: Path | None = None) -> Path:
    """Run every missing (schema, mode, method, experiment, scenario, seed) row and append the results.

    ``cfg["threads"]`` optionally caps the BLAS and OpenMP thread pools (None uses all cores).
    """
    limit = threadpool_limits(limits=int(cfg["threads"])) if cfg.get("threads") else nullcontext()
    with limit:
        return _run_all(cfg, data_dir, results_dir)


def _run_all(cfg: dict, data_dir: Path | None, results_dir: Path | None) -> Path:
    data_dir = Path(data_dir or cfg["data_dir"])
    results_dir = Path(results_dir or cfg["results_dir"])
    runs_path = results_dir / "runs.csv"
    attack_path = results_dir / "per_attack.csv"
    _check_header(runs_path, RUN_COLUMNS)
    _check_header(attack_path, PER_ATTACK_COLUMNS)
    tier = cfg["schema"]
    features = list(features_for(tier))
    chash = config_hash(cfg)

    tables, infos = load_tables(cfg, data_dir)
    keys = list(tables)
    if not keys:
        raise ValueError(f"no datasets support schema {tier!r}")
    log.info("datasets (%s schema, %d features): %s  [config %s]", tier, len(features), ", ".join(keys), chash)
    experiments = [e for e in cfg["experiments"] if e != "fewshot"]
    scenarios = build_scenarios(experiments, keys, infos)
    fewshot = build_scenarios(["fewshot"], keys, infos) if "fewshot" in cfg["experiments"] else []

    usable: list[str] = []
    for name in cfg["methods"]:
        try:
            make_method(name, 0, cfg["threads"], cfg["device"])
            usable.append(name)
        except MethodUnavailable as exc:
            log.warning("skipping method %s: %s", name, exc)

    done = _done_keys(runs_path)
    seeds = list(cfg["seeds"])
    n_main = len(seeds) * len(cfg["modes"]) * len(scenarios) * len(usable)
    n_few = len(seeds) * len(fewshot) * len(cfg["fewshot_k"]) * len(FEWSHOT_KINDS)
    total = n_main + n_few
    log.info("%d runs planned (%d main, %d few-shot)", total, n_main, n_few)

    counter = 0
    for seed in seeds:
        splits = split_tables(tables, cfg["test_size"], seed)
        for mode in cfg["modes"]:
            for sc in scenarios:
                for name in usable:
                    counter += 1
                    key = (tier, mode, name, sc.experiment, sc.name, str(seed))
                    if key in done:
                        log.info("[%d/%d] skip (done) %s %s %s %s seed=%d", counter, total, mode, name, sc.experiment, sc.name, seed)
                        continue
                    method = make_method(name, seed, cfg["threads"], cfg["device"])
                    if mode == "inductive" and not method.supports_inductive:
                        continue
                    row, attack_rows = _run_one(name, sc, seed, splits, features, tier, mode, chash, cfg)
                    _append(attack_path, attack_rows, PER_ATTACK_COLUMNS)
                    _append(runs_path, [row], RUN_COLUMNS)
                    done.add(key)
                    log.info(
                        "[%d/%d] %s %s %s %s seed=%d roc_auc=%.4f bal_acc=%.4f fit=%.1fs",
                        counter, total, mode, name, sc.experiment, sc.name, seed,
                        row["roc_auc"], row["balanced_accuracy"], row["fit_seconds"],
                    )
        for sc in fewshot:
            counter += len(cfg["fewshot_k"]) * len(FEWSHOT_KINDS)
            # Each (budget, kind) row is resumed on its own: a missing control row is computed without
            # recomputing its sibling rows that already exist.
            missing: dict = {}
            for k in cfg["fewshot_k"]:
                kinds = [kind for kind in FEWSHOT_KINDS
                         if (tier, "fewshot", fewshot_method(kind, k), "fewshot", sc.name, str(seed)) not in done]
                if kinds:
                    missing[k] = kinds
            if not missing:
                log.info("skip (done) few-shot %s seed=%d", sc.name, seed)
                continue
            rows, attack_rows = _run_fewshot(sc, seed, splits, features, chash, cfg, missing)
            for row, arows in zip(rows, attack_rows):
                _append(attack_path, arows, PER_ATTACK_COLUMNS)
                _append(runs_path, [row], RUN_COLUMNS)
                done.add((tier, "fewshot", row["method"], "fewshot", sc.name, str(seed)))
                log.info("few-shot %s k=%s seed=%d roc_auc=%.4f bal_acc=%.4f",
                         sc.name, row["fewshot_k"], seed, row["roc_auc"], row["balanced_accuracy"])
    log.info("results written to %s", runs_path)
    return runs_path


def _finish_row(base: dict, y_test: np.ndarray, scores: np.ndarray, pred: np.ndarray, attack_type: np.ndarray,
                fit_seconds: float, predict_seconds: float, sc: Scenario, seed: int, mode: str, name: str,
                chash: str, tier: str, n_train: int, pool: bool) -> tuple[dict, list[dict]]:
    metrics = evaluate(y_test, scores, pred)
    true_prior = float(y_test.mean())
    row = {
        "config_hash": chash,
        "schema": tier,
        "mode": mode,
        "method": name,
        "experiment": sc.experiment,
        "scenario": sc.name,
        "transfer_type": sc.transfer_type,
        "sources": "+".join(sc.sources),
        "target": sc.target,
        "seed": seed,
        "pool": pool,
        "n_train": n_train,
        "n_test": int(len(y_test)),
        "attack_fraction_test": true_prior,
        **metrics,
        "true_prior": true_prior,
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        **base,
    }
    per_attack = per_attack_recall(y_test, pred, attack_type)
    attack_rows = [
        {
            "config_hash": chash,
            "schema": tier,
            "mode": mode,
            "method": name,
            "experiment": sc.experiment,
            "scenario": sc.name,
            "seed": seed,
            "attack_type": r.attack_type,
            "n": int(r.n),
            "detected": int(r.detected),
            "recall": float(r.recall),
        }
        for r in per_attack.itertuples()
    ]
    return row, attack_rows


def _blank_base() -> dict:
    return {"est_prior": math.nan, "prior_abs_error": math.nan, "n_features": math.nan,
            "anchor_purity": math.nan, "shift_score": math.nan, "fewshot_k": math.nan}


def _run_one(name: str, sc: Scenario, seed: int, splits: dict, features: list[str], tier: str, mode: str,
             chash: str, cfg: dict) -> tuple[dict, list[dict]]:
    """Fit one method on one scenario and evaluate it on the target test split."""
    method = make_method(name, seed, cfg["threads"], cfg["device"])
    sources = [_domain(s, splits[s][0], features) for s in sc.sources]
    target_train, target_test = splits[sc.target]
    use_pool = mode == "transductive" and method.transductive and sc.experiment != "indomain"
    pool = target_train[features].to_numpy(dtype=np.float64) if use_pool else None
    if getattr(getattr(method, "config", None), "oracle", "") and pool is not None:
        method.oracle_pool_labels = target_train["label"].to_numpy().astype(int)  # diagnostic variants only
    t0 = time.perf_counter()
    method.fit(sources, pool)
    fit_seconds = time.perf_counter() - t0
    X_test = target_test[features].to_numpy(dtype=np.float64)
    t1 = time.perf_counter()
    scores = method.decision_scores(X_test)
    pred = method.predict(X_test)
    predict_seconds = time.perf_counter() - t1
    y_test = target_test["label"].to_numpy().astype(int)
    base = _blank_base()
    base["n_features"] = float(method.info.get("n_features", len(features)))
    base["est_prior"] = float(method.info.get("pi_t", math.nan))
    if np.isfinite(base["est_prior"]):
        base["prior_abs_error"] = abs(base["est_prior"] - float(y_test.mean()))
    anchor = getattr(method, "anchor_idx_", None)
    if anchor is not None and pool is not None:
        y_pool = target_train["label"].to_numpy().astype(int)
        base["anchor_purity"] = float(np.mean(y_pool[anchor] == 0))
    return _finish_row(base, y_test, scores, pred, target_test["attack_type"].to_numpy(), fit_seconds,
                       predict_seconds, sc, seed, mode, name, chash, tier, sum(len(s.y) for s in sources),
                       pool is not None)


def _run_fewshot(sc: Scenario, seed: int, splits: dict, features: list[str], chash: str, cfg: dict,
                 missing: dict) -> tuple[list[dict], list[list[dict]]]:
    """Few-shot rows for the (budget k -> kinds) pairs in ``missing``, with the shift score of the target pool.

    The source-pretrained CatBoost is only fitted when a ``catboost`` row is missing.
    """
    sources = [_domain(s, splits[s][0], features) for s in sc.sources]
    target_train, target_test = splits[sc.target]
    source_X = np.vstack([s.X for s in sources])
    pool = target_train[features].to_numpy(dtype=np.float64)
    shift = shift_score(source_X, pool, seed)
    X_lab = pool
    y_lab = target_train["label"].to_numpy().astype(int)
    X_test = target_test[features].to_numpy(dtype=np.float64)
    y_test = target_test["label"].to_numpy().astype(int)
    scores_by_kind: dict[str, dict[float, np.ndarray]] = {}
    seconds_by_kind: dict[str, float] = {}
    for kind in FEWSHOT_KINDS:
        ks = [k for k, kinds in missing.items() if kind in kinds]
        if not ks:
            continue
        t0 = time.perf_counter()
        if kind == "catboost":
            scores_by_kind[kind] = fewshot_scores(sources, (X_lab, y_lab), X_test, ks, seed, cfg["threads"], cfg["device"])
        else:
            scores_by_kind[kind] = targetonly_scores((X_lab, y_lab), X_test, ks, seed, cfg["threads"], cfg["device"])
        seconds_by_kind[kind] = (time.perf_counter() - t0) / len(ks)
    rows, attack_rows = [], []
    for k, kinds in missing.items():
        for kind in FEWSHOT_KINDS:
            if kind not in kinds:
                continue
            scores = scores_by_kind[kind][k]
            pred = (scores >= 0.5).astype(int)
            base = _blank_base()
            base["shift_score"] = shift
            base["fewshot_k"] = float(k)
            base["n_features"] = float(len(features))
            base["est_prior"] = math.nan
            row, arows = _finish_row(base, y_test, scores, pred, target_test["attack_type"].to_numpy(),
                                     seconds_by_kind[kind], 0.0, sc, seed, "fewshot", fewshot_method(kind, k),
                                     chash, cfg["schema"], sum(len(s.y) for s in sources), True)
            rows.append(row)
            attack_rows.append(arows)
    return rows, attack_rows
