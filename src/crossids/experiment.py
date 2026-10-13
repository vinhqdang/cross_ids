"""Experiment protocols (in-domain, pairwise, LODO), resumable runs and result files."""

from __future__ import annotations

import copy
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
from .methods import METHOD_NAMES, make_method
from .methods.base import Domain, MethodUnavailable
from .metrics import METRIC_NAMES, evaluate, per_attack_recall
from .schema import SOURCES, features_for, supports

log = logging.getLogger("crossids")

EXPERIMENT_NAMES = ("indomain", "pairwise", "lodo")

DEFAULT_CONFIG: dict = {
    "source": "real",
    "datasets": list(SOURCES),
    "max_rows": 100_000,
    "seeds": [0, 1, 2],
    "schema": "core",
    "test_size": 0.3,
    "transductive": True,
    "methods": list(METHOD_NAMES),
    "experiments": list(EXPERIMENT_NAMES),
    "data_dir": "data",
    "results_dir": "results",
    "synthetic": {"n_domains": 3, "n": 4000, "seed": 0},
    "threads": None,
}

RUN_COLUMNS = [
    "method",
    "experiment",
    "scenario",
    "transfer_type",
    "sources",
    "target",
    "seed",
    "schema",
    "transductive",
    "pool",
    "n_train",
    "n_test",
    "attack_fraction_test",
    *METRIC_NAMES,
    "true_prior",
    "est_prior",
    "n_features",
    "fit_seconds",
    "predict_seconds",
]
PER_ATTACK_COLUMNS = ["method", "experiment", "scenario", "seed", "attack_type", "n", "detected", "recall"]


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Read a YAML config and merge it over the defaults."""
    with open(path, encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(user)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    unknown = [m for m in cfg["methods"] if m not in METHOD_NAMES]
    if unknown:
        raise ValueError(f"unknown methods in config: {unknown}")
    unknown = [e for e in cfg["experiments"] if e not in EXPERIMENT_NAMES]
    if unknown:
        raise ValueError(f"unknown experiments in config: {unknown}")
    if cfg["schema"] not in ("minimal", "core", "extended"):
        raise ValueError(f"unknown schema {cfg['schema']!r}")
    return cfg


@dataclass(frozen=True)
class Scenario:
    """One train/target configuration of an experiment."""

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
    """Return the harmonised tables and metadata of the datasets that support the configured schema."""
    tier = cfg["schema"]
    tables: dict[str, pd.DataFrame] = {}
    infos: dict[str, DatasetInfo] = {}
    if cfg["source"] == "synthetic":
        params = cfg["synthetic"]
        tables = synthetic_tables(**params)
        infos = synthetic_info(list(tables))
        return tables, infos
    for key in datasets or cfg["datasets"]:
        info = real_info(key)
        if not supports(key, tier):
            log.info("skipping %s: does not support schema %s", key, tier)
            continue
        tables[key] = load_processed(key, data_dir)
        infos[key] = info
    return tables, infos


def build_scenarios(experiments: list[str], keys: list[str], infos: dict[str, DatasetInfo]) -> list[Scenario]:
    """Enumerate the in-domain, pairwise and LODO scenarios over the available datasets."""
    scenarios: list[Scenario] = []
    for exp in experiments:
        if exp == "indomain":
            for k in keys:
                scenarios.append(Scenario("indomain", k, (k,), k, "in-domain"))
        elif exp == "pairwise":
            for a in keys:
                for b in keys:
                    if a != b:
                        tt = transfer_type(infos[a], infos[b])
                        scenarios.append(Scenario("pairwise", f"{a}->{b}", (a,), b, tt))
        elif exp == "lodo":
            for b in keys:
                others = tuple(k for k in keys if k != b)
                if others:
                    scenarios.append(Scenario("lodo", f"leave-{b}-out", others, b, "lodo"))
        else:
            raise ValueError(f"unknown experiment {exp!r}")
    return scenarios


def split_tables(tables: dict[str, pd.DataFrame], test_size: float, seed: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Stratified train/test split of every table."""
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key, frame in tables.items():
        train, test = train_test_split(frame, test_size=test_size, stratify=frame["label"], random_state=seed)
        out[key] = (train.reset_index(drop=True), test.reset_index(drop=True))
    return out


def _done_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["method", "experiment", "scenario", "seed"])
    return {(str(r.method), str(r.experiment), str(r.scenario), int(r.seed)) for r in frame.itertuples()}


def _append(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _domain(key: str, frame: pd.DataFrame, features: list[str]) -> Domain:
    meta = frame[["attack_type", "dataset"]].reset_index(drop=True)
    return Domain(name=key, X=frame[features].to_numpy(dtype=np.float64), y=frame["label"].to_numpy(), meta=meta)


def run_experiments(
    cfg: dict,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Run every (method, scenario, seed) combination not yet present in runs.csv and append the results.

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
    tier = cfg["schema"]
    features = list(features_for(tier))

    tables, infos = load_tables(cfg, data_dir)
    keys = list(tables)
    if not keys:
        raise ValueError("no datasets available for this schema")
    log.info("datasets (%s schema, %d features): %s", tier, len(features), ", ".join(keys))
    scenarios = build_scenarios(cfg["experiments"], keys, infos)
    done = _done_keys(runs_path)
    usable_methods: list[str] = []
    for name in cfg["methods"]:
        try:
            make_method(name, 0)
            usable_methods.append(name)
        except MethodUnavailable as exc:
            log.warning("skipping method %s: %s", name, exc)
    total = len(cfg["seeds"]) * len(scenarios) * len(usable_methods)
    log.info("%d scenarios x %d methods x %d seeds = %d runs", len(scenarios), len(usable_methods), len(cfg["seeds"]), total)

    counter = 0
    for seed in cfg["seeds"]:
        splits = split_tables(tables, cfg["test_size"], seed)
        for sc in scenarios:
            for name in usable_methods:
                counter += 1
                key = (name, sc.experiment, sc.name, int(seed))
                if key in done:
                    log.info("[%d/%d] skip (done) %s %s %s seed=%d", counter, total, name, sc.experiment, sc.name, seed)
                    continue
                row, attack_rows = _run_one(name, sc, seed, splits, features, tier, cfg["transductive"])
                _append(runs_path, [row], RUN_COLUMNS)
                _append(attack_path, attack_rows, PER_ATTACK_COLUMNS)
                done.add(key)
                log.info(
                    "[%d/%d] %s %s %s seed=%d roc_auc=%.4f bal_acc=%.4f fit=%.1fs",
                    counter, total, name, sc.experiment, sc.name, seed,
                    row["roc_auc"], row["balanced_accuracy"], row["fit_seconds"],
                )
    log.info("results written to %s", runs_path)
    return runs_path


def _run_one(
    name: str,
    sc: Scenario,
    seed: int,
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    tier: str,
    transductive: bool,
) -> tuple[dict, list[dict]]:
    """Fit one method on one scenario and evaluate it on the target test split."""
    method = make_method(name, seed)
    sources = [_domain(s, splits[s][0], features) for s in sc.sources]
    target_train, target_test = splits[sc.target]
    use_pool = transductive and method.transductive and sc.experiment != "indomain"
    pool = target_train[features].to_numpy(dtype=np.float64) if use_pool else None
    t0 = time.perf_counter()
    method.fit(sources, pool)
    fit_seconds = time.perf_counter() - t0
    X_test = target_test[features].to_numpy(dtype=np.float64)
    t1 = time.perf_counter()
    proba = method.predict_proba(X_test)
    predict_seconds = time.perf_counter() - t1
    y_test = target_test["label"].to_numpy().astype(int)
    pred = (proba >= 0.5).astype(int)
    metrics = evaluate(y_test, proba, pred)
    row = {
        "method": name,
        "experiment": sc.experiment,
        "scenario": sc.name,
        "transfer_type": sc.transfer_type,
        "sources": "+".join(sc.sources),
        "target": sc.target,
        "seed": seed,
        "schema": tier,
        "transductive": bool(transductive and method.transductive),
        "pool": pool is not None,
        "n_train": int(sum(len(s.y) for s in sources)),
        "n_test": int(len(y_test)),
        "attack_fraction_test": float(y_test.mean()),
        **metrics,
        "true_prior": float(y_test.mean()),
        "est_prior": float(method.info.get("pi_t", math.nan)),
        "n_features": float(method.info.get("n_features", len(features))),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
    }
    per_attack = per_attack_recall(y_test, pred, target_test["attack_type"].to_numpy())
    attack_rows = [
        {
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
