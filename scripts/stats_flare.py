"""Paired statistics for the FLARE-2 few-label benchmark.

Reads results/flare/raw and writes results/flare/stats/*.csv and stats.md.

Unit of analysis is the target dataset (8 targets), because draws of one
target share the same pool and are not independent. For every k and for the
mean over k, the per-target mean difference (FLARE-2 minus baseline) is tested
with an exact two-sided Wilcoxon signed-rank test; p-values are Holm-adjusted
over the baselines. 95% intervals bootstrap the targets. Class-level tables use
(target, class) pairs as units, averaged over k.
"""
from __future__ import annotations

import glob
import itertools
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAW = "results/flare/raw"
OUT = "results/flare/stats"
BASELINES = ["rf", "catboost-target", "catboost-ft", "lr", "knn"]
KS = [4, 8, 16, 32, 64, 128]
CKS = [2, 4, 8, 16]
RNG = np.random.default_rng(0)


def load_eval(seeds: tuple[int, ...] = (0,)) -> pd.DataFrame:
    """Binary results of FLARE-2 and the baselines for the given meta-training seeds (default: seed 0 only)."""
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/eval_*.csv"))]
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(["method", "target", "k", "draw", "seed"])
    return d[d.method.isin(["flare2"] + BASELINES) & d.seed.isin(seeds)]


def load_class() -> pd.DataFrame:
    variant = re.compile(r"_(f2m|f2mbinary|f2noperm|f2insrc)_s\d+\.csv$")  # ablation files repeat the baselines
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/class_*_s0.csv"))
              if not os.path.basename(f).startswith("class_supports") and not variant.search(os.path.basename(f))]
    d = pd.concat(frames, ignore_index=True)
    d = d[d.metric != "confusion"]
    return d.drop_duplicates(["task", "method", "target", "seed", "k", "draw", "metric", "cls"])


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return list(out)


def boot_ci(x: np.ndarray, n: int = 5000) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.allclose(x, 0):
        return 1.0
    return float(wilcoxon(x, method="exact" if len(x) <= 25 else "approx").pvalue)


def paired_table(units: pd.DataFrame, group_cols: list[str], value: str, label: str) -> pd.DataFrame:
    """units: one row per (unit, baseline, group) with column 'diff'."""
    rows = []
    for g, sub in units.groupby(group_cols):
        for b, sb in sub.groupby("baseline"):
            x = sb["diff"].to_numpy()
            lo, hi = boot_ci(x)
            rows.append(dict(zip(group_cols, g if isinstance(g, tuple) else (g,)),
                             baseline=b, n_units=len(x), mean_diff=x.mean(), ci_lo=lo, ci_hi=hi,
                             wins=int((x > 0).sum()), losses=int((x < 0).sum()), p=wil(x), metric=label))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for g, sub in t.groupby(group_cols):
        t.loc[sub.index, "p_holm"] = holm(sub.p.tolist())
    return t


def binary_stats(d: pd.DataFrame) -> pd.DataFrame:
    per = (d.groupby(["method", "target", "k"]).roc_auc.mean().reset_index())
    f = per[per.method == "flare2"].set_index(["target", "k"]).roc_auc
    rows = []
    for b in BASELINES:
        pb = per[per.method == b].set_index(["target", "k"]).roc_auc
        j = pd.concat([f.rename("f"), pb.rename("b")], axis=1).dropna().reset_index()
        j["diff"] = j.f - j.b
        j["baseline"] = b
        rows.append(j)
    u = pd.concat(rows)
    u = u[u.k.isin(KS)]
    by_k = paired_table(u, ["k"], "diff", "roc_auc")
    avg = u.groupby(["target", "baseline"], as_index=False)["diff"].mean()
    avg["scope"] = "mean over k"
    overall = paired_table(avg, ["scope"], "diff", "roc_auc")
    low = u[u.k <= 16].groupby(["target", "baseline"], as_index=False)["diff"].mean()
    low["scope"] = "k<=16"
    high = u[u.k >= 32].groupby(["target", "baseline"], as_index=False)["diff"].mean()
    high["scope"] = "k>=32"
    ranges = paired_table(pd.concat([low, high]), ["scope"], "diff", "roc_auc")
    return by_k, pd.concat([overall, ranges]), u


def class_stats(c: pd.DataFrame) -> pd.DataFrame:
    base = {"rf": "rf", "catboost-target": "catboost", "lr": "lr", "knn": "knn"}
    out = []
    for task, metric in [("perclass", "roc_auc"), ("perclass", "recall"), ("perclass", "fpr"), ("nway", "macro_f1")]:
        s = c[(c.task == task) & (c.metric == metric) & c.k.isin(CKS)]
        s = s.groupby(["method", "target", "cls", "k"], dropna=False).value.mean().reset_index()
        f = s[s.method == "flare2"].set_index(["target", "cls", "k"]).value
        for name, m in base.items():
            pb = s[s.method == m].set_index(["target", "cls", "k"]).value
            j = pd.concat([f.rename("f"), pb.rename("b")], axis=1).dropna().reset_index()
            if j.empty:
                continue
            j["diff"] = j.f - j.b
            j["baseline"] = name
            j["task"] = f"{task}:{metric}"
            out.append(j)
    u = pd.concat(out)
    # one unit per (target, class); k averaged within, plus a per-k view
    unit = u.groupby(["task", "baseline", "target", "cls"], dropna=False, as_index=False)["diff"].mean()
    rows = []
    for (task, b), sb in unit.groupby(["task", "baseline"]):
        x = sb["diff"].to_numpy()
        lo, hi = boot_ci(x)
        # target-level view: average classes inside a target first
        xt = sb.groupby("target")["diff"].mean().to_numpy()
        rows.append(dict(task=task, baseline=b, n_units=len(x), mean_diff=x.mean(), ci_lo=lo, ci_hi=hi,
                         wins=int((x > 0).sum()), losses=int((x < 0).sum()), p=wil(x),
                         n_targets=len(xt), target_mean_diff=xt.mean(), target_wins=int((xt > 0).sum()),
                         target_p=wil(xt)))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for task, sub in t.groupby("task"):
        t.loc[sub.index, "p_holm"] = holm(sub.p.tolist())
    return t, u


def per_target_binary(u: pd.DataFrame) -> pd.DataFrame:
    return (u[u.baseline.isin(["rf", "catboost-target"])]
            .pivot_table(index=["target", "baseline"], columns="k", values="diff").round(3))


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    d = load_eval()
    by_k, summary, u = binary_stats(d)
    c = load_class()
    ctab, cu = class_stats(c)
    pt = per_target_binary(u)
    by_k.to_csv(f"{OUT}/binary_by_k.csv", index=False)
    summary.to_csv(f"{OUT}/binary_summary.csv", index=False)
    ctab.to_csv(f"{OUT}/class_summary.csv", index=False)
    pt.to_csv(f"{OUT}/binary_per_target_diff.csv")
    targets = sorted(d[d.method == "flare2"].target.unique())
    with open(f"{OUT}/stats.md", "w") as fh:
        fh.write(f"# FLARE-2 paired statistics\n\nTargets ({len(targets)}): {', '.join(targets)}.\n\n")
        fh.write("## Binary ROC-AUC, FLARE-2 minus baseline (units = targets)\n\n")
        fh.write(md(summary[["scope", "baseline", "n_units", "mean_diff", "ci_lo", "ci_hi", "wins", "losses", "p", "p_holm"]]))
        fh.write("\n\n### Per k\n\n")
        fh.write(md(by_k[["k", "baseline", "mean_diff", "ci_lo", "ci_hi", "wins", "losses", "p", "p_holm"]]))
        fh.write("\n\n### Per-target mean difference vs RF and CatBoost-target\n\n")
        fh.write(pt.reset_index().to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\n## Class level (units = target x class, averaged over k in 2..16)\n\n")
        fh.write(md(ctab[["task", "baseline", "n_units", "mean_diff", "ci_lo", "ci_hi", "wins", "losses", "p", "p_holm",
                          "n_targets", "target_mean_diff", "target_wins", "target_p"]]))
        fh.write("\n")
    print(open(f"{OUT}/stats.md").read())


if __name__ == "__main__":
    main()
