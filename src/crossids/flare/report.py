"""Summary of the FLARE few-shot evaluation (DESIGN.md §18).

Reads ``<runs>/eval_*.csv`` and ``<runs>/per_attack_<target>.csv`` (written by ``flare.evaluate``) and the Part I
benchmark table ``<benchmark>/runs.csv`` (read only). Writes CSV tables and figures to ``--out``:

* learning curves vs k per target and averaged over targets (ROC-AUC, balanced accuracy), with the best label-free
  Part I method and the full-label CatBoost as horizontal reference lines;
* the k = 0 zero-shot references (CatBoost on the meta-training datasets, and the Part I pairwise mean and best
  source), drawn as points at the k = 0 position of the learning curves;
* Wilcoxon signed-rank tests of each method against target-only CatBoost per k, over (target, draw) pairs;
* per-attack-family recall at k = 16, using ``crossids.analysis.attack_family``.

Example::

    python -m crossids.flare.report --runs runs/flare --out results/flare
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from ..analysis import INK, MUTED, PALETTE, SEQUENTIAL_CMAP, STYLE, attack_family  # noqa: E402
from .evaluate import PER_ATTACK_K  # noqa: E402

FLARE_METHODS = ("flare", "flare-al", "flare-kcenter-raw", "protonet", "flare-noA1", "flare-noA3", "flare-rowmlp",
                 "flare-synthetic", "flare-E1", "flare-u", "flare+cb", "flare-u+cb")
BASELINE_METHODS = ("catboost-target", "rf", "lr", "knn", "catboost-ft", "tabpfn")
REFERENCE = "catboost-fulllabel"
ZERO_SHOT = ("catboost-lodo", "catboost-pairwise-mean", "catboost-pairwise-best")  # k = 0 references (section 20.3)
WILCOXON_METHODS = ("flare", "flare-al", "flare-kcenter-raw", "protonet", "flare-u", "flare+cb", "flare-u+cb")
METHOD_ORDER = ("flare", "flare-u", "flare+cb", "flare-u+cb", "flare-al", "flare-kcenter-raw", "flare-E1",
                "flare-noA1", "flare-noA3", "flare-rowmlp", "flare-synthetic", "protonet", "catboost-target",
                "catboost-ft", "rf", "lr", "knn", "tabpfn", *ZERO_SHOT)
K0_POSITION = 2  # k = 0 is drawn at this x on the log-2 axis, labelled "0"
FAMILY_ORDER = ("benign", "web", "ddos", "dos", "scan", "botnet", "bruteforce", "exploit_malware", "other")


def _read_glob(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_eval(runs: Path) -> pd.DataFrame:
    frame = _read_glob(str(runs / "eval_*.csv"))
    if frame.empty:
        raise SystemExit(f"no eval_*.csv in {runs}")
    return frame


def learning_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (method, target, k): mean and sd over draws and seeds, with the counts."""
    fl = frame[frame["k"] > 0]
    g = fl.groupby(["method", "target", "k"])
    out = g.agg(roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                pr_auc=("pr_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
                balanced_accuracy_sd=("balanced_accuracy", "std"), f1=("f1", "mean"),
                n=("roc_auc", "size")).reset_index()
    return out


def zero_shot_table(frame: pd.DataFrame) -> pd.DataFrame:
    """k = 0 references per (method, target): mean over seeds of each metric (one row per target)."""
    z = frame[(frame["k"] == 0) & (frame["method"].isin(ZERO_SHOT))]
    if z.empty:
        return pd.DataFrame(columns=["method", "target", "k", "roc_auc", "balanced_accuracy", "pr_auc", "f1", "n"])
    g = z.groupby(["method", "target"]).agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
                                           pr_auc=("pr_auc", "mean"), f1=("f1", "mean"), n=("roc_auc", "size"))
    g = g.reset_index()
    g["k"] = 0
    return g


def average_over_targets(table: pd.DataFrame) -> pd.DataFrame:
    """Mean over targets of the per-target means (each target counts once)."""
    return table.groupby(["method", "k"]).agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
                                              pr_auc=("pr_auc", "mean"), f1=("f1", "mean"),
                                              targets=("target", "nunique")).reset_index()


def partI_reference(benchmark: Path) -> pd.DataFrame:
    """Best label-free Part I method per target: LODO, transductive, seed 0, maximum mean ROC-AUC over methods."""
    path = benchmark / "runs.csv"
    if not path.exists():
        return pd.DataFrame(columns=["target", "method", "roc_auc"])
    runs = pd.read_csv(path)
    sel = runs[(runs["experiment"] == "lodo") & (runs["mode"] == "transductive") & (runs["seed"] == 0)]
    rows = []
    for target, grp in sel.groupby("target"):
        means = grp.groupby("method")["roc_auc"].mean()
        if means.empty:
            continue
        rows.append({"target": target, "method": means.idxmax(), "roc_auc": float(means.max())})
    return pd.DataFrame(rows, columns=["target", "method", "roc_auc"])


def fulllabel_reference(frame: pd.DataFrame) -> pd.DataFrame:
    fl = frame[frame["method"] == REFERENCE]
    return fl.groupby("target").agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean")).reset_index()


def wilcoxon_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired tests vs catboost-target per k. Pairs are (target, draw), with metrics averaged over seeds first."""
    fl = frame[frame["k"] > 0]
    rows = []
    for metric in ("roc_auc", "balanced_accuracy"):
        base = fl[fl["method"] == "catboost-target"].groupby(["k", "target", "draw"])[metric].mean()
        for method in WILCOXON_METHODS:
            if method not in set(fl["method"]):
                continue  # only methods that were evaluated
            mine = fl[fl["method"] == method].groupby(["k", "target", "draw"])[metric].mean()
            ks_mine = set(mine.index.get_level_values("k")) if len(mine) else set()
            ks_base = set(base.index.get_level_values("k")) if len(base) else set()
            for k in sorted(fl["k"].unique()):
                if k not in ks_mine or k not in ks_base:
                    continue  # partially evaluated method: test only the k values it has
                a = mine.xs(k, level="k")
                b = base.xs(k, level="k")
                common = a.index.intersection(b.index)
                x, y = a.loc[common].to_numpy(), b.loc[common].to_numpy()
                diff = x - y
                stat, p = (np.nan, np.nan)
                if len(diff) >= 2 and np.any(diff != 0):
                    res = wilcoxon(x, y, zero_method="wilcox")
                    stat, p = float(res.statistic), float(res.pvalue)
                rows.append({"metric": metric, "method": method, "k": int(k), "pairs": int(len(common)),
                             "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                             "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                             "wilcoxon_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def family_recall(per_attack: pd.DataFrame, k: int = PER_ATTACK_K) -> pd.DataFrame:
    """Recall per attack family at k, pooled over targets, draws and seeds."""
    pa = per_attack[per_attack["k"] == k].copy()
    pa["family"] = pa["attack_type"].map(attack_family)
    g = pa.groupby(["method", "family"]).agg(n=("n", "sum"), detected=("detected", "sum")).reset_index()
    g["recall"] = g["detected"] / g["n"]
    return g


def _order(methods) -> list[str]:
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in known)


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=160, facecolor="white", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_learning(table: pd.DataFrame, avg: pd.DataFrame, partI: pd.DataFrame, full: pd.DataFrame,
                  metric: str, out: Path, zero: pd.DataFrame | None = None) -> None:
    """Learning curves per target and averaged. ``zero`` holds the k = 0 references, drawn at K0_POSITION."""
    zero = pd.DataFrame(columns=table.columns) if zero is None or zero.empty else zero
    zero_avg = zero.groupby(["method"]).agg(**{metric: (metric, "mean")}).reset_index() if len(zero) else zero
    targets = sorted(table["target"].unique())
    methods = _order(set(table["method"].unique()) | set(zero["method"].unique()))
    colour = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
    with plt.rc_context(STYLE):
        ncols = 3
        nrows = int(np.ceil((len(targets) + 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        panels = [(t, table[table["target"] == t], zero[zero["target"] == t]) for t in targets]
        panels += [("average over targets", avg, zero_avg)]
        for ax, (title, data, zdata) in zip(axes.flat, panels):
            for m in methods:
                d = data[data["method"] == m].sort_values("k")
                if len(d):
                    ax.plot(d["k"], d[metric], marker="o", ms=3.5, lw=1.4, color=colour[m], label=m)
                zm = zdata[zdata["method"] == m]
                if len(zm):
                    ax.plot([K0_POSITION] * len(zm), zm[metric], marker="D", ms=5, lw=0, color=colour[m],
                            markeredgecolor=INK, label=m if not len(d) else None)
            if title != "average over targets":
                ref = partI[partI["target"] == title]
                if len(ref) and metric == "roc_auc":
                    ax.axhline(float(ref["roc_auc"].iloc[0]), ls=":", color=MUTED, lw=1,
                               label=f"Part I best label-free ({ref['method'].iloc[0]})")
                fr = full[full["target"] == title]
                if len(fr):
                    ax.axhline(float(fr[metric].iloc[0]), ls="--", color=INK, lw=1, label="CatBoost, full labels")
            ax.set_xscale("log", base=2)
            ax.set_xticks([K0_POSITION, 4, 8, 16, 32, 64, 128])
            ax.set_xticklabels(["0", "4", "8", "16", "32", "64", "128"])
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("labelled target flows k")
            ax.set_ylabel(metric.replace("_", " "))
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        _save(fig, out, f"learning_curves_{metric}")


def plot_family(fam: pd.DataFrame, out: Path) -> None:
    pivot = fam.pivot(index="method", columns="family", values="recall")
    pivot = pivot.reindex(index=_order(pivot.index), columns=[f for f in FAMILY_ORDER if f in pivot.columns])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 2.5, 0.45 * len(pivot.index) + 1.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 else INK)
        ax.set_title(f"Recall per attack family at k = {PER_ATTACK_K} (pooled over targets)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="recall")
        _save(fig, out, "per_attack_family_k16")


def run(runs: Path, out: Path, benchmark: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame = load_eval(runs)
    table = learning_table(frame)
    avg = average_over_targets(table)
    partI = partI_reference(benchmark)
    full = fulllabel_reference(frame)
    zero = zero_shot_table(frame)
    zero_avg = average_over_targets(zero.assign(k=0)) if len(zero) else zero
    table.to_csv(out / "learning_by_target.csv", index=False)
    avg.to_csv(out / "learning_average.csv", index=False)
    zero.to_csv(out / "zero_shot_by_target.csv", index=False)
    wil = wilcoxon_table(frame)
    wil.to_csv(out / "wilcoxon_vs_catboost_target.csv", index=False)
    partI.to_csv(out / "partI_best_label_free.csv", index=False)
    full.to_csv(out / "catboost_fulllabel.csv", index=False)
    plot_learning(table, avg, partI, full, "roc_auc", out, zero=zero)
    plot_learning(table, avg, partI, full, "balanced_accuracy", out, zero=zero)
    pa = _read_glob(str(runs / "per_attack_*.csv"))
    if not pa.empty:
        fam = family_recall(pa)
        fam.to_csv(out / "per_attack_family_k16.csv", index=False)
        plot_family(fam, out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m crossids.flare.report", description="Summarise the FLARE evaluation")
    p.add_argument("--runs", default="runs/flare", help="folder with eval_<target>.csv and per_attack_<target>.csv")
    p.add_argument("--out", default="results/flare", help="output folder")
    p.add_argument("--benchmark", default="results", help="folder with the Part I runs.csv (read only)")
    a = p.parse_args(argv)
    run(Path(a.runs), Path(a.out), Path(a.benchmark))


if __name__ == "__main__":
    main()
