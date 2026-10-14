"""Summary of the attack-class evaluation of FLARE-2 (DESIGN.md section 21.5).

Reads ``<runs>/class_*.csv`` written by ``crossids.flare.evaluate_class`` and writes CSV tables and figures to ``--out``:

* N-way learning curves: macro-F1 and balanced accuracy against k per target and averaged over targets;
* per-class detection learning curves: ROC-AUC (mean over the target's classes) against k, with recall at 0.5 and FPR;
* Wilcoxon signed-rank tests of each method against target-only CatBoost per k: N-way over (target, draw) pairs with
  seeds averaged first, per-class over (target, class, draw) pairs;
* per-class confusion-matrix heatmaps at k = 8 for each target (mean over draws, rows normalised by true class).

Example::

    python -m crossids.flare.report_class --runs runs/flare --out results/flare/class
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

from ..analysis import INK, MUTED, PALETTE, SEQUENTIAL_CMAP, STYLE  # noqa: E402

REFERENCE = "catboost"
METHOD_ORDER = ("flare2", "flare2-binary", "flare2-noperm", "flare2-insrc", "flare2-ctx", "flare", "protonet",
                "catboost", "rf", "lr", "knn")
KS = (2, 4, 8, 16, 32)
HEATMAP_K = 8
NWAY_METRICS = ("macro_f1", "balanced_accuracy")
PERCLASS_METRICS = ("roc_auc", "recall", "fpr")


def load_class(runs: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(runs / "class_*.csv")))
    if not files:
        raise SystemExit(f"no class_*.csv in {runs}")
    return pd.concat([pd.read_csv(f, dtype={"cls": str, "pred": str}, keep_default_na=False) for f in files],
                     ignore_index=True)


def _order(methods) -> list[str]:
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in known)


def nway_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (method, target, k): mean and sd of macro-F1 and balanced accuracy over draws and seeds."""
    nw = frame[(frame["task"] == "nway") & (frame["metric"].isin(NWAY_METRICS))]
    wide = nw.pivot_table(index=["method", "target", "seed", "k", "draw"], columns="metric", values="value").reset_index()
    g = wide.groupby(["method", "target", "k"])
    out = g.agg(macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                balanced_accuracy=("balanced_accuracy", "mean"), n=("macro_f1", "size")).reset_index()
    return out


def perclass_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (method, target, k): per-class metrics averaged over classes (each class over draws and seeds first)."""
    pc = frame[(frame["task"] == "perclass") & (frame["metric"].isin(PERCLASS_METRICS))]
    per_cls = pc.groupby(["method", "target", "k", "cls", "metric"])["value"].mean().reset_index()
    out = per_cls.groupby(["method", "target", "k", "metric"])["value"].mean().unstack("metric").reset_index()
    return out.rename(columns={m: f"pc_{m}" for m in PERCLASS_METRICS})


def average_over_targets(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return table.groupby(["method", "k"]).agg(**{c: (c, "mean") for c in cols},
                                             targets=("target", "nunique")).reset_index()


def _wilcoxon_rows(task: str, metric: str, pairs: pd.DataFrame, methods) -> list[dict]:
    """Paired tests of each method against CatBoost per k. ``pairs`` holds (method, k, unit, value)."""
    rows = []
    base = pairs[pairs["method"] == REFERENCE]
    for method in methods:
        if method == REFERENCE:
            continue
        mine = pairs[pairs["method"] == method]
        if mine.empty:
            continue
        for k in sorted(mine["k"].unique()):
            a = mine[mine["k"] == k].set_index("unit")["value"]
            b = base[base["k"] == k].set_index("unit")["value"]
            common = a.index.intersection(b.index)
            x, y = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            diff = x - y
            stat, p = (np.nan, np.nan)
            if len(diff) >= 2 and np.any(diff != 0):
                res = wilcoxon(x, y, zero_method="wilcox")
                stat, p = float(res.statistic), float(res.pvalue)
            rows.append({"task": task, "metric": metric, "method": method, "k": int(k), "pairs": int(len(common)),
                         "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                         "mean_diff": float(np.mean(diff)) if len(diff) else np.nan,
                         "wilcoxon_stat": stat, "p_value": p})
    return rows


def wilcoxon_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairs: N-way (target, draw) with seeds averaged; per-class (target, class, draw) with seeds averaged."""
    nw = frame[(frame["task"] == "nway") & (frame["metric"] == "macro_f1")]
    nw_pairs = nw.groupby(["method", "k", "target", "draw"])["value"].mean().reset_index()
    nw_pairs["unit"] = nw_pairs["target"] + "|" + nw_pairs["draw"].astype(str)
    pc = frame[(frame["task"] == "perclass") & (frame["metric"] == "roc_auc")]
    pc_pairs = pc.groupby(["method", "k", "target", "cls", "draw"])["value"].mean().reset_index()
    pc_pairs["unit"] = pc_pairs["target"] + "|" + pc_pairs["cls"] + "|" + pc_pairs["draw"].astype(str)
    methods = _order(set(frame["method"]))
    rows = _wilcoxon_rows("nway", "macro_f1", nw_pairs, methods)
    rows += _wilcoxon_rows("perclass", "roc_auc", pc_pairs, methods)
    return pd.DataFrame(rows)


def confusion_k(frame: pd.DataFrame, k: int = HEATMAP_K) -> pd.DataFrame:
    """Mean confusion counts over draws and seeds at k, with a row-normalised share per (method, target)."""
    cm = frame[(frame["task"] == "nway") & (frame["metric"] == "confusion") & (frame["k"] == k)]
    g = cm.groupby(["method", "target", "cls", "pred"])["value"].mean().reset_index()
    g["share"] = g["value"] / g.groupby(["method", "target", "cls"])["value"].transform("sum")
    return g


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=160, facecolor="white", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_curves(table: pd.DataFrame, avg: pd.DataFrame, metric: str, ylabel: str, out: Path, stem: str) -> None:
    targets = sorted(table["target"].unique())
    methods = _order(set(table["method"]))
    colour = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}
    with plt.rc_context(STYLE):
        ncols = 3
        nrows = int(np.ceil((len(targets) + 1) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        panels = [(t, table[table["target"] == t]) for t in targets] + [("average over targets", avg)]
        for ax, (title, data) in zip(axes.flat, panels):
            for m in methods:
                d = data[data["method"] == m].sort_values("k")
                if len(d):
                    ax.plot(d["k"], d[metric], marker="o", ms=3.5, lw=1.4, color=colour[m], label=m)
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(KS))
            ax.set_xticklabels([str(k) for k in KS])
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("labelled target flows per class k")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        _save(fig, out, stem)


def plot_confusion(conf: pd.DataFrame, method: str, out: Path) -> None:
    targets = sorted(conf["target"].unique())
    if not targets:
        return
    ncols = min(3, len(targets))
    nrows = int(np.ceil(len(targets) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False)
        im = None
        for ax, target in zip(axes.flat, targets):
            d = conf[(conf["method"] == method) & (conf["target"] == target)]
            classes = sorted(set(d["cls"]) | set(d["pred"]), key=lambda c: (c != "benign", c))
            pivot = d.pivot(index="cls", columns="pred", values="share").reindex(index=classes, columns=classes)
            mat = pivot.fillna(0.0).to_numpy(dtype=float)
            im = ax.imshow(mat, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=7)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if mat[i, j] > 0.6 else INK)
            ax.set_title(target, fontsize=10)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
        for ax in list(axes.flat)[len(targets):]:
            ax.axis("off")
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="row share")
        fig.suptitle(f"{method}: N-way confusion at k = {HEATMAP_K} (mean over draws, rows = true class)", fontsize=10)
        _save(fig, out, f"confusion_k{HEATMAP_K}_{method}")


def run(runs: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame = load_class(runs)
    nw = nway_curves(frame)
    pc = perclass_curves(frame)
    nw.to_csv(out / "nway_by_target.csv", index=False)
    pc.to_csv(out / "perclass_by_target.csv", index=False)
    nw_avg = average_over_targets(nw, ["macro_f1", "balanced_accuracy"])
    pc_avg = average_over_targets(pc, ["pc_roc_auc", "pc_recall", "pc_fpr"])
    nw_avg.to_csv(out / "nway_average.csv", index=False)
    pc_avg.to_csv(out / "perclass_average.csv", index=False)
    wil = wilcoxon_table(frame)
    wil.to_csv(out / "wilcoxon_vs_catboost.csv", index=False)
    conf = confusion_k(frame)
    conf.to_csv(out / f"confusion_k{HEATMAP_K}.csv", index=False)
    if len(nw):
        plot_curves(nw, nw_avg, "macro_f1", "macro-F1", out, "nway_macro_f1_vs_k")
    if len(pc):
        plot_curves(pc, pc_avg, "pc_roc_auc", "per-class ROC-AUC", out, "perclass_roc_auc_vs_k")
    for method in _order(set(conf["method"])):
        plot_confusion(conf, method, out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m crossids.flare.report_class", description="Summarise the attack-class evaluation")
    p.add_argument("--runs", default="runs/flare", help="folder with class_<target>.csv files")
    p.add_argument("--out", default="results/flare/class", help="output folder")
    a = p.parse_args(argv)
    run(Path(a.runs), Path(a.out))


if __name__ == "__main__":
    main()
