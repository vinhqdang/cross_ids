"""Build the LaTeX tables and figures of the FLARE-2 paper from results/flare.

Every number in paper/flare/tables comes from this script, so the manuscript
cannot drift from the CSVs. Usage: python scripts/make_paper_tables.py
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import stats_flare as sf  # noqa: E402

OUT = "paper/flare"
TAB = f"{OUT}/tables"
FIG = f"{OUT}/figures"
NAMES = {"cicids2017": "CIC-IDS2017", "cicids2018": "CIC-IDS2018", "ctu13": "CTU-13", "insdn": "InSDN",
         "iot23": "IoT-23", "toniot": "ToN-IoT", "unswnb15": "UNSW-NB15", "usbids1": "USB-IDS-1"}
ORDER = ["cicids2017", "cicids2018", "ctu13", "insdn", "iot23", "toniot", "unswnb15", "usbids1"]
METHODS = [("flare2", "FLARE-2"), ("rf", "RF"), ("catboost-target", "CatBoost (target only)"),
           ("catboost-ft", "CatBoost (source + fine-tune)"), ("lr", "LR"), ("knn", "kNN")]
TKS = [4, 16, 64]


def write(name: str, text: str) -> None:
    with open(f"{TAB}/{name}.tex", "w") as fh:
        fh.write(text)


def fmt(x: float, bold: bool = False) -> str:
    s = "--" if not np.isfinite(x) else f"{x:.3f}"
    return f"\\textbf{{{s}}}" if bold and np.isfinite(x) else s


def pfmt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def binary_table(d: pd.DataFrame) -> None:
    per = d.groupby(["method", "target", "k"]).roc_auc.mean()
    rows = []
    for t in ORDER:
        line = [NAMES[t]]
        for k in TKS:
            vals = {m: per.get((m, t, k), np.nan) for m, _ in METHODS}
            best = max(vals.values())
            line += [fmt(vals[m], vals[m] >= best - 1e-9) for m, _ in METHODS[:4]]
        rows.append(line)
    head = " & ".join(["Target"] + [f"{s}" for _ in TKS for s in ["F2", "RF", "CB", "CB-ft"]])
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    ks = " & ".join(f"\\multicolumn{{4}}{{c}}{{$k={k}$}}" for k in TKS)
    write("binary", "\\begin{tabular}{l" + "cccc" * len(TKS) + "}\n\\toprule\n& " + ks + " \\\\\n"
          + "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n" + head + " \\\\\n\\midrule\n"
          + body + " \\\\\n\\bottomrule\n\\end{tabular}\n")
    # zero-shot references
    zs = d[d.k == 0].groupby(["method", "target"]).roc_auc.mean().unstack(0)
    cols = [("catboost-lodo", "CatBoost LODO"), ("catboost-pairwise-mean", "pairwise mean"),
            ("catboost-pairwise-best", "pairwise best"), ("catboost-fulllabel", "full label")]
    lines = []
    for t in ORDER:
        v = [zs.loc[t, c] if (t in zs.index and c in zs.columns) else np.nan for c, _ in cols]
        lines.append(" & ".join([NAMES[t]] + [fmt(x) for x in v]))
    write("zeroshot", "\\begin{tabular}{lcccc}\n\\toprule\nTarget & " + " & ".join(n for _, n in cols)
          + " \\\\\n\\midrule\n" + " \\\\\n".join(lines) + " \\\\\n\\bottomrule\n\\end{tabular}\n")


def stats_table() -> None:
    s = pd.read_csv("results/flare/stats/binary_summary.csv")
    names = {"catboost-target": "CatBoost (target)", "catboost-ft": "CatBoost (fine-tune)", "rf": "RF", "lr": "LR",
             "knn": "kNN"}
    lines = []
    for scope in ["mean over k", "k<=16", "k>=32"]:
        for b in ["rf", "catboost-target", "catboost-ft", "lr", "knn"]:
            r = s[(s.scope == scope) & (s.baseline == b)].iloc[0]
            lines.append(" & ".join([scope.replace("<=", "$\\leq$").replace(">=", "$\\geq$"), names[b],
                                     f"{r.mean_diff:+.3f}", f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
                                     f"{int(r.wins)}/{int(r.losses)}", pfmt(r.p_holm)]))
    write("stats", "\\begin{tabular}{llcccc}\n\\toprule\nScope & Baseline & $\\Delta$ROC-AUC & 95\\% CI & W/L & $p_{\\text{Holm}}$"
          " \\\\\n\\midrule\n" + " \\\\\n".join(lines) + " \\\\\n\\bottomrule\n\\end{tabular}\n")


def class_table() -> None:
    s = pd.read_csv("results/flare/stats/class_summary.csv")
    names = {"rf": "RF", "catboost-target": "CatBoost", "lr": "LR", "knn": "kNN"}
    tasks = [("perclass:roc_auc", "Per-class ROC-AUC"), ("perclass:recall", "Per-class recall@0.5"),
             ("perclass:fpr", "Per-class FPR@0.5"), ("nway:macro_f1", "N-way macro-F1")]
    lines = []
    for key, label in tasks:
        for b in ["rf", "catboost-target", "lr", "knn"]:
            r = s[(s.task == key) & (s.baseline == b)].iloc[0]
            lines.append(" & ".join([label, names[b], f"{r.target_mean_diff:+.3f}", f"{int(r.target_wins)}/{int(r.n_targets)}",
                                     pfmt(r.target_p), f"{r.mean_diff:+.3f}", f"{int(r.n_units)}"]))
    write("class", "\\begin{tabular}{llccccc}\n\\toprule\nMetric & Baseline & $\\Delta$ (targets) & FLARE-2 higher & $p$ & "
          "$\\Delta$ (units) & units \\\\\n\\midrule\n" + " \\\\\n".join(lines) + " \\\\\n\\bottomrule\n\\end{tabular}\n")


def class_by_target() -> None:
    c = sf.load_class()
    s = c[(c.task == "nway") & (c.metric == "macro_f1") & (c.k == 16) & c.method.isin(["flare2", "rf", "catboost", "lr", "knn"])]
    p = s.groupby(["target", "method"]).value.mean().unstack()
    r = c[(c.task == "perclass") & (c.metric == "roc_auc") & (c.k == 4) & c.method.isin(["flare2", "rf", "catboost"])]
    q = r.groupby(["target", "method"]).value.mean().unstack()
    lines = []
    for t in ORDER:
        if t not in p.index:
            continue
        lines.append(" & ".join([NAMES[t], fmt(q.loc[t, "flare2"], True), fmt(q.loc[t, "rf"]), fmt(q.loc[t, "catboost"]),
                                 fmt(p.loc[t, "flare2"], True), fmt(p.loc[t, "rf"]), fmt(p.loc[t, "catboost"])]))
    write("classtarget", "\\begin{tabular}{lcccccc}\n\\toprule\n& \\multicolumn{3}{c}{Per-class ROC-AUC, $k{=}4$} & "
          "\\multicolumn{3}{c}{N-way macro-F1, $k{=}16$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
          "Target & F2 & RF & CB & F2 & RF & CB \\\\\n\\midrule\n" + " \\\\\n".join(lines) + " \\\\\n\\bottomrule\n\\end{tabular}\n")


def curve_figure(d: pd.DataFrame) -> None:
    per = d.groupby(["method", "target", "k"]).roc_auc.mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    style = {"flare2": ("C3", "-", "o"), "rf": ("C0", "--", "s"), "catboost-target": ("C2", "--", "^"),
             "catboost-ft": ("C1", ":", "v"), "lr": ("C4", ":", "x"), "knn": ("C5", ":", "+")}
    for m, name in METHODS:
        s = per[(per.method == m) & per.k.isin(sf.KS)]
        g = s.groupby("k").roc_auc.mean()
        col, ls, mk = style[m]
        ax[0].plot(g.index, g.values, color=col, ls=ls, marker=mk, ms=4, label=name)
    zs = d[d.k == 0].groupby("method").roc_auc.mean()
    if "catboost-lodo" in zs.index:
        ax[0].axhline(zs["catboost-lodo"], color="gray", lw=0.8, ls="-.", label="CatBoost zero-shot (LODO)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("labelled target flows $k$")
    ax[0].set_ylabel("ROC-AUC (mean over 8 targets)")
    ax[0].legend(fontsize=6, frameon=False, loc="lower right")
    f = per[per.method == "flare2"].set_index(["target", "k"]).roc_auc
    for b, col in [("rf", "C0"), ("catboost-target", "C2")]:
        pb = per[per.method == b].set_index(["target", "k"]).roc_auc
        diff = (f - pb).dropna().reset_index()
        g = diff.groupby("k").roc_auc.mean()
        ax[1].plot(g.index, g.values, color=col, marker="o", ms=4, label=f"vs {dict(METHODS)[b]}")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("labelled target flows $k$")
    ax[1].set_ylabel("FLARE-2 $-$ baseline")
    ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/curves.pdf")
    plt.close(fig)


def pertarget_figure(d: pd.DataFrame) -> None:
    per = d.groupby(["method", "target", "k"]).roc_auc.mean().reset_index()
    f = per[per.method == "flare2"].set_index(["target", "k"]).roc_auc
    r = per[per.method == "rf"].set_index(["target", "k"]).roc_auc
    diff = (f - r).dropna().reset_index()
    pv = diff.pivot(index="target", columns="k", values="roc_auc").reindex(ORDER)
    pv = pv[[k for k in sf.KS if k in pv.columns]]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    v = np.nanmax(np.abs(pv.values))
    im = ax.imshow(pv.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), [NAMES[t] for t in pv.index], fontsize=7)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xlabel("labelled target flows $k$")
    fig.colorbar(im, ax=ax, label="FLARE-2 $-$ RF (ROC-AUC)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/heatmap.pdf")
    plt.close(fig)


ABL_ROWS = [("flare2", "FLARE-2 (reference)"), ("flare2-noperm", "no slot permutation"),
            ("flare2-insrc", "in-source validation"), ("flare2-m", "loss mask (reference)"),
            ("flare2-m-binary", "loss mask, binary tasks only")]
ABL_TARGETS = ["ctu13", "cicids2017"]


def _abl_binary(method: str, target: str, k: int) -> float:
    fs = glob.glob(f"{sf.RAW}/eval_{target}_*.csv")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[(d.method == method) & (d.k == k) & (d.seed == 0)]
    return d.roc_auc.mean() if len(d) else np.nan


def _abl_class(method: str, target: str, task: str, metric: str, k: int) -> float:
    fs = glob.glob(f"{sf.RAW}/class_{target}_*.csv")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[(d.method == method) & (d.task == task) & (d.metric == metric) & (d.k == k) & (d.seed == 0)]
    return d.value.mean() if len(d) else np.nan


def ablation_table() -> None:
    """Seed 0, two targets: binary ROC-AUC at k=4/16/64, N-way macro-F1 (k=16), per-class ROC-AUC (k=4)."""
    lines = [r"\begin{tabular}{l" + "rrrrr" * len(ABL_TARGETS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{NAMES[t]}}}" for t in ABL_TARGETS) + " \\\\",
             "Variant & " + " & ".join(["$k{=}4$ & $k{=}16$ & $k{=}64$ & F1@16 & AUC@4"] * len(ABL_TARGETS)) + " \\\\",
             r"\midrule"]
    for m, label in ABL_ROWS:
        cells = []
        for t in ABL_TARGETS:
            vals = [_abl_binary(m, t, k) for k in (4, 16, 64)]
            vals += [_abl_class(m, t, "nway", "macro_f1", 16), _abl_class(m, t, "perclass", "roc_auc", 4)]
            cells += [fmt(v) for v in vals]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("ablation", chr(10).join(lines))


def main() -> None:
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{sf.RAW}/eval_*.csv"))], ignore_index=True)
    d = d.drop_duplicates(["method", "target", "k", "draw", "seed"])
    binary_table(d)
    stats_table()
    class_table()
    ablation_table()
    class_by_target()
    curve_figure(d)
    pertarget_figure(d)


if __name__ == "__main__":
    main()
