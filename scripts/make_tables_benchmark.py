"""Build the LaTeX tables and copy the figures of the benchmark paper (paper) from results/.

Every number in paper/tables comes from results/runs.csv, results/analysis and results/controls, so the
tables cannot drift from the data. Usage: python scripts/make_tables_benchmark.py
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

OUT = "paper"
TAB = f"{OUT}/tables"
FIG = f"{OUT}/figures"
RUNS = "results/runs.csv"
BS = chr(92)


def write(name: str, lines: list[str]) -> None:
    with open(f"{TAB}/{name}.tex", "w", encoding="utf8") as fh:
        fh.write(chr(10).join(lines) + chr(10))


def tex(s: str) -> str:
    return s.replace("_", BS + "_").replace("&", BS + "&").replace("%", BS + "%")


def scenario_means(r: pd.DataFrame, experiment: str, mode: str, value: str = "roc_auc") -> pd.DataFrame:
    """Seed-averaged value per (method, scenario) for one experiment and mode."""
    g = r[(r.experiment == experiment) & (r["mode"] == mode)]
    return g.groupby(["method", "scenario", "transfer_type"])[value].mean().reset_index()


def mean_sd(x: pd.Series) -> str:
    return f"{x.mean():.3f}"


LABELS = {"lr": "LR", "rf": "RF", "hgb": "HGB", "catboost": "CatBoost", "xgboost": "XGBoost", "mlp": "MLP",
          "coral": "CORAL", "dann": "DANN", "rift": "RIFT", "rift-st": "RIFT + self-training",
          "rift-noDN": "RIFT $-$ domain norm.", "rift-noAnchor": "RIFT $-$ benign anchor",
          "rift-noIFS": "RIFT $-$ invariant selection", "rift-noLSC": "RIFT $-$ prior correction",
          "rift-poolInit": "RIFT, pooled init.", "rift-robust": "RIFT, robust scaling",
          "rift-standard": "RIFT, z-score scaling", "rift-oraclePrior": "RIFT, true prior (diagnostic)",
          "rift-oracleAnchor": "RIFT, true prior and benign anchor (diagnostic)"}


def main_table(r: pd.DataFrame) -> None:
    methods = ["lr", "rf", "hgb", "catboost", "xgboost", "mlp", "coral", "dann", "rift", "rift-st"]
    ind = scenario_means(r, "indomain", "transductive")
    pw = scenario_means(r, "pairwise", "transductive")
    lo = scenario_means(r, "lodo", "transductive")
    lines = [BS + r"begin{tabular}{lrrrrr}", BS + "toprule",
             r"Method & In-domain & Same family & Same extractor & Cross-extractor & LODO " + BS + BS,
             r" & (7) & (4 pairs) & (12 pairs) & (26 pairs) & (7) " + BS + BS, BS + "midrule"]
    for m in methods:
        cells = [mean_sd(ind[ind.method == m].roc_auc)]
        for tt in ["same-family", "same-extractor", "cross-extractor"]:
            cells.append(mean_sd(pw[(pw.method == m) & (pw.transfer_type == tt)].roc_auc))
        cells.append(mean_sd(lo[lo.method == m].roc_auc))
        lines.append(f"{LABELS[m]} & " + " & ".join(cells) + " " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    write("main", lines)


def rift_table(r: pd.DataFrame) -> None:
    methods = ["hgb", "rift", "rift-noDN", "rift-noAnchor", "rift-noIFS", "rift-noLSC", "rift-poolInit",
               "rift-robust", "rift-standard", "rift-st", "rift-oraclePrior", "rift-oracleAnchor"]
    pw = scenario_means(r, "pairwise", "transductive")
    lo = scenario_means(r, "lodo", "transductive")
    pwb = scenario_means(r, "pairwise", "transductive", "balanced_accuracy")
    err = r[(r.experiment == "pairwise") & (r["mode"] == "transductive")].groupby("method").prior_abs_error.mean()
    lines = [BS + r"begin{tabular}{lrrrr}", BS + "toprule",
             r"Method & Pairwise AUC & Pairwise bal. acc. & LODO AUC & Prior error " + BS + BS, BS + "midrule"]
    for m in methods:
        e = err.get(m, np.nan)
        e_s = "--" if not np.isfinite(e) else f"{e:.3f}"
        lines.append(f"{LABELS[m]} & {mean_sd(pw[pw.method == m].roc_auc)} & {mean_sd(pwb[pwb.method == m].balanced_accuracy)}"
                     f" & {mean_sd(lo[lo.method == m].roc_auc)} & {e_s} " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    write("rift", lines)


def target_table(r: pd.DataFrame) -> None:
    """In-domain versus leave-one-dataset-out ROC-AUC of CatBoost per target."""
    ind = scenario_means(r, "indomain", "inductive")
    lo = scenario_means(r, "lodo", "inductive")
    names = {"cicids2017": "CIC-IDS2017", "cicids2018": "CIC-IDS2018", "insdn": "InSDN", "iot23": "IoT-23",
             "toniot": "ToN-IoT", "unswnb15": "UNSW-NB15", "usbids1": "USB-IDS-1"}
    lines = [BS + r"begin{tabular}{lrrr}", BS + "toprule", r"Target & In-domain & LODO CatBoost & LODO HGB " + BS + BS,
             BS + "midrule"]
    for k, n in names.items():
        a = ind[(ind.method == "catboost") & (ind.scenario == k)].roc_auc.mean()
        b = lo[(lo.method == "catboost") & (lo.scenario == f"leave-{k}-out")].roc_auc.mean()
        c = lo[(lo.method == "hgb") & (lo.scenario == f"leave-{k}-out")].roc_auc.mean()
        lines.append(f"{n} & {a:.3f} & {b:.3f} & {c:.3f} " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    write("targets", lines)


def fewshot_table() -> None:
    f = pd.read_csv("results/controls/fewshot_fair.csv")
    lines = [BS + r"begin{tabular}{lrrrr}", BS + "toprule",
             r"Setting & Extra trees & 0.1\% labels & 1\% labels & 5\% labels " + BS + BS, BS + "midrule"]
    for variant, label in [("continued", "Source-pretrained, continued"), ("targetonly", "Target labels only")]:
        for it in [100, 300, 500]:
            g = f[(f.variant == variant) & (f.iterations == it)].set_index("k")
            lines.append(f"{label} & {it} & {g.loc[0.1, 'roc_auc_mean']:.3f} & {g.loc[1.0, 'roc_auc_mean']:.3f}"
                         f" & {g.loc[5.0, 'roc_auc_mean']:.3f} " + BS + BS)
            label = ""
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    write("fewshot", lines)


def datasets_table() -> None:
    import json

    rows = [("cicids2017", "CIC-IDS2017", "UNB CIC", "CICFlowMeter", "core"),
            ("cicids2018", "CIC-IDS2018", "UNB CIC", "CICFlowMeter", "core"),
            ("usbids1", "USB-IDS-1", "Univ. of Sannio", "CICFlowMeter", "core"),
            ("insdn", "InSDN", "UCD", "CICFlowMeter", "core"),
            ("unswnb15", "UNSW-NB15", "UNSW Canberra", "Argus, Bro", "core"),
            ("toniot", "ToN-IoT", "UNSW Canberra", "Zeek", "core"),
            ("iot23", "IoT-23", "Stratosphere, CTU", "Zeek", "core"),
            ("ctu13", "CTU-13", "Stratosphere, CTU", "Argus", "minimal")]
    lines = [BS + r"begin{tabular}{lllcc}", BS + "toprule",
             r"Dataset & Laboratory & Extractor & Attack prior & Feature tier " + BS + BS, BS + "midrule"]
    for key, name, lab, ext, tier in rows:
        path = f"data/processed/{key}.json"
        prior = json.load(open(path))["prior_distinct"] if os.path.exists(path) else float("nan")
        lines.append(f"{name} & {lab} & {ext} & {prior:.3f} & {tier} " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    write("datasets", lines)


def family_table() -> None:
    """Same-attack-family transfer between datasets (scripts/family_transfer.py)."""
    path = "results/controls/family_transfer.csv"
    if not os.path.exists(path):
        return
    f = pd.read_csv(path)
    names = {"dos": "DoS", "ddos": "DDoS", "scan": "Scan", "bruteforce": "Brute force",
             "exploit_malware": "Exploit / malware", "botnet": "Botnet"}
    lines = [BS + r"begin{tabular}{lrrrrr}", BS + "toprule",
             r"Family & Cells & Median cross-dataset AUC & Min & Max & Median in-domain AUC " + BS + BS, BS + "midrule"]
    for fam in ["dos", "ddos", "scan", "bruteforce", "exploit_malware", "botnet"]:
        g = f[f.family == fam]
        if len(g):
            lines.append(f"{names[fam]} & {len(g)} & {g.auc_cross.median():.3f} & {g.auc_cross.min():.3f} & "
                         f"{g.auc_cross.max():.3f} & {g.auc_in_source.median():.3f} " + BS + BS)
    lines.append(BS + "midrule")
    lines.append(f"All & {len(f)} & {f.auc_cross.median():.3f} & {f.auc_cross.min():.3f} & {f.auc_cross.max():.3f} & "
                 f"{f.auc_in_source.median():.3f} " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    write("family", lines)


def copy_figures() -> None:
    pairs = {"results/analysis/fig02_feature_sign_heatmap.pdf": "feature_sign.pdf",
             "results/analysis/fig03_pairwise_matrix_transductive.pdf": "pairwise_matrix.pdf",
             "results/analysis/fig07_fewshot_curve.pdf": "fewshot_curve.pdf",
             "results/analysis/fig08_prior_estimation.pdf": "prior_estimation.pdf",
             "results/controls/concept_kappa.pdf": "concept_kappa.pdf",
             "results/controls/separability_confusion.pdf": "separability.pdf"}
    for src, dst in pairs.items():
        if os.path.exists(src):
            shutil.copyfile(src, f"{FIG}/{dst}")


def main() -> None:
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    r = pd.read_csv(RUNS, low_memory=False)
    main_table(r)
    rift_table(r)
    target_table(r)
    fewshot_table()
    datasets_table()
    family_table()
    copy_figures()


if __name__ == "__main__":
    main()
