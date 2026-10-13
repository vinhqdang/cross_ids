"""Paper analysis: attack-family taxonomy, feature sign consistency, transfer matrices, rankings and figures.

Everything here is read-only with respect to ``results/runs.csv`` and ``results/per_attack.csv``; outputs go to
a separate directory (``results/analysis`` by default). Missing experiments or modes are reported and skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .schema import CORE_FEATURES, MINIMAL_FEATURES, SOURCES, features_for, supports  # noqa: E402

log = logging.getLogger("crossids.analysis")

# ---------------------------------------------------------------------------------------------------------------------
# Attack-family taxonomy
# ---------------------------------------------------------------------------------------------------------------------

FAMILY_ORDER = ("benign", "dos", "ddos", "scan", "bruteforce", "web", "botnet", "exploit_malware", "other")
ATTACK_FAMILY_ORDER = tuple(f for f in FAMILY_ORDER if f != "benign")

# Each rule: exact names, lower-case prefixes and lower-case keywords (substring). Matching is case-insensitive.
# Evaluation order matters: web is checked before bruteforce, so "Brute Force -Web" lands in web.
ATTACK_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "benign": {"exact": ("benign", "normal")},
    "web": {
        "prefix": ("web attack", "web-attack", "brute force -web", "brute force -xss"),
        "keyword": ("xss", "injection"),
    },
    "ddos": {"prefix": ("ddos",)},
    "dos": {"prefix": ("dos", "hulk", "slowloris", "slowhttptest", "tcpflood")},
    "scan": {"exact": ("scan", "scanning", "probe", "reconnaissance"), "keyword": ("portscan",)},
    "botnet": {
        "exact": ("bot", "botnet", "filedownload", "okiru", "mirai", "torii"),
        "prefix": ("botnet", "c&c", "okiru"),
    },
    "bruteforce": {
        "exact": ("bfa",),
        "keyword": ("patator", "bruteforce", "brute force", "password"),
    },
    "exploit_malware": {
        "exact": ("exploits", "shellcode", "worms", "backdoor", "u2r", "heartbleed", "ransomware"),
        "prefix": ("backdoor", "exploit"),
        "keyword": ("infilt",),
    },
}
# Names whose family is genuinely unclear; they map to 'other' on purpose and are listed explicitly.
EXPLICIT_OTHER = frozenset({"attack", "analysis", "generic", "fuzzers", "mitm", "unknown"})
DEFENCE_VARIANTS = ("NoDefense", "Reqtimeout", "Evasive", "Security2")


def attack_family(name: object) -> str:
    """Map an attack_type string onto one of ``FAMILY_ORDER``. Unknown names fall back to ``other``."""
    text = str(name).strip().lower()
    if text in EXPLICIT_OTHER:
        return "other"
    for family, rules in ATTACK_FAMILIES.items():
        if text in rules.get("exact", ()):
            return family
        if any(text.startswith(p) for p in rules.get("prefix", ())):
            return family
        if any(k in text for k in rules.get("keyword", ())):
            return family
    return "other"


def defence_variant(name: object) -> str | None:
    """USB-IDS-1 defence suffix (NoDefense, Reqtimeout, Evasive, Security2), or None for other datasets."""
    text = str(name).strip()
    for variant in DEFENCE_VARIANTS:
        if text.lower().endswith("-" + variant.lower()):
            return variant
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Constants: datasets, methods, metrics, plotting style
# ---------------------------------------------------------------------------------------------------------------------

DATASET_ORDER = ("cicids2017", "cicids2018", "usbids1", "insdn", "unswnb15", "toniot", "iot23", "ctu13")
DEFAULT_METHODS = ("hgb", "catboost", "dann", "rift", "rift-noDN")
METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc", "fpr")
RUN_KEYS = ["method", "mode", "experiment", "scenario", "seed"]
SIGN_FEATURES = tuple(CORE_FEATURES) + tuple(f for f in MINIMAL_FEATURES if f not in CORE_FEATURES)

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e4e1"
DIVERGING = ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f9d0cf", "#e34948", "#9e1f1f"]
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING_CMAP = LinearSegmentedColormap.from_list("crossids_diverging", DIVERGING)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("crossids_sequential", SEQUENTIAL)
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#9a9893",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "legend.fontsize": 8,
}


def _colours(names: list[str]) -> dict[str, str]:
    """Assign fixed-order palette colours. Never cycles: more than eight series is a programming error."""
    if len(names) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} series per figure, got {len(names)}")
    return {name: PALETTE[i] for i, name in enumerate(names)}


def _target_of(scenario: object) -> str:
    text = str(scenario)
    if "->" in text:
        return text.split("->")[-1]
    if text.startswith("leave-") and text.endswith("-out"):
        return text[len("leave-") : -len("-out")]
    return text


def _source_of(scenario: object) -> str | None:
    text = str(scenario)
    return text.split("->")[0] if "->" in text else None


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _pm(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "–"
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows: the required experiment or mode is not in the results yet._\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = []
    for row in df.itertuples(index=False):
        cells = [_fmt(v, digits).replace("|", "\\|") for v in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body]) + "\n"


class _Report:
    """Collects written files and skipped sections for the README."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.files: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def table(self, stem: str, df: pd.DataFrame, description: str, title: str, note: str = "") -> None:
        df.to_csv(self.out / f"{stem}.csv", index=False)
        md = [f"# {title}\n"]
        if note:
            md.append(note.strip() + "\n")
        md.append(_md(df))
        (self.out / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")
        self.files.append((f"{stem}.csv / {stem}.md", description))

    def figure(self, stem: str, fig: plt.Figure, description: str) -> None:
        fig.savefig(self.out / f"{stem}.png", dpi=200, facecolor="white", bbox_inches="tight")
        fig.savefig(self.out / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
        plt.close(fig)
        self.files.append((f"{stem}.png / {stem}.pdf", description))

    def skip(self, section: str, reason: str) -> None:
        log.warning("skipping %s: %s", section, reason)
        self.skipped.append((section, reason))


# ---------------------------------------------------------------------------------------------------------------------
# Loading (read-only)
# ---------------------------------------------------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_runs(results_dir: Path) -> pd.DataFrame:
    runs = _read_csv(results_dir / "runs.csv")
    if runs.empty:
        return runs
    runs = runs.drop_duplicates(RUN_KEYS, keep="last").reset_index(drop=True)
    if "prior_abs_error" not in runs and {"est_prior", "true_prior"} <= set(runs.columns):
        runs["prior_abs_error"] = (runs["est_prior"] - runs["true_prior"]).abs()
    elif "prior_abs_error" not in runs:
        runs["prior_abs_error"] = np.nan
    return runs


def load_per_attack(results_dir: Path) -> pd.DataFrame:
    pa = _read_csv(results_dir / "per_attack.csv")
    if pa.empty:
        return pa
    pa = pa.drop_duplicates(["method", "mode", "experiment", "scenario", "seed", "attack_type"], keep="last")
    pa = pa.reset_index(drop=True)
    pa["target"] = pa["scenario"].map(_target_of)
    pa["source"] = pa["scenario"].map(_source_of)
    pa["family"] = pa["attack_type"].map(attack_family)
    pa["defence"] = pa["attack_type"].map(defence_variant)
    return pa


def load_processed(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Full harmonised frames keyed by dataset name (file stem). Empty dict when none are cached."""
    folder = data_dir / "processed"
    if not folder.exists():
        return {}
    return {p.stem: pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))}


def families_by_dataset(processed: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    """Attack families present in each dataset (label == 1 rows only)."""
    out: dict[str, set[str]] = {}
    for name, df in processed.items():
        attacks = df.loc[df["label"] == 1, "attack_type"].astype(str).unique()
        out[name] = {attack_family(a) for a in attacks} - {"benign"}
    return out


def _ordered(names: list[str]) -> list[str]:
    known = [d for d in DATASET_ORDER if d in names]
    return known + sorted(n for n in names if n not in DATASET_ORDER)


# ---------------------------------------------------------------------------------------------------------------------
# 1. Attack-family taxonomy
# ---------------------------------------------------------------------------------------------------------------------


def taxonomy_table(processed: dict[str, pd.DataFrame], per_attack: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in processed.items():
        grouped = df.assign(attack_type=df["attack_type"].astype(str)).groupby("attack_type").size()
        for attack, count in grouped.items():
            rows.append({"attack_type": attack, "dataset": name, "rows": int(count)})
    seen = pd.DataFrame(rows)
    names = set(seen["attack_type"]) if not seen.empty else set()
    if not per_attack.empty:
        names |= set(per_attack["attack_type"].astype(str))
    table = []
    for attack in sorted(names, key=lambda a: (attack_family(a), a.lower())):
        ds = sorted(seen.loc[seen["attack_type"] == attack, "dataset"]) if not seen.empty else []
        table.append(
            {
                "attack_type": attack,
                "family": attack_family(attack),
                "defence": defence_variant(attack) or "",
                "datasets": ", ".join(ds),
                "rows": int(seen.loc[seen["attack_type"] == attack, "rows"].sum()) if not seen.empty else 0,
                "in_per_attack": bool(not per_attack.empty and attack in set(per_attack["attack_type"].astype(str))),
            }
        )
    return pd.DataFrame(table)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Feature sign consistency
# ---------------------------------------------------------------------------------------------------------------------








def _annotated_heatmap(
    ax: plt.Axes, mat: pd.DataFrame, cmap: LinearSegmentedColormap, norm: Normalize, cell_fmt: str = "{:.2f}"
) -> None:
    data = mat.to_numpy(dtype=float)
    cm = cmap.with_extremes(bad="#ffffff")
    ax.imshow(np.ma.masked_invalid(data), cmap=cm, norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns), rotation=40, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(list(mat.index))
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not math.isfinite(value):
                continue
            r, g, b, _ = cm(norm(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(
                j,
                i,
                cell_fmt.format(value),
                ha="center",
                va="center",
                fontsize=7,
                color=INK if luminance > 0.42 else "#ffffff",
            )


# ---------------------------------------------------------------------------------------------------------------------
# 3. Pairwise transfer matrices
# ---------------------------------------------------------------------------------------------------------------------


def pairwise_matrix_long(runs: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for mode in _ordered_modes(runs):
        for method in methods:
            pw = runs[(runs["experiment"] == "pairwise") & (runs["method"] == method) & (runs["mode"] == mode)]
            ind = runs[(runs["experiment"] == "indomain") & (runs["method"] == method) & (runs["mode"] == mode)]
            if pw.empty and ind.empty:
                continue
            pairs = pw.assign(source=pw["scenario"].map(_source_of), target=pw["scenario"].map(_target_of))
            cells = pairs.groupby(["source", "target"])["roc_auc"].mean()
            diag = (
                ind.assign(target=ind["scenario"].map(_target_of)).groupby("target")["roc_auc"].mean()
                if not ind.empty
                else pd.Series(dtype=float)
            )
            for (src, tgt), value in cells.items():
                rows.append(_matrix_row(method, mode, src, tgt, value, False))
            for tgt, value in diag.items():
                rows.append(_matrix_row(method, mode, tgt, tgt, value, True))
    return pd.DataFrame(rows)


def _matrix_row(method: str, mode: str, src: str, tgt: str, value: float, diagonal: bool) -> dict:
    return {
        "method": method,
        "mode": mode,
        "source": src,
        "target": tgt,
        "roc_auc": float(value),
        "in_domain_diagonal": diagonal,
    }


def _ordered_modes(runs: pd.DataFrame) -> list[str]:
    present = set(runs["mode"].unique()) if not runs.empty else set()
    return [m for m in ("transductive", "inductive") if m in present]


def _matrix_for(long: pd.DataFrame, method: str, mode: str) -> pd.DataFrame:
    sub = long[(long["method"] == method) & (long["mode"] == mode)]
    datasets = _ordered(list(set(sub["source"]) | set(sub["target"])))
    wide = pd.DataFrame(np.nan, index=datasets, columns=datasets)
    for row in sub.itertuples(index=False):
        wide.loc[row.source, row.target] = row.roc_auc
    return wide


def _plot_pairwise_grid(long: pd.DataFrame, mode: str, methods: list[str]) -> plt.Figure | None:
    methods = [m for m in methods if not long[(long["method"] == m) & (long["mode"] == mode)].empty]
    if not methods:
        return None
    ncols = min(3, len(methods))
    nrows = math.ceil(len(methods) / ncols)
    size = _matrix_for(long, methods[0], mode).shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1.2, (0.52 * size + 1.3) * nrows), squeeze=False)
    off_means = []
    norm = Normalize(vmin=0.0, vmax=1.0)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, method in zip(axes.ravel(), methods):
        ax.set_visible(True)
        wide = _matrix_for(long, method, mode)
        _annotated_heatmap(ax, wide, SEQUENTIAL_CMAP, norm, cell_fmt="{:.2f}")
        ax.set_xlabel("target", color=MUTED)
        ax.set_ylabel("source", color=MUTED)
        for i, dataset in enumerate(wide.index):
            if dataset in wide.columns:
                j = list(wide.columns).index(dataset)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
        off = wide.to_numpy(dtype=float).copy()
        for i, dataset in enumerate(wide.index):
            if dataset in wide.columns:
                off[i, list(wide.columns).index(dataset)] = np.nan
        vals = off[np.isfinite(off)]
        if vals.size:
            off_means.append(f"{method} {vals.mean():.2f}")
        ax.set_title(method, loc="left")
    takeaway = ", ".join(off_means)
    fig.suptitle(
        f"Off-diagonal transfer ROC-AUC (mean): {takeaway}  [{mode}]; boxed cells are in-domain",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# 4. Summary and ranking
# ---------------------------------------------------------------------------------------------------------------------

GROUP_KEYS = ["experiment", "mode", "transfer_type", "method"]


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [*METRICS, "prior_abs_error"]
    per_scen = runs.groupby([*GROUP_KEYS, "scenario"], as_index=False)[cols].mean()
    rows = []
    for keys, grp in per_scen.groupby(GROUP_KEYS, sort=True):
        rec = dict(zip(GROUP_KEYS, keys))
        rec["n_scenarios"] = int(grp["scenario"].nunique())
        for col in cols:
            vals = grp[col].dropna().to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(vals.mean()) if vals.size else np.nan
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summary_markdown(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[GROUP_KEYS + ["n_scenarios"]].copy()
    for col in [*METRICS, "prior_abs_error"]:
        out[col] = [_pm(m, s) for m, s in zip(summary[f"{col}_mean"], summary[f"{col}_std"])]
    return out


def _pivot_scenarios(runs: pd.DataFrame, experiment: str, mode: str) -> pd.DataFrame:
    sub = runs[(runs["experiment"] == experiment) & (runs["mode"] == mode)]
    if sub.empty:
        return pd.DataFrame()
    return sub.groupby(["scenario", "method"])["roc_auc"].mean().unstack("method")


def ranking_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment in ("indomain", "pairwise", "lodo"):
        for mode in _ordered_modes(runs):
            pivot = _pivot_scenarios(runs, experiment, mode)
            if pivot.empty:
                continue
            common = pivot.dropna()
            basis = "common scenarios" if len(common) else "all scenarios"
            table = common if len(common) else pivot
            means = table.mean()
            ranks = means.rank(ascending=False, method="min").astype(int)
            for method in means.index:
                rows.append(
                    {
                        "experiment": experiment,
                        "mode": mode,
                        "method": method,
                        "rank": int(ranks[method]),
                        "mean_roc_auc": float(means[method]),
                        "n_scenarios": int(table[method].notna().sum()),
                        "rank_basis": basis,
                        "n_scenarios_common": int(len(common)),
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(["experiment", "mode", "rank", "method"]).reset_index(drop=True) if not out.empty else out


def friedman_test(runs: pd.DataFrame, experiment: str, mode: str) -> dict | None:
    """Friedman test over blocks (targets for lodo, pairs for pairwise) with the largest method set that is complete."""
    pivot = _pivot_scenarios(runs, experiment, mode)
    if pivot.empty:
        return None
    coverage = pivot.notna().sum().sort_values(ascending=False)
    coverage = coverage[coverage > 0]
    for k in range(len(coverage), 2, -1):
        methods = list(coverage.index[:k])
        complete = pivot[methods].dropna()
        if len(complete) >= 3:
            res = stats.friedmanchisquare(*[complete[m].to_numpy() for m in methods])
            avg_rank = complete.rank(axis=1, ascending=False).mean()
            return {
                "experiment": experiment,
                "mode": mode,
                "methods": methods,
                "n_blocks": int(len(complete)),
                "chi2": float(res.statistic),
                "p_value": float(res.pvalue),
                "avg_rank": avg_rank,
            }
    return None


def friedman_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stat_rows, rank_rows = [], []
    for experiment, block in (("lodo", "targets"), ("pairwise", "pairs")):
        for mode in _ordered_modes(runs):
            res = friedman_test(runs, experiment, mode)
            if res is None:
                continue
            stat_rows.append(
                {
                    "experiment": experiment,
                    "mode": mode,
                    "blocks": block,
                    "n_blocks": res["n_blocks"],
                    "n_methods": len(res["methods"]),
                    "methods": ", ".join(res["methods"]),
                    "chi2": res["chi2"],
                    "p_value": res["p_value"],
                }
            )
            for method, rank in res["avg_rank"].sort_values().items():
                rank_rows.append(
                    {"experiment": experiment, "mode": mode, "method": method, "average_rank": float(rank)}
                )
    return pd.DataFrame(stat_rows), pd.DataFrame(rank_rows)


# ---------------------------------------------------------------------------------------------------------------------
# 5. Family transfer
# ---------------------------------------------------------------------------------------------------------------------












# ---------------------------------------------------------------------------------------------------------------------
# 6. USB-IDS-1 defence variants
# ---------------------------------------------------------------------------------------------------------------------


def defence_table(per_attack: pd.DataFrame) -> pd.DataFrame:
    d = per_attack[(per_attack["target"] == "usbids1") & per_attack["defence"].notna()]
    if d.empty:
        return pd.DataFrame()
    per_seed = d.groupby(["method", "mode", "experiment", "scenario", "seed", "defence"], as_index=False)[
        ["n", "detected"]
    ].sum()
    per_seed["recall"] = per_seed["detected"] / per_seed["n"].where(per_seed["n"] > 0)
    per_scen = per_seed.groupby(["method", "mode", "experiment", "scenario", "defence"], as_index=False)[
        "recall"
    ].mean()
    out = per_scen.groupby(["method", "mode", "experiment", "defence"])["recall"].agg(["mean", "std", "count"])
    out = out.reset_index().rename(columns={"mean": "recall_mean", "std": "recall_std", "count": "n_scenarios"})
    out["_d"] = out["defence"].map({v: i for i, v in enumerate(DEFENCE_VARIANTS)})
    return out.sort_values(["method", "mode", "experiment", "_d"]).drop(columns="_d").reset_index(drop=True)


def _plot_defence(defence: pd.DataFrame, mode: str, experiment: str, methods: list[str]) -> plt.Figure | None:
    sub = defence[(defence["mode"] == mode) & (defence["experiment"] == experiment)]
    methods = [m for m in methods if m in set(sub["method"])]
    if not methods:
        return None
    colours = _colours(list(DEFENCE_VARIANTS))
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(methods), 3.6))
    width = 0.8 / len(DEFENCE_VARIANTS)
    worst = []
    for i, variant in enumerate(DEFENCE_VARIANTS):
        vals, errs = [], []
        for method in methods:
            row = sub[(sub["method"] == method) & (sub["defence"] == variant)]
            vals.append(float(row["recall_mean"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["recall_std"].iloc[0]) if len(row) else np.nan)
        xs = np.arange(len(methods)) + (i - (len(DEFENCE_VARIANTS) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.92, color=colours[variant], label=variant, linewidth=0)
        ax.errorbar(xs, vals, yerr=np.nan_to_num(errs), fmt="none", ecolor=INK, elinewidth=0.6, capsize=0)
    for method in methods:
        cur = sub[sub["method"] == method].dropna(subset=["recall_mean"])
        if len(cur):
            worst.append(cur.loc[cur["recall_mean"].idxmin(), "defence"])
    hardest = pd.Series(worst).mode().iloc[0] if worst else "n/a"
    hardest_count = int((pd.Series(worst) == hardest).sum()) if worst else 0
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall", color=MUTED)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title(
        f"USB-IDS-1 ({experiment}, {mode}): {hardest} gives the lowest recall for "
        f"{hardest_count} of {len(worst)} methods",
        loc="left",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# 7. Few-shot curve and shift score
# ---------------------------------------------------------------------------------------------------------------------


SHIFT_METHODS = ("hgb", "catboost", "dann", "rift-noDN")
FEWSHOT_SERIES_LABEL = {"fewshot-catboost": "fewshot-catboost (continued)", "fewshot-targetonly": "fewshot-targetonly (control)"}


def _fewshot_series(method: pd.Series) -> pd.Series:
    """Series name of a few-shot row: the method without its ``@k`` budget suffix."""
    return method.astype(str).str.split("@").str[0]


def fewshot_curve(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over scenarios of few-shot ROC-AUC and balanced accuracy per series and budget k."""
    fs = runs[runs["experiment"] == "fewshot"].copy()
    if fs.empty:
        return pd.DataFrame()
    fs["method"] = _fewshot_series(fs["method"])
    per_scen = fs.groupby(["method", "fewshot_k", "scenario"], as_index=False)[
        ["roc_auc", "balanced_accuracy"]
    ].mean()
    out = per_scen.groupby(["method", "fewshot_k"]).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        n_scenarios=("scenario", "nunique"),
    )
    return out.reset_index()


def fewshot_scenarios(runs: pd.DataFrame) -> set[str]:
    """Scenarios covered by the few-shot rows (few-shot rows have their own mode label, 'fewshot')."""
    return set(runs.loc[runs["experiment"] == "fewshot", "scenario"])


def zero_shot_references(runs: pd.DataFrame, mode: str, scenarios: set[str]) -> pd.DataFrame:
    """Mean pairwise ROC-AUC per method on the same scenarios that the few-shot rows cover."""
    pw = runs[(runs["experiment"] == "pairwise") & (runs["mode"] == mode) & runs["scenario"].isin(scenarios)]
    if pw.empty:
        return pd.DataFrame()
    per_scen = pw.groupby(["method", "scenario"])["roc_auc"].mean().reset_index()
    return per_scen.groupby("method")["roc_auc"].agg(["mean", "count"]).reset_index()


def _plot_fewshot(curve: pd.DataFrame, refs: pd.DataFrame, mode: str) -> plt.Figure:
    """Few-shot curves with two zero-shot reference lines: pairwise CatBoost and the best other label-free method."""
    series = list(dict.fromkeys(curve["method"]))
    catboost_ref = refs[refs["method"] == "catboost"] if len(refs) else refs
    others = refs[refs["method"] != "catboost"] if len(refs) else refs
    best = others.sort_values(["count", "mean"], ascending=[False, False]).iloc[0] if len(others) else None
    colours = _colours(series + ["ref-catboost", "ref-best"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    panels = (
        ("roc_auc", "ROC-AUC"),
        ("balanced_accuracy", "balanced accuracy"),
    )
    for ax, (metric, label) in zip(axes, panels):
        for name in series:
            sub = curve[curve["method"] == name].sort_values("fewshot_k")
            ax.errorbar(
                sub["fewshot_k"],
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_std"].fillna(0.0),
                color=colours[name],
                marker="o",
                markersize=4,
                capsize=2,
                elinewidth=0.6,
                label=FEWSHOT_SERIES_LABEL.get(name, name),
            )
        if metric == "roc_auc" and len(catboost_ref):
            ax.axhline(float(catboost_ref["mean"].iloc[0]), color=colours["ref-catboost"], linestyle=":",
                       linewidth=1.1, label="zero-shot catboost (pairwise)")
        if metric == "roc_auc" and best is not None:
            ax.axhline(float(best["mean"]), color=colours["ref-best"], linestyle="--", linewidth=1.0,
                       label=f"best label-free zero-shot: {best['method']}")
        ax.set_xscale("log")
        ax.set_xlabel("labelled target samples per class, k (log scale)", color=MUTED)
        ax.set_ylabel(label, color=MUTED)
        ax.set_ylim(0, 1.02)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 3), bbox_to_anchor=(0.5, -0.02), fontsize=8)
    k_values = sorted(curve["fewshot_k"].unique())
    axes[0].set_xticks(k_values)
    axes[0].set_xticklabels([f"{k:g}" for k in k_values])
    top_k = curve["fewshot_k"].max()
    top = curve[curve["fewshot_k"] == top_k].set_index("method")["roc_auc_mean"]
    cat_top = float(top.get("fewshot-catboost", np.nan))
    tgt_top = float(top.get("fewshot-targetonly", np.nan))
    zero_cat = float(catboost_ref["mean"].iloc[0]) if len(catboost_ref) else float("nan")
    tgt_text = f"{tgt_top:.2f}" if np.isfinite(tgt_top) else "not yet run"
    fig.suptitle(
        f"Few-shot ROC-AUC at k={top_k:g}: catboost {cat_top:.2f}, targetonly {tgt_text}; "
        f"zero-shot catboost {zero_cat:.2f} ({mode} pairwise references)",
        x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    return fig


def shift_by_scenario_seed(runs: pd.DataFrame) -> pd.DataFrame:
    """Shift score per (scenario, seed), taken from the few-shot rows (the value is repeated over budgets k)."""
    fs = runs[(runs["experiment"] == "fewshot") & runs["shift_score"].notna()]
    if fs.empty:
        return pd.DataFrame()
    return fs.groupby(["scenario", "seed"], as_index=False)["shift_score"].mean()


def shift_scores(runs: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise zero-shot ROC-AUC per (method, scenario, seed) joined to the shift score, and Spearman rho per method."""
    shift = shift_by_scenario_seed(runs)
    if shift.empty:
        return pd.DataFrame(), pd.DataFrame()
    pw = runs[(runs["experiment"] == "pairwise") & (runs["mode"] == mode)]
    zs = pw.groupby(["method", "scenario", "seed"], as_index=False)["roc_auc"].mean()
    points = zs.merge(shift, on=["scenario", "seed"], how="inner")
    if points.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for method, grp in points.groupby("method"):
        rho = _spearman(grp["shift_score"], grp["roc_auc"])
        rows.append({"method": method, "mode": mode, "n_points": len(grp), "spearman_rho": rho})
    overall = {"method": "all methods", "mode": mode, "n_points": len(points),
               "spearman_rho": _spearman(points["shift_score"], points["roc_auc"])}
    return points, pd.DataFrame([overall, *rows])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    keep = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def _plot_shift(points: pd.DataFrame, rho_table: pd.DataFrame, mode: str, methods: list[str]) -> plt.Figure:
    overall = float(rho_table.loc[rho_table["method"] == "all methods", "spearman_rho"].iloc[0])
    shown = [m for m in methods if m in set(points["method"])]
    colours = _colours(shown)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for method in shown:
        sub = points[points["method"] == method]
        ax.scatter(sub["shift_score"], sub["roc_auc"], s=18, color=colours[method], label=method, linewidths=0)
    ax.set_xlabel("IsolationForest shift score (target pool vs sources)", color=MUTED)
    ax.set_ylabel("zero-shot pairwise ROC-AUC", color=MUTED)
    if not np.isfinite(overall) or abs(overall) < 0.1:
        verdict = "No clear relation between shift and zero-shot ROC-AUC"
    else:
        verdict = ("Larger shift goes with lower" if overall < 0 else "Larger shift goes with higher") + " zero-shot ROC-AUC"
    ax.set_title(f"{verdict} (Spearman rho = {overall:.2f}, {mode})", loc="left", fontsize=9)
    if len(shown) >= 2:
        ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# 8. Prior estimation
# ---------------------------------------------------------------------------------------------------------------------


def prior_table(runs: pd.DataFrame) -> pd.DataFrame:
    pr = runs[runs["est_prior"].notna() & runs["true_prior"].notna()].copy()
    if pr.empty:
        return pd.DataFrame()
    rows = []
    for (method, experiment, mode), grp in pr.groupby(["method", "experiment", "mode"]):
        est = grp["est_prior"].to_numpy(dtype=float)
        true = grp["true_prior"].to_numpy(dtype=float)
        r = np.nan
        if len(grp) >= 3 and est.std() > 0 and true.std() > 0:
            r = float(stats.pearsonr(true, est).statistic)
        rows.append(
            {
                "method": method,
                "experiment": experiment,
                "mode": mode,
                "n": int(len(grp)),
                "mean_true_prior": float(true.mean()),
                "mean_est_prior": float(est.mean()),
                "bias_est_minus_true": float((est - true).mean()),
                "mean_abs_error": float(np.abs(est - true).mean()),
                "pearson_r": r,
            }
        )
    return pd.DataFrame(rows)


def _plot_prior(runs: pd.DataFrame, table: pd.DataFrame) -> plt.Figure | None:
    pr = runs[runs["est_prior"].notna() & runs["true_prior"].notna()]
    experiments = [e for e in ("pairwise", "lodo") if e in set(pr["experiment"])]
    if not experiments:
        return None
    methods = list(dict.fromkeys(pr["method"]))[:8]
    colours = _colours(methods)
    fig, axes = plt.subplots(1, len(experiments), figsize=(4.0 * len(experiments) + 0.8, 4.0), squeeze=False)
    r_text = []
    for ax, experiment in zip(axes.ravel(), experiments):
        sub = pr[pr["experiment"] == experiment]
        for method in methods:
            m = sub[sub["method"] == method]
            if len(m):
                ax.scatter(m["true_prior"], m["est_prior"], s=14, color=colours[method], label=method, linewidths=0)
        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("true target prior (attack fraction)", color=MUTED)
        ax.set_ylabel("estimated prior", color=MUTED)
        ax.set_title(experiment, loc="left")
        row = table[table["experiment"] == experiment]
        if len(row):
            r_text.append(f"{experiment} r={row['pearson_r'].dropna().mean():.2f}" if row["pearson_r"].notna().any()
                          else experiment)
    if len(methods) >= 2:
        axes.ravel()[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("RIFT prior estimates vs true prior, dashed line is exact estimation (" + ", ".join(r_text) + ")",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------------------------


def _feature_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rebuild every schema feature that the cached base columns allow, in log1p space, for one dataset.

    Base columns are stored log1p-transformed; they are inverted with expm1, the derived features are computed
    as in ``schema.harmonise_with_stats``, and the result is transformed back with log1p. ROC-AUC is invariant
    to this monotone map, so the values are equivalent to the harmonised ones.
    """
    raw = {c: np.expm1(df[c].to_numpy(dtype=float)) for c in ("duration", "fwd_pkts", "bwd_pkts", "fwd_bytes",
                                                             "bwd_bytes", "total_pkts", "total_bytes")
           if c in df.columns}
    d: dict[str, np.ndarray] = {}
    if "duration" in raw:
        d["duration"] = raw["duration"]
    if "fwd_pkts" in raw and "bwd_pkts" in raw:
        d["fwd_pkts"], d["bwd_pkts"] = raw["fwd_pkts"], raw["bwd_pkts"]
    if "total_pkts" in raw:
        d["total_pkts"] = raw["total_pkts"]
    elif "fwd_pkts" in d:
        d["total_pkts"] = d["fwd_pkts"] + d["bwd_pkts"]
    if "fwd_bytes" in raw and "bwd_bytes" in raw:
        d["fwd_bytes"], d["bwd_bytes"] = raw["fwd_bytes"], raw["bwd_bytes"]
        d["total_bytes"] = d["fwd_bytes"] + d["bwd_bytes"]
    elif "total_bytes" in raw:
        d["total_bytes"] = raw["total_bytes"]
    if "duration" in d:
        dur = np.maximum(d["duration"], 1e-6)
        if "total_bytes" in d:
            d["bytes_per_s"] = d["total_bytes"] / dur
        if "total_pkts" in d:
            d["pkts_per_s"] = d["total_pkts"] / dur
    if "total_bytes" in d and "total_pkts" in d:
        d["bytes_per_pkt"] = d["total_bytes"] / np.maximum(d["total_pkts"], 1.0)
    if "fwd_bytes" in d:
        d["fwd_bwd_byte_ratio"] = d["fwd_bytes"] / (d["bwd_bytes"] + 1.0)
    if "fwd_pkts" in d:
        d["fwd_bytes_per_pkt"] = d["fwd_bytes"] / np.maximum(d["fwd_pkts"], 1.0)
        d["bwd_bytes_per_pkt"] = d["bwd_bytes"] / np.maximum(d["bwd_pkts"], 1.0)
        d["fwd_bwd_pkt_ratio"] = d["fwd_pkts"] / (d["bwd_pkts"] + 1.0)
    return {k: np.log1p(np.maximum(v, 0.0)) for k, v in d.items() if k in SIGN_FEATURES}


def feature_auc_table(processed: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Univariate ROC-AUC of every rebuildable schema feature on every cached dataset."""
    rows = []
    for name, df in processed.items():
        if "label" not in df or df["label"].nunique() < 2:
            continue
        y = df["label"].to_numpy()
        for feat, x in _feature_frame(df).items():
            keep = np.isfinite(x)
            if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
                continue
            rows.append({"dataset": name, "feature": feat, "roc_auc": float(roc_auc_score(y[keep], x[keep])),
                         "n": int(keep.sum())})
    return pd.DataFrame(rows, columns=["dataset", "feature", "roc_auc", "n"])


def feature_sign_summary(auc: pd.DataFrame, n_datasets: int) -> pd.DataFrame:
    """Sign of each feature across the datasets where it is available.

    ``consistent_sign_across_datasets`` is only defined for features available in at least two datasets; a
    feature seen in one dataset cannot be checked for consistency and is reported as ``single dataset``.
    """
    rows = []
    for feat in SIGN_FEATURES:
        vals = auc.loc[auc["feature"] == feat, "roc_auc"].dropna().to_numpy()
        n_pos = int((vals > 0.5).sum())
        n_neg = int((vals < 0.5).sum())
        if vals.size == 0:
            sign = "absent"
        elif vals.size == 1:
            sign = "single dataset"
        elif n_pos == vals.size:
            sign = "+"
        elif n_neg == vals.size:
            sign = "-"
        else:
            sign = "mixed"
        rows.append(
            {
                "feature": feat,
                "datasets_available": int(vals.size),
                "datasets_total": n_datasets,
                "n_auc_above_0.5": n_pos,
                "n_auc_below_0.5": n_neg,
                "min_auc": float(vals.min()) if vals.size else np.nan,
                "max_auc": float(vals.max()) if vals.size else np.nan,
                "sign": sign,
                "consistent_sign_across_datasets": vals.size >= 2 and sign in ("+", "-"),
            }
        )
    return pd.DataFrame(rows)


def _plot_sign_heatmap(auc: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    wide = auc.pivot_table(index="dataset", columns="feature", values="roc_auc", aggfunc="first")
    datasets = _ordered(list(wide.index))
    feats = [f for f in SIGN_FEATURES if f in wide.columns]
    mat = wide.reindex(index=datasets, columns=feats)
    fig = plt.figure(figsize=(max(7.5, 0.62 * len(feats) + 2), 0.45 * len(datasets) + 2.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(datasets), 1.1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    _annotated_heatmap(ax, mat, DIVERGING_CMAP, norm, cell_fmt="{:.2f}")
    multi = summary[summary["datasets_available"] >= 2]
    n_consistent = int(multi["consistent_sign_across_datasets"].sum())
    ax.set_title(
        f"{n_consistent} of {len(multi)} features available in at least two datasets keep the same side of AUC 0.5",
        loc="left",
    )
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    sign_row = summary.set_index("feature").reindex(feats)["sign"]
    ax2.set_xlim(-0.5, len(feats) - 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["sign"])
    ax2.grid(False)
    labels = {"+": "+ all", "-": "- all", "single dataset": "1 dataset", "mixed": "mixed"}
    for j, feat in enumerate(feats):
        s = sign_row.get(feat, "absent")
        ax2.text(j, 0.5, labels.get(s, s), ha="center", va="center", fontsize=7, color=INK)
    ax2.set_xticks(range(len(feats)))
    ax2.set_xticklabels(feats, rotation=40, ha="right")
    ax2.tick_params(axis="x", labelbottom=True)
    plt.setp(ax.get_xticklabels(), visible=False)
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("univariate ROC-AUC (0.5 = no signal)", color=MUTED)
    cbar.outline.set_visible(False)
    return fig


def family_pair_frame(per_attack: pd.DataFrame, runs: pd.DataFrame, fams: dict[str, set[str]]) -> pd.DataFrame:
    """Per (method, mode, scenario, family): recall and informedness (recall - target FPR), averaged over seeds.

    ``seen`` is True when the source dataset contains the family. Informedness uses the ``fpr`` column of the
    same scenario and seed in runs.csv, so it removes the effect of a model that flags everything.
    """
    pw = per_attack[(per_attack["experiment"] == "pairwise") & per_attack["family"].notna()]
    pw = pw[pw["family"] != "benign"]
    if pw.empty:
        return pd.DataFrame()
    pw = pw.assign(seen=[fam in fams.get(src, set()) for src, fam in zip(pw["source"], pw["family"])])
    per_seed = pw.groupby(["method", "mode", "scenario", "source", "target", "seed", "family", "seen"],
                          as_index=False)[["n", "detected"]].sum()
    per_seed["recall"] = per_seed["detected"] / per_seed["n"].where(per_seed["n"] > 0)
    if not runs.empty and "fpr" in runs:
        fpr = runs.loc[runs["experiment"] == "pairwise", ["method", "mode", "scenario", "seed", "fpr"]]
    else:
        fpr = pd.DataFrame(columns=["method", "mode", "scenario", "seed", "fpr"])
    per_seed = per_seed.merge(fpr, on=["method", "mode", "scenario", "seed"], how="left")
    per_seed["fpr"] = pd.to_numeric(per_seed["fpr"], errors="coerce")
    per_seed["informedness"] = per_seed["recall"] - per_seed["fpr"]
    return per_seed.groupby(["method", "mode", "scenario", "source", "target", "family", "seen"], as_index=False)[
        ["recall", "informedness"]
    ].mean()


def family_seen_unseen(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Secondary, unpaired view: pooled over all pairs. Confounded by target and flag rate; reported for reference."""
    if pair_frame.empty:
        return pd.DataFrame()
    rows = []
    for (method, mode, family), grp in pair_frame.groupby(["method", "mode", "family"]):
        seen = grp[grp["seen"]]
        unseen = grp[~grp["seen"]]
        rows.append(
            {
                "method": method,
                "mode": mode,
                "family": family,
                "seen_informedness": float(seen["informedness"].mean()) if len(seen) else np.nan,
                "unseen_informedness": float(unseen["informedness"].mean()) if len(unseen) else np.nan,
                "seen_recall": float(seen["recall"].mean()) if len(seen) else np.nan,
                "unseen_recall": float(unseen["recall"].mean()) if len(unseen) else np.nan,
                "seen_n_pairs": int(len(seen)),
                "unseen_n_pairs": int(len(unseen)),
            }
        )
    out = pd.DataFrame(rows)
    out["raw_recall_gap"] = out["seen_recall"] - out["unseen_recall"]
    out["_order"] = out["family"].map({f: i for i, f in enumerate(ATTACK_FAMILY_ORDER)}).fillna(99)
    return out.sort_values(["method", "mode", "_order"]).drop(columns="_order").reset_index(drop=True)


def paired_family_cells(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Within-target comparison: for each (method, mode, target, family) present in the target, the mean
    informedness over sources that contain the family minus the mean over sources that do not. A cell counts
    only when it has at least one source on each side."""
    rows = []
    if pair_frame.empty:
        return pd.DataFrame(rows)
    for (method, mode, target, family), grp in pair_frame.groupby(["method", "mode", "target", "family"]):
        seen = grp.loc[grp["seen"], "informedness"].dropna()
        unseen = grp.loc[~grp["seen"], "informedness"].dropna()
        if seen.empty or unseen.empty:
            continue
        rows.append(
            {
                "method": method,
                "mode": mode,
                "target": target,
                "family": family,
                "n_sources_with_family": int(seen.size),
                "n_sources_without_family": int(unseen.size),
                "informedness_with_family": float(seen.mean()),
                "informedness_without_family": float(unseen.mean()),
                "paired_difference": float(seen.mean() - unseen.mean()),
            }
        )
    return pd.DataFrame(rows)


def _wilcoxon_p(diffs: np.ndarray) -> float:
    d = diffs[np.isfinite(diffs) & (diffs != 0)]
    if d.size < 1:
        return float("nan")
    try:
        return float(stats.wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


def paired_family_stats(cells: pd.DataFrame) -> pd.DataFrame:
    """Per family (and pooled over families) mean paired difference, 95% CI, cell count and Wilcoxon p-value."""
    rows = []
    if cells.empty:
        return pd.DataFrame(rows)
    for (method, mode), grp in cells.groupby(["method", "mode"]):
        for family, sub in [*grp.groupby("family"), ("all", grp)]:
            d = sub["paired_difference"].to_numpy(dtype=float)
            n = int(d.size)
            half = np.nan
            if n >= 2:
                half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n))
            rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "family": family,
                    "n_cells": n,
                    "n_targets": int(sub["target"].nunique()),
                    "mean_paired_difference": float(d.mean()),
                    "ci95_half_width": half,
                    "wilcoxon_p": _wilcoxon_p(d),
                }
            )
    out = pd.DataFrame(rows)
    out["_order"] = out["family"].map({f: i for i, f in enumerate((*ATTACK_FAMILY_ORDER, "all"))})
    return out.sort_values(["method", "mode", "_order"]).drop(columns="_order").reset_index(drop=True)


def source_target_family_matrix(pair_frame: pd.DataFrame, fams: dict[str, set[str]], method: str, mode: str):
    """Cell (source family, target family): mean informedness of the target family over pairs whose source
    contains the source family (also the mean raw recall, for reference)."""
    pp = pair_frame[(pair_frame["method"] == method) & (pair_frame["mode"] == mode)]
    if pp.empty:
        return pd.DataFrame()
    src_fam = pd.DataFrame(
        [(ds, f) for ds, families in fams.items() for f in families], columns=["source", "source_family"]
    )
    merged = pp.merge(src_fam, on="source")
    if merged.empty:
        return pd.DataFrame()
    cells = (
        merged.groupby(["source_family", "family"])
        .agg(informedness_mean=("informedness", "mean"), recall_mean=("recall", "mean"), n_pairs=("recall", "size"))
        .reset_index()
    )
    return cells.rename(columns={"family": "target_family"})


def _plot_family_matrix(cells: pd.DataFrame, method: str, mode: str) -> plt.Figure:
    rows = [f for f in ATTACK_FAMILY_ORDER if f in set(cells["source_family"])]
    cols = [f for f in ATTACK_FAMILY_ORDER if f in set(cells["target_family"])]
    wide = cells.pivot(index="source_family", columns="target_family", values="informedness_mean")
    wide = wide.reindex(index=rows, columns=cols)
    same = cells[cells["source_family"] == cells["target_family"]]["informedness_mean"]
    other = cells[cells["source_family"] != cells["target_family"]]["informedness_mean"]
    fig, ax = plt.subplots(figsize=(0.9 * len(cols) + 2.4, 0.6 * len(rows) + 1.8))
    _annotated_heatmap(ax, wide, DIVERGING_CMAP, TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), cell_fmt="{:+.2f}")
    ax.set_xlabel("target family (informedness = recall - FPR)", color=MUTED)
    ax.set_ylabel("source family present in the source dataset", color=MUTED)
    same_txt = f"{same.mean():+.2f}" if len(same) else "n/a"
    other_txt = f"{other.mean():+.2f}" if len(other) else "n/a"
    ax.set_title(
        f"{method} ({mode}): mean target informedness is {same_txt} when the source has the same family, "
        f"{other_txt} otherwise",
        loc="left",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def _plot_paired_family(paired: pd.DataFrame, mode: str) -> plt.Figure | None:
    sub = paired[paired["mode"] == mode]
    methods = list(dict.fromkeys(sub["method"]))
    if not methods:
        return None
    ncols = min(3, len(methods))
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols + 0.8, 2.6 * nrows + 0.9), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    pooled = []
    for ax, method in zip(axes.ravel(), methods):
        ax.set_visible(True)
        m = sub[sub["method"] == method]
        fam_rows = m[m["family"] != "all"]
        families = [f for f in ATTACK_FAMILY_ORDER if f in set(fam_rows["family"])]
        pos = {f: i for i, f in enumerate(families)}
        ys = [pos[f] for f in families]
        vals = fam_rows.set_index("family")["mean_paired_difference"].reindex(families).to_numpy()
        halves = fam_rows.set_index("family")["ci95_half_width"].reindex(families).fillna(0.0).to_numpy()
        ax.axvline(0.0, color=MUTED, linewidth=0.8)
        ax.errorbar(vals, ys, xerr=halves, fmt="o", color=PALETTE[0], markersize=4, capsize=2, elinewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(families)
        ax.invert_yaxis()
        ax.set_xlabel("paired difference in informedness", color=MUTED)
        total = m[m["family"] == "all"]
        if len(total):
            d = float(total["mean_paired_difference"].iloc[0])
            p = float(total["wilcoxon_p"].iloc[0])
            pooled.append(d)
            p_txt = f"{p:.2g}" if math.isfinite(p) else "n/a"
            ax.set_title(f"{method}: pooled {d:+.2f}, p={p_txt}", loc="left", fontsize=9)
        else:
            ax.set_title(method, loc="left", fontsize=9)
    overall = float(np.mean(pooled)) if pooled else float("nan")
    direction = "higher" if overall > 0 else "lower"
    fig.suptitle(
        f"Within the same target, sources containing a family give {direction} informedness on it "
        f"(mean paired difference {overall:+.2f}, {mode}); bars are 95% CIs",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def run_analysis(
    results_dir: Path | str,
    data_dir: Path | str,
    out_dir: Path | str,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> dict[str, list]:
    """Produce all analysis tables and figures. Returns the written files and the skipped sections."""
    results_dir, data_dir, out = Path(results_dir), Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = _Report(out)
    runs = load_runs(results_dir)
    per_attack = load_per_attack(results_dir)
    processed = load_processed(data_dir)
    fams = families_by_dataset(processed)
    methods = tuple(methods)
    if len(methods) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} methods per figure")
    modes = _ordered_modes(runs)
    if runs.empty:
        rep.skip("runs-based sections", "results/runs.csv is missing or empty")
    if per_attack.empty:
        rep.skip("per-attack sections", "results/per_attack.csv is missing or empty")
    if not processed:
        rep.skip("data-based sections", "no data/processed/*.parquet files found")

    # 1. taxonomy
    tax = taxonomy_table(processed, per_attack)
    if not tax.empty:
        other = tax.loc[tax["family"] == "other", "attack_type"].tolist()
        rep.table(
            "01_attack_taxonomy",
            tax,
            "attack_type -> family map with defence variant and datasets in which it occurs",
            "Attack-family taxonomy",
            note=f"Attack types mapped to `other` ({len(other)}): " + ", ".join(f"`{o}`" for o in other),
        )

    # 2. feature sign consistency
    if processed:
        auc = feature_auc_table(processed)
        if not auc.empty:
            summary = feature_sign_summary(auc, n_datasets=len(processed))
            wide = auc.pivot_table(index="dataset", columns="feature", values="roc_auc", aggfunc="first")
            wide = wide.reindex(index=_ordered(list(wide.index)))
            rep.table(
                "02_feature_sign_summary",
                summary,
                "univariate ROC-AUC sign of each core feature across datasets",
                "Feature sign consistency",
                note="ROC-AUC is computed on the cached 100k-row samples; `sign` is `+` when every available "
                "dataset has AUC > 0.5, `-` when every one has AUC < 0.5, otherwise `mixed`. CTU-13 only "
                "supports the minimal tier, so core-only features are absent there.",
            )
            rep.table(
                "02_feature_auc_long",
                auc,
                "univariate ROC-AUC per dataset and feature (long form)",
                "Univariate feature ROC-AUC",
            )
            fig = _plot_sign_heatmap(auc, summary)
            rep.figure(
                "fig02_feature_sign_heatmap",
                fig,
                "diverging heatmap of univariate ROC-AUC (datasets x features), centred at 0.5, with sign row",
            )
    else:
        rep.skip("feature sign consistency", "no processed datasets")

    # 3. pairwise transfer matrices
    if not runs.empty:
        long = pairwise_matrix_long(runs, methods)
        if not long.empty:
            rep.table(
                "03_pairwise_matrix_long",
                long,
                "source x target ROC-AUC per mode and method; in_domain_diagonal marks the in-domain value",
                "Pairwise transfer matrices (long form)",
            )
            for mode in modes:
                fig = _plot_pairwise_grid(long, mode, list(methods))
                if fig is not None:
                    rep.figure(
                        f"fig03_pairwise_matrix_{mode}",
                        fig,
                        f"source x target ROC-AUC heatmaps for {mode} mode (boxed diagonal = in-domain)",
                    )
        else:
            rep.skip("pairwise matrices", "no pairwise rows in runs.csv")

        # 4. summary and ranking
        summary = summary_table(runs)
        rep.table(
            "04_summary_long",
            summary,
            "mean and std over scenarios (scenario values are seed means) per method x experiment x mode x transfer type",
            "Summary by method, experiment, mode and transfer type (long form)",
        )
        rep.table(
            "04_summary",
            summary_markdown(summary),
            "mean ± std over scenarios for ROC-AUC, PR-AUC, balanced accuracy, F1, MCC, FPR and prior error",
            "Summary: mean ± std over scenarios",
            note="Values are mean ± std over scenarios; each scenario value is the mean over seeds. "
            "`prior_abs_error` is |est_prior - true_prior| and is blank for methods without a prior estimate.",
        )
        rank = ranking_table(runs)
        rep.table(
            "04_method_ranking",
            rank,
            "method ranks by mean ROC-AUC per experiment and mode",
            "Method ranking by ROC-AUC",
            note="Rank uses the scenarios on which every method has a result (`common scenarios`); if no "
            "scenario is shared, all available scenarios are used and the basis column says so.",
        )
        fried, fried_ranks = friedman_tables(runs)
        if fried.empty:
            rep.skip("Friedman tests", "fewer than 3 complete blocks for lodo or pairwise")
        else:
            rep.table(
                "04_friedman",
                fried,
                "Friedman tests over targets (lodo) and pairs (pairwise), with the largest complete method set",
                "Friedman tests",
                note="Blocks with a missing method are dropped; methods with the lowest coverage are excluded "
                "first. A test needs at least 3 complete blocks.",
            )
            rep.table(
                "04_friedman_average_ranks",
                fried_ranks,
                "average rank per method over blocks (1 = best)",
                "Friedman average ranks",
            )

        # 6. USB-IDS-1 defence variants
        defence = defence_table(per_attack)
        if defence.empty:
            rep.skip("USB-IDS-1 defence variants", "no usbids1 defence rows in per_attack.csv")
        else:
            rep.table(
                "06_usbids1_defence",
                defence,
                "recall per USB-IDS-1 defence variant (usbids1 as target), per method",
                "USB-IDS-1 defence variants",
                note="Recall is pooled over the attack tools (Hulk, Slowloris, Slowhttptest, TCPFlood) that share "
                "each defence suffix, averaged over seeds and then over scenarios.",
            )
            for mode in modes:
                experiment = "pairwise" if (defence["experiment"] == "pairwise").any() else "indomain"
                fig = _plot_defence(defence, mode, experiment, list(methods))
                if fig is not None:
                    rep.figure(
                        f"fig06_usbids1_defence_{mode}",
                        fig,
                        f"recall per USB-IDS-1 defence variant for {mode} mode ({experiment})",
                    )

        # 8. prior estimation
        prior = prior_table(runs)
        if prior.empty:
            rep.skip("prior estimation", "no rows with est_prior in runs.csv")
        else:
            rep.table(
                "08_prior_estimation",
                prior,
                "RIFT prior estimate vs true prior: bias, error and correlation per method",
                "Prior estimation summary",
            )
            fig = _plot_prior(runs, prior)
            if fig is not None:
                rep.figure("fig08_prior_estimation", fig, "estimated vs true target prior for RIFT variants")

        # 7. few-shot
        curve = fewshot_curve(runs)
        if curve.empty:
            rep.skip("few-shot curve", "no experiment == 'fewshot' rows in runs.csv yet")
            rep.skip("shift score vs zero-shot", "no few-shot rows with shift_score yet")
        else:
            rep.table(
                "07_fewshot_curve",
                curve,
                "few-shot ROC-AUC and balanced accuracy per series (fewshot-catboost continued on k% target "
                "rows; fewshot-targetonly trained on the same rows alone) and budget k",
                "Few-shot curve",
            )
            # the zero-shot reference lines use the transductive pairwise rows when present, else inductive
            ref_mode = "transductive" if (runs["experiment"].eq("pairwise") & runs["mode"].eq("transductive")).any() else "inductive"
            refs = zero_shot_references(runs, ref_mode, fewshot_scenarios(runs))
            fig = _plot_fewshot(curve, refs, ref_mode)
            rep.figure("fig07_fewshot_curve", fig,
                       f"few-shot curves for both series with zero-shot reference lines ({ref_mode} pairwise)")
            for mode in modes:
                points, rho = shift_scores(runs, mode)
                if points.empty:
                    rep.skip(f"shift score vs zero-shot ({mode})", "no shift_score values joined to pairwise rows")
                    continue
                rep.table(
                    f"07_shift_spearman_{mode}",
                    rho,
                    f"Spearman rho between the IsolationForest shift score and zero-shot pairwise ROC-AUC per "
                    f"(scenario, seed) ({mode})",
                    f"Shift score vs zero-shot ROC-AUC ({mode})",
                )
                fig = _plot_shift(points, rho, mode, list(SHIFT_METHODS))
                rep.figure(f"fig07b_shift_vs_zeroshot_{mode}", fig,
                           f"shift score vs zero-shot ROC-AUC scatter with Spearman rho ({mode})")

    # 5. family transfer (informedness, paired within target)
    if not per_attack.empty and fams:
        pair_frame = family_pair_frame(per_attack, runs, fams)
        if pair_frame.empty:
            rep.skip("family transfer", "no pairwise rows in per_attack.csv")
        else:
            seen = family_seen_unseen(pair_frame)
            rep.table(
                "05_family_seen_unseen",
                seen,
                "secondary, unpaired: family informedness and recall on pairs whose source does / does not contain it",
                "Family transfer, unpaired (secondary)",
                note="Pooled over all pairs, so it is confounded by target dataset and by how often each model "
                "flags attacks. Use `05_family_paired_stats` for the main comparison. `informedness` is recall minus "
                "the scenario FPR from runs.csv.",
            )
            cells = paired_family_cells(pair_frame)
            paired = paired_family_stats(cells)
            if paired.empty:
                rep.skip("paired family comparison", "no (target, family) cell has sources on both sides")
            else:
                rep.table(
                    "05_family_paired_cells",
                    cells,
                    "within-target paired differences of informedness per (method, mode, target, family)",
                    "Family transfer: paired cells",
                    note="A cell needs at least one source containing the family and one that does not for the same "
                    "target. `paired_difference` = mean over sources with the family minus mean over sources without.",
                )
                rep.table(
                    "05_family_paired_stats",
                    paired,
                    "per family: mean paired difference, 95% CI, number of cells, Wilcoxon signed-rank p-value",
                    "Family transfer: paired statistics",
                    note="`all` pools every cell across families. Wilcoxon signed-rank tests whether the paired "
                    "differences are centred on zero (two-sided).",
                )
                for mode in modes:
                    fig = _plot_paired_family(paired, mode)
                    if fig is not None:
                        rep.figure(
                            f"fig05_family_paired_{mode}",
                            fig,
                            f"paired informedness difference per family, sources with vs without the family ({mode})",
                        )
            if "hgb" in set(pair_frame["method"]):
                for mode in modes:
                    cells_hgb = source_target_family_matrix(pair_frame, fams, "hgb", mode)
                    if cells_hgb.empty:
                        continue
                    rep.table(
                        f"05_source_target_family_hgb_{mode}",
                        cells_hgb,
                        f"hgb mean informedness of each target family, grouped by source family present ({mode})",
                        f"Source-family x target-family informedness for hgb ({mode})",
                    )
                    rep.figure(
                        f"fig05b_source_target_family_hgb_{mode}",
                        _plot_family_matrix(cells_hgb, "hgb", mode),
                        f"hgb source-family x target-family informedness matrix ({mode})",
                    )
            else:
                rep.skip("source x target family matrix", "hgb has no pairwise rows yet")

    readme = _readme(rep, runs, processed)
    (out / "README.md").write_text(readme, encoding="utf-8")
    log.info("analysis written to %s (%d files, %d skipped sections)", out, len(rep.files), len(rep.skipped))
    return {"files": rep.files, "skipped": rep.skipped, "out": out}


def _readme(rep: _Report, runs: pd.DataFrame, processed: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Analysis outputs",
        "",
        "Generated by `crossids analyze`. Each figure is written as PNG (200 dpi) and PDF. Tables are written as "
        "CSV and Markdown. Re-running the command overwrites these files; `runs.csv` and `per_attack.csv` are only read.",
        "",
        "## Coverage of the current results",
        "",
    ]
    if runs.empty:
        lines.append("- runs.csv: not available")
    else:
        cov = runs.groupby(["experiment", "mode"]).agg(rows=("method", "size"), methods=("method", "nunique"),
                                                       scenarios=("scenario", "nunique"),
                                                       seeds=("seed", "nunique"))
        lines.append("| experiment | mode | rows | methods | scenarios | seeds |")
        lines.append("|---|---|---|---|---|---|")
        for (exp, mode), row in cov.iterrows():
            lines.append(f"| {exp} | {mode} | {row.rows} | {row.methods} | {row.scenarios} | {row.seeds} |")
        lines.append("")
    if processed:
        lines.append("Processed datasets: " + ", ".join(sorted(processed)))
        lines.append("")
    lines += ["## Outputs", "", "| file | what it shows |", "|---|---|"]
    lines += [f"| `{name}` | {desc} |" for name, desc in rep.files]
    lines += ["", "## Sections not produced", ""]
    lines += [f"- **{sec}**: {why}" for sec, why in rep.skipped] or ["- none"]
    lines += [
        "",
        "## Conventions",
        "",
        "- Attack-family rules are in `crossids.analysis.ATTACK_FAMILIES`; `other` is used only for names listed in "
        "`EXPLICIT_OTHER` or unmatched names (the unit test enforces this for every cached attack type).",
        "- Figures follow one palette: categorical colours in fixed order (never cycled), a diverging blue-red scale "
        "centred at 0.5 for feature AUC, and a sequential blue scale for recall and transfer ROC-AUC.",
        "- Seeds are averaged within a scenario before scenario-level statistics are computed.",
        "",
    ]
    return "\n".join(lines)
