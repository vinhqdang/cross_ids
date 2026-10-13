"""Summary tables from runs.csv and per_attack.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES

REPORT_GROUPS = ["mode", "experiment", "transfer_type", "method"]
DISPLAY = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1": "F1",
    "balanced_accuracy": "Bal. acc.",
    "mcc": "MCC",
    "detection_rate": "Recall",
    "fpr": "FPR",
    "prior_abs_error": "abs prior err",
}
SUMMARY_METRICS = (*METRIC_NAMES, "prior_abs_error")


def _with_defaults(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    if "mode" not in runs:
        runs["mode"] = "transductive"
    if "prior_abs_error" not in runs:
        if {"est_prior", "true_prior"} <= set(runs.columns):
            runs["prior_abs_error"] = (runs["est_prior"] - runs["true_prior"]).abs()
        else:
            runs["prior_abs_error"] = np.nan
    return runs


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Average over scenarios within each seed, then mean and std over seeds, per mode/experiment/transfer type/method."""
    runs = _with_defaults(runs)
    per_seed = runs.groupby([*REPORT_GROUPS, "seed"], as_index=False)[list(SUMMARY_METRICS)].mean()
    rows = []
    for keys, grp in per_seed.groupby(REPORT_GROUPS, sort=True):
        record = dict(zip(REPORT_GROUPS, keys))
        selector = np.ones(len(runs), dtype=bool)
        for col, val in record.items():
            selector &= (runs[col] == val).to_numpy()
        record["n_seeds"] = int(grp["seed"].nunique())
        record["n_runs"] = int(selector.sum())
        record["n_scenarios"] = int(runs.loc[selector, "scenario"].nunique())
        for metric in SUMMARY_METRICS:
            values = grp[metric].to_numpy(dtype=float)
            finite = np.isfinite(values)
            record[f"{metric}_mean"] = float(np.mean(values[finite])) if finite.any() else float("nan")
            record[f"{metric}_std"] = float(np.std(values[finite], ddof=1)) if finite.sum() > 1 else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def _cell(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.3f} ± {std:.3f}"


def _markdown_table(summary: pd.DataFrame) -> str:
    header = "| Method | " + " | ".join(DISPLAY[m] for m in SUMMARY_METRICS) + " | Seeds | Scenarios |"
    sep = "|" + "---|" * (len(SUMMARY_METRICS) + 3)
    lines = [header, sep]
    for r in summary.itertuples():
        cells = [_cell(getattr(r, f"{m}_mean"), getattr(r, f"{m}_std")) for m in SUMMARY_METRICS]
        lines.append(f"| {r.method} | " + " | ".join(cells) + f" | {r.n_seeds} | {r.n_scenarios} |")
    return "\n".join(lines)


def build_markdown(summary: pd.DataFrame) -> str:
    """Markdown report: one table per mode, experiment and transfer type."""
    parts = [
        "# Results",
        "",
        "Mean ± std over seeds. Pairwise, LODO and few-shot values average over scenarios within each seed.",
        "Absolute prior error is |estimated − true| target attack fraction (RIFT variants only).",
        "",
    ]
    order_exp = [e for e in ("indomain", "pairwise", "lodo", "fewshot") if e in set(summary["experiment"])]
    for mode in [m for m in ("transductive", "inductive", "fewshot") if m in set(summary["mode"])]:
        for exp in order_exp:
            sub = summary[(summary["mode"] == mode) & (summary["experiment"] == exp)]
            if sub.empty:
                continue
            for tt in sorted(sub["transfer_type"].unique()):
                block = sub[sub["transfer_type"] == tt].sort_values("method")
                parts.append(f"## {mode} / {exp} / {tt} ({int(block['n_scenarios'].max())} scenarios)")
                parts.append("")
                parts.append(_markdown_table(block))
                parts.append("")
    return "\n".join(parts)


def write_report(results_dir: Path) -> tuple[Path, Path]:
    """Read runs.csv from results_dir and write summary.csv and summary.md."""
    results_dir = Path(results_dir)
    runs_path = results_dir / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found; run `crossids run` first")
    runs = pd.read_csv(runs_path)
    summary = summarise(runs)
    csv_path = results_dir / "summary.csv"
    md_path = results_dir / "summary.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    return csv_path, md_path
