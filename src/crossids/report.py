"""Summary tables from runs.csv and per_attack.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES

REPORT_GROUPS = ["experiment", "transfer_type", "method"]
DISPLAY = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1": "F1",
    "balanced_accuracy": "Bal. acc.",
    "mcc": "MCC",
    "detection_rate": "Recall",
    "fpr": "FPR",
}


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Average over scenarios within each seed, then mean and std over seeds."""
    per_seed = runs.groupby([*REPORT_GROUPS, "seed"], as_index=False)[list(METRIC_NAMES)].mean()
    rows = []
    for keys, grp in per_seed.groupby(REPORT_GROUPS, sort=True):
        record = dict(zip(REPORT_GROUPS, keys))
        record["n_seeds"] = int(grp["seed"].nunique())
        record["n_runs"] = int(((runs["experiment"] == record["experiment"])
                                & (runs["transfer_type"] == record["transfer_type"])
                                & (runs["method"] == record["method"])).sum())
        for metric in METRIC_NAMES:
            values = grp[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
            record[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def _cell(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.3f} ± {std:.3f}"


def _markdown_table(summary: pd.DataFrame) -> str:
    header = "| Method | " + " | ".join(DISPLAY[m] for m in METRIC_NAMES) + " | Seeds |"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 2)
    lines = [header, sep]
    for r in summary.itertuples():
        cells = [_cell(getattr(r, f"{m}_mean"), getattr(r, f"{m}_std")) for m in METRIC_NAMES]
        lines.append(f"| {r.method} | " + " | ".join(cells) + f" | {r.n_seeds} |")
    return "\n".join(lines)


def build_markdown(summary: pd.DataFrame, runs: pd.DataFrame) -> str:
    """Markdown report with one table per experiment and transfer type."""
    parts = ["# Results", "", "Mean ± std over seeds. Pairwise and LODO values average over scenarios within each seed.", ""]
    order = [e for e in ("indomain", "pairwise", "lodo") if e in set(summary["experiment"])]
    for exp in order:
        sub = summary[summary["experiment"] == exp]
        for tt in sorted(sub["transfer_type"].unique()):
            block = sub[sub["transfer_type"] == tt].sort_values("method")
            n_scen = runs[(runs["experiment"] == exp) & (runs["transfer_type"] == tt)]["scenario"].nunique()
            parts.append(f"## {exp} / {tt} ({n_scen} scenarios)")
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
    md_path.write_text(build_markdown(summary, runs), encoding="utf-8")
    return csv_path, md_path
