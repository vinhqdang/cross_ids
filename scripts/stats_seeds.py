"""Seed-level statistics: FLARE-2 minus each baseline over meta-training seeds 0-2.

The seed also fixes the train/test split and the support draws of the target, so every seed is a full repeat of the
benchmark for FLARE-2 and for the baselines. Reads results/flare/raw and writes results/flare/stats/seeds.csv and
seeds.md. Per seed: mean over targets and k of the per-target difference. Across seeds: mean, standard deviation, and
an exact Wilcoxon test over targets on the seed-averaged difference.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import stats_flare as sf  # noqa: E402

SEEDS = (0, 1, 2)


def seed_table(d: pd.DataFrame, scope: str = "mean over k", ks: tuple[int, ...] | None = None) -> pd.DataFrame:
    """One row per (baseline, seed) plus a pooled row per baseline, over the given values of k (default: all)."""
    if ks is not None:
        d = d[d.k.isin(ks)]
    per = d.groupby(["method", "target", "seed", "k"]).roc_auc.mean().reset_index()
    wide = per.pivot_table(index=["target", "seed", "k"], columns="method", values="roc_auc")
    rows = []
    for b in sf.BASELINES:
        diff = (wide["flare2"] - wide[b]).dropna().rename("diff").reset_index()
        ps = diff.groupby(["seed", "target"])["diff"].mean().reset_index()
        for seed in SEEDS:
            x = ps[ps.seed == seed]["diff"].to_numpy()
            rows.append({"scope": scope, "baseline": b, "seed": str(seed), "n_targets": len(x), "mean_diff": x.mean(),
                         "wins": int((x > 0).sum()), "p": sf.wil(x)})
        # seed-averaged difference per target, then the target-level test
        full = ps.groupby("target").filter(lambda g: set(g.seed) == set(SEEDS))
        avg = full.groupby("target")["diff"].mean().to_numpy()
        means = [ps[ps.seed == s]["diff"].mean() for s in SEEDS]
        lo, hi = sf.boot_ci(avg) if len(avg) else (np.nan, np.nan)
        rows.append({"scope": scope, "baseline": b, "seed": "mean of seeds", "n_targets": len(avg), "mean_diff": avg.mean(),
                     "wins": int((avg > 0).sum()), "p": sf.wil(avg), "ci_lo": lo, "ci_hi": hi,
                     "sd_across_seeds": float(np.std(means, ddof=1))})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    m = out.seed == "mean of seeds"
    out.loc[m, "p_holm"] = sf.holm(out.loc[m, "p"].tolist())
    return out


def main() -> None:
    d = sf.load_eval(SEEDS)
    os.makedirs(sf.OUT, exist_ok=True)
    t = pd.concat([seed_table(d), seed_table(d, "k<=16", (4, 8, 16)), seed_table(d, "k>=32", (32, 64, 128))],
                  ignore_index=True)
    t.to_csv(f"{sf.OUT}/seeds.csv", index=False)
    with open(f"{sf.OUT}/seeds.md", "w", encoding="utf8") as fh:
        fh.write(sf.md(t))
    print(sf.md(t))


if __name__ == "__main__":
    main()
