"""FLARE-2 versus the TabPFN baselines (v3 and v3.5), binary ROC-AUC, unit = target.

Seed 0 compares FLARE-2 with both versions on all 8 targets; Holm adjusts over the two versions within each scope.
Seeds 0-2 compare FLARE-2 with v3.5 only (v3 was run for seed 0). Writes results/flare/stats/tabpfn.csv and
tabpfn_seeds.csv. The main five-baseline family of stats_flare.py is left unchanged.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import stats_flare as sf  # noqa: E402

SCOPES = {"mean over k": (4, 8, 16, 32, 64, 128), "k<=16": (4, 8, 16), "k>=32": (32, 64, 128)}
VERSIONS = ["tabpfn-v3", "tabpfn-v3.5"]


def load(methods: list[str], seeds: tuple[int, ...]) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{sf.RAW}/eval_*.csv"))]
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["method", "target", "k", "draw", "seed"])
    return d[d.method.isin(methods) & d.seed.isin(seeds)]


def target_diffs(d: pd.DataFrame, other: str, ks: tuple[int, ...], seeds: tuple[int, ...]) -> pd.Series:
    """FLARE-2 minus ``other`` per target: mean over k of the mean over draws, then over the given seeds."""
    d = d[d.k.isin(ks)]
    per = d.groupby(["method", "target", "seed", "k"]).roc_auc.mean().reset_index()
    wide = per.pivot_table(index=["target", "seed", "k"], columns="method", values="roc_auc")
    diff = (wide["flare2"] - wide[other]).dropna().rename("diff").reset_index()
    full = diff.groupby("target").filter(lambda g: set(g.seed) == set(seeds))
    return full.groupby("target")["diff"].mean()


def summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(sf.OUT, exist_ok=True)
    seed0 = load(["flare2", *VERSIONS, "rf", "catboost-target"], (0,))
    rows = []
    for scope, ks in SCOPES.items():
        p = []
        recs = []
        for v in VERSIONS:
            x = target_diffs(seed0, v, ks, (0,)).to_numpy()
            lo, hi = sf.boot_ci(x)
            recs.append({"scope": scope, "baseline": v, "n_targets": len(x), "mean_diff": x.mean(), "ci_lo": lo,
                         "ci_hi": hi, "wins": int((x > 0).sum()), "p": sf.wil(x)})
            p.append(recs[-1]["p"])
        for r, ph in zip(recs, sf.holm(p)):
            rows.append({**r, "p_holm": ph})
    t = pd.DataFrame(rows)
    t.to_csv(f"{sf.OUT}/tabpfn.csv", index=False)

    allseeds = load(["flare2", "tabpfn-v3.5"], (0, 1, 2))
    rows = []
    for scope, ks in SCOPES.items():
        x = target_diffs(allseeds, "tabpfn-v3.5", ks, (0, 1, 2)).to_numpy()
        lo, hi = sf.boot_ci(x)
        per_seed = [target_diffs(allseeds[allseeds.seed == s], "tabpfn-v3.5", ks, (s,)).mean() for s in (0, 1, 2)]
        rows.append({"scope": scope, "baseline": "tabpfn-v3.5", "n_targets": len(x), "mean_diff": x.mean(),
                     "ci_lo": lo, "ci_hi": hi, "wins": int((x > 0).sum()), "p": sf.wil(x),
                     "sd_across_seeds": float(np.std(per_seed, ddof=1)), "seed0": per_seed[0], "seed1": per_seed[1],
                     "seed2": per_seed[2]})
    ts = pd.DataFrame(rows)
    ts.to_csv(f"{sf.OUT}/tabpfn_seeds.csv", index=False)
    print(sf.md(t))
    print(sf.md(ts))
    per_t = seed0.groupby(["target", "method"]).roc_auc.mean().unstack()[["flare2", "rf", "catboost-target", *VERSIONS]]
    print(per_t.round(3).to_string())
    print(per_t.mean().round(3).to_string())


if __name__ == "__main__":
    main()
