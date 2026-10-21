"""Does FLARE-2 read the rule from the support labels? Controls on the support set of the binary task.

For each target, k and draw, ROC-AUC on the target test split with (a) the true support labels, (b) the support
labels randomly permuted among the support rows (class counts kept), (c) the two class labels swapped, scored as
P(benign), and (d) a support set of benign rows only is not possible, so we also report (e) the other draws' support
rows given to this draw's queries (support from the same target, different draw). A model that reads the rule from the
labels gives about 0.5 under (b) and 1 - AUC under (c). Usage:
    python scripts/incontext_controls.py --seed 1 --ks 16 64 --draws 5 --out results/flare/raw/incontext_s1.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from crossids.flare import evaluate as ev
from crossids.flare.data import load_split, target_features
from crossids.flare.data import target_keys

log = logging.getLogger("incontext")


def run_target(target: str, seed: int, ks: tuple[int, ...], draws: int, device: str, runs: Path) -> list[dict]:
    ctx = ev.EvalContext(runs_dir=runs, supports_dir=runs, device=device, threads=4, al_pool=0, ks=ks, n_draws=10,
                         benchmark_dir=Path("results"))
    split = load_split(target, seed, "data", features=target_features(target))
    st = ev.SeedState(target, seed, split, ctx, sources_fn=lambda: [])
    model = st.model("flare2")
    if model is None:
        return []
    sup = ev.load_or_draw_supports(runs / f"supports_{target}_s{seed}.json", target, seed, split.y_train, ks, 10)
    y, Xt = split.y_test, split.X_test
    rows = []
    for k in ks:
        for d in range(draws):
            idx = sup[k][d]
            Xs, ys = split.X_train[idx], split.y_train[idx]
            rng = np.random.default_rng([seed, k, d, 99])
            other = sup[k][(d + 1) % 10]
            variants = {
                "true": roc_auc_score(y, model.predict_proba(Xs, ys, Xt, seed=d)),
                "shuffled": roc_auc_score(y, model.predict_proba(Xs, rng.permutation(ys), Xt, seed=d)),
                "swapped": roc_auc_score(y, 1 - model.predict_proba(Xs, 1 - ys, Xt, seed=d)),
                "other_draw": roc_auc_score(y, model.predict_proba(split.X_train[other], split.y_train[other], Xt,
                                                                  seed=d)),
            }
            for name, auc in variants.items():
                rows.append({"target": target, "seed": seed, "k": k, "draw": d, "control": name, "roc_auc": auc})
            log.info("%s k=%d draw=%d %s", target, k, d, {n: round(v, 3) for n, v in variants.items()})
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ks", nargs="+", type=int, default=[16, 64])
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--targets", nargs="+", default=list(target_keys()))
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs-dir", default="runs/flare")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    done = set(pd.read_csv(out).target) if out.exists() else set()
    for t in a.targets:
        if t in done:
            continue
        rows = run_target(t, a.seed, tuple(a.ks), a.draws, a.device, Path(a.runs_dir))
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
