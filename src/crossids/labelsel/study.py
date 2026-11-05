"""Few-label study on one target: strategies x classifiers, on the common-schema tables of the benchmark."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..datasets import load_processed
from ..flare.data import target_features
from ..schema import SOURCES, supports
from ..schemaflow.episodes import rank_ref
from .strategies import STRATEGIES

KS = [4, 8, 16, 32, 64]
warnings.filterwarnings("ignore")


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")


def source_model(target, feats, per_source=3000, seed=0, data_dir="data"):
    """Forest trained on the other datasets (each rank-normalised within itself) on the target's feature list."""
    Zs, ys = [], []
    tier = "core" if supports(target, "core") else "minimal"
    rng = np.random.default_rng(seed)
    for key in SOURCES:
        if key == target or not supports(key, tier):
            continue
        fr = load_processed(key, Path(data_dir))
        idx = rng.choice(len(fr), min(per_source, len(fr)), replace=False)
        Xk = fr[feats].to_numpy(np.float32)[idx]
        Zs.append(rank_ref(Xk, Xk))
        ys.append(fr["label"].to_numpy(np.int64)[idx])
    rf = RandomForestClassifier(200, min_samples_leaf=3, random_state=seed, n_jobs=1)
    return rf.fit(np.concatenate(Zs), np.concatenate(ys))


def run(target, splits=(0, 1, 2), draws=4, pool_n=2000, test_n=6000, data_dir="data", strategies=None):
    frame = load_processed(target, Path(data_dir))
    feats = list(target_features(target))
    X = frame[feats].to_numpy(np.float32)
    y = frame["label"].to_numpy(np.int64)
    rows = []
    need_src = any(n.startswith("src") for n in (strategies or STRATEGIES))
    src = source_model(target, feats, data_dir=data_dir) if need_src else None
    for sp in splits:
        rng0 = np.random.default_rng(100 + sp)
        perm = rng0.permutation(len(X))
        pool, test = perm[:pool_n], perm[pool_n : pool_n + test_n]
        Xp, yp, Xt, yt = X[pool], y[pool], X[test], y[test]
        Zp, Zt = rank_ref(Xp, Xp), rank_ref(Xp, Xt)
        ctx = {"src": src.predict_proba(Zp)[:, 1] + 1e-9 * np.random.default_rng(sp).random(len(Zp))} if src else None
        if ctx:
            rows.append(dict(target=target, split=sp, strategy="source-score", k=0, draw=0, method="pool_auc",
                             auc=auc(yp, ctx["src"]), n_att=int(yp.sum()), pool_att=float(yp.mean())))
        for name in strategies or STRATEGIES:
            fn = STRATEGIES[name]
            for k in KS:
                for d in range(draws):
                    rng = np.random.default_rng(10_000 * sp + 100 * d + k)
                    sel = fn(Zp, k, rng, lambda idx: yp[idx], ctx=ctx)
                    ys = yp[sel]
                    two = len(np.unique(ys)) == 2
                    if two:
                        rf = RandomForestClassifier(200, random_state=d, n_jobs=1).fit(Xp[sel], ys)
                        s_rf = rf.predict_proba(Xt)[:, 1]
                        lr = LogisticRegression(class_weight="balanced", max_iter=500).fit(Zp[sel], ys)
                        s_lr = lr.decision_function(Zt)
                    else:
                        s_rf = s_lr = np.zeros(len(yt))
                    for m, s in (("rf", s_rf), ("lr", s_lr)):
                        rows.append(dict(target=target, split=sp, strategy=name, k=k, draw=d, method=m,
                                         auc=auc(yt, s), n_att=int(ys.sum()), pool_att=float(yp.mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = run(a.target, draws=a.draws, strategies=a.strategies)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df[df.method == "rf"].groupby(["strategy", "k"]).auc.mean().unstack("k").round(3))
