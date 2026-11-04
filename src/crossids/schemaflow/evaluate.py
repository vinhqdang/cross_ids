"""Few-label evaluation on a held-out target: label-selection strategies x classifiers, ROC-AUC over draws."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .episodes import rank_ref
from .model import SchemaFlow
from .train import KEYS, load_domains

KS = [4, 8, 16, 32, 64]
STRATS = ["random", "kmedoid", "kmedoid-emb"]


def _medoids(Z, centers, labels):
    out = []
    for c in range(len(centers)):
        idx = np.flatnonzero(labels == c)
        if len(idx):
            out.append(idx[np.argmin(((Z[idx] - centers[c]) ** 2).sum(1))])
    return np.asarray(out)


def select(strategy, k, rng, Zf, Ze):
    n = len(Zf)
    if strategy == "random":
        return rng.choice(n, k, replace=False)
    Z = Ze if strategy == "kmedoid-emb" else Zf
    if strategy in ("kmedoid", "kmedoid-emb"):
        km = KMeans(k, n_init=1, random_state=int(rng.integers(1 << 30))).fit(Z)
        sel = _medoids(Z, km.cluster_centers_, km.labels_)
    else:  # rarecluster: 2k clusters, take medoids of the k smallest
        km = KMeans(2 * k, n_init=1, random_state=int(rng.integers(1 << 30))).fit(Z)
        sizes = np.bincount(km.labels_, minlength=2 * k)
        order = np.argsort(sizes)[:k]
        sel = np.asarray([np.flatnonzero(km.labels_ == c)[np.argmin(((Z[km.labels_ == c] - km.cluster_centers_[c]) ** 2).sum(1))] for c in order if (km.labels_ == c).any()])
    if len(sel) < k:  # top up with random rows
        rest = np.setdiff1d(np.arange(n), sel)
        sel = np.concatenate([sel, rng.choice(rest, k - len(sel), replace=False)])
    return sel[:k]


@torch.no_grad()
def sf_scores(model, vs, ys, vq, e):
    f = vs.shape[1]
    t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
    sm = torch.ones(1, len(ys), dtype=torch.bool)
    fm = torch.ones(1, f, dtype=torch.bool)
    lp = model(t(vs), t(ys), sm, t(vq), t(e), fm)[0]
    return (lp[:, 1] - lp[:, 0]).numpy()


@torch.no_grad()
def embed_rows(model, v, e):
    f = v.shape[1]
    return model.encode(torch.from_numpy(v)[None], torch.from_numpy(e)[None], torch.ones(1, f, dtype=torch.bool))[0].numpy()


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")


def evaluate(target, ckpt, notext=False, draws=10, pool_n=2000, test_n=6000, seed=0, data_dir="data"):
    dom = load_domains([target], data_dir, notext=notext)[target]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(dom.X))
    pool, test = perm[:pool_n], perm[pool_n : pool_n + test_n]
    Xp, yp, Xt, yt = dom.X[pool], dom.y[pool], dom.X[test], dom.y[test]
    Zp, Zt = rank_ref(Xp, Xp), rank_ref(Xp, Xt)
    model = SchemaFlow()
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    Ep = embed_rows(model, Zp, dom.emb)
    Et = np.concatenate([embed_rows(model, Zt[i : i + 1000], dom.emb) for i in range(0, len(Zt), 1000)])
    rows = []
    for strat in STRATS:
        for k in KS:
            for d in range(draws):
                r = np.random.default_rng(1000 * d + k)
                sel = select(strat, k, r, Zp, Ep)
                ys = yp[sel]
                s_sf = sf_scores(model, Zp[sel], ys, Zt, dom.emb)
                rf = RandomForestClassifier(200, min_samples_leaf=1, random_state=d, n_jobs=4)
                if len(np.unique(ys)) == 2:
                    rf.fit(Xp[sel], ys)
                    s_rf = rf.predict_proba(Xt)[:, 1]
                else:
                    s_rf = np.full(len(yt), 0.5)
                def lr_scores(A, B):
                    if len(np.unique(ys)) < 2:
                        return np.full(len(B), 0.5)
                    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(A[sel], ys)
                    return clf.decision_function(B)

                s_sflr = lr_scores(Ep, Et)
                s_lrr = lr_scores(Zp, Zt)
                rows += [
                    dict(target=target, strategy=strat, k=k, draw=d, method="schemaflow-lr", auc=auc(yt, s_sflr), n_att=int(ys.sum())),
                    dict(target=target, strategy=strat, k=k, draw=d, method="lr-rank", auc=auc(yt, s_lrr), n_att=int(ys.sum())),
                    dict(target=target, strategy=strat, k=k, draw=d, method="schemaflow", auc=auc(yt, s_sf), n_att=int(ys.sum())),
                    dict(target=target, strategy=strat, k=k, draw=d, method="rf", auc=auc(yt, s_rf), n_att=int(ys.sum())),
                ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--notext", action="store_true")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = evaluate(a.target, a.ckpt, a.notext, a.draws)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["strategy", "method", "k"]).auc.mean().unstack("k").round(3))
