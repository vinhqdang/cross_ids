"""Same-attack-family transfer between datasets: does one attack family transfer across laboratories?

For every attack family present in at least two core datasets (at least MIN_ROWS rows each), a gradient-boosting model
is trained on benign plus that family in the source dataset and scored on benign plus that family in the target
dataset (ROC-AUC). The in-domain reference is the same task with a random 70/30 split inside the source. If the same
family transfers well, the benchmark failure is mainly a change of attack mix; if it does not, the conditional
distribution of the label given the flow statistics differs. Usage:
    python scripts/family_transfer.py --out results/controls/family_transfer.csv
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from crossids import datasets as ds
from crossids.analysis import attack_family
from crossids.schema import features_for

CORE = ["cicids2017", "cicids2018", "usbids1", "insdn", "unswnb15", "toniot", "iot23"]
MIN_ROWS = 200
CAP = 20_000
SEED = 0


def sample_rows(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """At most CAP benign and CAP family rows."""
    parts = []
    for lab in (0, 1):
        g = frame[frame.label == lab]
        if len(g) > CAP:
            g = g.iloc[rng.choice(len(g), CAP, replace=False)]
        parts.append(g)
    return pd.concat(parts)


def fit_auc(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> float:
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                           class_weight="balanced", random_state=SEED)
    model.fit(train[feats].to_numpy(), train.label.to_numpy())
    return float(roc_auc_score(test.label.to_numpy(), model.predict_proba(test[feats].to_numpy())[:, 1]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    feats = list(features_for("core"))
    rng = np.random.default_rng(SEED)
    tables = {}
    for key in CORE:
        t = ds.load_processed(key, Path(a.data_dir))
        t = t.assign(family=t.attack_type.map(attack_family))
        t = t.assign(family=np.where(t.label == 0, "benign", t.family))
        tables[key] = t
    families = sorted({f for t in tables.values() for f in t.family.unique()} - {"benign", "other"})
    rows = []
    for fam in families:
        have = [k for k, t in tables.items() if (t.family == fam).sum() >= MIN_ROWS]
        for src, tgt in itertools.permutations(have, 2):
            def task(key: str) -> pd.DataFrame:
                t = tables[key]
                return sample_rows(t[(t.family == fam) | (t.label == 0)].assign(label=lambda d: (d.family == fam).astype(int)),
                                   rng)

            s, t = task(src), task(tgt)
            idx = rng.permutation(len(s))
            cut = int(0.7 * len(s))
            auc_in = fit_auc(s.iloc[idx[:cut]], s.iloc[idx[cut:]], feats)
            rows.append({"family": fam, "source": src, "target": tgt, "n_source_family": int(s.label.sum()),
                         "n_target_family": int(t.label.sum()), "auc_cross": fit_auc(s, t, feats),
                         "auc_in_source": auc_in})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.groupby("family").auc_cross.agg(["count", "median", "min", "max"]).round(3))
    print("cells", len(out), "median cross", round(out.auc_cross.median(), 3), "below 0.5:", round((out.auc_cross < 0.5).mean(), 2),
          ">=0.9:", round((out.auc_cross >= 0.9).mean(), 2), "median in-domain", round(out.auc_in_source.median(), 3))


if __name__ == "__main__":
    main()
