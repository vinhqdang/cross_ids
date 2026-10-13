#!/usr/bin/env python3
"""Quick CatBoost cross-dataset intrusion detection experiment (binary benign/attack)."""
import argparse
import functools
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "quick_catboost"
SAMPLE_MAX = 100_000
CHUNK = 500_000
MIN_PER_LABEL = 1_000
SEED = 0

SHORT = {
    "cicids2017": "cic17",
    "cicids2018": "cic18",
    "usbids1": "usb",
    "insdn": "insdn",
    "unswnb15": "unsw",
    "toniot": "ton",
    "iot23": "iot23",
}
CIC = ("cicids2017", "cicids2018", "usbids1", "insdn")
ZEEK = ("toniot", "iot23")
STOP = {"of", "the"}
TOKEN_MAP = {
    "pkts": "packet", "pkt": "packet", "packets": "packet",
    "byts": "byte", "bytes": "byte", "byt": "byte",
    "len": "length", "tot": "total", "seg": "segment", "blk": "bulk",
    "cnt": "count", "cwr": "cwe", "avg": "average", "var": "variance",
    "forward": "fwd", "backward": "bwd", "b": "bulk",
}
EXCLUDE = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
           "Destination Port", "Protocol", "Timestamp", "Label"]
CORE_BASE = ["duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes"]
CORE_FEATS = CORE_BASE + [
    "total_pkts", "total_bytes", "bytes_per_s", "pkts_per_s",
    "fwd_bytes_per_pkt", "bwd_bytes_per_pkt", "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio",
]
CORE_RAW = {
    "cic": {"duration": "Flow Duration", "fwd_pkts": "Total Fwd Packets",
            "bwd_pkts": "Total Backward Packets", "fwd_bytes": "Total Length of Fwd Packets",
            "bwd_bytes": "Total Length of Bwd Packets"},
    "unswnb15": {"duration": "dur", "fwd_pkts": "spkts", "bwd_pkts": "dpkts",
                 "fwd_bytes": "sbytes", "bwd_bytes": "dbytes"},
    "toniot": {"duration": "duration", "fwd_pkts": "src_pkts", "bwd_pkts": "dst_pkts",
               "fwd_bytes": "src_bytes", "bwd_bytes": "dst_bytes"},
    "iot23": {"duration": "duration", "fwd_pkts": "orig_pkts", "bwd_pkts": "resp_pkts",
              "fwd_bytes": "orig_bytes", "bwd_bytes": "resp_bytes"},
}
LABEL_COLS = {"unswnb15": ["label", "attack_cat"], "toniot": ["label", "type"], "iot23": ["label"]}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def norm_key(name):
    s = name.strip().lower().replace("totlen", "total length")
    out = []
    for tok in re.split(r"[^a-z0-9]+", s):
        if tok and tok not in STOP:
            out.extend(TOKEN_MAP.get(tok, tok).split())
    return tuple(sorted(out))


def key_map(names):
    km = {}
    for n in names:
        if "duplicated" in n.lower():
            continue
        km.setdefault(norm_key(n), n)
    return km


def sources(name):
    """Return list of source tuples (kind, path, member), or None when the dataset is unavailable."""
    if name == "cicids2017":
        files = sorted((RAW / "cicids2017").glob("*.parquet"))
        return [("parquet", str(p), None) for p in files] or None
    if name == "cicids2018":
        files = sorted((RAW / "cicids2018").glob("*_TrafficForML_CICFlowMeter.csv"))
        return [("csv", str(p), None) for p in files] or None
    if name == "usbids1":
        files = [RAW / "usbids1" / f"USB-IDS-1-{s}.csv.gz" for s in ("TRAINING", "VALIDATION", "TEST")]
        if not all(f.exists() for f in files):
            return None
        return [("csv", str(f), None) for f in files]
    if name == "insdn":
        zp = RAW / "insdn" / "InSDN_DatasetCSV.zip"
        if not zp.exists():
            log(f"NOTE: insdn skipped, {zp} is missing")
            return None
        try:
            with zipfile.ZipFile(zp) as z:
                bad = z.testzip()
                members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        except zipfile.BadZipFile:
            log(f"NOTE: insdn skipped, {zp} is not a valid zip (incomplete download)")
            return None
        if bad or not members:
            log(f"NOTE: insdn skipped, archive is corrupt or has no CSV members ({bad})")
            return None
        return [("zip", str(zp), m) for m in members]
    if name == "unswnb15":
        p = RAW / "unswnb15" / "UNSW_NB15_training-set.csv"
        return [("csv", str(p), None)] if p.exists() else None
    if name == "toniot":
        p = RAW / "toniot" / "train_test_network.csv"
        return [("csv", str(p), None)] if p.exists() else None
    if name == "iot23":
        files = sorted((RAW / "iot23").glob("*.parquet"))
        return [("parquet", str(p), None) for p in files] or None
    raise ValueError(name)


def read_header(src):
    kind, path, member = src
    if kind == "parquet":
        return [str(n) for n in pq.read_schema(path).names]
    if kind == "csv":
        return list(pd.read_csv(path, nrows=0).columns)
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        return list(pd.read_csv(f, nrows=0).columns)


@functools.lru_cache(maxsize=None)
def get_rmap(src):
    return {n.strip(): n for n in read_header(src)}


def iter_raw(src, cols, wanted):
    """Yield chunks holding the stripped names in `wanted`; `cols` are the matching raw names."""
    kind, path, member = src
    if kind == "parquet":
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=CHUNK, columns=cols):
            df = batch.to_pandas()
            df.columns = [c.strip() for c in df.columns]
            yield df[wanted]
        return
    if kind == "csv":
        yield from _csv_chunks(path, cols, wanted)
        return
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        yield from _csv_chunks(f, cols, wanted)


def _csv_chunks(obj, cols, wanted):
    for df in pd.read_csv(obj, usecols=cols, chunksize=CHUNK, low_memory=False):
        df.columns = [c.strip() for c in df.columns]
        yield df[wanted]


def process_chunk(chunk, name, mode, want):
    if name in CIC:
        chunk = chunk[chunk["Label"].astype(str).str.strip() != "Label"]
        lab = chunk["Label"].astype(str).str.strip()
        is_attack = ~lab.str.lower().isin(["benign", "normal"]).to_numpy()
        at = lab.to_numpy()
    elif name == "unswnb15":
        lab01 = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        is_attack = lab01 == 1
        raw_at = chunk["attack_cat"].fillna("").astype(str).str.strip().to_numpy()
        at = np.where(raw_at == "", np.where(is_attack, "Unknown", "Normal"), raw_at)
    elif name == "toniot":
        lab01 = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        is_attack = lab01 == 1
        raw_at = chunk["type"].fillna("").astype(str).str.strip().to_numpy()
        at = np.where(raw_at == "", np.where(is_attack, "Unknown", "normal"), raw_at)
    else:
        lab = chunk["label"].astype(str).str.strip()
        is_attack = ~lab.str.startswith("Benign").to_numpy()
        at = lab.to_numpy()

    out = pd.DataFrame(index=range(len(chunk)))
    zeek = name in ZEEK
    for f, raw in want.items():
        s = chunk[raw].reset_index(drop=True)
        if zeek:
            s = s.mask(s.astype(str).str.strip() == "-")
        num = pd.to_numeric(s, errors="coerce")
        out[f] = num.fillna(0) if zeek else num
    if mode == "core":
        if name in CIC:
            out["duration"] = out["duration"] / 1e6
        d = np.maximum(out["duration"], 1e-6)
        tp = out["fwd_pkts"] + out["bwd_pkts"]
        tb = out["fwd_bytes"] + out["bwd_bytes"]
        out["total_pkts"] = tp
        out["total_bytes"] = tb
        out["bytes_per_s"] = tb / d
        out["pkts_per_s"] = tp / d
        out["fwd_bytes_per_pkt"] = out["fwd_bytes"] / np.maximum(out["fwd_pkts"], 1)
        out["bwd_bytes_per_pkt"] = out["bwd_bytes"] / np.maximum(out["bwd_pkts"], 1)
        out["fwd_bwd_pkt_ratio"] = out["fwd_pkts"] / (out["bwd_pkts"] + 1)
        out["fwd_bwd_byte_ratio"] = out["fwd_bytes"] / (out["bwd_bytes"] + 1)
        out[CORE_FEATS] = np.log1p(out[CORE_FEATS].clip(lower=0))
    feats = list(want) if mode == "cic" else CORE_FEATS
    out = out.replace([np.inf, -np.inf], np.nan)
    out["label"] = is_attack.astype(int)
    out["attack_type"] = at
    out = out.dropna(subset=feats)
    out = out.drop_duplicates(subset=feats + ["label", "attack_type"])
    return out.reset_index(drop=True), feats


def allocate(counts, budget):
    total = sum(counts.values())
    if total <= budget:
        return dict(counts)
    floors = {c: min(n, MIN_PER_LABEL) for c, n in counts.items()}
    if sum(floors.values()) > budget:
        return {c: int(n * budget / total) for c, n in counts.items()}
    lo, hi = 0, max(counts.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        s = sum(min(n, max(mid, floors[c])) for c, n in counts.items())
        if s <= budget:
            lo = mid
        else:
            hi = mid - 1
    return {c: min(n, max(lo, floors[c])) for c, n in counts.items()}


class Reservoir:
    """Per attack_type uniform reservoir (priority sampling) with exact stream counts."""

    def __init__(self, cap):
        self.cap = cap
        self.rng = np.random.default_rng(SEED)
        self.pools = {}
        self.counts = Counter()

    def add(self, df):
        for at, g in df.groupby("attack_type", sort=False):
            self.counts[at] += len(g)
            g = g.assign(_u=self.rng.random(len(g)))
            prev = self.pools.get(at)
            if prev is not None:
                g = pd.concat([prev, g], ignore_index=True)
            if len(g) > self.cap:
                g = g.nlargest(self.cap, "_u")
            self.pools[at] = g

    def finish(self, feats):
        quotas = allocate(self.counts, SAMPLE_MAX)
        parts = []
        for at, pool in self.pools.items():
            pool = pool.drop(columns="_u").drop_duplicates(subset=feats + ["label", "attack_type"])
            q = min(quotas.get(at, 0), len(pool))
            if q > 0:
                parts.append(pool.sample(n=q, random_state=SEED))
        return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_dataset(name, mode, cic_pairs, srcs):
    if name not in CIC and mode == "cic":
        raise SystemExit(f"--features cic requires CIC-format datasets only; {name} is not CIC-format")
    res = Reservoir(SAMPLE_MAX)
    raw_rows = 0
    kept_rows = 0
    feats = None
    for src in srcs:
        rmap = get_rmap(src)
        if name in CIC:
            km = key_map(rmap)
            if mode == "cic":
                want = {f: km[k] for f, k in cic_pairs}
            else:
                want = {b: km[norm_key(CORE_RAW["cic"][b])] for b in CORE_BASE}
            label_cols = ["Label"]
        else:
            want = {b: CORE_RAW[name][b] for b in CORE_BASE}
            label_cols = LABEL_COLS[name]
        for c in list(want.values()) + label_cols:
            if c not in rmap:
                raise SystemExit(f"{name}: column '{c}' missing in {src[1]}")
        wanted = list(want.values()) + label_cols
        cols = [rmap[w] for w in wanted]
        for chunk in iter_raw(src, cols, wanted):
            raw_rows += len(chunk)
            df, feats = process_chunk(chunk, name, mode, want)
            kept_rows += len(df)
            res.add(df)
        log(f"  {name}: {Path(src[1]).name}{'/' + src[2] if src[2] else ''} done "
            f"(raw {raw_rows}, kept {kept_rows})")
    sample = res.finish(feats)
    comp = (sample.groupby(["attack_type", "label"]).size().reset_index(name="n")
            .sort_values(["label", "n"], ascending=[True, False]))
    info = {"raw_rows": raw_rows, "clean_rows": kept_rows, "sampled": len(sample),
            "benign": int((sample.label == 0).sum()), "attack": int((sample.label == 1).sum()),
            "types": int(sample.attack_type.nunique()), "composition": comp}
    return sample, info


def strat_key(df):
    return df["attack_type"] if df["attack_type"].value_counts().min() >= 2 else df["label"]


def evaluate(model, df, feats):
    y = df["label"].to_numpy()
    p = model.predict_proba(df[feats])[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both = 0 < y.sum() < len(y)
    m = {
        "n": int(len(y)), "n_attack": int(y.sum()), "attack_prior": float(y.mean()),
        "roc_auc": roc_auc_score(y, p) if has_both else np.nan,
        "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_attack": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "mcc": matthews_corrcoef(y, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    g = pd.DataFrame({"attack_type": df["attack_type"].to_numpy(), "label": y, "pred": pred}).groupby("attack_type")
    per = pd.DataFrame({"label": g["label"].max(), "n": g.size(), "pred_attack": g["pred"].sum()})
    per["pred_benign"] = per["n"] - per["pred_attack"]
    per["pct_flagged"] = 100.0 * per["pred_attack"] / per["n"]
    per = per.reset_index().sort_values(["label", "attack_type"])
    per = per[["attack_type", "pred_benign", "pred_attack", "n", "pct_flagged"]]
    return m, per


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{v:.4f}"
    return str(v)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--features", choices=["cic", "core"], required=True)
    args = ap.parse_args()
    for d in args.train + args.test:
        if d not in SHORT:
            ap.error(f"unknown dataset {d}; choose from {list(SHORT)}")
    mode = args.features

    srcs = {}
    for d in dict.fromkeys(args.train + args.test):
        s = sources(d)
        if s is None:
            log(f"NOTE: dataset {d} unavailable, skipped")
        else:
            srcs[d] = s
    train = [d for d in args.train if d in srcs]
    test = [d for d in args.test if d in srcs]
    if not train or not test:
        raise SystemExit("need at least one available train and one available test dataset")
    run_name = f"{'+'.join(SHORT[d] for d in train)}__to__{'+'.join(SHORT[d] for d in test)}__{mode}"
    run_ds = train + test

    display = {}
    cic_pairs = []
    if mode == "cic":
        non_cic = [d for d in run_ds if d not in CIC]
        if non_cic:
            raise SystemExit(f"--features cic needs CIC-format datasets only; got {non_cic}")
        common, first_km = None, None
        for d in run_ds:
            ks = None
            for src in srcs[d]:
                km = key_map(get_rmap(src))
                if first_km is None:
                    first_km = km
                ks = set(km) if ks is None else ks & set(km)
            common = ks if common is None else common & ks
        excl = {norm_key(n) for n in EXCLUDE}
        keys = [k for k in first_km if k in common and k not in excl]
        cic_pairs = [(f"f{i:03d}", k) for i, k in enumerate(keys)]
        display = {f: first_km[k] for f, k in cic_pairs}
        log(f"Common CIC features across {run_ds} ({len(cic_pairs)}): " + ", ".join(display.values()))
        feats = [f for f, _ in cic_pairs]
    else:
        feats = CORE_FEATS
        display = {f: f for f in feats}

    data, infos = {}, {}
    for d in run_ds:
        log(f"Loading {d} ({mode})")
        data[d], infos[d] = load_dataset(d, mode, cic_pairs, srcs[d])
        log(f"  {d}: sampled {infos[d]['sampled']} rows, attack {infos[d]['attack']}, types {infos[d]['types']}")

    tr_parts, in_domain = [], {}
    for d in train:
        a, b = train_test_split(data[d], test_size=0.3, stratify=strat_key(data[d]), random_state=SEED)
        tr_parts.append(a)
        in_domain[d] = b
    pool = pd.concat(tr_parts, ignore_index=True)
    fit_df, es_df = train_test_split(pool, test_size=0.1, stratify=strat_key(pool), random_state=SEED)

    log(f"Training CatBoost on {len(fit_df)} rows, early stopping on {len(es_df)} rows")
    t_fit = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="quick_catboost_")
    model = CatBoostClassifier(
        iterations=500, depth=6, auto_class_weights="Balanced", thread_count=4,
        verbose=0, random_seed=0, early_stopping_rounds=50, use_best_model=True,
        train_dir=tmp_dir,
    )
    try:
        model.fit(fit_df[feats], fit_df["label"].to_numpy(), eval_set=(es_df[feats], es_df["label"].to_numpy()))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    fit_sec = time.time() - t_fit

    evals = []
    for d in train:
        evals.append(("in-domain", f"{d} 30% holdout", in_domain[d]))
    for d in test:
        evals.append(("cross-dataset", d, data[d]))

    metric_rows, cm_tables, attack_tables = [], {}, {}
    for kind, name, df in evals:
        m, per = evaluate(model, df, feats)
        metric_rows.append({"run": run_name, "kind": kind, "eval_set": name, **m})
        cm_tables[name] = pd.DataFrame({
            "": ["True benign (0)", "True attack (1)"],
            "Pred benign (0)": [m["tn"], m["fn"]],
            "Pred attack (1)": [m["fp"], m["tp"]],
        })
        attack_tables[name] = per

    metrics = pd.DataFrame(metric_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / f"{run_name}_metrics.csv", index=False)

    mt = metrics[["kind", "eval_set", "n", "attack_prior", "roc_auc", "pr_auc", "accuracy",
                  "balanced_accuracy", "f1_attack", "precision", "recall", "fpr", "mcc"]].copy()
    mt.columns = ["Kind", "Eval set", "n", "Attack prior", "ROC-AUC", "PR-AUC", "Accuracy",
                  "Bal. acc", "F1 (attack)", "Precision", "Recall (DR)", "FPR", "MCC"]

    imp = model.get_feature_importance()
    order = np.argsort(-imp)[:10]
    imp_df = pd.DataFrame({"Rank": range(1, 11),
                           "Feature": [display[feats[i]] for i in order],
                           "Importance": [imp[i] for i in order]})

    comp_rows = []
    for d in run_ds:
        r = infos[d]
        comp_rows.append({"Dataset": d, "Role": "train" if d in train else "test",
                          "Raw rows scanned": r["raw_rows"], "Clean rows": r["clean_rows"],
                          "Sampled": r["sampled"], "Benign": r["benign"], "Attack": r["attack"],
                          "Attack types": r["types"]})
    comp_df = pd.DataFrame(comp_rows)
    total_sec = time.time() - t0

    L = []
    L.append(f"# Quick CatBoost: {run_name}")
    L.append("")
    L.append(f"- Train datasets: {', '.join(train)}")
    L.append(f"- Test datasets: {', '.join(test)}")
    L.append(f"- Feature mode: `{mode}` ({len(feats)} features)")
    L.append(f"- Sampling: at most {SAMPLE_MAX} rows per dataset, stratified by original label, seed {SEED}")
    L.append("- Cleaning: inf replaced by NaN, NaN rows dropped on model features, duplicates dropped")
    L.append("- Model: CatBoostClassifier(iterations=500, depth=6, auto_class_weights='Balanced', "
             f"random_seed=0); early stopping on a 10% slice of pooled 70% splits; best iteration "
             f"{model.get_best_iteration()}; threshold 0.5")
    L.append(f"- Training rows: {len(fit_df)} fit, {len(es_df)} early-stopping; fit time {fit_sec:.1f} s; "
             f"total runtime {total_sec:.1f} s")
    L.append("")
    L.append("## Feature set")
    L.append("")
    L.append(", ".join(display[f] for f in feats))
    L.append("")
    L.append("## Sample composition")
    L.append("")
    L.append(md(comp_df))
    L.append("")
    for d in run_ds:
        L.append(f"Per attack type in sample for {d}:")
        L.append("")
        L.append(md(infos[d]["composition"].rename(columns={"attack_type": "Attack type", "label": "Binary label", "n": "n"})))
        L.append("")
    L.append("## Metrics")
    L.append("")
    L.append(md(mt))
    L.append("")
    L.append("## Confusion matrices [[TN, FP], [FN, TP]]")
    L.append("")
    for name, cm in cm_tables.items():
        m = metrics[metrics.eval_set == name].iloc[0]
        L.append(f"### {name}")
        L.append("")
        L.append(f"TN={m.tn}, FP={m.fp}, FN={m.fn}, TP={m.tp}")
        L.append("")
        L.append(md(cm))
        L.append("")
    L.append("## Per-attack-type breakdown")
    L.append("")
    for name, per in attack_tables.items():
        L.append(f"### {name}")
        L.append("")
        L.append(md(per.rename(columns={"attack_type": "True attack type", "pred_benign": "Pred benign",
                                        "pred_attack": "Pred attack", "pct_flagged": "% flagged"})))
        L.append("")
    L.append("## Top-10 CatBoost feature importances")
    L.append("")
    L.append(md(imp_df))
    L.append("")
    report = "\n".join(L)

    (OUT / f"{run_name}.md").write_text(report, encoding="utf-8")
    print(report)
    log(f"Wrote {OUT / (run_name + '.md')} and {OUT / (run_name + '_metrics.csv')} in {total_sec:.1f} s")


if __name__ == "__main__":
    main()
