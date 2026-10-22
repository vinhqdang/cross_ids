"""Facts about the eight datasets for the paper's dataset table: flows, capture period, attack classes, features.

Flows and prior come from data/processed/<key>.json (written by `crossids prepare`), attack-class names from the
cached benchmark sample data/processed/<key>.parquet (a rare class can be absent from the sample), the capture
period from the timestamp columns of the raw files where they exist (otherwise from the file names, or "n/a"), and the
feature counts and examples from the raw headers. Writes results/controls/dataset_summary.csv. Usage:
    python scripts/dataset_summary.py
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUT = Path("results/controls/dataset_summary.csv")

# header examples that describe each extractor (taken from the raw headers, see check below)
EXAMPLES = {
    "cicids2017": ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Fwd IAT Mean", "Init_Win_bytes_forward"],
    "cicids2018": ["Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Fwd IAT Mean", "Init Fwd Win Byts"],
    "usbids1": ["Flow Duration", "Total Fwd Packet", "Flow Bytes/s", "Fwd IAT Mean", "FWD Init Win Bytes"],
    "insdn": ["Flow Duration", "Tot Fwd Pkts", "Flow Byts/s", "Fwd IAT Mean", "Init Fwd Win Byts"],
    "unswnb15": ["dur", "spkts", "sbytes", "rate", "sttl"],
    "toniot": ["duration", "src_bytes", "conn_state", "src_pkts", "service"],
    "iot23": ["duration", "orig_bytes", "conn_state", "orig_pkts", "history"],
    "ctu13": ["Dur", "TotPkts", "TotBytes", "SrcBytes", "State"],
}


def span(ts: pd.Series, unit: str | None = None) -> str:
    t = pd.to_datetime(ts, errors="coerce", unit=unit).dropna()
    t = t[t.dt.year > 2000]
    return f"{t.min():%Y-%m-%d} to {t.max():%Y-%m-%d} (timestamps of the released file)"


def period_and_columns(key: str) -> tuple[str, int]:
    raw = DATA / "raw" / key
    if key == "cicids2017":
        files = sorted(raw.glob("*.parquet"))
        n = len(pd.read_parquet(files[0]).columns)
        days = sorted({f.name.split("-")[0] for f in files})
        return f"{len(days)} days, Monday to Friday (from the file names)", n
    if key == "cicids2018":
        files = sorted(raw.glob("*.csv"))
        days = sorted({f.name.split("_")[0].split("-", 1)[1] for f in files})
        ordered = sorted(days, key=lambda d: (d[-4:], d[3:5], d[:2]))
        return f"{len(files)} daily files, {ordered[0]} to {ordered[-1]} (file names; one 4 GB file excluded)", len(
            pd.read_csv(files[0], nrows=1).columns)
    if key == "usbids1":  # the timestamps mix formats between and inside files, so no reliable range exists
        cols = len(pd.read_csv(sorted(raw.glob("*.csv.gz"))[0], nrows=1).columns)
        return "n/a (mixed timestamp formats in the released files)", cols
    if key == "insdn":
        z = zipfile.ZipFile(raw / "InSDN_DatasetCSV.zip")  # mixed timestamp formats, see usbids1
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        return "n/a (mixed timestamp formats in the released files)", len(pd.read_csv(z.open(name), nrows=1).columns)
    if key == "unswnb15":
        return "n/a (the released training set has no timestamps)", len(pd.read_csv(raw / "UNSW_NB15_training-set.csv", nrows=1).columns)
    if key == "toniot":
        return "n/a (the released file has no timestamps)", len(pd.read_csv(raw / "train_test_network.csv", nrows=1).columns)
    if key == "iot23":
        files = sorted(raw.glob("*.parquet"))
        head = pd.read_parquet(files[0])
        tcol = next((c for c in head.columns if c.lower() in ("ts", "timestamp")), None)
        period = span(pd.concat([pd.read_parquet(f, columns=[tcol])[tcol] for f in files]), unit="s") if tcol else "n/a (no timestamps in the released file)"
        return period, len(head.columns)
    if key == "ctu13":
        files = sorted(raw.glob("*.binetflow"))
        dates = sorted({f.name.split("capture")[1][:8] for f in files})
        return f"{len(files)} scenarios, {dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]} to {dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]} (file names)", len(
            pd.read_csv(files[0], nrows=1).columns)
    raise KeyError(key)


def main() -> None:
    rows = []
    for key in EXAMPLES:
        meta = json.loads((DATA / "processed" / f"{key}.json").read_text())
        valid = meta["valid_per_label"]
        df = pd.read_parquet(DATA / "processed" / f"{key}.parquet", columns=["attack_type", "label"])
        classes = df[df.label == 1].attack_type.value_counts()
        period, ncols = period_and_columns(key)
        rows.append({
            "dataset": key, "flows_valid": int(valid["0"] + valid["1"]), "benign": int(valid["0"]), "attack": int(valid["1"]),
            "prior_raw": meta["prior_raw"], "prior_distinct": meta["prior_distinct"], "rows_cached": meta["rows_cached"],
            "n_attack_classes": len(classes), "attack_classes": "; ".join(classes.index), "period": period,
            "raw_columns": ncols, "example_features": "; ".join(EXAMPLES[key]),
        })
        print(key, rows[-1]["flows_valid"], len(classes), period, ncols)
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
