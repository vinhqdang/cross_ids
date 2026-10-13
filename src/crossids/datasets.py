"""Dataset registry: download, streamed loading, stratified sampling, caching and synthetic data."""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .methods.base import Domain
from .schema import (
    EXTENDED_FEATURES,
    MINIMAL_FEATURES,
    SOURCES,
    TIER_ORDER,
    dataset_features,
    harmonise,
    source_columns,
)

SYNTHETIC_FEATURES: tuple[str, ...] = tuple(dict.fromkeys((*MINIMAL_FEATURES, *EXTENDED_FEATURES)))

log = logging.getLogger("crossids")

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
USB_BASE = "https://idsdata.ding.unisannio.it/datasets/USBIDS1TTV/"
INSDN_URL = "https://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip"
CTU_URL = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{n}/detailed-bidirectional-flow-labels/{file}"

CIC2017_FILES = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
    "Monday-WorkingHours.pcap_ISCX.csv.parquet",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
    "Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
    "Wednesday-workingHours.pcap_ISCX.csv.parquet",
)
CIC2018_FILES = (
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
)
USB_FILES = ("USB-IDS-1-TRAINING.csv.gz", "USB-IDS-1-VALIDATION.csv.gz", "USB-IDS-1-TEST.csv.gz")
IOT23_FILES = (
    "train-00000-of-00003-47134d3a10206e3c.parquet",
    "train-00001-of-00003-7fb0937ba34dd762.parquet",
    "train-00002-of-00003-78be3fca5c692525.parquet",
)
CTU_CAPTURES: dict[int, str] = {
    42: "capture20110810.binetflow",
    43: "capture20110811.binetflow",
    44: "capture20110812.binetflow",
    45: "capture20110815.binetflow",
    46: "capture20110815-2.binetflow",
    47: "capture20110816.binetflow",
    48: "capture20110816-2.binetflow",
    49: "capture20110816-3.binetflow",
    50: "capture20110817.binetflow",
    51: "capture20110818.binetflow",
    52: "capture20110818-2.binetflow",
    53: "capture20110819.binetflow",
    54: "capture20110815-3.binetflow",
}


@dataclass(frozen=True)
class RawFile:
    """A remote file and the local name it is stored under in data/raw/{key}/."""

    url: str
    name: str


def _hf(repo: str, path: str) -> str:
    return HF.format(repo=repo, path=path)


RAW_FILES: dict[str, tuple[RawFile, ...]] = {
    "cicids2017": tuple(RawFile(_hf("bvsam/cic-ids-2017", f"machine_learning/{n}"), n) for n in CIC2017_FILES),
    "cicids2018": tuple(RawFile(_hf("c01dsnap/CIC-IDS2018", n), n) for n in CIC2018_FILES),
    "usbids1": tuple(RawFile(USB_BASE + n, n) for n in USB_FILES),
    "insdn": (RawFile(INSDN_URL, "InSDN_DatasetCSV.zip"),),
    "unswnb15": (RawFile(_hf("Mouwiya/UNSW-NB15", "UNSW_NB15_training-set.csv"), "UNSW_NB15_training-set.csv"),),
    "toniot": (RawFile(_hf("codymlewis/TON_IoT_network", "train_test_network.csv"), "train_test_network.csv"),),
    "iot23": tuple(RawFile(_hf("19kmunz/iot-23-preprocessed-allcolumns", f"data/{n}"), n) for n in IOT23_FILES),
    "ctu13": tuple(
        RawFile(CTU_URL.format(n=n, file=f), f"scenario{n}-{f}") for n, f in sorted(CTU_CAPTURES.items())
    ),
}


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata the experiment runner needs for one dataset or synthetic domain."""

    key: str
    family: str
    extractor: str
    tiers: tuple[str, ...]


def real_info(key: str) -> DatasetInfo:
    spec = SOURCES[key]
    return DatasetInfo(key, spec.family, spec.extractor, spec.tiers)


def raw_dir(data_dir: Path, key: str) -> Path:
    return Path(data_dir) / "raw" / key


def processed_path(data_dir: Path, key: str) -> Path:
    return Path(data_dir) / "processed" / f"{key}.parquet"


def raw_paths(data_dir: Path, key: str) -> list[Path]:
    return [raw_dir(data_dir, key) / rf.name for rf in RAW_FILES[key]]


def download_file(url: str, dest: Path, retries: int = 3, chunk_bytes: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial ``.part`` file. Skips non-empty existing files."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip existing %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "crossids/0.1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                resumed = start > 0 and resp.status == 206
                mode = "ab" if resumed else "wb"
                with open(part, mode) as fh:
                    while True:
                        block = resp.read(chunk_bytes)
                        if not block:
                            break
                        fh.write(block)
            os.replace(part, dest)
            log.info("downloaded %s in %.1fs", dest.name, time.time() - t0)
            return dest
        except OSError as exc:
            log.warning("download of %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt == retries:
                raise
    return dest


def download(keys: list[str], data_dir: Path) -> None:
    """Download the raw files of every requested dataset into data/raw/{key}/."""
    for key in keys:
        if key not in RAW_FILES:
            raise KeyError(f"unknown dataset {key!r}")
        for rf in RAW_FILES[key]:
            download_file(rf.url, raw_dir(data_dir, key) / rf.name)


def _open_binary(path: Path) -> IO[bytes]:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _csv_chunks(opener: Callable[[], IO[bytes]], key: str, chunk_rows: int) -> Iterator[pd.DataFrame]:
    with opener() as fh:
        header = pd.read_csv(fh, nrows=0, encoding="utf-8-sig").columns
    cols = source_columns(key, header)
    with opener() as fh:
        yield from pd.read_csv(
            fh, usecols=cols, chunksize=chunk_rows, encoding="utf-8-sig", encoding_errors="replace", low_memory=False
        )


def iter_raw_chunks(path: Path, key: str, chunk_rows: int = 200_000) -> Iterator[pd.DataFrame]:
    """Yield raw data in chunks, reading only the columns the schema needs."""
    path = Path(path)
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        cols = source_columns(key, pf.schema_arrow.names)
        for batch in pf.iter_batches(batch_size=chunk_rows, columns=cols):
            yield batch.to_pandas()
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv") and not m.is_dir()]
            for member in members:
                yield from _csv_chunks(lambda m=member: zf.open(m), key, chunk_rows)
    else:
        yield from _csv_chunks(lambda: _open_binary(path), key, chunk_rows)


class StratifiedReservoir:
    """Bottom-k sample per binary label with exact duplicate removal inside the sample pool."""

    def __init__(self, k: int, features: tuple[str, ...], seed: int):
        self.k = k
        self.features = list(features)
        self.rng = np.random.default_rng(seed)
        self.pool: pd.DataFrame | None = None
        self.counts = np.zeros(2, dtype=np.int64)

    def add(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.counts += np.bincount(df["label"].to_numpy(dtype=np.int64), minlength=2)[:2]
        tagged = df.assign(_u=self.rng.random(len(df)))
        pool = tagged if self.pool is None else pd.concat([self.pool, tagged], ignore_index=True)
        pool = pool.drop_duplicates(subset=[*self.features, "label"], keep="first")
        parts = [pool[pool["label"] == c].nsmallest(self.k, "_u") for c in (0, 1)]
        self.pool = pd.concat(parts, ignore_index=True)

    def sample(self, max_rows: int, seed: int) -> pd.DataFrame:
        if self.pool is None:
            raise ValueError("no rows were read")
        pool = self.pool.drop(columns="_u")
        if len(pool) <= max_rows:
            return pool.reset_index(drop=True)
        total = self.counts.sum()
        frac_attack = self.counts[1] / total if total else 0.5
        pos = pool[pool["label"] == 1]
        neg = pool[pool["label"] == 0]
        n_pos = min(int(round(max_rows * frac_attack)), len(pos))
        n_neg = min(max_rows - n_pos, len(neg))
        n_pos = min(max_rows - n_neg, len(pos))
        out = pd.concat(
            [pos.sample(n=n_pos, random_state=seed), neg.sample(n=n_neg, random_state=seed)],
            ignore_index=True,
        )
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare(key: str, data_dir: Path, max_rows: int, seed: int = 0, chunk_rows: int = 200_000) -> Path:
    """Harmonise, sample and cache one dataset as data/processed/{key}.parquet."""
    if key not in SOURCES:
        raise KeyError(f"unknown dataset {key!r}")
    paths = raw_paths(data_dir, key)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw files for {key}: {[p.name for p in missing]}; run `crossids download`")
    features = dataset_features(key)
    res = StratifiedReservoir(max_rows, features, seed)
    t0 = time.time()
    for path in paths:
        log.info("reading %s", path.name)
        for chunk in iter_raw_chunks(path, key, chunk_rows):
            res.add(harmonise(chunk, key))
        log.info("  seen so far per label %s (%.0fs)", res.counts.tolist(), time.time() - t0)
    table = res.sample(max_rows, seed)
    out = processed_path(data_dir, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    meta = {
        "dataset": key,
        "rows_valid_per_label": res.counts.tolist(),
        "rows_cached": int(len(table)),
        "attack_fraction_full": float(res.counts[1] / max(res.counts.sum(), 1)),
        "attack_fraction_cached": float(table["label"].mean()) if len(table) else float("nan"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log.info("cached %s: %d rows", out, len(table))
    return out


def load_processed(key: str, data_dir: Path) -> pd.DataFrame:
    """Load a cached harmonised table."""
    path = processed_path(data_dir, key)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `crossids prepare --datasets {key}` first")
    return pd.read_parquet(path)


def synthetic_tables(
    n_domains: int = 3,
    n: int = 4000,
    seed: int = 0,
    priors: tuple[float, ...] = (0.2, 0.5, 0.7),
    n_informative: int = 6,
) -> dict[str, pd.DataFrame]:
    """Domains that share one class-conditional mechanism but differ in distortions and attack prior.

    Every domain applies its own monotone distortion (scale, offset, power) to all
    features. Feature ``spurious`` separates the classes in the first domain only.
    """
    rng = np.random.default_rng(seed)
    d = len(SYNTHETIC_FEATURES)
    shift = np.zeros(d)
    shift[:n_informative] = 1.0
    spurious = d - 1
    tables: dict[str, pd.DataFrame] = {}
    for i in range(n_domains):
        name = f"syn{i}"
        prior = priors[i % len(priors)]
        y = (rng.random(n) < prior).astype(np.int64)
        z = rng.normal(size=(n, d)) + y[:, None] * shift
        z[:, spurious] = rng.normal(size=n) + (2.0 * y if i == 0 else 0.0)
        scale = np.exp(rng.normal(0.0, 0.5, size=d))
        offset = rng.normal(0.0, 1.0, size=d)
        power = rng.uniform(0.7, 1.4, size=d)
        u = scale * z + offset
        x = np.sign(u) * np.abs(u) ** power
        frame = pd.DataFrame(x, columns=list(SYNTHETIC_FEATURES))
        frame["label"] = y
        frame["attack_type"] = np.where(y == 1, "attack", "benign")
        frame["dataset"] = name
        tables[name] = frame
    return tables


def synthetic_info(names: list[str]) -> dict[str, DatasetInfo]:
    """Metadata for synthetic domains: distinct families, one shared extractor, all tiers."""
    return {
        name: DatasetInfo(name, f"syn{i}", "synthetic", TIER_ORDER) for i, name in enumerate(names)
    }


def make_synthetic_domains(n_domains: int = 3, n: int = 4000, seed: int = 0) -> list[Domain]:
    """Synthetic domains as Domain objects over the extended feature set."""
    tables = synthetic_tables(n_domains=n_domains, n=n, seed=seed)
    domains: list[Domain] = []
    for name, frame in tables.items():
        meta = frame[["attack_type", "dataset"]].reset_index(drop=True)
        domains.append(
            Domain(name=name, X=frame[list(SYNTHETIC_FEATURES)].to_numpy(dtype=np.float64),
                   y=frame["label"].to_numpy(), meta=meta)
        )
    return domains
