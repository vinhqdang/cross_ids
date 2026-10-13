"""Common Flow Schema: feature tiers, source spellings and harmonisation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

MINIMAL_BASE = ("duration", "total_pkts", "fwd_bytes", "bwd_bytes")
MINIMAL_DERIVED = ("total_bytes", "bytes_per_s", "pkts_per_s", "bytes_per_pkt", "fwd_bwd_byte_ratio")
MINIMAL_FEATURES = MINIMAL_BASE + MINIMAL_DERIVED

CORE_BASE = ("duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes")
CORE_DERIVED = (
    "total_pkts",
    "total_bytes",
    "bytes_per_s",
    "pkts_per_s",
    "fwd_bytes_per_pkt",
    "bwd_bytes_per_pkt",
    "fwd_bwd_pkt_ratio",
    "fwd_bwd_byte_ratio",
)
CORE_FEATURES = CORE_BASE + CORE_DERIVED

EXTENDED_EXTRA = ("fwd_iat_mean", "bwd_iat_mean", "fwd_init_win", "bwd_init_win")
EXTENDED_FEATURES = CORE_FEATURES + EXTENDED_EXTRA

TIERS: dict[str, tuple[str, ...]] = {
    "minimal": MINIMAL_FEATURES,
    "core": CORE_FEATURES,
    "extended": EXTENDED_FEATURES,
}
TIER_ORDER = ("minimal", "core", "extended")

# Spelling tables: canonical name -> header names exactly as they occur in the source files.
SPELLINGS: dict[str, dict[str, tuple[str, ...]]] = {
    "cic2017": {
        "duration": ("Flow Duration",),
        "fwd_pkts": ("Total Fwd Packets",),
        "bwd_pkts": ("Total Backward Packets",),
        "fwd_bytes": ("Total Length of Fwd Packets",),
        "bwd_bytes": ("Total Length of Bwd Packets",),
        "fwd_iat_mean": ("Fwd IAT Mean",),
        "bwd_iat_mean": ("Bwd IAT Mean",),
        "fwd_init_win": ("Init_Win_bytes_forward",),
        "bwd_init_win": ("Init_Win_bytes_backward",),
    },
    "cic2018": {
        "duration": ("Flow Duration",),
        "fwd_pkts": ("Tot Fwd Pkts",),
        "bwd_pkts": ("Tot Bwd Pkts",),
        "fwd_bytes": ("TotLen Fwd Pkts",),
        "bwd_bytes": ("TotLen Bwd Pkts",),
        "fwd_iat_mean": ("Fwd IAT Mean",),
        "bwd_iat_mean": ("Bwd IAT Mean",),
        "fwd_init_win": ("Init Fwd Win Byts",),
        "bwd_init_win": ("Init Bwd Win Byts",),
    },
    "sannio": {
        "duration": ("Flow Duration",),
        "fwd_pkts": ("Total Fwd Packet",),
        "bwd_pkts": ("Total Bwd packets",),
        "fwd_bytes": ("Total Length of Fwd Packet",),
        "bwd_bytes": ("Total Length of Bwd Packet",),
        "fwd_iat_mean": ("Fwd IAT Mean",),
        "bwd_iat_mean": ("Bwd IAT Mean",),
        "fwd_init_win": ("FWD Init Win Bytes",),
        "bwd_init_win": ("Bwd Init Win Bytes",),
    },
    "argus": {
        "duration": ("dur",),
        "fwd_pkts": ("spkts",),
        "bwd_pkts": ("dpkts",),
        "fwd_bytes": ("sbytes",),
        "bwd_bytes": ("dbytes",),
        "fwd_iat_mean": ("sinpkt",),
        "bwd_iat_mean": ("dinpkt",),
        "fwd_init_win": ("swin",),
        "bwd_init_win": ("dwin",),
    },
    "zeek_ton": {
        "duration": ("duration",),
        "fwd_pkts": ("src_pkts",),
        "bwd_pkts": ("dst_pkts",),
        "fwd_bytes": ("src_bytes",),
        "bwd_bytes": ("dst_bytes",),
    },
    "zeek_iot23": {
        "duration": ("duration",),
        "fwd_pkts": ("orig_pkts",),
        "bwd_pkts": ("resp_pkts",),
        "fwd_bytes": ("orig_bytes",),
        "bwd_bytes": ("resp_bytes",),
    },
    "ctu13": {
        "duration": ("Dur",),
        "total_pkts": ("TotPkts",),
        "fwd_bytes": ("SrcBytes",),
        "total_bytes": ("TotBytes",),
    },
}

ZERO_IF_DASH = ("duration", "fwd_bytes", "bwd_bytes")


@dataclass(frozen=True)
class SourceSpec:
    """How one dataset maps onto the Common Flow Schema."""

    key: str
    family: str
    extractor: str
    spelling: str
    tiers: tuple[str, ...]
    time_scale: float
    iat_scale: float
    label_rule: str
    zero_missing: bool = False


SOURCES: dict[str, SourceSpec] = {
    "cicids2017": SourceSpec("cicids2017", "cic", "cicflowmeter", "cic2017", TIER_ORDER, 1e-6, 1e-6, "cic"),
    "cicids2018": SourceSpec("cicids2018", "cic", "cicflowmeter", "cic2018", TIER_ORDER, 1e-6, 1e-6, "cic"),
    "usbids1": SourceSpec("usbids1", "sannio", "cicflowmeter", "sannio", TIER_ORDER, 1e-6, 1e-6, "cic"),
    "insdn": SourceSpec("insdn", "ucd", "cicflowmeter", "cic2018", TIER_ORDER, 1e-6, 1e-6, "insdn"),
    "unswnb15": SourceSpec("unswnb15", "unsw", "argus", "argus", TIER_ORDER, 1.0, 1e-3, "argus"),
    "toniot": SourceSpec("toniot", "unsw", "zeek", "zeek_ton", ("minimal", "core"), 1.0, 1.0, "ton", True),
    "iot23": SourceSpec("iot23", "ctu", "zeek", "zeek_iot23", ("minimal", "core"), 1.0, 1.0, "iot23", True),
    "ctu13": SourceSpec("ctu13", "ctu", "argus", "ctu13", ("minimal",), 1.0, 1.0, "ctu13"),
}

LABEL_SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "cic": ("Label",),
    "insdn": ("Label",),
    "ctu13": ("Label",),
    "argus": ("label", "attack_cat"),
    "ton": ("label", "type"),
    "iot23": ("label",),
}


def normalise_name(name: str) -> str:
    """Return a column name with BOM and surrounding whitespace removed, lower-cased."""
    return str(name).replace("﻿", "").strip().lower()


def resolve_names(columns: Iterable[str], wanted: Iterable[str]) -> dict[str, str]:
    """Map wanted header names to the actual column names, case-insensitively."""
    lookup = {normalise_name(c): c for c in columns}
    found: dict[str, str] = {}
    for name in wanted:
        actual = lookup.get(normalise_name(name))
        if actual is not None:
            found[name] = actual
    return found


def dataset_features(dataset: str) -> tuple[str, ...]:
    """Ordered union of the feature tiers a dataset supports."""
    feats: list[str] = []
    for tier in SOURCES[dataset].tiers:
        feats.extend(f for f in TIERS[tier] if f not in feats)
    return tuple(feats)


def features_for(tier: str) -> tuple[str, ...]:
    """Feature names of one schema tier."""
    if tier not in TIERS:
        raise ValueError(f"unknown schema tier {tier!r}")
    return TIERS[tier]


def supports(dataset: str, tier: str) -> bool:
    """True when the dataset provides the given tier."""
    return tier in SOURCES[dataset].tiers


def source_columns(dataset: str, columns: Iterable[str]) -> list[str]:
    """Raw column names needed to harmonise a dataset, resolved against a header."""
    spec = SOURCES[dataset]
    cols = list(columns)
    wanted = [raw for raw in _spelling_names(spec)]
    found = list(resolve_names(cols, wanted).values())
    found += list(resolve_names(cols, LABEL_SOURCE_COLUMNS[spec.label_rule]).values())
    return list(dict.fromkeys(found))


def _spelling_names(spec: SourceSpec) -> list[str]:
    names: list[str] = []
    for header_names in SPELLINGS[spec.spelling].values():
        names.extend(header_names)
    return names


def _numeric(series: pd.Series, dash_is_zero: bool) -> np.ndarray:
    values = series
    if dash_is_zero:
        mask = values.astype(str).str.strip().eq("-")
        values = values.astype(object).mask(mask, 0)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")


def _base_columns(df: pd.DataFrame, spec: SourceSpec, tier_base: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract canonical base columns in SI units from raw data."""
    raw = resolve_names(df.columns, _spelling_names(spec))
    table = SPELLINGS[spec.spelling]
    out: dict[str, np.ndarray] = {}
    for canon, headers in table.items():
        actual = next((raw[h] for h in headers if h in raw), None)
        if actual is None:
            continue
        out[canon] = _numeric(df[actual], spec.zero_missing and canon in ZERO_IF_DASH)
    if "total_bytes" in out and "bwd_bytes" not in out:
        out["bwd_bytes"] = np.maximum(out["total_bytes"] - out["fwd_bytes"], 0.0)
    if "fwd_pkts" in out and "bwd_pkts" in out:
        out["total_pkts"] = out["fwd_pkts"] + out["bwd_pkts"]
    if "duration" in out:
        out["duration"] = out["duration"] * spec.time_scale
    for name in ("fwd_iat_mean", "bwd_iat_mean"):
        if name in out:
            out[name] = out[name] * spec.iat_scale
    missing = [c for c in tier_base if c not in out]
    if missing:
        raise KeyError(f"{spec.key}: missing source columns for {missing}")
    return out


def _labels(df: pd.DataFrame, spec: SourceSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (binary label, attack_type, keep mask) for raw rows."""
    cols = resolve_names(df.columns, LABEL_SOURCE_COLUMNS[spec.label_rule])
    n = len(df)
    keep = np.ones(n, dtype=bool)
    rule = spec.label_rule
    if rule in ("cic", "insdn", "ctu13", "iot23"):
        lab = df[cols[LABEL_SOURCE_COLUMNS[rule][0]]].astype(str).str.strip()
        if rule == "cic":
            keep = (lab != "Label").to_numpy()
            y = (lab.str.lower() != "benign").to_numpy()
        elif rule == "insdn":
            y = (lab.str.lower() != "normal").to_numpy()
        elif rule == "iot23":
            y = (~lab.str.lower().str.startswith("benign")).to_numpy()
        else:
            text = lab.str.lower()
            is_botnet = text.str.contains("botnet", regex=False).to_numpy()
            is_normal = text.str.contains("normal", regex=False).to_numpy()
            keep = is_botnet | is_normal
            y = is_botnet
        return y.astype(np.int8), lab.to_numpy(object), keep
    if rule == "argus":
        y_raw = pd.to_numeric(df[cols["label"]], errors="coerce")
        keep = y_raw.notna().to_numpy()
        y = y_raw.fillna(0).to_numpy().astype(np.int8)
        cat = df[cols["attack_cat"]].astype(str).str.strip()
        empty = cat.isin(["", "nan", "None"]).to_numpy()
        attack = np.where(empty, np.where(y == 1, "Unknown", "Normal"), cat.to_numpy(object))
        return y, attack.astype(object), keep
    if rule == "ton":
        y_raw = pd.to_numeric(df[cols["label"]], errors="coerce")
        keep = y_raw.notna().to_numpy()
        y = y_raw.fillna(0).to_numpy().astype(np.int8)
        attack = df[cols["type"]].astype(str).str.strip().to_numpy(object)
        return y, attack, keep
    raise ValueError(f"unknown label rule {rule!r}")


def harmonise(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Map a raw frame of one dataset onto the Common Flow Schema.

    Returns every feature the dataset supports (log1p transformed), plus
    ``label`` (1 = attack), ``attack_type`` and ``dataset``. Rows with missing
    base values or without a usable label are dropped.
    """
    spec = SOURCES[dataset]
    tier_base = list(MINIMAL_BASE)
    if supports(dataset, "core"):
        tier_base += ["fwd_pkts", "bwd_pkts"]
    if supports(dataset, "extended"):
        tier_base += list(EXTENDED_EXTRA)
    base = _base_columns(df, spec, tuple(tier_base))
    y, attack, keep = _labels(df, spec)
    values = {k: base[k] for k in tier_base if k in base}
    base_mat = pd.DataFrame(values).replace([np.inf, -np.inf], np.nan)
    keep = keep & base_mat.notna().all(axis=1).to_numpy()
    base_mat = base_mat.clip(lower=0)[keep]
    d = {k: base_mat[k].to_numpy(dtype="float64") for k in base_mat.columns}
    d["total_pkts"] = d["fwd_pkts"] + d["bwd_pkts"] if "fwd_pkts" in d else d["total_pkts"]
    dur = np.maximum(d["duration"], 1e-6)
    d["total_bytes"] = d["fwd_bytes"] + d["bwd_bytes"]
    d["bytes_per_s"] = d["total_bytes"] / dur
    d["pkts_per_s"] = d["total_pkts"] / dur
    d["bytes_per_pkt"] = d["total_bytes"] / np.maximum(d["total_pkts"], 1.0)
    d["fwd_bwd_byte_ratio"] = d["fwd_bytes"] / (d["bwd_bytes"] + 1.0)
    if "fwd_pkts" in d:
        d["fwd_bytes_per_pkt"] = d["fwd_bytes"] / np.maximum(d["fwd_pkts"], 1.0)
        d["bwd_bytes_per_pkt"] = d["bwd_bytes"] / np.maximum(d["bwd_pkts"], 1.0)
        d["fwd_bwd_pkt_ratio"] = d["fwd_pkts"] / (d["bwd_pkts"] + 1.0)
    feats = dataset_features(dataset)
    out = pd.DataFrame({f: np.log1p(d[f]) for f in feats})
    out["label"] = y[keep].astype(np.int8)
    out["attack_type"] = attack[keep]
    out["dataset"] = dataset
    return out.reset_index(drop=True)
