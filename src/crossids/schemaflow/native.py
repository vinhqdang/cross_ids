"""Native-schema loaders: each dataset keeps its own numeric columns, with the column name and a short description.

Nothing is mapped to a common schema. A feature is identified only by its text (name plus description), which
:mod:`crossids.schemaflow.text` turns into an embedding. Ports, addresses, timestamps and sequence numbers are dropped.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CAP_PER_CLASS = 15000

ARGUS = {
    "dur": "total duration of the flow in seconds",
    "spkts": "number of packets from source to destination",
    "dpkts": "number of packets from destination to source",
    "sbytes": "number of bytes from source to destination",
    "dbytes": "number of bytes from destination to source",
    "rate": "packets per second over the flow",
    "sttl": "time to live value of the source",
    "dttl": "time to live value of the destination",
    "sload": "source bits per second",
    "dload": "destination bits per second",
    "sloss": "source packets retransmitted or dropped",
    "dloss": "destination packets retransmitted or dropped",
    "sinpkt": "source inter-packet arrival time",
    "dinpkt": "destination inter-packet arrival time",
    "sjit": "source jitter",
    "djit": "destination jitter",
    "swin": "source TCP window advertisement",
    "dwin": "destination TCP window advertisement",
    "tcprtt": "TCP connection round trip time",
    "synack": "time between the SYN and SYN-ACK packets",
    "ackdat": "time between the SYN-ACK and ACK packets",
    "smean": "mean size of the packets sent by the source",
    "dmean": "mean size of the packets sent by the destination",
    "trans_depth": "depth of the HTTP request-response transaction",
    "response_body_len": "size of the HTTP response body",
    "ct_srv_src": "connections with the same service and source address in the last 100 connections",
    "ct_state_ttl": "connections with the same state and TTL range",
    "ct_dst_ltm": "connections with the same destination address in the last 100 connections",
    "ct_src_dport_ltm": "connections with the same source address and destination port in the last 100 connections",
    "ct_dst_sport_ltm": "connections with the same destination address and source port in the last 100 connections",
    "ct_dst_src_ltm": "connections with the same source and destination addresses in the last 100 connections",
    "is_ftp_login": "whether the FTP session is logged in",
    "ct_ftp_cmd": "number of FTP commands in the session",
    "ct_flw_http_mthd": "number of HTTP flows with methods such as GET and POST",
    "ct_src_ltm": "connections with the same source address in the last 100 connections",
    "ct_srv_dst": "connections with the same service and destination address in the last 100 connections",
}
ZEEK = {
    "duration": "total duration of the connection in seconds",
    "src_bytes": "number of payload bytes sent by the source",
    "dst_bytes": "number of payload bytes sent by the destination",
    "missed_bytes": "number of bytes missed in content gaps",
    "src_pkts": "number of packets sent by the source",
    "dst_pkts": "number of packets sent by the destination",
    "src_ip_bytes": "number of IP-level bytes sent by the source",
    "dst_ip_bytes": "number of IP-level bytes sent by the destination",
}
NETFLOW = {
    "Dur": "total duration of the flow in seconds",
    "TotPkts": "total number of packets in the flow in both directions",
    "TotBytes": "total number of bytes in the flow in both directions",
    "SrcBytes": "number of bytes sent by the source",
    "sTos": "type of service value of the source",
    "dTos": "type of service value of the destination",
}
CIC_ABBR = {
    "TotLen": "total length of",
    "Tot": "total", "Pkts": "packets", "Pkt": "packet", "Byts": "bytes", "Len": "length", "Fwd": "forward",
    "Bwd": "backward", "IAT": "inter-arrival time", "Flag": "flag", "Cnt": "count", "Seg": "segment",
    "Avg": "average", "Std": "standard deviation", "Var": "variance", "Win": "window", "Init": "initial",
    "Act": "active", "Blk": "bulk", "PSH": "push", "URG": "urgent", "FIN": "FIN", "SYN": "SYN", "RST": "reset",
    "ACK": "acknowledgement", "ECE": "ECN echo", "CWE": "congestion window reduced", "Subflow": "subflow",
    "Min": "minimum", "Max": "maximum", "Down/Up": "download to upload", "Byts/s": "bytes per second",
    "Pkts/s": "packets per second", "Byts/b": "bytes per bulk", "Pkts/b": "packets per bulk",
}
CIC_RENAME = {
    "Total Fwd Packets": "Tot Fwd Pkts", "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts", "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Init_Win_bytes_forward": "Init Fwd Win Byts", "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "Flow Bytes/s": "Flow Byts/s", "Flow Packets/s": "Flow Pkts/s", "Fwd Packets/s": "Fwd Pkts/s",
    "Bwd Packets/s": "Bwd Pkts/s", "Packet Length Mean": "Pkt Len Mean", "Packet Length Std": "Pkt Len Std",
    "Packet Length Variance": "Pkt Len Var", "Min Packet Length": "Pkt Len Min", "Max Packet Length": "Pkt Len Max",
    "Average Packet Size": "Pkt Size Avg", "Avg Fwd Segment Size": "Fwd Seg Size Avg",
    "Avg Bwd Segment Size": "Bwd Seg Size Avg", "Fwd Header Length": "Fwd Header Len",
    "Bwd Header Length": "Bwd Header Len", "min_seg_size_forward": "Fwd Seg Size Min",
    "act_data_pkt_fwd": "Fwd Act Data Pkts", "Down/Up Ratio": "Down/Up Ratio",
}
CIC_DROP = re.compile(
    r"^(Flow ID|Src IP|Source IP|Dst IP|Destination IP|Src Port|Source Port|Dst Port|Destination Port|Timestamp|"
    r"Protocol|Label|Fwd Header Length\.1|Unnamed.*|.*Bulk.*|.*Blk.*|.*Byts/b.*|.*Pkts/b.*|CWE Flag Count|"
    r"Fwd URG Flags|Bwd URG Flags|Fwd PSH Flags|Bwd PSH Flags)$"
)


def describe_cic(name: str) -> str:
    name = CIC_RENAME.get(name.strip(), name.strip())
    words = []
    for w in name.replace("/", " / ").split():
        words.append(CIC_ABBR.get(w, w.lower()) if w != "/" else "per")
    return " ".join(words)


def _balance(df: pd.DataFrame, label: np.ndarray, rng: np.random.Generator):
    keep = []
    for c in (0, 1):
        idx = np.flatnonzero(label == c)
        if len(idx) > CAP_PER_CLASS:
            idx = rng.choice(idx, CAP_PER_CLASS, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return df.iloc[keep], label[keep]


def _finish(df: pd.DataFrame, label: np.ndarray, desc: dict[str, str], seed: int):
    rng = np.random.default_rng(seed)
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().mean() > 0.9]
    df = df.loc[:, df.nunique() > 1]
    df, label = _balance(df, label, rng)
    names = list(df.columns)
    texts = [f"{n}: {desc[n]}" if n in desc else f"{n}: {describe_cic(n)}" for n in names]
    return df.to_numpy(np.float32), label.astype(np.int64), names, texts


def load_native(key: str, data_dir: str | Path = "data", seed: int = 0):
    """Return X (n, F), y (n,), column names, feature texts for dataset ``key``."""
    raw = Path(data_dir) / "raw" / key
    if key == "unswnb15":
        df = pd.read_csv(raw / "UNSW_NB15_training-set.csv")
        y = df["label"].to_numpy()
        feats = [c for c in ARGUS if c in df.columns]
        return _finish(df[feats], y, ARGUS, seed)
    if key == "toniot":
        df = pd.read_csv(raw / "train_test_network.csv", low_memory=False)
        y = df["label"].to_numpy()
        feats = [c for c in ZEEK if c in df.columns]
        return _finish(df[feats], y, ZEEK, seed)
    if key == "insdn":
        z = zipfile.ZipFile(raw / "InSDN_DatasetCSV.zip")
        parts = [pd.read_csv(io.BytesIO(z.read(n)), low_memory=False) for n in z.namelist() if n.endswith(".csv")]
        df = pd.concat(parts, ignore_index=True)
        y = (df["Label"].astype(str).str.strip().str.lower() != "normal").to_numpy().astype(int)
        feats = [c for c in df.columns if not CIC_DROP.match(c)]
        return _finish(df[feats], y, {}, seed)
    if key == "cicids2017":
        parts = [pd.read_parquet(p) for p in sorted(raw.glob("*.parquet"))]
        df = pd.concat(parts, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
        y = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").to_numpy().astype(int)
        feats = [c for c in df.columns if not CIC_DROP.match(c)]
        return _finish(df[feats], y, {}, seed)
    if key == "ctu13":
        parts = []
        for p in sorted(raw.glob("*.binetflow")):
            d = pd.read_csv(p, low_memory=False)
            lab = d["Label"].astype(str).str.lower()
            d = d[lab.str.contains("botnet|normal")].copy()
            d["_y"] = d["Label"].astype(str).str.lower().str.contains("botnet").astype(int)
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        y = df["_y"].to_numpy()
        feats = [c for c in NETFLOW if c in df.columns]
        return _finish(df[feats], y, NETFLOW, seed)
    raise KeyError(key)
