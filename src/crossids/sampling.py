"""Distinct-row sampling: seeded hashes, per-label KMV sketches and the natural distinct prior."""

from __future__ import annotations

import numpy as np
import pandas as pd

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_FNV = np.uint64(0x100000001B3)
_LABEL_MUL = np.uint64(0xC2B2AE3D27D4EB4F)
_SCALE = 2.0**53


def row_hashes(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """64-bit seeded hash of each row's features (rounded to 1e-6) combined with its label."""
    q = np.round(np.asarray(features, dtype=np.float64) * 1e6).astype(np.int64).view(np.uint64)
    n = q.shape[0]
    with np.errstate(over="ignore"):
        h = np.full(n, np.uint64(seed) * _GOLDEN + _GOLDEN, dtype=np.uint64)
        for j in range(q.shape[1]):
            h = (h ^ q[:, j]) * _FNV
            h ^= h >> np.uint64(29)
        h = (h ^ (np.asarray(labels).astype(np.uint64) + np.uint64(1)) * _LABEL_MUL)
        z = h + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
        z ^= z >> np.uint64(31)
    return z


def hash_to_unit(keys: np.ndarray) -> np.ndarray:
    """Map 64-bit keys to uniform values in [0, 1)."""
    return (keys >> np.uint64(11)).astype(np.float64) / _SCALE


class KMVSketch:
    """Bottom-k sketch over distinct row keys: keeps the k smallest unit hashes."""

    def __init__(self, k: int):
        self.k = int(k)
        self.frame: pd.DataFrame | None = None

    def __len__(self) -> int:
        return 0 if self.frame is None else len(self.frame)

    def update(self, candidates: pd.DataFrame) -> None:
        """Merge candidate rows (with ``_key`` and ``_u`` columns); duplicates share a key and are kept once."""
        if candidates.empty:
            return
        merged = candidates if self.frame is None else pd.concat([self.frame, candidates], ignore_index=True)
        merged = merged.drop_duplicates(subset="_key", keep="first")
        self.frame = merged.nsmallest(self.k, "_u").reset_index(drop=True)

    def distinct_estimate(self) -> float:
        """Exact count while the sketch is not full, otherwise the KMV estimate (k - 1) / u_(k)."""
        n = len(self)
        if n < self.k:
            return float(n)
        u_k = float(self.frame["_u"].max())
        return (self.k - 1) / u_k if u_k > 0 else float("inf")


def plan_counts(d_benign: float, d_attack: float, s_benign: int, s_attack: int, max_rows: int) -> tuple[int, int]:
    """Largest cached size n (<= max_rows) whose distinct prior can be realised from the sketches.

    Returns ``(n_benign, n_attack)`` with ``n_attack = round(n * pi)`` where
    ``pi = d_attack / (d_attack + d_benign)``. A short class shrinks ``n``
    rather than being back-filled from the other class.
    """
    total = d_benign + d_attack
    if total <= 0:
        return 0, 0
    pi = d_attack / total
    n_target = int(min(max_rows, round(total)))
    for n in range(n_target, 0, -1):
        n1 = int(round(n * pi))
        n0 = n - n1
        if n1 <= s_attack and n0 <= s_benign:
            return n0, n1
    return 0, 0


class DistinctSampler:
    """Streaming sampler over harmonised chunks: distinct-row sketches per label and a final stratified sample."""

    def __init__(self, max_rows: int, features: tuple[str, ...], seed: int):
        self.max_rows = int(max_rows)
        self.features = list(features)
        self.seed = int(seed)
        self.sketches = {0: KMVSketch(self.max_rows), 1: KMVSketch(self.max_rows)}
        self.valid = {0: 0, 1: 0}
        self.rows_raw = 0
        self.unlabelled = 0
        self.dropped_nan = {0: 0, 1: 0}

    def add(self, frame: pd.DataFrame, stats: dict | None = None) -> None:
        """Add one harmonised chunk, plus the drop statistics returned by ``harmonise_with_stats``."""
        if stats is not None:
            self.rows_raw += stats["raw"]
            self.unlabelled += stats["unlabelled"]
            for c in (0, 1):
                self.dropped_nan[c] += stats["dropped_nan_by_label"][c]
        if frame.empty:
            return
        labels = frame["label"].to_numpy(dtype=np.int64)
        keys = row_hashes(frame[self.features].to_numpy(dtype=np.float64), labels, self.seed)
        units = hash_to_unit(keys)
        for c in (0, 1):
            mask = labels == c
            self.valid[c] += int(mask.sum())
            if not mask.any():
                continue
            part = frame.loc[mask].reset_index(drop=True)
            part["_key"] = keys[mask].view(np.int64)
            part["_u"] = units[mask]
            self.sketches[c].update(part)

    def distinct(self) -> dict[int, float]:
        return {c: self.sketches[c].distinct_estimate() for c in (0, 1)}

    def finalize(self) -> tuple[pd.DataFrame, dict]:
        """Return the cached stratified table and the metadata that describes it."""
        dist = self.distinct()
        d0, d1 = dist[0], dist[1]
        s0, s1 = len(self.sketches[0]), len(self.sketches[1])
        n0, n1 = plan_counts(d0, d1, s0, s1, self.max_rows)
        parts = []
        for c, count in ((0, n0), (1, n1)):
            if count <= 0 or self.sketches[c].frame is None:
                continue
            chosen = self.sketches[c].frame.nsmallest(count, "_u")
            parts.append(chosen)
        cols = [*self.features, "label", "attack_type", "dataset"]
        if parts:
            table = pd.concat(parts, ignore_index=True)[cols]
            table = table.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        else:
            table = pd.DataFrame(columns=cols)
        n = len(table)
        meta = {
            "rows_raw": self.rows_raw,
            "unlabelled_dropped": self.unlabelled,
            "raw_per_label": {str(c): self.valid[c] + self.dropped_nan[c] for c in (0, 1)},
            "valid_per_label": {str(c): self.valid[c] for c in (0, 1)},
            "dropped_nan_per_label": {str(c): self.dropped_nan[c] for c in (0, 1)},
            "distinct_est_per_label": {str(c): dist[c] for c in (0, 1)},
            "prior_raw": float(self.valid[1] / max(self.valid[0] + self.valid[1], 1)),
            "prior_distinct": float(d1 / (d0 + d1)) if (d0 + d1) > 0 else float("nan"),
            "rows_cached": n,
            "cached_per_label": {"0": int(n0), "1": int(n1)},
            "prior_cached": float(table["label"].mean()) if n else float("nan"),
        }
        return table, meta
