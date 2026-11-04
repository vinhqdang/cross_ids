"""Feature-text embeddings (frozen sentence encoder), cached on disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: list[str], cache: str | Path = "data/text_cache.json") -> np.ndarray:
    cache = Path(cache)
    store = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in dict.fromkeys(texts) if t not in store]
    if todo:
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(MODEL)
        vec = enc.encode(todo, normalize_embeddings=True, show_progress_bar=False)
        store.update({t: v.tolist() for t, v in zip(todo, vec)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(store))
    return np.asarray([store[t] for t in texts], dtype=np.float32)
