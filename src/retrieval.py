"""Known-artwork recognition: FAISS cosine search over a reference database.

PCA is deliberately NOT applied here: raw embeddings keep the fine detail
needed to tell near-identical paintings apart. Test both in your experiments.
"""
from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

import config

INDEX_PATH = config.MODELS_DIR / "reference.index"
REF_META_PATH = config.MODELS_DIR / "reference_meta.csv"


def build_index(embeddings: np.ndarray, meta: pd.DataFrame) -> None:
    """embeddings must be L2-normalised, so inner product == cosine similarity."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(INDEX_PATH))
    meta.reset_index(drop=True).to_csv(REF_META_PATH, index=False)


def search(embedding: np.ndarray, k: int = config.TOP_K) -> list[dict]:
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_csv(REF_META_PATH)
    sims, ids = index.search(embedding.astype("float32"), k)
    results = []
    for sim, i in zip(sims[0], ids[0]):
        if i < 0:
            continue
        row = meta.iloc[int(i)].to_dict()
        row["similarity"] = float(sim)
        results.append(row)
    return results


def is_known(matches: list[dict], threshold: float = config.KNOWN_THRESHOLD) -> bool:
    return bool(matches) and matches[0]["similarity"] >= threshold
