"""Leakage-safe splits: group by artist so no artist appears in both train and test."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config


def load_embeddings(extractor: str):
    X = np.load(config.EMBED_DIR / f"{extractor}_embeddings.npy")
    meta = pd.read_csv(config.EMBED_DIR / f"{extractor}_meta.csv")
    meta["period"] = meta["year"].apply(config.year_to_period)
    return X, meta


def group_split(meta: pd.DataFrame, test_size: float = 0.2):
    """Return (train_idx, test_idx), grouped by artist."""
    groups = meta["artist"].fillna("unknown").values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_SEED)
    return next(gss.split(meta, groups=groups))
