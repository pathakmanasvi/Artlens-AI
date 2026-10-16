"""Fit, save and load PCA. Always fit on TRAIN embeddings only."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.decomposition import PCA

import config

PCA_PATH = config.MODELS_DIR / "pca_model.pkl"


def fit_pca(train_embeddings: np.ndarray, n_components: float | int | None = None) -> PCA:
    """n_components: int = fixed count, float in (0,1) = variance to keep."""
    n = n_components if n_components is not None else config.PCA_VARIANCE
    pca = PCA(n_components=n, svd_solver="full", random_state=config.RANDOM_SEED)
    pca.fit(train_embeddings)
    return pca


def save_pca(pca: PCA, path=PCA_PATH) -> None:
    joblib.dump(pca, path)


def load_pca(path=PCA_PATH) -> PCA:
    return joblib.load(path)


def explained_variance_curve(train_embeddings: np.ndarray) -> np.ndarray:
    """Cumulative explained variance for plotting the 'how many components?' curve."""
    full = PCA(random_state=config.RANDOM_SEED).fit(train_embeddings)
    return np.cumsum(full.explained_variance_ratio_)
