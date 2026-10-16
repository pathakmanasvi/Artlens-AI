"""Style and period classifiers (PCA features -> SVM)."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from src.pca_pipeline import load_pca

STYLE_PATH = config.MODELS_DIR / "style_classifier.pkl"
PERIOD_PATH = config.MODELS_DIR / "period_classifier.pkl"


def make_classifier() -> SVC:
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, probability=True, class_weight="balanced",
            random_state=config.RANDOM_SEED),
    )


def save(clf, path) -> None:
    joblib.dump(clf, path)


def load(path):
    return joblib.load(path)


def predict_top(clf, features: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k (label, probability) for a single feature row."""
    probs = clf.predict_proba(features)[0]
    idx = np.argsort(probs)[::-1][:k]
    classes = clf.classes_
    return [(str(classes[i]), float(probs[i])) for i in idx]


def predict_style_period(embedding: np.ndarray) -> dict:
    """embedding: (1, D) raw (pre-PCA) embedding."""
    pca = load_pca()
    feats = pca.transform(embedding)
    style = predict_top(load(STYLE_PATH), feats)
    period = predict_top(load(PERIOD_PATH), feats)
    return {"style": style, "period": period}
