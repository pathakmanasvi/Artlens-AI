"""Step 3: train and save PCA + style classifier + period classifier (CLIP features).

Usage: python -m scripts.train_classifiers [--components 0.95]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import config
from scripts.splits import group_split, load_embeddings
from src import classifier
from src.pca_pipeline import fit_pca, save_pca


def main(components: float) -> None:
    X, meta = load_embeddings(config.EXTRACTOR)
    tr, te = group_split(meta)

    n = components if components < 1 else int(components)
    pca = fit_pca(X[tr], n)          # train only: no leakage
    save_pca(pca)
    print(f"PCA kept {pca.n_components_} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance).")
    Ftr, Fte = pca.transform(X[tr]), pca.transform(X[te])

    for target, path in (("style", classifier.STYLE_PATH), ("period", classifier.PERIOD_PATH)):
        ytr, yte = meta[target].values[tr], meta[target].values[te]
        clf = classifier.make_classifier().fit(Ftr, ytr)
        pred = clf.predict(Fte)
        print(f"\n=== {target} ===")
        print(classification_report(yte, pred, zero_division=0))
        classifier.save(clf, path)

        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax, xticks_rotation=90, colorbar=False)
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / f"confusion_{target}.png", dpi=150)
        plt.close()

        if target == "period":
            # Mean absolute error in years using the bin's midpoint (coarse but useful)
            def mid(label):
                return 1375 if label == "before 1400" else int(label.split("-")[0]) + 25
            err = np.mean([abs(mid(a) - mid(b)) for a, b in zip(yte, pred)])
            print(f"Mean year error (bin midpoints): {err:.0f} years")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=float, default=config.PCA_VARIANCE)
    main(ap.parse_args().components)
