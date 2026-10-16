"""Step 2: PCA experiments. Produces the plots/tables for your report.

 - cumulative explained-variance curve
 - 2D PCA scatter coloured by style
 - style accuracy / macro-F1 vs number of components, and vs no PCA
 - CLIP vs ResNet50 comparison (if both embeddings exist)

Usage: python -m scripts.run_pca_experiment
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

import config
from scripts.splits import group_split, load_embeddings
from src.classifier import make_classifier
from src.pca_pipeline import explained_variance_curve


def evaluate(Xtr, ytr, Xte, yte):
    clf = make_classifier().fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def main():
    rows = []
    for extractor in ("clip", "resnet50"):
        try:
            X, meta = load_embeddings(extractor)
        except FileNotFoundError:
            print(f"Skipping {extractor}: run build_embeddings first.")
            continue

        tr, te = group_split(meta)
        Xtr, Xte = X[tr], X[te]
        ytr, yte = meta["style"].values[tr], meta["style"].values[te]

        # Baseline: no PCA
        acc, f1 = evaluate(Xtr, ytr, Xte, yte)
        rows.append({"extractor": extractor, "components": "none", "accuracy": acc, "macro_f1": f1})

        # PCA at several sizes (fit on TRAIN only)
        for n in (10, 25, 50, 100, 200):
            n = min(n, Xtr.shape[0] - 1, Xtr.shape[1])
            pca = PCA(n_components=n, random_state=config.RANDOM_SEED).fit(Xtr)
            acc, f1 = evaluate(pca.transform(Xtr), ytr, pca.transform(Xte), yte)
            rows.append({"extractor": extractor, "components": n, "accuracy": acc, "macro_f1": f1})

        if extractor == "clip":
            cum = explained_variance_curve(Xtr)
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(cum) + 1), cum)
            plt.axhline(config.PCA_VARIANCE, ls="--", c="grey")
            plt.xlabel("Components")
            plt.ylabel("Cumulative explained variance")
            plt.title("PCA explained variance (CLIP, train set)")
            plt.tight_layout()
            plt.savefig(config.REPORTS_DIR / "pca_variance_curve.png", dpi=150)
            plt.close()

            pts = PCA(n_components=2, random_state=config.RANDOM_SEED).fit(Xtr).transform(X)
            plt.figure(figsize=(7, 6))
            for style in sorted(meta["style"].unique()):
                m = (meta["style"] == style).values
                plt.scatter(pts[m, 0], pts[m, 1], s=8, label=style, alpha=0.7)
            plt.legend(fontsize=7, markerscale=2)
            plt.title("2D PCA of CLIP embeddings by style")
            plt.tight_layout()
            plt.savefig(config.REPORTS_DIR / "pca_2d_scatter.png", dpi=150)
            plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(config.REPORTS_DIR / "pca_experiment_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
