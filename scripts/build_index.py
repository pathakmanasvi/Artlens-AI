"""Step 4: build the FAISS reference index of KNOWN artworks.

Uses rows where is_reference == 1 in the metadata (or all rows if the column is absent).

Usage: python -m scripts.build_index
"""
import config
from scripts.splits import load_embeddings
from src.retrieval import build_index


def main() -> None:
    X, meta = load_embeddings(config.EXTRACTOR)
    if "is_reference" in meta.columns:
        mask = meta["is_reference"].fillna(0).astype(int).values == 1
        X, meta = X[mask], meta[mask]
    cols = [c for c in ("title", "artist", "year", "style", "source", "image_path") if c in meta.columns]
    build_index(X, meta[cols])
    print(f"Indexed {len(meta)} reference artworks.")


if __name__ == "__main__":
    main()
