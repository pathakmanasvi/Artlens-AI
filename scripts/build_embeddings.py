"""Step 1: embed every image listed in data/artwork_metadata.csv.

CSV columns required:
  image_path (relative to data/raw_images), title, artist, year, style, source
Optional: is_reference (1 = goes into the known-artwork database)

Usage: python -m scripts.build_embeddings [--extractor clip|resnet50]
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from src.feature_extraction import embed_images


def main(extractor: str) -> None:
    meta = pd.read_csv(config.METADATA_CSV)
    meta = meta.dropna(subset=["image_path", "year", "style"]).reset_index(drop=True)
    meta["year"] = meta["year"].astype(int)

    embs, keep = [], []
    batch_paths, batch_idx = [], []

    def flush():
        if not batch_paths:
            return
        imgs = [Image.open(config.RAW_DIR / p).convert("RGB") for p in batch_paths]
        embs.append(embed_images(imgs, extractor))
        keep.extend(batch_idx)
        batch_paths.clear()
        batch_idx.clear()

    for i, p in tqdm(list(enumerate(meta["image_path"])), desc="Embedding"):
        if not (config.RAW_DIR / p).exists():
            continue
        batch_paths.append(p)
        batch_idx.append(i)
        if len(batch_paths) == 64:
            flush()
    flush()

    X = np.concatenate(embs)
    meta = meta.iloc[keep].reset_index(drop=True)
    np.save(config.EMBED_DIR / f"{extractor}_embeddings.npy", X)
    meta.to_csv(config.EMBED_DIR / f"{extractor}_meta.csv", index=False)
    print(f"Saved {X.shape} embeddings for {len(meta)} images.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default=config.EXTRACTOR, choices=["clip", "resnet50"])
    main(ap.parse_args().extractor)
