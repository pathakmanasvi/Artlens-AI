"""Optional starter: download public-domain paintings from the Met Open Access API.

Creates data/raw_images/met/*.jpg and appends rows to data/artwork_metadata.csv.
The Met API has no 'style' field, so you must map style/movement yourself
(e.g. from the artist or period) or merge with WikiArt labels.

Usage: python -m scripts.download_met_sample --query "impressionism" --limit 100
"""
import argparse
import time

import pandas as pd
import requests

import config

BASE = "https://collectionapi.metmuseum.org/public/collection/v1"


def main(query: str, limit: int) -> None:
    out_dir = config.RAW_DIR / "met"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = requests.get(f"{BASE}/search",
                       params={"q": query, "hasImages": "true", "isPublicDomain": "true",
                               "medium": "Paintings"},
                       timeout=30).json().get("objectIDs") or []
    rows = []
    for oid in ids[:limit]:
        try:
            o = requests.get(f"{BASE}/objects/{oid}", timeout=30).json()
            url = o.get("primaryImageSmall")
            year = o.get("objectBeginDate")
            if not url or not year:
                continue
            fname = f"met/{oid}.jpg"
            (config.RAW_DIR / fname).write_bytes(requests.get(url, timeout=60).content)
            rows.append({"image_path": fname, "title": o.get("title"), "artist": o.get("artistDisplayName") or "unknown",
                         "year": year, "style": query, "source": "met", "is_reference": 1})
            time.sleep(0.15)  # be polite to the API
        except Exception as e:
            print(f"skip {oid}: {e}")

    new = pd.DataFrame(rows)
    if config.METADATA_CSV.exists():
        new = pd.concat([pd.read_csv(config.METADATA_CSV), new], ignore_index=True)
    new.to_csv(config.METADATA_CSV, index=False)
    print(f"Added {len(rows)} rows -> {config.METADATA_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    main(a.query, a.limit)
