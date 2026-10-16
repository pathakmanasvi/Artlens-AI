# ArtLens AI

Photograph a painting or drawing and get: known-artwork identification (or a style/period estimate), approximate age, and an LLM-written description.

## Pipeline

```
Photo -> quality check (blur/glare) -> canvas crop -> CLIP embedding
   |-- FAISS search on reference DB -> similarity >= threshold? -> KNOWN: title, artist, year from metadata
   '-- otherwise: PCA -> SVM (style + 50-year period bin) -> UNKNOWN: estimate + confidence
Vision LLM (image + metadata or estimate) -> description JSON
Streamlit UI
```

Design choices worth defending in your report:
- **CLIP** is the main extractor; **ResNet50** is the ImageNet baseline for comparison.
- **PCA** is used for the classifiers (fit on train only). Retrieval uses raw embeddings, since PCA can discard fine detail. Test both.
- **Artist-grouped splits** prevent leakage between train and test.
- Unknown-artwork dates are **style estimates, not proof of age**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add ANTHROPIC_API_KEY
```

## Data

Create `data/artwork_metadata.csv` with columns:

| column | meaning |
|---|---|
| image_path | path relative to `data/raw_images/` |
| title, artist | text (artist is used for leakage-safe splitting) |
| year | integer creation year |
| style | label such as Impressionism, Baroque... |
| source | wikiart / met / aic |
| is_reference | 1 = include in the known-artwork index |

Sources: WikiArt (style labels), Met Open Access and Art Institute of Chicago APIs (dates, metadata). `scripts/download_met_sample.py` is a starter for the Met API. Check each source's license. Aim for 3-5k images across 5-8 styles plus ~500-1000 reference paintings, and add some non-Western works if you can.

## Run order

```bash
python -m scripts.build_embeddings --extractor clip
python -m scripts.build_embeddings --extractor resnet50     # baseline
python -m scripts.run_pca_experiment                        # plots + results table -> reports/
python -m scripts.train_classifiers                          # saves PCA + classifiers
python -m scripts.build_index                               # FAISS reference index
streamlit run app.py
```

## Evaluation checklist

- Style/period: accuracy, macro-F1, confusion matrix, mean year error (`train_classifiers`)
- PCA: variance curve, with vs without PCA, CLIP vs ResNet50 (`run_pca_experiment`)
- Retrieval: Top-1 / Top-5 accuracy on held-out photos of reference paintings
- **Phone-photo test set**: 50-100 of your own photos of printed/framed art. Use it to tune `KNOWN_THRESHOLD` in `config.py`
- LLM-only baseline: send the same images to a vision LLM with no pipeline and compare period accuracy and hallucination rate

## Layout

```
app.py                 Streamlit UI
config.py              paths, thresholds, model names
src/                   preprocessing, feature_extraction, pca_pipeline, classifier,
                       retrieval, llm_description, pipeline
scripts/               build_embeddings, splits, run_pca_experiment, train_classifiers,
                       build_index, download_met_sample
data/ models/ reports/ generated artefacts (git-ignored)
notebooks/             exploration (empty)
```
