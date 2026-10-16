"""Central configuration for ArtLens AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Paths ----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_images"
PROCESSED_DIR = DATA_DIR / "processed_images"
EMBED_DIR = DATA_DIR / "embeddings"
METADATA_CSV = DATA_DIR / "artwork_metadata.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, EMBED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Embedding model ----
# "clip" is the main extractor; "resnet50" is kept as an ImageNet baseline.
EXTRACTOR = "clip"
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
IMAGE_SIZE = 224

# ---- PCA ----
PCA_VARIANCE = 0.95          # keep components explaining this much variance
PCA_MAX_COMPONENTS = 200

# ---- Known vs unknown decision ----
# Cosine similarity of the top match above this => treat as a KNOWN artwork.
# Tune this on your own phone-photo test set (see scripts/tune_threshold.py idea in README).
KNOWN_THRESHOLD = 0.85
TOP_K = 5

# ---- Quality checks ----
BLUR_THRESHOLD = 80.0        # variance of Laplacian; lower = blurrier
GLARE_FRACTION = 0.05        # share of near-white pixels that triggers a glare warning

# ---- Period bins (century labels derived from year) ----
def year_to_period(year: int) -> str:
    """Bin a year into a label: 'before 1400', then 50-year bins ('1850-1899')."""
    year = int(year)
    if year < 1400:
        return "before 1400"
    start = year // 50 * 50
    return f"{start}-{start + 49}"

# ---- LLM ----
LLM_MODEL = "claude-sonnet-5-5"   # any vision-capable model works; swap in describe.py
LLM_MAX_TOKENS = 800

RANDOM_SEED = 42
