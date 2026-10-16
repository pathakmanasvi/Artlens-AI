"""Turn images into embedding vectors with CLIP (main) or ResNet50 (baseline)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import config

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_cache: dict = {}


def _load_clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model.eval().to(_DEVICE)
    return model, preprocess


def _load_resnet():
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()          # output = 2048-d feature vector
    model.eval().to(_DEVICE)
    return model, weights.transforms()


def get_extractor(name: str | None = None):
    name = name or config.EXTRACTOR
    if name not in _cache:
        _cache[name] = _load_clip() if name == "clip" else _load_resnet()
    return _cache[name]


@torch.no_grad()
def embed_images(images: list[Image.Image], name: str | None = None, batch_size: int = 32) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalised embeddings."""
    name = name or config.EXTRACTOR
    model, tfm = get_extractor(name)
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([tfm(im.convert("RGB")) for im in images[i:i + batch_size]]).to(_DEVICE)
        feats = model.encode_image(batch) if name == "clip" else model(batch)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


def embed_one(img: Image.Image, name: str | None = None) -> np.ndarray:
    return embed_images([img], name)
