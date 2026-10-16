"""End-to-end analysis: photo -> quality -> embed -> known? -> estimate -> description."""
from __future__ import annotations

from datetime import date

from PIL import Image

import config
from src import classifier, retrieval
from src.feature_extraction import embed_one
from src.llm_description import describe
from src.preprocessing import preprocess, quality_report


def analyze(img: Image.Image, crop: bool = True, use_llm: bool = True) -> dict:
    quality = quality_report(img)
    prepared = preprocess(img, crop=crop)
    emb = embed_one(prepared)

    matches = retrieval.search(emb)
    known = retrieval.is_known(matches)

    result: dict = {
        "quality": quality,
        "similar_works": matches,
        "identification": "known" if known else "unknown",
    }

    if known:
        top = matches[0]
        year = int(top["year"]) if str(top.get("year", "")).strip() not in ("", "nan") else None
        result.update({
            "title": top.get("title"),
            "artist": top.get("artist"),
            "year": year,
            "age_years": (date.today().year - year) if year else None,
            "style": top.get("style"),
            "confidence": top["similarity"],
            "note": "Matched against the reference database; date comes from museum metadata.",
        })
        estimate = None
    else:
        estimate = classifier.predict_style_period(emb)
        result.update({
            "style_estimate": estimate["style"],
            "period_estimate": estimate["period"],
            "confidence": estimate["period"][0][1],
            "note": "No confident match. Style/period is a visual estimate, not proof of when it was made.",
        })

    result["description"] = describe(prepared, known_match=matches[0] if known else None,
                                     estimate=estimate) if use_llm else None
    return result
