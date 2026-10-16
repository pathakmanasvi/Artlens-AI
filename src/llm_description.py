"""Vision-LLM description, grounded in retrieved metadata or the classifier's estimate."""
from __future__ import annotations

import base64
import io
import json
import os

from dotenv import load_dotenv
from PIL import Image

import config

load_dotenv()

SYSTEM = (
    "You are an art historian assistant. Describe only what is visible in the image. "
    "Use the provided context as ground truth when present; if it conflicts with what you see, say so. "
    "If something is uncertain, say it is uncertain. Never invent titles, dates or artists. "
    "Reply with a single JSON object and nothing else."
)

SCHEMA_HINT = (
    '{"subject": str, "composition": str, "colors_and_technique": str, '
    '"mood": str, "historical_context": str, "uncertainty_notes": str}'
)


def _to_b64(img: Image.Image, max_side: int = 1024) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def build_context(known_match: dict | None, estimate: dict | None) -> str:
    if known_match:
        return (
            "VERIFIED MATCH from reference database: "
            f"title={known_match.get('title')}, artist={known_match.get('artist')}, "
            f"year={known_match.get('year')}, style={known_match.get('style')}."
        )
    if estimate:
        style = ", ".join(f"{l} ({p:.0%})" for l, p in estimate["style"])
        period = ", ".join(f"{l} ({p:.0%})" for l, p in estimate["period"])
        return (
            "NO KNOWN MATCH. Classifier estimates (not verified): "
            f"style: {style}; period: {period}. Confirm or correct from the visuals."
        )
    return "No extra context."


def describe(img: Image.Image, known_match: dict | None = None, estimate: dict | None = None) -> dict:
    """Return the description as a dict. Falls back gracefully if no API key / error."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY not set. Add it to .env to enable descriptions."}
    try:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": _to_b64(img)}},
                    {"type": "text",
                     "text": f"{build_context(known_match, estimate)}\n\nDescribe this artwork as JSON with keys: {SCHEMA_HINT}"},
                ],
            }],
        )
        text = msg.content[0].text.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Model returned non-JSON output.", "raw": text}
    except Exception as e:  # network, auth, rate limit...
        return {"error": f"LLM call failed: {e}"}
