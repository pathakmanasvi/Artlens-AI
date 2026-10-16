"""Image loading, quality checks and canvas cropping for phone photos."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

import config


def load_image(path_or_file) -> Image.Image:
    """Load from a path or file-like object (e.g. Streamlit upload) as RGB."""
    return Image.open(path_or_file).convert("RGB")


def quality_report(img: Image.Image) -> dict:
    """Return blur/glare metrics plus human-readable warnings."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_frac = float((gray > 245).mean())
    warnings = []
    if blur_score < config.BLUR_THRESHOLD:
        warnings.append("Image looks blurry. Hold the camera steady and retake.")
    if glare_frac > config.GLARE_FRACTION:
        warnings.append("Strong glare detected. Tilt the artwork or change the lighting.")
    return {"blur_score": blur_score, "glare_fraction": glare_frac, "warnings": warnings}


def auto_crop_canvas(img: Image.Image) -> Image.Image:
    """Best-effort: find the largest 4-sided contour and warp it flat.

    Falls back to the original image if nothing convincing is found.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 0.25 * h * w:
        return img
    approx = cv2.approxPolyDP(best, 0.02 * cv2.arcLength(best, True), True)
    if len(approx) != 4:
        return img

    pts = approx.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    rect = np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")
    tl, tr, br, bl = rect
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < 50 or out_h < 50:
        return img
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    warped = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(rect, dst), (out_w, out_h))
    return Image.fromarray(warped)


def preprocess(img: Image.Image, crop: bool = True) -> Image.Image:
    """Full preprocessing used at inference time."""
    return auto_crop_canvas(img) if crop else img
