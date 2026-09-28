"""Light cleanup for stamped or faded scans. No extra models."""

from __future__ import annotations

from PIL import Image, ImageOps


def prepare_scan(image: Image.Image) -> Image.Image:
    """Autocontrast and a small deskew. Safe to run on every OCR page."""
    rgb = image.convert("RGB")
    gray = ImageOps.autocontrast(rgb.convert("L"), cutoff=1)
    angle = _skew_angle(gray)
    if 0.35 <= abs(angle) <= 8.0:
        gray = gray.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    return gray.convert("RGB")


def _skew_angle(gray: Image.Image) -> float:
    """Score a few small rotations by how peaked the horizontal ink projection is."""
    sample = gray
    if max(sample.size) > 900:
        sample.thumbnail((900, 900), Image.Resampling.BILINEAR)
    best_angle = 0.0
    best_score = -1.0
    for tenths in range(-40, 41, 5):
        angle = tenths / 10.0
        rotated = sample if angle == 0 else sample.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
        score = _projection_peak(rotated)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _projection_peak(gray: Image.Image) -> float:
    pixels = gray.load()
    width, height = gray.size
    rows = []
    for y in range(height):
        ink = 0
        for x in range(0, width, 2):
            if pixels[x, y] < 140:
                ink += 1
        rows.append(ink)
    if not rows:
        return 0.0
    mean = sum(rows) / len(rows)
    return sum((value - mean) ** 2 for value in rows) / len(rows)
