"""Optional recognition check. Downloads models on first run.

Set GROCK_OCR_RUN_OCR=1 to execute it.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GROCK_OCR_RUN_OCR") != "1",
    reason="set GROCK_OCR_RUN_OCR=1 to download models and run OCR",
)


def test_rapidocr_reads_a_rendered_word():
    from PIL import Image, ImageDraw

    from local_ocr.ocr import RapidOcrEngine

    image = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(image).text((20, 40), "FATURA 123", fill="black")
    blocks = RapidOcrEngine().recognize(image)
    found = " ".join(block.text for block in blocks).upper()
    assert "FATURA" in found or "123" in found
