from __future__ import annotations

from grock_ocr.models import BBox, Page, TextBlock


class NoEmbed:
    def embed(self, text: str):
        return None


class FakeOcr:
    def __init__(self, text: str = "SCANNED") -> None:
        self.text = text
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        width, height = image.size
        return [
            TextBlock(
                text=self.text,
                bbox=BBox(10, 10, min(width, 200), 40),
                confidence=0.91,
                page=1,
            )
        ]


def block(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> TextBlock:
    return TextBlock(text=text, bbox=BBox(x0, y0, x1, y1), confidence=1.0, page=page)


def page(blocks: list[TextBlock], width: float = 300, height: float = 400, number: int = 1) -> Page:
    return Page(number=number, width=width, height=height, source="text_layer", blocks=blocks)
