"""Public result types for local_ocr."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BBox:
    """Axis-aligned box. Origin is the top-left of the page."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class TextBlock:
    text: str
    bbox: BBox
    confidence: float = 1.0
    page: int = 1


@dataclass
class Page:
    number: int
    width: float
    height: float
    source: str
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks).strip()


@dataclass
class FieldValue:
    """One extracted field.

    ``source`` is ``anchor`` (learned template), ``pattern`` (built-in
    English/Portuguese rules), ``extractor`` (a function registered by the
    caller), or ``labeled`` (a value supplied to ``learn`` that could not be
    located in the page).
    """

    value: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class ReadResult:
    text: str
    pages: list[Page]
    language: str
    doc_type: str
    issuer: str | None
    confidence: float
    fields: dict[str, FieldValue]
    source: str
    path: str | None = None
    page: int | None = None

    def to_dict(self) -> dict:
        pages = []
        for page in self.pages:
            pages.append(
                {
                    "number": page.number,
                    "width": page.width,
                    "height": page.height,
                    "source": page.source,
                    "blocks": [
                        {
                            "text": block.text,
                            "bbox": block.bbox.to_list(),
                            "confidence": block.confidence,
                            "page": block.page,
                        }
                        for block in page.blocks
                    ],
                }
            )
        return {
            "path": self.path,
            "page": self.page,
            "text": self.text,
            "language": self.language,
            "doc_type": self.doc_type,
            "issuer": self.issuer,
            "confidence": self.confidence,
            "source": self.source,
            "fields": {name: item.to_dict() for name, item in self.fields.items()},
            "pages": pages,
        }


def document_text(pages: list[Page]) -> str:
    parts = [page.text for page in pages if page.text]
    return "\n\n".join(parts).strip()


def all_blocks(pages: list[Page]) -> list[TextBlock]:
    return [block for page in pages for block in page.blocks]
