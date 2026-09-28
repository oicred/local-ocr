"""Load images and PDFs into pages of text blocks.

Digital PDF pages keep their text layer. A page with almost no real text is
rendered and passed to the OCR callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageSequence

from local_ocr.models import BBox, Page, TextBlock

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
PDF_SUFFIX = ".pdf"
MIN_TEXT_LAYER_ALNUM = 40
MAX_RENDER_SIDE = 4000
RENDER_DPI = 300


@dataclass
class _Char:
    ch: str
    x0: float
    y0: float
    x1: float
    y1: float


def alnum_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def needs_ocr(text: str) -> bool:
    return alnum_count(text) < MIN_TEXT_LAYER_ALNUM


def load_document(
    path: str | Path,
    recognize,
    page_numbers: set[int] | None = None,
) -> list[Page]:
    """Read ``path`` into pages.

    ``recognize`` accepts a PIL image and returns ``TextBlock`` items in
    image coordinates. Page numbers are assigned here. ``page_numbers`` is
    a 1-based set; other pages are skipped.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == PDF_SUFFIX:
        return _load_pdf(source, recognize, page_numbers)
    if suffix in IMAGE_SUFFIXES:
        return _load_image(source, recognize, page_numbers)
    raise ValueError(f"Unsupported file type '{suffix}'. Use a PDF or an image.")


def _wanted(number: int, page_numbers: set[int] | None) -> bool:
    return page_numbers is None or number in page_numbers


def _load_image(path: Path, recognize, page_numbers: set[int] | None) -> list[Page]:
    pages: list[Page] = []
    with Image.open(path) as image:
        frames = [frame.copy() for frame in ImageSequence.Iterator(image)]
    if not frames:
        frames = [Image.open(path)]
    for index, frame in enumerate(frames, start=1):
        if not _wanted(index, page_numbers):
            continue
        rgb = frame.convert("RGB")
        width, height = rgb.size
        blocks = [_with_page(block, index) for block in recognize(rgb)]
        pages.append(
            Page(
                number=index,
                width=float(width),
                height=float(height),
                source="ocr",
                blocks=blocks,
            )
        )
    return pages


def _load_pdf(path: Path, recognize, page_numbers: set[int] | None) -> list[Page]:
    import pypdfium2 as pdfium

    pages: list[Page] = []
    document = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(document)):
            number = index + 1
            if not _wanted(number, page_numbers):
                continue
            page = document[index]
            try:
                pages.append(_load_pdf_page(page, number, recognize))
            finally:
                page.close()
    finally:
        document.close()
    return pages


def _load_pdf_page(page, number: int, recognize) -> Page:
    width, height = page.get_size()
    text, blocks = _text_layer(page, number, float(width), float(height))
    if not needs_ocr(text):
        return Page(
            number=number,
            width=float(width),
            height=float(height),
            source="text_layer",
            blocks=blocks,
        )
    image = _render_page(page)
    pixel_width, pixel_height = image.size
    ocr_blocks = [_with_page(block, number) for block in recognize(image)]
    return Page(
        number=number,
        width=float(pixel_width),
        height=float(pixel_height),
        source="ocr",
        blocks=ocr_blocks,
    )


def _render_page(page):
    width, height = page.get_size()
    scale = RENDER_DPI / 72
    longest = max(float(width), float(height), 1.0) * scale
    if longest > MAX_RENDER_SIDE:
        scale = MAX_RENDER_SIDE / max(float(width), float(height), 1.0)
    bitmap = page.render(scale=scale)
    try:
        return bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()


def _text_layer(page, number: int, width: float, height: float) -> tuple[str, list[TextBlock]]:
    textpage = page.get_textpage()
    try:
        count = textpage.count_chars()
        chars: list[_Char] = []
        for index in range(count):
            piece = textpage.get_text_range(index, 1) or ""
            if not piece or piece == "\x00":
                continue
            try:
                box = textpage.get_charbox(index)
            except RuntimeError:
                continue
            left, bottom, right, top = (float(value) for value in box)
            chars.append(
                _Char(
                    ch=piece,
                    x0=left,
                    y0=height - top,
                    x1=right,
                    y1=height - bottom,
                )
            )
        blocks = _blocks_from_chars(chars, number)
        if blocks:
            return "\n".join(block.text for block in blocks), blocks
        fallback = textpage.get_text_bounded() or ""
        fallback = fallback.replace("\r", "\n").strip()
        if not fallback:
            return "", []
        block = TextBlock(
            text=fallback,
            bbox=BBox(0.0, 0.0, width, min(height, 40.0)),
            confidence=1.0,
            page=number,
        )
        return fallback, [block]
    finally:
        textpage.close()


def _blocks_from_chars(chars: list[_Char], page_number: int) -> list[TextBlock]:
    visible = [item for item in chars if item.ch.strip() or item.ch.isspace()]
    if not visible:
        return []
    ordered = sorted(visible, key=lambda item: ((item.y0 + item.y1) / 2, item.x0))
    lines: list[list[_Char]] = []
    line_cy = 0.0
    line_h = 0.0
    for item in ordered:
        cy = (item.y0 + item.y1) / 2
        height = max(item.y1 - item.y0, 1.0)
        if not lines or abs(cy - line_cy) > max(line_h * 0.6, 2.0):
            lines.append([item])
            line_cy = cy
            line_h = height
        else:
            lines[-1].append(item)
            count = len(lines[-1])
            line_cy = (line_cy * (count - 1) + cy) / count
            line_h = max(line_h, height)

    blocks: list[TextBlock] = []
    for line in lines:
        text = _line_text(line)
        if not text.strip():
            continue
        blocks.append(
            TextBlock(
                text=text,
                bbox=BBox(
                    min(item.x0 for item in line),
                    min(item.y0 for item in line),
                    max(item.x1 for item in line),
                    max(item.y1 for item in line),
                ),
                confidence=1.0,
                page=page_number,
            )
        )
    return blocks


def _line_text(chars: list[_Char]) -> str:
    ordered = sorted(chars, key=lambda item: item.x0)
    # Glyph boxes are tighter than the advance width, so a narrow character
    # such as "1" or "i" looks like a word gap. When the PDF already stored
    # space characters, trust those. Otherwise use the median glyph width.
    has_space = any(item.ch.isspace() for item in ordered)
    widths = [item.x1 - item.x0 for item in ordered if not item.ch.isspace() and item.x1 > item.x0]
    median_width = sorted(widths)[len(widths) // 2] if widths else 4.0
    gap_limit = max(median_width * 0.8, 1.0)
    parts: list[str] = []
    previous: _Char | None = None
    for item in ordered:
        if item.ch.isspace():
            if parts and not parts[-1].endswith(" "):
                parts.append(" ")
            previous = item
            continue
        if not has_space and previous is not None and not (parts and parts[-1].endswith(" ")):
            gap = item.x0 - previous.x1
            if gap > gap_limit:
                parts.append(" ")
        parts.append(item.ch)
        previous = item
    return " ".join("".join(parts).split())


def _with_page(block: TextBlock, page_number: int) -> TextBlock:
    return TextBlock(
        text=block.text,
        bbox=block.bbox,
        confidence=block.confidence,
        page=page_number,
    )


def aggregate_source(pages: list[Page]) -> str:
    sources = {page.source for page in pages}
    if sources == {"text_layer"}:
        return "text_layer"
    if sources == {"ocr"}:
        return "ocr"
    if not sources:
        return "ocr"
    return "mixed"
