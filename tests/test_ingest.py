from __future__ import annotations

from pathlib import Path

import pytest

from grock_ocr.ingest import load_document, needs_ocr
from grock_ocr.models import BBox, TextBlock
from grock_ocr.ocr import parse_rapidocr_output
from tests.helpers import FakeOcr
from tests.pdf_bytes import blank_pdf, pages_pdf, text_pdf

INVOICE_LINE = (
    "EDP Comercial Fatura N FT 2024/123 Data de emissao 15/01/2024 "
    "Data de vencimento 15/02/2024 NIF 123456789 IVA 10,35 Total a pagar 45,20 EUR"
)


def test_short_text_needs_ocr():
    assert needs_ocr("scan")
    assert not needs_ocr(INVOICE_LINE)


def test_digital_pdf_uses_text_layer(tmp_path: Path):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(text_pdf(INVOICE_LINE))
    ocr = FakeOcr()
    pages = load_document(path, ocr.recognize)
    assert ocr.calls == 0
    assert pages[0].source == "text_layer"
    assert "FT 2024/123" in pages[0].text
    assert "123456789" in pages[0].text


def test_blank_pdf_is_rendered_for_ocr(tmp_path: Path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(blank_pdf())
    ocr = FakeOcr("Fatura lida por ocr")
    pages = load_document(path, ocr.recognize)
    assert ocr.calls == 1
    assert pages[0].source == "ocr"
    assert pages[0].text == "Fatura lida por ocr"


def test_image_uses_ocr(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "card.png"
    Image.new("RGB", (80, 40), "white").save(path)
    ocr = FakeOcr("Cartao de Cidadao")
    pages = load_document(path, ocr.recognize)
    assert ocr.calls == 1
    assert pages[0].source == "ocr"
    assert pages[0].width == 80


def test_load_one_page_skips_the_rest(tmp_path: Path):
    path = tmp_path / "two.pdf"
    path.write_bytes(pages_pdf(["PAGE ONE HAS ENOUGH TEXT FOR THE LAYER " * 3, "PAGE TWO ALSO HAS ENOUGH TEXT HERE " * 3]))
    ocr = FakeOcr()
    pages = load_document(path, ocr.recognize, page_numbers={2})
    assert ocr.calls == 0
    assert len(pages) == 1
    assert pages[0].number == 2
    assert "TWO" in pages[0].text


def test_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(path, FakeOcr().recognize)


def test_parse_rapidocr_output_boxes():
    class Output:
        txts = ("Olá fatura",)
        scores = (0.87,)
        boxes = [[[1, 2], [30, 2], [30, 12], [1, 12]]]

    blocks = parse_rapidocr_output(Output())
    assert blocks[0].text == "Olá fatura"
    assert blocks[0].confidence == pytest.approx(0.87)
    assert blocks[0].bbox == BBox(1, 2, 30, 12)


def test_parse_legacy_rows():
    rows = [([[0, 0], [8, 0], [8, 4], [0, 4]], "Total", 0.5)]
    blocks = parse_rapidocr_output(rows)
    assert blocks == [
        TextBlock(text="Total", bbox=BBox(0, 0, 8, 4), confidence=0.5, page=1)
    ]
