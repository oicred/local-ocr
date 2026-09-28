from __future__ import annotations

import json
from pathlib import Path

import pytest

from grock_ocr.cli import main
from grock_ocr.reader import DocumentReader
from tests.helpers import FakeOcr, NoEmbed
from tests.pdf_bytes import pages_pdf

PAGE_A = (
    "EDP Comercial Fatura N FT 2024/123 Data de emissao 15/01/2024 "
    "NIF 123456789 Total a pagar 45,20 EUR"
)
PAGE_B = (
    "Schindler Factura Original NIF 502353740 Total a pagar 337,14 "
    "Data de emissao 28/01/2026"
)


def _factory(store):
    return DocumentReader(store_path=store, embedder=NoEmbed(), recognizer=FakeOcr())


def test_version_prints_package_and_schema(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "0.2.0" in text
    assert "store schema 1" in text


def test_read_pages_prints_json(tmp_path: Path, capsys):
    pdf = tmp_path / "anexo.pdf"
    pdf.write_bytes(pages_pdf([PAGE_A, PAGE_B]))
    code = main(
        ["--store", str(tmp_path / "store"), "read", str(pdf), "--pages"],
        reader_factory=_factory,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["page"] for item in payload] == [1, 2]
    assert payload[0]["doc_type"] == "invoice"
    assert payload[0]["issuer"] == "edp"
    assert "45,20" in payload[0]["fields"]["total"]["value"]
    assert payload[1]["issuer"] == "schindler"
    assert set(payload[0]) >= {"page", "doc_type", "issuer", "fields"}


def test_learn_page_from_the_command(tmp_path: Path, capsys):
    pdf = tmp_path / "anexo.pdf"
    pdf.write_bytes(pages_pdf([PAGE_A, PAGE_B]))
    code = main(
        [
            "--store",
            str(tmp_path / "store"),
            "learn",
            str(pdf),
            "--type",
            "invoice",
            "--issuer",
            "schindler",
            "--page",
            "2",
            "--field",
            "total=337,14",
        ],
        reader_factory=_factory,
    )
    assert code == 0
    learned = json.loads(capsys.readouterr().out)
    assert learned["page"] == 2
    assert learned["issuer"] == "schindler"
    assert learned["fields"]["total"]["source"] == "anchor"
