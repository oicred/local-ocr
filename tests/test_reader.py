from __future__ import annotations

import json
from pathlib import Path

import pytest

from grock_ocr import DocumentReader, FieldValue
from grock_ocr.models import ReadResult
from tests.helpers import FakeOcr, NoEmbed
from tests.pdf_bytes import pages_pdf, text_pdf

EDP = (
    "EDP Comercial Fatura N FT 2024/123 Data de emissao 15/01/2024 "
    "Data de vencimento 15/02/2024 NIF 123456789 IVA 10,35 Total a pagar 45,20 EUR"
)
EDP_NEXT = (
    "EDP Comercial Fatura N FT 2024/124 Data de emissao 02/02/2024 "
    "Data de vencimento 02/03/2024 NIF 123456789 IVA 20,00 Total a pagar 99,00 EUR"
)
NOS = (
    "NOS Comunicacoes Recibo N RC 88 Data de emissao 02/02/2024 "
    "Total a pagar 10,00 EUR NIF 509111222"
)


def _reader(tmp_path: Path) -> DocumentReader:
    return DocumentReader(store_path=tmp_path / "store", embedder=NoEmbed(), recognizer=FakeOcr())


def test_read_digital_invoice_without_ocr(tmp_path: Path):
    path = tmp_path / "edp.pdf"
    path.write_bytes(text_pdf(EDP))
    result = _reader(tmp_path).read(path)
    assert result.source == "text_layer"
    assert result.doc_type == "invoice"
    assert result.issuer == "edp"
    assert result.language == "pt"
    assert result.fields["invoice_number"].value == "FT 2024/123"
    assert "45,20" in result.fields["total"].value
    payload = json.dumps(result.to_dict())
    assert "123456789" in payload


def test_learn_then_read_matches_issuer_and_anchor(tmp_path: Path):
    first = tmp_path / "jan.pdf"
    second = tmp_path / "feb.pdf"
    other = tmp_path / "nos.pdf"
    first.write_bytes(text_pdf(EDP))
    second.write_bytes(text_pdf(EDP_NEXT))
    other.write_bytes(text_pdf(NOS))
    reader = _reader(tmp_path)
    learned = reader.learn(
        first,
        doc_type="invoice",
        issuer="edp",
        fields={"invoice_number": "FT 2024/123", "total": "45,20", "nif": "123456789"},
    )
    assert learned.issuer == "edp"
    assert learned.fields["total"].source == "anchor"
    matched = reader.read(second)
    assert matched.issuer == "edp"
    assert matched.doc_type == "invoice"
    assert "99,00" in matched.fields["total"].value
    assert matched.fields["total"].source == "anchor"
    assert reader.read(other).issuer != "edp"


def test_register_extractor_overrides_a_field(tmp_path: Path):
    path = tmp_path / "edp.pdf"
    path.write_bytes(text_pdf(EDP))
    reader = _reader(tmp_path)

    def custom(result: ReadResult) -> dict[str, str]:
        assert result.doc_type == "invoice"
        return {"total": "FROM-EXTRACTOR"}

    reader.register_extractor("invoice", None, custom)
    result = reader.read(path)
    assert result.fields["total"] == FieldValue("FROM-EXTRACTOR", 0.9, "extractor")


def test_read_many(tmp_path: Path):
    one = tmp_path / "a.pdf"
    two = tmp_path / "b.pdf"
    one.write_bytes(text_pdf(EDP))
    two.write_bytes(text_pdf(NOS))
    results = list(_reader(tmp_path).read_many([one, two]))
    assert [item.doc_type for item in results] == ["invoice", "receipt"]


CAIXA = (
    "Comprovativo de Operacao Caixadirecta Empresas Exmo Senhor "
    "Pagamento de servicos e compras Conta origem 0000000000001"
)
SCHINDLER = (
    "Schindler Factura Original N YA11 BE01/0355785624 Data 28.01.2026 "
    "NIF 502353740 Total a pagar 337,14 IVA 22%"
)
GARDEN = (
    "Nota de lancamento de despesa "
    "Fornecedor : Garden Co "
    "Data de emissao : 01-03-2026 Rubrica : Jardim Valor 95,00 Descricao : marco 2026"
)


def test_read_pages_splits_a_mixed_annex(tmp_path: Path):
    path = tmp_path / "anexo.pdf"
    path.write_bytes(pages_pdf([CAIXA, SCHINDLER, GARDEN]))
    results = _reader(tmp_path).read_pages(path)
    assert [item.page for item in results] == [1, 2, 3]
    assert [item.doc_type for item in results] == ["bank_payment", "invoice", "expense_note"]
    assert results[0].issuer == "caixa"
    assert results[1].issuer == "schindler"
    assert results[1].fields["issuer_name"].value == "Schindler"
    assert "95,00" in results[2].fields["total"].value


def test_learn_one_page_of_a_mixed_pdf(tmp_path: Path):
    first = tmp_path / "anexo.pdf"
    later = tmp_path / "later.pdf"
    first.write_bytes(pages_pdf([CAIXA, SCHINDLER]))
    later.write_bytes(text_pdf(SCHINDLER.replace("337,14", "411,31")))
    reader = _reader(tmp_path)
    taught = reader.learn(
        first,
        doc_type="invoice",
        issuer="schindler",
        fields={"total": "337,14"},
        page=2,
    )
    assert taught.page == 2
    assert taught.issuer == "schindler"
    assert taught.doc_type == "invoice"
    matched = reader.read(later)
    assert matched.issuer == "schindler"
    assert "411,31" in matched.fields["total"].value


def test_read_missing_page(tmp_path: Path):
    path = tmp_path / "one.pdf"
    path.write_bytes(text_pdf(EDP))
    with pytest.raises(ValueError):
        _reader(tmp_path).read(path, page=9)
