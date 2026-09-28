from __future__ import annotations

from local_ocr.extract.accounting import extract_expense_note
from local_ocr.extract.anchors import apply_anchors, build_anchors
from local_ocr.extract.identity import extract_identity, mrz_check
from local_ocr.extract.invoice import extract_invoice
from tests.helpers import block, page

PT_LINE = (
    "EDP Comercial Fatura N FT 2024/123 Data de emissão 15/01/2024 "
    "Data de vencimento 15/02/2024 NIF 123456789 IVA 10,35 Total a pagar 45,20 EUR"
)


def test_portuguese_invoice_line():
    fields = extract_invoice([block(PT_LINE, 0, 0, 500, 20)])
    assert fields["invoice_number"].value == "FT 2024/123"
    assert fields["issue_date"].value == "15/01/2024"
    assert fields["due_date"].value == "15/02/2024"
    assert fields["nif"].value == "123456789"
    assert "45,20" in fields["total"].value
    assert "10,35" in fields["tax"].value
    assert fields["currency"].value == "EUR"
    assert fields["total"].source == "pattern"
    assert fields["issuer_name"].value == "EDP Comercial"


def test_tax_is_not_copied_from_the_same_total():
    fields = extract_invoice(
        [block("MEO Empresas IVA 167,66 Total a pagar 167,66 EUR", 0, 0, 400, 20)]
    )
    assert "167,66" in fields["total"].value
    assert "tax" not in fields


def test_schindler_issuer_not_madeira():
    fields = extract_invoice(
        [
            block("Madeira Rampa do Pico", 10, 10, 200, 28),
            block("Schindler Factura Original Total a pagar 337,14 IVA 22%", 10, 40, 400, 60),
        ]
    )
    assert fields["issuer_name"].value == "Schindler"
    assert fields["tax"].value.startswith("22")


def test_expense_note_fields():
    fields = extract_expense_note(
        [
            block("Nota de lancamento de despesa", 10, 10, 220, 28),
            block("Fornecedor : Garden Co", 10, 40, 280, 58),
            block("Rubrica : Jardim", 10, 70, 180, 88),
            block("Valor : 95,00", 10, 100, 160, 118),
            block("Descricao : marco 2026", 10, 130, 220, 148),
        ]
    )
    assert fields["supplier"].value.startswith("Garden Co")
    assert "95,00" in fields["total"].value
    assert fields["category"].value == "Jardim"


def test_english_invoice_blocks():
    blocks = [
        block("Acme Ltd", 10, 10, 120, 28),
        block("Invoice number INV-99", 10, 40, 220, 58),
        block("Issue date 2024-03-02", 10, 70, 220, 88),
        block("Due date 2024-03-30", 10, 100, 220, 118),
        block("VAT 4.00", 10, 130, 120, 148),
        block("Total due 24.00 USD", 10, 160, 220, 178),
    ]
    fields = extract_invoice(blocks)
    assert fields["issuer_name"].value == "Acme Ltd"
    assert fields["invoice_number"].value == "INV-99"
    assert fields["issue_date"].value == "2024-03-02"
    assert fields["due_date"].value == "2024-03-30"
    assert "24.00" in fields["total"].value
    assert fields["currency"].value == "USD"
    assert "4.00" in fields["tax"].value


def test_identity_labels():
    blocks = [
        block("NOME", 10, 10, 80, 28),
        block("JOAO SILVA", 10, 34, 180, 52),
        block("DATA DE NASCIMENTO", 10, 70, 180, 88),
        block("01/02/1990", 10, 94, 120, 112),
        block("DOCUMENTO", 10, 130, 140, 148),
        block("12345678 9 ZZ4", 10, 154, 220, 172),
        block("VALIDADE", 10, 190, 120, 208),
        block("01/02/2030", 10, 214, 140, 232),
        block("NACIONALIDADE", 10, 250, 160, 268),
        block("PRT", 10, 274, 80, 292),
    ]
    fields = extract_identity(blocks, "\n".join(item.text for item in blocks))
    assert fields["full_name"].value == "JOAO SILVA"
    assert fields["birth_date"].value == "01/02/1990"
    assert fields["expiry_date"].value == "01/02/2030"
    assert fields["nationality"].value == "PRT"
    assert "12345678" in fields["document_number"].value


def test_passport_mrz():
    text = "\n".join(
        [
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
        ]
    )
    assert mrz_check("L898902C3", "6")
    fields = extract_identity([], text)
    assert fields["full_name"].value == "ANNA MARIA ERIKSSON"
    assert fields["document_number"].value == "L898902C3"
    assert fields["nationality"].value == "UTO"
    assert fields["birth_date"].value == "1974-08-12"
    assert fields["expiry_date"].value == "2012-04-15"
    assert fields["full_name"].confidence > 0.9


def test_identity_card_mrz():
    text = "\n".join(
        [
            "I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
        ]
    )
    fields = extract_identity([], text)
    assert fields["full_name"].value == "ANNA MARIA ERIKSSON"
    assert fields["document_number"].value == "D23145890"
    assert fields["nationality"].value == "UTO"
    assert fields["birth_date"].value == "1974-08-12"
    assert fields["expiry_date"].value == "2012-04-15"


def test_same_block_anchor_reads_a_new_total():
    taught = page([block("Total a pagar 45,20 EUR", 10, 40, 240, 60)])
    anchors = build_anchors([taught], {"total": "45,20"})
    assert anchors and anchors[0]["same_block"] is True
    fresh = page([block("Total a pagar 99,00 EUR", 10, 40, 240, 60)])
    fields = apply_anchors([fresh], anchors)
    assert "99,00" in fields["total"].value
    assert fields["total"].source == "anchor"


def test_separate_block_anchor_uses_the_label_position():
    taught = page(
        [
            block("Total", 10, 100, 70, 120),
            block("45,20 EUR", 90, 100, 190, 120),
        ]
    )
    anchors = build_anchors([taught], {"total": "45,20"})
    assert anchors and anchors[0]["same_block"] is False
    fresh = page(
        [
            block("Total", 10, 100, 70, 120),
            block("12,50 EUR", 90, 100, 190, 120),
        ]
    )
    fields = apply_anchors([fresh], anchors)
    assert "12,50" in fields["total"].value
