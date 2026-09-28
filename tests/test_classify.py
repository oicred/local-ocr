from __future__ import annotations

from grock_ocr.classify import (
    DocumentFeatures,
    coarse_type,
    match_issuer,
)
from grock_ocr.store import LearnedExample
from grock_ocr.textutil import detect_language


def _features(text: str, tokens: list[str], nifs: list[str], layout: list[float] | None = None, embedding=None):
    return DocumentFeatures(
        text=text,
        tokens=tokens,
        nifs=nifs,
        layout=layout or [1.0] + [0.0] * 15,
        embedding=embedding,
    )


def test_coarse_invoice_and_receipt():
    kind, confidence = coarse_type("Fatura FT 2024/1 Total a pagar 10,00 EUR")
    assert kind == "invoice"
    assert confidence > 0.5
    kind, _confidence = coarse_type("Recibo de pagamento no valor de 8,00 EUR")
    assert kind == "receipt"


def test_coarse_identity_passport_and_letter():
    assert coarse_type("CARTAO DE CIDADAO\nNOME JOAO SILVA")[0] == "identity"
    mrz = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"
    assert coarse_type(mrz)[0] == "passport"
    assert coarse_type("Contrato de prestacao de servicos\nClausula primeira do acordo")[0] == "contract"
    assert coarse_type("Exmo. Senhor\nAssunto: reuniao\nAtenciosamente")[0] == "letter"
    assert coarse_type("Dear Anna,\nThank you for your letter.\nSincerely")[0] == "letter"
    assert coarse_type("nothing useful here about the weather")[0] == "other"
    caixa = (
        "Comprovativo de Operacao Caixadirecta Empresas Exmo Senhor "
        "Pagamento de servicos e compras"
    )
    assert coarse_type(caixa)[0] == "bank_payment"
    assert coarse_type("Schindler NOTA DE CREDITO Original Factura Cancelada YA11")[0] == "credit_note"
    assert coarse_type("Nota de lancamento de despesa Fornecedor Garden Co Rubrica Jardim")[0] == "expense_note"
    assert coarse_type("FATURA/RECIBO ORIGINAL CONSUMO DE AGUA")[0] == "fatura_recibo"
    cgd = "Referencia ICGDPT0406 Canal Plataforma de Balcao Para credito da conta PT0035"
    assert coarse_type(cgd)[0] == "bank_payment"


def test_language_pt_and_en():
    assert detect_language("Fatura com nif e data de vencimento") == "pt"
    assert detect_language("The invoice and the vat amount due") == "en"


def test_issuer_matches_same_company_not_another():
    saved = LearnedExample(
        doc_type="invoice",
        issuer="edp",
        tokens=["comercial", "edp", "emissao", "eur"],
        nifs=["123456789"],
        layout=[1.0] + [0.0] * 15,
        anchors=[],
    )
    same = _features(
        "EDP Comercial fatura",
        ["comercial", "edp", "emissao", "eur"],
        ["123456789"],
    )
    other = _features(
        "NOS Comunicacoes recibo",
        ["comunicacoes", "emissao", "eur", "nos"],
        ["509111222"],
    )
    match = match_issuer(same, [saved])
    assert match.issuer == "edp"
    assert match.doc_type == "invoice"
    assert match_issuer(other, [saved]).issuer is None


def test_embedding_similarity_breaks_a_tie_with_shared_words():
    saved = LearnedExample(
        doc_type="invoice",
        issuer="edp",
        tokens=["emissao", "eur"],
        nifs=[],
        layout=[0.0] * 16,
        anchors=[],
        embedding=[1.0, 0.0, 0.0],
    )
    close = _features("fatura", ["emissao", "eur", "edp"], [], layout=[0.0] * 16, embedding=[0.99, 0.1, 0.0])
    far = _features("fatura", ["emissao", "eur", "nos"], [], layout=[0.0] * 16, embedding=[0.0, 1.0, 0.0])
    assert match_issuer(close, [saved]).issuer == "edp"
    assert match_issuer(far, [saved]).issuer is None
