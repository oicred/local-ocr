"""Run built-in extractors, then learned anchors."""

from __future__ import annotations

from grock_ocr.extract.accounting import extract_bank_payment, extract_expense_note
from grock_ocr.extract.anchors import apply_anchors
from grock_ocr.extract.identity import extract_identity
from grock_ocr.extract.invoice import extract_invoice
from grock_ocr.models import FieldValue, Page, all_blocks, document_text

_INVOICE_LIKE = {"invoice", "receipt", "fatura_recibo", "credit_note"}


def extract_fields(
    pages: list[Page],
    doc_type: str,
    anchors: list[dict] | None = None,
) -> dict[str, FieldValue]:
    blocks = all_blocks(pages)
    text = document_text(pages)
    fields: dict[str, FieldValue] = {}
    if doc_type in _INVOICE_LIKE:
        fields.update(extract_invoice(blocks))
    elif doc_type == "expense_note":
        fields.update(extract_expense_note(blocks))
    elif doc_type == "bank_payment":
        fields.update(extract_bank_payment(blocks, text))
    elif doc_type in {"identity", "passport"}:
        fields.update(extract_identity(blocks, text))
    fields.update(apply_anchors(pages, anchors or []))
    return fields
