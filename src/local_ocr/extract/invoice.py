"""Invoice and receipt fields in Portuguese and English."""

from __future__ import annotations

import re

from local_ocr.classify import guess_issuer, issuer_display_name
from local_ocr.extract.generic import currency_of, labeled_value
from local_ocr.models import FieldValue, TextBlock
from local_ocr.textutil import fold

INVOICE_NUMBER = [
    "numero da fatura",
    "número da fatura",
    "fatura n",
    "factura n",
    "invoice number",
    "invoice no",
    "invoice #",
]
ISSUE = [
    "data de emissao",
    "data de emissão",
    "data emissao",
    "issue date",
    "invoice date",
    "date of issue",
]
DUE = ["data de vencimento", "vencimento", "due date", "payment due"]
TOTAL = ["total a pagar", "valor a pagar", "valor total", "total due", "amount due", "grand total", "total"]
TAX = ["valor do iva", "total iva", "iva", "vat", "imposto"]
NIF_LABELS = ["nipc", "nif", "contribuinte", "tax id", "vat no", "vat number"]
STOP = INVOICE_NUMBER + ISSUE + DUE + TOTAL + TAX + NIF_LABELS + [
    "fatura",
    "factura",
    "invoice",
    "recibo",
    "receipt",
]

_TITLES = (
    "fatura",
    "factura",
    "invoice",
    "recibo",
    "receipt",
    "original",
    "data",
    "moeda",
    "date",
    "currency",
    "atcud",
)
_MONEY_CORE = re.compile(r"(\d{1,3}(?:[.\s]\d{3})*[.,]\d{2}|\d+[.,]\d{2})")


def extract_invoice(blocks: list[TextBlock]) -> dict[str, FieldValue]:
    fields: dict[str, FieldValue] = {}
    _put(fields, "invoice_number", labeled_value(blocks, INVOICE_NUMBER, "code", STOP), 0.74)
    _put(fields, "issue_date", labeled_value(blocks, ISSUE, "date", STOP), 0.74)
    _put(fields, "due_date", labeled_value(blocks, DUE, "date", STOP), 0.7)
    _put(fields, "total", labeled_value(blocks, TOTAL, "money", STOP), 0.76)
    _put(fields, "tax", labeled_value(blocks, TAX, "percent_or_money", STOP), 0.7)
    _put(fields, "nif", labeled_value(blocks, NIF_LABELS, "nif", STOP), 0.8)
    _drop_tax_if_same_as_total(fields, blocks)
    issuer = _issuer_name(blocks)
    if issuer:
        fields["issuer_name"] = FieldValue(issuer, 0.72, "pattern")
    total = fields.get("total")
    if total:
        currency = currency_of(total.value)
        if currency:
            fields["currency"] = FieldValue(currency, 0.7, "pattern")
    return fields


def _drop_tax_if_same_as_total(fields: dict[str, FieldValue], blocks: list[TextBlock]) -> None:
    tax = fields.get("tax")
    total = fields.get("total")
    if not tax or not total:
        return
    if "%" in tax.value:
        return
    if _money_core(tax.value) != _money_core(total.value):
        return
    percent = labeled_value(blocks, TAX, "percent", STOP)
    if percent and "%" in percent:
        fields["tax"] = FieldValue(percent, 0.7, "pattern")
        return
    del fields["tax"]


def _money_core(value: str) -> str:
    match = _MONEY_CORE.search(value.replace(" ", ""))
    return match.group(1) if match else value.strip()


def _issuer_name(blocks: list[TextBlock]) -> str | None:
    text = "\n".join(block.text for block in blocks)
    known = issuer_display_name(guess_issuer(text))
    if known:
        return known
    return _issuer_line(blocks)


def _issuer_line(blocks: list[TextBlock]) -> str | None:
    skipped = ("total", "nif", "emissao", "vencimento", "invoice number", "iva", "vat", "moeda", "currency")
    for block in blocks:
        folded = fold(block.text).strip()
        if any(folded.startswith(title) for title in _TITLES):
            continue
        if any(token in folded for token in skipped):
            continue
        letters = sum(character.isalpha() for character in block.text)
        if letters >= 3 and len(block.text) <= 80:
            return " ".join(block.text.split())
    return None


def _put(fields: dict[str, FieldValue], name: str, value: str | None, confidence: float) -> None:
    if value:
        fields[name] = FieldValue(value, confidence, "pattern")
