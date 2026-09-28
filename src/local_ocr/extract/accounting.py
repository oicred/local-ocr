"""Portuguese bank proofs, expense notes, and receipt-invoices."""

from __future__ import annotations

from local_ocr.extract.generic import labeled_value
from local_ocr.models import FieldValue, TextBlock
from local_ocr.textutil import fold

STOP = [
    "fornecedor",
    "rubrica",
    "valor",
    "descricao",
    "data de emissao",
    "data de vencimento",
    "n lancamento",
    "conta origem",
    "tipo",
]


def extract_expense_note(blocks: list[TextBlock]) -> dict[str, FieldValue]:
    fields: dict[str, FieldValue] = {}
    _put(fields, "supplier", labeled_value(blocks, ["fornecedor"], "text", STOP), 0.8)
    _put(fields, "total", labeled_value(blocks, ["valor"], "money", STOP), 0.8)
    _put(
        fields,
        "issue_date",
        labeled_value(blocks, ["data de emissao", "data de emissão"], "date", STOP),
        0.74,
    )
    _put(fields, "category", labeled_value(blocks, ["rubrica"], "text", STOP), 0.7)
    _put(fields, "description", labeled_value(blocks, ["descricao", "descrição"], "text", STOP), 0.65)
    if "supplier" in fields:
        fields["issuer_name"] = FieldValue(fields["supplier"].value, 0.75, "pattern")
    return fields


def extract_bank_payment(blocks: list[TextBlock], text: str) -> dict[str, FieldValue]:
    fields: dict[str, FieldValue] = {}
    _put(
        fields,
        "operation_type",
        labeled_value(
            blocks,
            ["tipo", "dados da operacao", "dados da operação"],
            "text",
            STOP + ["conta origem", "conta"],
        ),
        0.6,
    )
    folded = fold(text)
    if "transferencia sepa" in folded:
        fields["operation_type"] = FieldValue("Transferencia SEPA nacional", 0.8, "pattern")
    elif "pagamento de servicos" in folded:
        fields["operation_type"] = FieldValue("Pagamento de servicos e compras", 0.8, "pattern")
    _put(
        fields,
        "account",
        labeled_value(blocks, ["conta origem", "conta"], "code", STOP),
        0.65,
    )
    _put(
        fields,
        "issue_date",
        labeled_value(blocks, ["data de emissao", "data de emissão"], "date", STOP),
        0.7,
    )
    return fields


def _put(fields: dict[str, FieldValue], name: str, value: str | None, confidence: float) -> None:
    if value:
        fields[name] = FieldValue(value, confidence, "pattern")
