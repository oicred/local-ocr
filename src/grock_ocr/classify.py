"""Coarse document type and learned issuer matching."""

from __future__ import annotations

from dataclasses import dataclass

from grock_ocr.models import Page
from grock_ocr.store import LearnedExample
from grock_ocr.textutil import (
    first_page_blocks,
    fold,
    has_mrz,
    header_text,
    layout_signature,
    nifs_of,
    tokens_of,
)

DOC_TYPES = (
    "invoice",
    "receipt",
    "fatura_recibo",
    "credit_note",
    "expense_note",
    "bank_payment",
    "identity",
    "passport",
    "contract",
    "letter",
    "bank_statement",
    "other",
)

# Phrases are matched on folded text. Longer, specific phrases weigh more.
_PHRASES: dict[str, list[tuple[str, int]]] = {
    "bank_payment": [
        ("comprovativo de operacao", 8),
        ("comprovativo", 5),
        ("caixadirecta", 6),
        ("caixa directa", 6),
        ("pagamento de servicos", 5),
        ("transferencia sepa", 6),
        ("para credito da conta", 6),
        ("plataforma de balcao", 5),
    ],
    "credit_note": [
        ("nota de credito", 8),
        ("credit note", 8),
        ("factura cancelada", 6),
        ("fatura cancelada", 6),
    ],
    "expense_note": [
        ("nota de lancamento de despesa", 10),
        ("nota de lancamento", 7),
        ("dados da despesa", 5),
    ],
    "fatura_recibo": [
        ("fatura-recibo", 8),
        ("fatura/recibo", 8),
        ("fatura recibo", 6),
        ("factura-recibo", 8),
    ],
    "invoice": [
        ("fatura", 3),
        ("factura", 3),
        ("invoice", 3),
        ("nota de debito", 2),
        ("vat invoice", 3),
    ],
    "receipt": [
        ("recibo", 3),
        ("receipt", 3),
        ("talao", 3),
        ("venda a dinheiro", 2),
    ],
    "identity": [
        ("cartao de cidadao", 6),
        ("bilhete de identidade", 6),
        ("documento de identificacao", 5),
        ("citizen card", 5),
        ("identity card", 5),
        ("carteira de identidade", 5),
    ],
    "passport": [
        ("passaporte", 5),
        ("passport", 5),
    ],
    "contract": [
        ("contrato", 3),
        ("contract", 3),
        ("acordo", 2),
        ("agreement", 2),
        ("clausula", 2),
    ],
    "letter": [
        ("exmo", 2),
        ("exma", 2),
        ("assunto", 2),
        ("atenciosamente", 2),
        ("dear ", 2),
        ("sincerely", 2),
    ],
    "bank_statement": [
        ("extrato", 4),
        ("account statement", 4),
        ("bank statement", 4),
        ("saldo disponivel", 3),
        ("movimentos", 2),
    ],
}

ISSUER_MATCH_THRESHOLD = 0.60

# Known public Portuguese issuers. First matching keyword or NIF wins.
KNOWN_ISSUERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("schindler", ("schindler",), ("502353740",)),
    ("eem", ("empresa de electricidade da madeira", "electricidade da madeira", "eem -"), ()),
    ("funchal", ("funchal.pt", "consumo de agua", "servicos camara"), ()),
    ("ctt", ("correios de portugal", "loja ctt"), ()),
    ("caixa", ("caixadirecta", "caixa directa", "caixa geral de depositos"), ()),
    ("meo", ("meoempresas", "meo empresas"), ()),
    ("irn", ("instituto dos registos", "registo predial"), ()),
    ("edp", ("edp comercial", "edp"), ()),
)

ISSUER_DISPLAY = {
    "schindler": "Schindler",
    "eem": "EEM",
    "funchal": "Camara do Funchal",
    "ctt": "CTT",
    "caixa": "Caixa",
    "meo": "MEO Empresas",
    "irn": "IRN",
    "edp": "EDP Comercial",
}


@dataclass
class DocumentFeatures:
    text: str
    tokens: list[str]
    nifs: list[str]
    layout: list[float]
    embedding: list[float] | None = None


@dataclass
class Classification:
    doc_type: str
    issuer: str | None
    confidence: float
    example: LearnedExample | None = None


def features_from_pages(pages: list[Page], text: str, embedding: list[float] | None) -> DocumentFeatures:
    blocks, height = first_page_blocks(pages)
    header = header_text(blocks, height)
    token_source = f"{header}\n{text[:1500]}"
    return DocumentFeatures(
        text=text,
        tokens=sorted(tokens_of(token_source)),
        nifs=sorted(nifs_of(text)),
        layout=layout_signature(blocks, height),
        embedding=embedding,
    )


def guess_issuer(text: str) -> str | None:
    folded = fold(text)
    found_nifs = nifs_of(text)
    for issuer, keywords, nifs in KNOWN_ISSUERS:
        if any(nif in found_nifs for nif in nifs):
            return issuer
        if any(keyword in folded for keyword in keywords):
            return issuer
    return None


def issuer_display_name(issuer: str | None) -> str | None:
    if not issuer:
        return None
    return ISSUER_DISPLAY.get(issuer, issuer.replace("_", " ").title())


def coarse_type(text: str) -> tuple[str, float]:
    folded = fold(text)
    head = folded[:700]
    scores = {name: 0 for name in _PHRASES}
    for name, phrases in _PHRASES.items():
        for phrase, weight in phrases:
            if phrase in head:
                scores[name] += weight * 2
            elif phrase in folded:
                scores[name] += weight
    if _passport_mrz(text):
        scores["passport"] += 6
    elif has_mrz(text) and scores["bank_payment"] == 0:
        scores["identity"] += 4
    if "cartao de cidadao" in folded or "bilhete de identidade" in folded:
        scores["passport"] = min(scores["passport"], 1)
        scores["identity"] += 4
    if scores["bank_payment"] >= 5:
        scores["identity"] = 0
        scores["letter"] = min(scores["letter"], 2)
    if scores["credit_note"] >= 6:
        scores["invoice"] = min(scores["invoice"], 2)
    if scores["expense_note"] >= 5:
        scores["invoice"] = min(scores["invoice"], 1)
        scores["letter"] = min(scores["letter"], 1)
    if scores["fatura_recibo"] >= 6:
        scores["invoice"] = min(scores["invoice"], 3)
        scores["receipt"] = min(scores["receipt"], 3)

    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score <= 0:
        return "other", 0.3
    if best == "receipt" and scores["invoice"] >= scores["receipt"] and scores["fatura_recibo"] < scores["invoice"]:
        best = "invoice"
        best_score = scores["invoice"]
    if best == "invoice" and scores["receipt"] > scores["invoice"] and scores["invoice"] == 0:
        best = "receipt"
    if best == "letter" and scores["bank_payment"] > 0:
        best = "bank_payment"
        best_score = max(best_score, scores["bank_payment"])
    confidence = min(0.95, 0.4 + 0.12 * best_score)
    return best, confidence


def match_issuer(
    features: DocumentFeatures,
    examples: list[LearnedExample],
    threshold: float = ISSUER_MATCH_THRESHOLD,
) -> Classification:
    doc_type, type_confidence = coarse_type(features.text)
    guessed = guess_issuer(features.text)
    best_example: LearnedExample | None = None
    best_score = 0.0
    for example in examples:
        score = _example_score(features, example)
        if score > best_score:
            best_score = score
            best_example = example
    if best_example is not None and best_score >= threshold:
        return Classification(
            doc_type=best_example.doc_type,
            issuer=best_example.issuer,
            confidence=min(0.99, best_score),
            example=best_example,
        )
    if (
        best_example is not None
        and best_score >= 0.45
        and doc_type == "other"
        and best_example.embedding
        and features.embedding
    ):
        return Classification(
            doc_type=best_example.doc_type,
            issuer=guessed,
            confidence=type_confidence,
            example=None,
        )
    return Classification(
        doc_type=doc_type,
        issuer=guessed,
        confidence=type_confidence,
        example=None,
    )


def _example_score(features: DocumentFeatures, example: LearnedExample) -> float:
    lexical = _jaccard(set(features.tokens), set(example.tokens))
    shared = set(features.nifs) & set(example.nifs)
    # A customer NIF is on every page of a condo annex. Only treat a NIF as
    # a strong hint when it is also a known supplier tax id.
    supplier = {nif for _name, _keys, nifs in KNOWN_ISSUERS for nif in nifs}
    if shared & supplier:
        lexical = max(lexical, 0.8)
    elif shared:
        lexical = min(1.0, lexical + 0.08)
    layout = _cosine(features.layout, example.layout)
    embed = None
    if features.embedding and example.embedding:
        embed = _cosine(features.embedding, example.embedding)
    if embed is None:
        return 0.65 * lexical + 0.35 * layout
    return 0.50 * embed + 0.35 * lexical + 0.15 * layout


def _passport_mrz(text: str) -> bool:
    for line in text.splitlines():
        compact = line.strip().replace(" ", "")
        if compact.startswith("P<") and len(compact) >= 30:
            return True
    return False


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_l = 0.0
    norm_r = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_l += a * a
        norm_r += b * b
    if norm_l <= 0 or norm_r <= 0:
        return 0.0
    return dot / ((norm_l ** 0.5) * (norm_r ** 0.5))
