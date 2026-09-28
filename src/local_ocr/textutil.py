"""Text folding, language hints, and fingerprints shared by classify and extract."""

from __future__ import annotations

import re
import unicodedata

from local_ocr.models import Page, TextBlock

_ACCENTS = str.maketrans(
    {
        "á": "a",
        "à": "a",
        "â": "a",
        "ã": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "ô": "o",
        "õ": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "ñ": "n",
    }
)

_TOKEN = re.compile(r"[a-z0-9]{3,}")
_NIF = re.compile(r"(?<!\d)(\d{9})(?!\d)")
_MRZ_LINE = re.compile(r"[A-Z0-9<]{28,44}")

_STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "for",
    "with",
    "this",
    "that",
    "from",
    "your",
    "are",
    "was",
    "you",
    "not",
    "uma",
    "para",
    "com",
    "que",
    "dos",
    "das",
    "por",
    "em",
    "nao",
    "não",
    "seu",
    "sua",
    "fatura",
    "factura",
    "invoice",
    "recibo",
    "receipt",
    "total",
    "data",
    "valor",
    "original",
    "pagar",
    "page",
    "pagina",
    "nif",
    "iva",
    "vat",
    "eur",
    "usd",
    "gbp",
    "emissao",
    "vencimento",
    "numero",
    "number",
    "date",
    "issue",
    "due",
}

_PT_MARKERS = {
    "de",
    "da",
    "do",
    "fatura",
    "factura",
    "recibo",
    "nif",
    "vencimento",
    "emissao",
    "emissão",
    "validade",
    "nome",
    "artigo",
    "quantidade",
}
_EN_MARKERS = {
    "the",
    "and",
    "invoice",
    "receipt",
    "vat",
    "due",
    "expiry",
    "amount",
    "statement",
    "dear",
}


def fold(text: str) -> str:
    """Lowercase and strip Portuguese accents without changing string length."""
    lowered = text.casefold()
    folded = lowered.translate(_ACCENTS)
    if len(folded) == len(lowered):
        return folded
    # casefold can expand a character (for example ß). Fall back to NFKD.
    normalized = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.translate(_ACCENTS)


def find_folded(haystack: str, needle: str) -> int:
    """Return the start index of ``needle`` in ``haystack`` after folding.

    The index refers to the folded strings. When folding preserves length it
    is also a valid index into ``haystack``.
    """
    return fold(haystack).find(fold(needle))


def tokens_of(text: str) -> set[str]:
    found = set(_TOKEN.findall(fold(text)))
    return {token for token in found if token not in _STOPWORDS and not token.isdigit()}


def nifs_of(text: str) -> set[str]:
    return set(_NIF.findall(text))


def detect_language(text: str) -> str:
    folded = fold(text)
    words = set(re.findall(r"[a-z]{2,}", folded))
    pt = len(words & {fold(word) for word in _PT_MARKERS})
    en = len(words & _EN_MARKERS)
    if pt == 0 and en == 0:
        return "mixed"
    if pt >= en + 1:
        return "pt"
    if en >= pt + 1:
        return "en"
    return "mixed"


def has_mrz(text: str) -> bool:
    for line in text.splitlines():
        compact = line.strip().replace(" ", "")
        if _MRZ_LINE.fullmatch(compact) and compact.count("<") >= 2:
            return True
    return False


def header_text(blocks: list[TextBlock], page_height: float, fraction: float = 0.4) -> str:
    if page_height <= 0:
        return "\n".join(block.text for block in blocks)
    limit = page_height * fraction
    chosen = [block.text for block in blocks if block.bbox.cy <= limit]
    if not chosen:
        chosen = [block.text for block in blocks[:8]]
    return "\n".join(chosen)


def layout_signature(blocks: list[TextBlock], page_height: float, bins: int = 16) -> list[float]:
    hist = [0.0] * bins
    if page_height <= 0 or not blocks:
        return hist
    for block in blocks:
        index = int(min(bins - 1, max(0, (block.bbox.cy / page_height) * bins)))
        hist[index] += 1.0
    total = sum(hist)
    if total <= 0:
        return hist
    return [value / total for value in hist]


def first_page_blocks(pages: list[Page]) -> tuple[list[TextBlock], float]:
    if not pages:
        return [], 0.0
    page = pages[0]
    return page.blocks, page.height
