"""Field anchors learned from values the caller labels."""

from __future__ import annotations

import re
from math import hypot

from local_ocr.extract.generic import extract_by_kind, find_label
from local_ocr.models import FieldValue, Page, TextBlock
from local_ocr.textutil import fold

FIELD_KIND = {
    "invoice_number": "code",
    "issue_date": "date",
    "due_date": "date",
    "total": "money",
    "tax": "percent_or_money",
    "nif": "nif",
    "currency": "text",
    "issuer_name": "text",
    "full_name": "text",
    "document_number": "code",
    "birth_date": "date",
    "expiry_date": "date",
    "nationality": "text",
}

FIELD_LABELS: dict[str, list[str]] = {
    "invoice_number": [
        "numero da fatura",
        "fatura n",
        "factura n",
        "invoice number",
        "invoice no",
    ],
    "issue_date": ["data de emissao", "data emissao", "issue date", "invoice date"],
    "due_date": ["data de vencimento", "vencimento", "due date"],
    "total": ["total a pagar", "valor total", "total due", "amount due", "total"],
    "tax": ["valor do iva", "iva", "vat"],
    "nif": ["nif", "nipc", "contribuinte", "tax id", "vat no"],
    "full_name": ["nome", "full name", "name"],
    "document_number": ["numero do documento", "document number", "document no", "documento"],
    "birth_date": ["data de nascimento", "date of birth", "nascimento"],
    "expiry_date": ["data de validade", "validade", "date of expiry", "expiry"],
    "nationality": ["nacionalidade", "nationality"],
    "issuer_name": [],
}


def build_anchors(pages: list[Page], fields: dict[str, str]) -> list[dict]:
    anchors: list[dict] = []
    for name, raw in fields.items():
        value = str(raw).strip()
        if not value:
            continue
        located = _locate(pages, value)
        if located is None:
            continue
        page, block, span = located
        kind = FIELD_KIND.get(name, "text")
        if _value_only(block.text, value):
            label_block = _label_neighbor(block, page.blocks)
            if label_block is None:
                continue
            anchors.append(
                {
                    "field": name,
                    "label": fold(label_block.text)[:80],
                    "page": page.number,
                    "same_block": False,
                    "dx": (block.bbox.cx - label_block.bbox.cx) / max(page.width, 1.0),
                    "dy": (block.bbox.cy - label_block.bbox.cy) / max(page.height, 1.0),
                    "kind": kind,
                }
            )
            continue
        label = _label_before(block.text, span[0], name)
        if not label:
            continue
        anchors.append(
            {
                "field": name,
                "label": label,
                "page": page.number,
                "same_block": True,
                "dx": 0.0,
                "dy": 0.0,
                "kind": kind,
            }
        )
    return anchors


def apply_anchors(pages: list[Page], anchors: list[dict]) -> dict[str, FieldValue]:
    fields: dict[str, FieldValue] = {}
    by_number = {page.number: page for page in pages}
    for anchor in anchors:
        page = by_number.get(int(anchor["page"]))
        if page is None and pages:
            page = pages[0]
        if page is None:
            continue
        if anchor.get("same_block"):
            value = _apply_same_block(page.blocks, anchor)
        else:
            value = _apply_offset(page, anchor)
        if value:
            fields[str(anchor["field"])] = FieldValue(value, 0.86, "anchor")
    return fields


def _locate(pages: list[Page], value: str):
    pattern = _value_regex(value)
    found = []
    for page in pages:
        for block in page.blocks:
            match = pattern.search(fold(block.text))
            if match:
                found.append((page, block, (match.start(), match.end()), match.start()))
    if not found:
        return None
    found.sort(key=lambda item: item[3])
    page, block, span, _start = found[0]
    return page, block, span


def _value_regex(value: str) -> re.Pattern[str]:
    parts: list[str] = []
    for char in fold(value):
        if char in ".,":
            parts.append(r"[.,]")
        elif char.isspace():
            parts.append(r"\s*")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts))


def _value_only(text: str, value: str) -> bool:
    folded = fold(text)
    match = _value_regex(value).search(folded)
    if not match:
        return False
    rest = (folded[: match.start()] + folded[match.end() :]).strip(" :.-€$£")
    for token in ("eur", "usd", "gbp"):
        rest = rest.replace(token, "")
    return rest.strip() == ""


def _label_neighbor(block: TextBlock, blocks: list[TextBlock]) -> TextBlock | None:
    height = max(block.bbox.height, 8.0)
    left = []
    above = []
    for other in blocks:
        if other is block:
            continue
        same_line = abs(other.bbox.cy - block.bbox.cy) <= height * 0.8
        if same_line and other.bbox.x1 <= block.bbox.x0 + 4:
            left.append(other)
            continue
        overlaps = not (other.bbox.x1 < block.bbox.x0 - 20 or other.bbox.x0 > block.bbox.x1 + 20)
        if other.bbox.y1 <= block.bbox.y0 + 4 and overlaps:
            above.append(other)
    if left:
        return max(left, key=lambda item: item.bbox.x1)
    if above:
        return max(above, key=lambda item: item.bbox.y1)
    return None


def _label_before(text: str, value_start: int, field: str) -> str | None:
    prefix = fold(text)[:value_start]
    best = None
    best_at = -1
    for label in FIELD_LABELS.get(field, []):
        span = find_label(prefix, label)
        if span and span[1] >= best_at:
            best = fold(label)
            best_at = span[1]
    if best:
        return best
    words = [word for word in prefix.strip().split() if word]
    if not words:
        return None
    return " ".join(words[-3:])


def _apply_same_block(blocks: list[TextBlock], anchor: dict) -> str | None:
    label = str(anchor["label"])
    kind = str(anchor.get("kind") or "text")
    for block in blocks:
        span = find_label(block.text, label)
        if not span and fold(label) not in fold(block.text):
            continue
        if span:
            rest = fold(block.text)[span[1] :]
        else:
            folded = fold(block.text)
            rest = folded[folded.find(fold(label)) + len(fold(label)) :]
        value = extract_by_kind(rest, kind)
        if value:
            return value
    return None


def _apply_offset(page: Page, anchor: dict) -> str | None:
    label = str(anchor["label"])
    kind = str(anchor.get("kind") or "text")
    hits = [block for block in page.blocks if find_label(block.text, label) or fold(block.text).strip() == fold(label)]
    best: TextBlock | None = None
    best_dist = 10.0
    width = max(page.width, 1.0)
    height = max(page.height, 1.0)
    for label_block in hits:
        pred_x = label_block.bbox.cx + float(anchor.get("dx") or 0) * width
        pred_y = label_block.bbox.cy + float(anchor.get("dy") or 0) * height
        for other in page.blocks:
            if other is label_block:
                continue
            dist = hypot((other.bbox.cx - pred_x) / width, (other.bbox.cy - pred_y) / height)
            if dist < best_dist:
                best_dist = dist
                best = other
    if best is None or best_dist > 0.15:
        return None
    return extract_by_kind(best.text, kind) or " ".join(best.text.split())[:80]
