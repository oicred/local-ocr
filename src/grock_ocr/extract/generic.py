"""Shared patterns for dates, money, tax ids, and label lookup."""

from __future__ import annotations

import re

from grock_ocr.models import TextBlock
from grock_ocr.textutil import fold

MONEY = re.compile(
    r"(?:€|eur|usd|gbp|£|\$)?\s*\d{1,3}(?:[.\s,]\d{3})*(?:[.,]\d{2})\s*(?:€|eur|usd|gbp|£|\$)?",
    re.IGNORECASE,
)
DATE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+de\s+[a-z]+\s+de\s+\d{4})\b",
    re.IGNORECASE,
)
NIF = re.compile(r"(?<!\d)(\d{9})(?!\d)")
CODE = re.compile(
    r"[A-Z]{1,6}(?:[\s./-]*\d[\w./-]*)+|\d{3,}(?:\s+\d+)*",
    re.IGNORECASE,
)
PERCENT = re.compile(r"\d{1,2}(?:[.,]\d+)?\s*%")


def label_pattern(label: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in fold(label).split()]
    body = r"\s+".join(parts)
    return re.compile(rf"(?:^|[^a-z0-9]){body}(?:[^a-z0-9]|$)")


def find_label(text: str, label: str) -> tuple[int, int] | None:
    match = label_pattern(label).search(fold(text))
    if not match:
        return None
    # The pattern allows one leading boundary character. Keep the label span.
    start = match.start()
    end = match.end()
    folded = fold(text)
    while start < end and not folded[start].isalnum():
        start += 1
    while end > start and not folded[end - 1].isalnum():
        end -= 1
    if start >= end:
        return None
    return start, end


def cut_before_labels(text: str, labels: list[str], skip: str | None = None) -> str:
    folded = fold(text)
    cut = len(folded)
    for label in labels:
        if skip and fold(label) == fold(skip):
            continue
        span = find_label(folded, label)
        if span and span[0] > 0:
            cut = min(cut, span[0])
    return text[:cut] if len(folded) == len(text) else folded[:cut]


def extract_by_kind(text: str, kind: str) -> str | None:
    snippet = text.strip(" \t:.-")
    if not snippet:
        return None
    if kind == "money":
        match = MONEY.search(snippet)
        return _clean(match.group(0)) if match else None
    if kind == "date":
        match = DATE.search(snippet)
        return _clean(match.group(0)) if match else None
    if kind == "nif":
        match = NIF.search(snippet)
        return match.group(1) if match else None
    if kind == "code":
        match = CODE.search(snippet)
        return _clean(match.group(0)) if match else None
    if kind == "percent":
        percent = PERCENT.search(snippet)
        return _clean(percent.group(0)) if percent else None
    if kind == "percent_or_money":
        percent = PERCENT.search(snippet)
        if percent:
            return _clean(percent.group(0))
        money = MONEY.search(snippet)
        return _clean(money.group(0)) if money else None
    cleaned = " ".join(snippet.split())
    return cleaned[:80] if cleaned else None


def currency_of(text: str) -> str | None:
    folded = fold(text)
    if "€" in text or "eur" in folded:
        return "EUR"
    if "$" in text or "usd" in folded:
        return "USD"
    if "£" in text or "gbp" in folded:
        return "GBP"
    return None


def labeled_value(
    blocks: list[TextBlock],
    labels: list[str],
    kind: str,
    stop_labels: list[str],
) -> str | None:
    ordered = sorted(labels, key=lambda label: len(label), reverse=True)
    for label in ordered:
        for block in blocks:
            span = find_label(block.text, label)
            if not span:
                continue
            if _rejected_prefix(block.text, span[0], label):
                continue
            rest = cut_before_labels(_rest_after(block.text, span[1]), stop_labels, skip=label)
            value = extract_by_kind(rest, kind)
            if value:
                return value
            neighbor = _neighbor_text(block, blocks)
            if neighbor:
                value = extract_by_kind(neighbor, kind)
                if value:
                    return value
                if kind == "text":
                    return " ".join(neighbor.split())[:80]
    return None


def _rest_after(original: str, folded_end: int) -> str:
    folded = fold(original)
    if len(folded) == len(original):
        return original[folded_end:]
    return folded[folded_end:]


def _rejected_prefix(text: str, start: int, label: str) -> bool:
    """Skip 'subtotal' when the label is 'total'."""
    if fold(label) != "total":
        return False
    prefix = fold(text)[max(0, start - 3) : start]
    return prefix.endswith("sub")


def _neighbor_text(block: TextBlock, blocks: list[TextBlock]) -> str | None:
    height = max(block.bbox.height, 8.0)
    same_line = []
    below = []
    for other in blocks:
        if other is block:
            continue
        same = abs(other.bbox.cy - block.bbox.cy) <= height * 0.8
        if same and other.bbox.x0 >= block.bbox.x0 - 2:
            same_line.append(other)
            continue
        overlaps = other.bbox.x1 >= block.bbox.x0 and other.bbox.x0 <= block.bbox.x1 + 40
        if other.bbox.y0 >= block.bbox.y0 and overlaps:
            below.append(other)
    if same_line:
        chosen = min(same_line, key=lambda item: item.bbox.x0)
        return chosen.text
    if below:
        chosen = min(below, key=lambda item: item.bbox.y0)
        return chosen.text
    return None


def _clean(value: str) -> str:
    return " ".join(value.replace("€", " € ").split())
