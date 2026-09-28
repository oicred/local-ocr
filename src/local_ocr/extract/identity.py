"""Identity cards and passports, including ICAO MRZ lines."""

from __future__ import annotations

import re

from local_ocr.extract.generic import labeled_value
from local_ocr.models import FieldValue, TextBlock

NAME = ["nome / name", "full name", "given names", "surname", "apelido", "nome"]
DOCUMENT = [
    "numero do documento",
    "número do documento",
    "document number",
    "document no",
    "id no",
    "documento",
]
BIRTH = ["data de nascimento", "date of birth", "nascimento", "birth"]
EXPIRY = ["data de validade", "date of expiry", "valido ate", "válido até", "validade", "expiry"]
NATIONALITY = ["nacionalidade", "nationality"]
STOP = NAME + DOCUMENT + BIRTH + EXPIRY + NATIONALITY

_MRZ = re.compile(r"[A-Z0-9<]{28,44}")
_WEIGHTS = (7, 3, 1)


def extract_identity(blocks: list[TextBlock], text: str) -> dict[str, FieldValue]:
    fields: dict[str, FieldValue] = {}
    mrz, confidence = parse_mrz(text)
    for key, value in mrz.items():
        fields[key] = FieldValue(value, confidence, "pattern")
    _fill(fields, "full_name", labeled_value(blocks, NAME, "text", STOP), 0.72)
    _fill(fields, "document_number", labeled_value(blocks, DOCUMENT, "code", STOP), 0.72)
    _fill(fields, "birth_date", labeled_value(blocks, BIRTH, "date", STOP), 0.72)
    _fill(fields, "expiry_date", labeled_value(blocks, EXPIRY, "date", STOP), 0.72)
    _fill(fields, "nationality", labeled_value(blocks, NATIONALITY, "text", STOP), 0.66)
    return fields


def parse_mrz(text: str) -> tuple[dict[str, str], float]:
    lines = []
    for raw in text.splitlines():
        compact = raw.strip().upper().replace(" ", "")
        if _MRZ.fullmatch(compact):
            lines.append(compact)
    if len(lines) >= 2 and lines[0].startswith("P<") and len(lines[0]) >= 30 and len(lines[1]) >= 28:
        return _parse_td3(lines[0], lines[1])
    if len(lines) >= 3 and lines[0][0] in {"I", "A", "C"} and len(lines[0]) >= 28:
        return _parse_td1(lines[0], lines[1], lines[2])
    return {}, 0.0


def _parse_td3(line1: str, line2: str) -> tuple[dict[str, str], float]:
    line1 = line1.ljust(44, "<")
    line2 = line2.ljust(44, "<")
    surname, given = _split_name(line1[5:])
    full = " ".join(part for part in (given, surname) if part).strip()
    document = line2[0:9].replace("<", "")
    nationality = line2[10:13].replace("<", "")
    birth = _mrz_date(line2[13:19], birth=True)
    expiry = _mrz_date(line2[21:27], birth=False)
    valid = mrz_check(line2[0:9], line2[9])
    fields: dict[str, str] = {}
    if full:
        fields["full_name"] = full
    if document:
        fields["document_number"] = document
    if nationality:
        fields["nationality"] = nationality
    if birth:
        fields["birth_date"] = birth
    if expiry:
        fields["expiry_date"] = expiry
    return fields, 0.93 if valid else 0.7


def _parse_td1(line1: str, line2: str, line3: str) -> tuple[dict[str, str], float]:
    line1 = line1.ljust(30, "<")
    line2 = line2.ljust(30, "<")
    document = line1[5:14].replace("<", "")
    birth = _mrz_date(line2[0:6], birth=True)
    expiry = _mrz_date(line2[8:14], birth=False)
    nationality = line2[15:18].replace("<", "")
    surname, given = _split_name(line3)
    full = " ".join(part for part in (given, surname) if part).strip()
    valid = mrz_check(line1[5:14], line1[14])
    fields: dict[str, str] = {}
    if full:
        fields["full_name"] = full
    if document:
        fields["document_number"] = document
    if nationality:
        fields["nationality"] = nationality
    if birth:
        fields["birth_date"] = birth
    if expiry:
        fields["expiry_date"] = expiry
    return fields, 0.93 if valid else 0.7


def _split_name(raw: str) -> tuple[str, str]:
    primary = raw.split("<<", 1)
    surname = primary[0].replace("<", " ").strip()
    given = primary[1].replace("<", " ").strip() if len(primary) > 1 else ""
    return " ".join(surname.split()), " ".join(given.split())


def _mrz_date(value: str, birth: bool) -> str:
    if len(value) < 6 or not value[:6].isdigit():
        return ""
    year = int(value[0:2])
    month = value[2:4]
    day = value[4:6]
    century = (1900 if year >= 40 else 2000) if birth else (2000 if year < 80 else 1900)
    return f"{century + year:04d}-{month}-{day}"


def mrz_check(data: str, check: str) -> bool:
    if not check.isdigit():
        return False
    total = 0
    for index, char in enumerate(data):
        total += _char_value(char) * _WEIGHTS[index % 3]
    return str(total % 10) == check


def _char_value(char: str) -> int:
    if char == "<":
        return 0
    if char.isdigit():
        return int(char)
    if char.isalpha():
        return ord(char.upper()) - 55
    return 0


def _fill(fields: dict[str, FieldValue], name: str, value: str | None, confidence: float) -> None:
    if value and name not in fields:
        fields[name] = FieldValue(value, confidence, "pattern")
