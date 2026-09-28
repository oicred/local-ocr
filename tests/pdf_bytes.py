"""Minimal PDF bytes with a single Helvetica text line."""

from __future__ import annotations


def text_pdf(text: str) -> bytes:
    """Build a one-page PDF whose text layer is ``text`` (WinAnsi / ASCII)."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 700 Td ({safe}) Tj ET".encode("ascii")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        (
            f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    return _assemble(objects)


def pages_pdf(texts: list[str]) -> bytes:
    """Build a multi-page digital PDF, one Helvetica line per page."""
    kids = []
    objects: dict[int, bytes] = {}
    font_id = 3 + 2 * len(texts)
    objects[1] = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    next_id = 3
    for text in texts:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 700 Td ({safe}) Tj ET".encode("ascii")
        page_id = next_id
        content_id = next_id + 1
        kids.append(f"{page_id} 0 R")
        objects[page_id] = (
            f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>\nendobj\n"
        ).encode("ascii")
        objects[content_id] = (
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        )
        next_id += 2
    objects[2] = (
        f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(texts)} >>\nendobj\n"
    ).encode("ascii")
    objects[font_id] = (
        f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ).encode("ascii")
    ordered = [objects[key] for key in sorted(objects)]
    return _assemble(ordered)


def blank_pdf() -> bytes:
    stream = b""
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << >> >>\nendobj\n"
        ),
        (
            f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        ),
    ]
    return _assemble(objects)


def _assemble(objects: list[bytes]) -> bytes:
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    chunks = []
    offsets = []
    cursor = len(header)
    for body in objects:
        offsets.append(cursor)
        chunks.append(body)
        cursor += len(body)
    xref_at = cursor
    size = len(objects) + 1
    xref = [f"xref\n0 {size}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")
    )
    return header + b"".join(chunks) + b"".join(xref) + trailer
