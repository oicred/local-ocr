from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from grock_ocr.store import SCHEMA_VERSION, ExampleStore

_EXAMPLES = """
CREATE TABLE examples (
    id INTEGER PRIMARY KEY,
    doc_type TEXT NOT NULL,
    issuer TEXT NOT NULL,
    preview TEXT NOT NULL DEFAULT '',
    embedding BLOB,
    tokens TEXT NOT NULL,
    nifs TEXT NOT NULL,
    layout TEXT NOT NULL,
    anchors TEXT NOT NULL,
    source_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
)
"""


def test_store_without_a_version_row_still_loads(tmp_path: Path):
    db_path = tmp_path / "examples.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(_EXAMPLES)
    conn.execute(
        """
        INSERT INTO examples (
            doc_type, issuer, preview, embedding, tokens, nifs, layout,
            anchors, source_name, created_at
        ) VALUES ('invoice', 'edp', '', NULL, '[]', '[]', '[]', '[]', 'old.pdf', '2026-01-01')
        """
    )
    conn.commit()
    conn.close()

    store = ExampleStore(tmp_path)
    assert store.schema_version == SCHEMA_VERSION == 1
    saved = store.all()
    assert saved[0].issuer == "edp"
    assert saved[0].doc_type == "invoice"


def test_newer_store_is_rejected(tmp_path: Path):
    store = ExampleStore(tmp_path)
    conn = sqlite3.connect(store.db_path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        ExampleStore(tmp_path)
