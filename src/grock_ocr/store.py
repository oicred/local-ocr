"""SQLite store of labeled document examples. Nothing leaves the machine."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
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

_META = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


@dataclass
class LearnedExample:
    doc_type: str
    issuer: str
    tokens: list[str]
    nifs: list[str]
    layout: list[float]
    anchors: list[dict]
    embedding: list[float] | None = None
    preview: str = ""
    source_name: str = ""
    id: int | None = None


class ExampleStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "examples.sqlite"
        self.schema_version = SCHEMA_VERSION
        with self._connect() as conn:
            self._ensure_version(conn)

    def add(self, example: LearnedExample) -> int:
        embedding = _pack(example.embedding)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO examples (
                    doc_type, issuer, preview, embedding, tokens, nifs, layout,
                    anchors, source_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    example.doc_type,
                    example.issuer,
                    example.preview[:2000],
                    embedding,
                    json.dumps(example.tokens),
                    json.dumps(example.nifs),
                    json.dumps(example.layout),
                    json.dumps(example.anchors),
                    example.source_name,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def all(self) -> list[LearnedExample]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, doc_type, issuer, preview, embedding, tokens, nifs,
                       layout, anchors, source_name
                FROM examples
                ORDER BY id
                """
            ).fetchall()
        return [_row_to_example(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_version(self, conn: sqlite3.Connection) -> None:
        conn.execute(_SCHEMA)
        conn.execute(_META)
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.schema_version = SCHEMA_VERSION
            return
        version = int(row["value"])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Store schema {version} is newer than this grock-ocr "
                f"(schema {SCHEMA_VERSION}). Upgrade grock-ocr before opening this store."
            )
        if version < SCHEMA_VERSION:
            version = _migrate(conn, version)
        self.schema_version = version


def _migrate(conn: sqlite3.Connection, version: int) -> int:
    """Upgrade a store written by an older grock-ocr. Version 1 is the current layout."""
    if version < 1:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        return SCHEMA_VERSION
    raise RuntimeError(
        f"No migration from store schema {version} to {SCHEMA_VERSION}."
    )


def _pack(values: list[float] | None) -> bytes | None:
    if not values:
        return None
    return np.asarray(values, dtype=np.float32).tobytes()


def _unpack(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    array = np.frombuffer(blob, dtype=np.float32)
    return [float(value) for value in array]


def _row_to_example(row: sqlite3.Row) -> LearnedExample:
    return LearnedExample(
        id=int(row["id"]),
        doc_type=row["doc_type"],
        issuer=row["issuer"],
        preview=row["preview"] or "",
        embedding=_unpack(row["embedding"]),
        tokens=json.loads(row["tokens"]),
        nifs=json.loads(row["nifs"]),
        layout=json.loads(row["layout"]),
        anchors=json.loads(row["anchors"]),
        source_name=row["source_name"] or "",
    )
