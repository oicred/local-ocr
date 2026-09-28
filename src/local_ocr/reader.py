"""Importable local OCR reader."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

from local_ocr.classify import Classification, features_from_pages, match_issuer
from local_ocr.embed import FastEmbedder
from local_ocr.extract import extract_fields
from local_ocr.extract.anchors import build_anchors
from local_ocr.ingest import aggregate_source, load_document
from local_ocr.models import FieldValue, Page, ReadResult, document_text
from local_ocr.ocr import RapidOcrEngine
from local_ocr.store import ExampleStore, LearnedExample
from local_ocr.textutil import detect_language, first_page_blocks

Extractor = Callable[[ReadResult], Mapping[str, str | FieldValue]]


class DocumentReader:
    """Read images and PDFs, classify them, and extract fields.

    Learned templates are stored under ``store_path`` (default ``~/.local_ocr``).
    Copy that folder, or pass the same path, to reuse them in another project.
    OCR and embedding models load on the first call that needs them.

    Mixed annexes: use ``read_pages`` for one result per page, and
    ``learn(..., page=3)`` to teach a single page.
    """

    def __init__(
        self,
        store_path: str | Path | None = None,
        *,
        embedder=None,
        recognizer=None,
    ) -> None:
        self.store_path = Path(store_path) if store_path is not None else Path.home() / ".local_ocr"
        self.store = ExampleStore(self.store_path)
        self.embedder = embedder if embedder is not None else FastEmbedder()
        self.recognizer = recognizer if recognizer is not None else RapidOcrEngine()
        self._extractors: list[tuple[str, str | None, Extractor]] = []

    def read(self, source: str | Path, page: int | None = None) -> ReadResult:
        """Read a file. Pass ``page`` (1-based) to classify that page only."""
        return self._read(source, taught=None, page=page)

    def read_pages(self, source: str | Path) -> list[ReadResult]:
        """OCR the file once and return one result per page."""
        path = Path(source)
        pages = load_document(path, self.recognizer.recognize)
        if not pages:
            return []
        return [self._finish(path, [item], taught=None) for item in pages]

    def read_many(self, sources: Iterable[str | Path]) -> Iterator[ReadResult]:
        for source in sources:
            yield self.read(source)

    def learn(
        self,
        source: str | Path,
        doc_type: str,
        issuer: str,
        fields: dict[str, str] | None = None,
        page: int | None = None,
    ) -> ReadResult:
        """Remember ``source`` (or one ``page``) as ``issuer`` and record field anchors."""
        if not str(doc_type).strip() or not str(issuer).strip():
            raise ValueError("doc_type and issuer are required")
        supplied = {str(key): str(value) for key, value in (fields or {}).items()}
        return self._read(
            source,
            taught=(str(doc_type).strip(), str(issuer).strip(), supplied),
            page=page,
        )

    def register_extractor(self, doc_type: str, issuer: str | None, fn: Extractor) -> None:
        """Register ``fn`` for a document type. ``issuer`` None applies to every issuer of that type.

        ``fn`` receives the ``ReadResult`` and returns a mapping of field names to
        strings or ``FieldValue``. Returned keys replace built-in and anchor values.
        """
        if not callable(fn):
            raise TypeError("extractor must be callable")
        self._extractors.append((doc_type, issuer, fn))

    def _read(
        self,
        source: str | Path,
        taught: tuple[str, str, dict[str, str]] | None,
        page: int | None,
    ) -> ReadResult:
        path = Path(source)
        wanted = {int(page)} if page is not None else None
        if page is not None and int(page) < 1:
            raise ValueError("page must be 1-based")
        pages = load_document(path, self.recognizer.recognize, page_numbers=wanted)
        if page is not None and not pages:
            raise ValueError(f"page {page} was not found in {path.name}")
        return self._finish(path, pages, taught)

    def _finish(
        self,
        path: Path,
        pages: list[Page],
        taught: tuple[str, str, dict[str, str]] | None,
    ) -> ReadResult:
        text = document_text(pages)
        language = detect_language(text)
        embedding = self.embedder.embed(_embed_text(pages, text))
        features = features_from_pages(pages, text, embedding)
        anchors: list[dict] = []
        if taught is None:
            classification = match_issuer(features, self.store.all())
            if classification.example is not None:
                anchors = list(classification.example.anchors)
        else:
            doc_type, issuer, supplied = taught
            anchors = build_anchors(pages, supplied)
            self.store.add(
                LearnedExample(
                    doc_type=doc_type,
                    issuer=issuer,
                    tokens=features.tokens,
                    nifs=features.nifs,
                    layout=features.layout,
                    anchors=anchors,
                    embedding=features.embedding,
                    preview=text[:1500],
                    source_name=path.name,
                )
            )
            classification = Classification(doc_type=doc_type, issuer=issuer, confidence=1.0)
        fields = extract_fields(pages, classification.doc_type, anchors)
        if taught is not None:
            _overlay_labeled(fields, taught[2], anchors)
        page_number = pages[0].number if len(pages) == 1 else None
        result = ReadResult(
            text=text,
            pages=pages,
            language=language,
            doc_type=classification.doc_type,
            issuer=classification.issuer,
            confidence=classification.confidence,
            fields=fields,
            source=aggregate_source(pages),
            path=str(path),
            page=page_number,
        )
        custom = self._lookup_extractor(result.doc_type, result.issuer)
        if custom is not None:
            _apply_custom(result, custom(result))
        return result

    def _lookup_extractor(self, doc_type: str, issuer: str | None) -> Extractor | None:
        exact = None
        fallback = None
        for saved_type, saved_issuer, fn in self._extractors:
            if saved_type != doc_type:
                continue
            if issuer is not None and saved_issuer == issuer:
                exact = fn
            elif saved_issuer is None:
                fallback = fn
        return exact or fallback


def _embed_text(pages: list[Page], text: str) -> str:
    blocks, _height = first_page_blocks(pages)
    page_text = "\n".join(block.text for block in blocks).strip()
    return page_text or text


def _overlay_labeled(fields: dict[str, FieldValue], supplied: dict[str, str], anchors: list[dict]) -> None:
    anchored = {str(item["field"]) for item in anchors}
    for name, value in supplied.items():
        source = "anchor" if name in anchored else "labeled"
        fields[name] = FieldValue(value, 1.0, source)


def _apply_custom(result: ReadResult, produced: Mapping[str, str | FieldValue]) -> None:
    for name, value in produced.items():
        if isinstance(value, FieldValue):
            result.fields[str(name)] = value
        else:
            result.fields[str(name)] = FieldValue(str(value), 0.9, "extractor")
