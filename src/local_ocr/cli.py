"""Command-line entry for a copied local-ocr folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from local_ocr import __version__
from local_ocr.models import ReadResult
from local_ocr.reader import DocumentReader
from local_ocr.store import SCHEMA_VERSION


def main(argv: list[str] | None = None, reader_factory=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        reader = _reader(args.store, reader_factory)
        if args.command == "read":
            payload = _read(reader, args)
        else:
            payload = _learn(reader, args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-ocr")
    parser.add_argument(
        "--version",
        action="version",
        version=f"local-ocr {__version__} (store schema {SCHEMA_VERSION})",
    )
    parser.add_argument("--store", type=Path, default=None, help="folder for learned templates")
    commands = parser.add_subparsers(dest="command", required=True)

    read_cmd = commands.add_parser("read", help="recognize a PDF or image")
    read_cmd.add_argument("file", type=Path)
    read_cmd.add_argument("--pages", action="store_true", help="one JSON object per page")
    read_cmd.add_argument("--page", type=int, default=None, help="1-based page to read")

    learn_cmd = commands.add_parser("learn", help="remember one file or page")
    learn_cmd.add_argument("file", type=Path)
    learn_cmd.add_argument("--type", required=True, dest="doc_type")
    learn_cmd.add_argument("--issuer", required=True)
    learn_cmd.add_argument("--page", type=int, default=None)
    learn_cmd.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="labeled field, repeat for each one",
    )
    return parser


def _reader(store: Path | None, reader_factory):
    if reader_factory is not None:
        return reader_factory(store)
    return DocumentReader(store_path=store)


def _read(reader: DocumentReader, args) -> dict | list[dict]:
    if args.pages and args.page is not None:
        raise ValueError("use either --pages or --page")
    if args.pages:
        return [_payload(item) for item in reader.read_pages(args.file)]
    return _payload(reader.read(args.file, page=args.page))


def _learn(reader: DocumentReader, args) -> dict:
    fields = dict(_parse_field(item) for item in args.field)
    result = reader.learn(
        args.file,
        doc_type=args.doc_type,
        issuer=args.issuer,
        fields=fields,
        page=args.page,
    )
    return _payload(result)


def _parse_field(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"field must be name=value, got {item!r}")
    name, value = item.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"field must be name=value, got {item!r}")
    return name, value.strip()


def _payload(result: ReadResult) -> dict:
    return {
        "page": result.page,
        "doc_type": result.doc_type,
        "issuer": result.issuer,
        "fields": {name: item.to_dict() for name, item in result.fields.items()},
    }
