# Changelog

## 0.2.0

- `grock-ocr read` and `grock-ocr learn` run the package from the command line, including one result per page.
- `grock-ocr --version` prints the package version and the store schema version.
- The learned-template database records schema version 1. A store created by 0.1.0, which has no version row, is stamped as version 1 and still opens.
- A store written by a newer grock-ocr is rejected with an upgrade message.
- Runtime dependencies are limited to the release series that was tested.

## 0.1.0

- Local OCR for images and PDFs in English and Portuguese. No cloud model and no LLM.
- Document types, issuer matching, and field anchors learned from examples you label.
- Per-page `read_pages` and `learn(..., page=N)` for a PDF where each page is a different document.
