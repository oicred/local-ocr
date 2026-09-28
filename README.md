# grock-ocr

Local OCR for images and PDFs in English and Portuguese. Other Python projects import it. Text stays on your machine: there is no cloud OCR service and no language model.

A digital PDF uses its text layer. A scan or a photo is read with PP-OCRv5 (Latin) through RapidOCR on ONNX Runtime. After you label one example, later files from the same company or the same ID layout reuse that template.

## Requirements

- Python 3.11 or newer
- Windows, macOS, or Linux
- Network access the first time you run OCR, so the models can download. Later runs stay offline.

## Setup

```bash
git clone https://github.com/oicred/grock-ocr.git
cd grock-ocr
```

From the project folder:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`[dev]` also installs pytest. To use the library without the tests, run `python -m pip install -e .` instead.

Check the install:

```bash
grock-ocr --version
```

That prints the package version and the store schema version, for example `grock-ocr 0.2.0 (store schema 1)`.

## First document

```bash
grock-ocr read invoice.pdf
grock-ocr read annex.pdf --pages
grock-ocr learn annex.pdf --type invoice --issuer acme --page 2 --field total=45.20
```

`python -m grock_ocr` is the same program. The first `read` or `learn` downloads the OCR model and a small multilingual embedding model into the local cache.

`--store` chooses where learned templates are saved. The default folder is `~/.grock_ocr` (`%USERPROFILE%\.grock_ocr` on Windows).

```bash
grock-ocr --store ./ocr-store read invoice.pdf
```

Each command prints JSON with `page`, `doc_type`, `issuer`, and `fields`.

## Python

```python
from grock_ocr import DocumentReader

reader = DocumentReader()

result = reader.read("invoice.pdf")
result.doc_type     # invoice, receipt, fatura_recibo, credit_note, expense_note,
                    # bank_payment, identity, passport, contract, letter, bank_statement, other
result.issuer
result.page         # set when a single page is returned
result.language     # pt, en, or mixed
result.text
result.fields["total"].value
result.fields["total"].confidence
result.fields["total"].source   # pattern, anchor, extractor, or labeled
result.to_dict()
```

When each page of a PDF is a different document, read them one by one. Label one page; later pages from the same issuer reuse that template.

```python
for page in reader.read_pages("annex.pdf"):
    print(page.page, page.doc_type, page.issuer, page.fields)

reader.learn(
    "annex.pdf",
    doc_type="invoice",
    issuer="acme",
    fields={"total": "45.20"},
    page=2,
)
```

Omit `page` to teach the whole file as one document.

Learning is local: SQLite plus ONNX embeddings. Built-in extractors still fill invoice, receipt, payment, expense-note, identity, and passport fields when no template matches. Passport machine-readable lines are parsed when they are present. Scans get a light contrast and deskew pass before OCR.

Images: png, jpg, jpeg, tiff, webp, bmp. A multipage TIFF is one page per frame.

## Use from another project

Install this package into that project's environment, then point every project at the same store folder. Copy the store folder when you want the learned templates on another machine.

```python
from grock_ocr import DocumentReader

reader = DocumentReader(store_path="ocr-store")
```

A custom extractor replaces fields for one type. Pass `issuer=None` to cover every issuer of that type.

```python
def acme_total(result):
    return {"total": result.fields["total"].value.replace(" ", "")}

reader.register_extractor("invoice", "acme", acme_total)
```

The built-in issuer list covers a few public utilities (EDP, MEO, CTT, Caixa, Schindler, and similar). Any other company is learned when you call `learn` with an `issuer` label. A store written by a newer release is refused until you upgrade this package. See `CHANGELOG.md`.

## Tests

```bash
pytest
```

Recognition against the real model is skipped unless you set `GROCK_OCR_RUN_OCR=1`. That run downloads the models.
