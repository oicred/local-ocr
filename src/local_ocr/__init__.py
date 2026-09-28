"""Local document OCR that other Python projects can import."""

from local_ocr.models import FieldValue, ReadResult
from local_ocr.reader import DocumentReader

__all__ = ["DocumentReader", "FieldValue", "ReadResult"]
__version__ = "0.2.0"
