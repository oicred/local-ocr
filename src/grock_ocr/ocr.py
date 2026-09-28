"""PP-OCRv5 Latin recognition through RapidOCR on ONNX Runtime."""

from __future__ import annotations

from grock_ocr.models import BBox, TextBlock


class RapidOcrEngine:
    """Lazy RapidOCR engine.

    The Latin PP-OCRv5 recognizer covers Portuguese and English. Text-line
    orientation classification is left on so rotated photos still read.
    Models download on the first call and stay in the local cache.
    """

    def __init__(self) -> None:
        self._engine = None

    def recognize(self, image) -> list[TextBlock]:
        from grock_ocr.preprocess import prepare_scan

        cleaned = prepare_scan(image) if hasattr(image, "convert") else image
        engine = self._load()
        array = _to_rgb_array(cleaned)
        try:
            output = engine(array)
        except TypeError:
            output = engine(array, use_cls=True)
        return parse_rapidocr_output(output)

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise ImportError(
                "rapidocr is required for OCR. Install grock-ocr with its dependencies."
            ) from exc
        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.LATIN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Global.use_cls": True,
        }
        self._engine = RapidOCR(params=params)
        return self._engine


def parse_rapidocr_output(output) -> list[TextBlock]:
    """Turn a RapidOCR result, or the older list shape, into text blocks."""
    if output is None:
        return []
    if isinstance(output, (list, tuple)):
        return _parse_legacy(output)
    texts = getattr(output, "txts", None) or []
    scores = getattr(output, "scores", None) or []
    boxes = getattr(output, "boxes", None)
    if boxes is None:
        return []
    blocks: list[TextBlock] = []
    for index, text in enumerate(texts):
        if text is None or not str(text).strip():
            continue
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else 0.0
        box = boxes[index]
        blocks.append(
            TextBlock(text=str(text).strip(), bbox=_bbox_from_box(box), confidence=score, page=1)
        )
    return blocks


def _parse_legacy(rows) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        box, text = row[0], row[1]
        score = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0
        if not str(text).strip():
            continue
        blocks.append(
            TextBlock(text=str(text).strip(), bbox=_bbox_from_box(box), confidence=score, page=1)
        )
    return blocks


def _bbox_from_box(box) -> BBox:
    first = box[0]
    if len(box) == 4 and not hasattr(first, "__len__"):
        return BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    points = [(float(point[0]), float(point[1])) for point in box]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _to_rgb_array(image):
    import numpy as np

    if hasattr(image, "convert"):
        image = image.convert("RGB")
        return np.array(image)
    return image
