"""Local multilingual embeddings. The model loads on the first call."""

from __future__ import annotations

_MODEL_NAME = "intfloat/multilingual-e5-small"
_PREFIX = "query: "


class FastEmbedder:
    """ONNX multilingual-e5-small via fastembed. No PyTorch."""

    def __init__(self) -> None:
        self._model = None

    def embed(self, text: str) -> list[float] | None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return None
        model = self._load()
        payload = f"{_PREFIX}{cleaned[:8000]}"
        vectors = list(model.embed([payload]))
        if not vectors:
            return None
        return [float(value) for value in vectors[0]]

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required to learn document types. Install local-ocr dependencies."
            ) from exc
        _ensure_model(TextEmbedding)
        self._model = TextEmbedding(model_name=_MODEL_NAME)
        return self._model


def _ensure_model(text_embedding) -> None:
    supported: set[str] = set()
    for item in text_embedding.list_supported_models():
        if isinstance(item, dict):
            name = item.get("model")
        else:
            name = getattr(item, "model", None)
        if name:
            supported.add(str(name))
    if _MODEL_NAME in supported:
        return
    from fastembed.common.model_description import ModelSource, PoolingType

    text_embedding.add_custom_model(
        model=_MODEL_NAME,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=_MODEL_NAME),
        dim=384,
        model_file="onnx/model.onnx",
    )
