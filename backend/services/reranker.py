"""Cross-encoder based reranking service."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None  # type: ignore[assignment]


class RerankerService:
    """Apply a cross-encoder reranker when the model is available."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        if CrossEncoder and model_name:
            self._model = CrossEncoder(model_name)
        else:
            self._model = None

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not self._model or not docs:
            return docs[:top_k]

        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]
