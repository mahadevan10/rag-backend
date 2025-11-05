"""Embedding service wrapper."""
from __future__ import annotations

from langchain_community.embeddings import SentenceTransformerEmbeddings


class EmbeddingService:
    """Thin wrapper around LangChain embeddings for dependency injection."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformerEmbeddings(model_name=model_name)
