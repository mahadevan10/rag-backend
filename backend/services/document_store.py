"""Vector store backed document persistence."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma as ChromaVectorStore
from langchain_core.documents import Document

from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentStore:
    """Wrapper around Chroma vector store with utility helpers."""

    def __init__(
        self,
        chroma_path: str,
        embedding_service: EmbeddingService,
        collection_name: str = "documents",
        reranker_service = None,  # Add reranker support
    ) -> None:
        self._chroma_path = chroma_path
        self._embedding_service = embedding_service
        self._collection_name = collection_name
        self._reranker = reranker_service
        self._vectorstore = self._create_vector_store()

    def _create_vector_store(self) -> ChromaVectorStore:
        return ChromaVectorStore(
            collection_name=self._collection_name,
            embedding_function=self._embedding_service.model,
            persist_directory=self._chroma_path,
        )

    def load_documents(self) -> List[Document]:
        return self._vectorstore.similarity_search("*", k=10_000)

    def get_retriever(self):  # pragma: no cover - simple pass-through
        return self._vectorstore.as_retriever()

    def add_documents(self, docs: List[Document]) -> None:
        self._vectorstore.add_documents(docs)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        use_reranking: bool = True,
        rerank_multiplier: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Similarity search with optional reranking.
        
        Args:
            query: Search query
            k: Number of final results to return
            use_reranking: Whether to use reranker (if available)
            rerank_multiplier: Fetch k*multiplier candidates before reranking
            **kwargs: Additional filter arguments (doc_id, page, etc.)
        
        Returns:
            List of most relevant documents
        """
        # Determine how many candidates to fetch
        initial_k = k * rerank_multiplier if (use_reranking and self._reranker) else k
        
        # Initial vector search
        candidates = self._vectorstore.similarity_search(
            query, 
            k=initial_k,
            **kwargs
        )
        
        # Rerank if available and enabled
        if use_reranking and self._reranker and len(candidates) > k:
            logger.info(f"Reranking {len(candidates)} candidates to top {k}")
            reranked = self._reranker.rerank(query, candidates, top_k=k)
            return reranked
        
        return candidates[:k]

    def clear(self) -> int:
        """Remove all stored vectors and delete persisted artefacts."""

        previous_count = len(self.load_documents())

        if self._vectorstore is not None:
            try:
                collection_name = getattr(self._vectorstore._collection, "name", self._collection_name)
                self._vectorstore._client.delete_collection(name=collection_name)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Failed to delete existing Chroma collection: %s", exc)
            
            # Recreate the vectorstore instead of setting to None
            self._vectorstore = self._create_vector_store()

        return previous_count

    def count(self) -> int:
        return len(self.load_documents())
