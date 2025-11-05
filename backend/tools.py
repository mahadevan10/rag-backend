"""Document retrieval and manipulation tools for the agentic RAG system."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from backend.services.document_store import DocumentStore
from backend.services.llm import LLMService
from backend.services.reranker import RerankerService

logger = logging.getLogger(__name__)


class AgentTools:
    """Collection of tools for the document QA agent."""

    def __init__(
        self,
        doc_store: DocumentStore,
        reranker: RerankerService,
        llm: LLMService,
    ):
        self.doc_store = doc_store
        self.reranker = reranker
        self.llm = llm
        self._vectorstore = doc_store._vectorstore

    def list_all_documents(self) -> List[str]:
        """List unique document filenames in the store."""
        all_docs = self.doc_store.load_documents()
        unique_filenames = {doc.metadata.get("filename", "unknown") for doc in all_docs}
        return sorted(unique_filenames)

    def get_document_overview(self, filename: str) -> Dict[str, Any]:
        """Get metadata overview for a specific document."""
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        # Extract metadata
        total_chunks = len(doc_chunks)
        pages = sorted(set(d.metadata.get("page", 0) for d in doc_chunks))
        
        # Get first chunk as sample
        sample_text = doc_chunks[0].page_content[:500] if doc_chunks else ""

        return {
            "filename": filename,
            "total_chunks": total_chunks,
            "pages": pages,
            "total_pages": len(pages),
            "sample_content": sample_text,
        }

    def select_document(self, filename: str) -> Dict[str, Any]:
        """Select and return overview of a specific document."""
        return self.get_document_overview(filename)

    def broad_search_all_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search across all documents with automatic query expansion and reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return after reranking
        """
        logger.info(f"Broad search: '{query}' (top_k={top_k})")
        
        # Initial search with reranking
        results = self.doc_store.similarity_search(
            query, 
            k=top_k,
            use_reranking=True,
            rerank_multiplier=4
        )

        # AUTO-EXPAND: If results are poor quality, try query expansion
        needs_expansion = False
        if not results:
            logger.info("No results found - triggering query expansion")
            needs_expansion = True
        elif self.reranker and results:
            # Check if best result has low confidence (reranker returns scores)
            # If reranker didn't improve much, query might be too narrow
            if len(results) < top_k // 2:  # Got less than half of requested results
                logger.info(f"Only {len(results)}/{top_k} results - expanding query")
                needs_expansion = True
        
        expanded_results = []
        if needs_expansion:
            # Expand query
            expansion = self.expand_query(query)
            expanded_terms = expansion.get("expanded_terms", [])
            
            logger.info(f"Searching with expanded terms: {expanded_terms[:3]}")
            
            # Search with each expanded term
            for expanded_query in expanded_terms[:3]:  # Try top 3 expansions
                expanded_results.extend(
                    self.doc_store.similarity_search(
                        expanded_query,
                        k=top_k,
                        use_reranking=False,  # Don't rerank yet
                        rerank_multiplier=2
                    )
                )
            
            # Combine original + expanded results
            all_results = results + expanded_results
            
            # Remove duplicates (same chunk_id)
            seen_ids = set()
            unique_results = []
            for doc in all_results:
                chunk_id = f"{doc.metadata.get('filename', '')}_{doc.metadata.get('chunk_id', '')}"
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_results.append(doc)
            
            # Final reranking with original query
            if self.reranker and len(unique_results) > top_k:
                logger.info(f"Reranking {len(unique_results)} results (original + expanded)")
                results = self.reranker.rerank(query, unique_results, top_k=top_k)
            else:
                results = unique_results[:top_k]

        chunks = []
        for doc in results:
            # Support both old (page) and new (page_number) metadata fields
            page = doc.metadata.get("page_number") or doc.metadata.get("page", "N/A")
            chunk = {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            }
            # Handle bridge chunks (spanning multiple pages)
            if doc.metadata.get("is_bridge_chunk"):
                page_2 = doc.metadata.get("page_number_secondary")
                chunk["page"] = f"{chunk['page']}-{page_2}" if page_2 else chunk["page"]
            chunks.append(chunk)

        return {
            "query": query,
            "num_results": len(chunks),
            "chunks": chunks,
            "reranked": self.reranker is not None,
            "query_expanded": needs_expansion,
            "expanded_terms": expanded_terms[:3] if needs_expansion else []
        }

    def search_specific_documents(
        self, query: str, doc_identifiers: List[str], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search within specific documents with reranking.
        
        Args:
            query: Search query
            doc_identifiers: List of document filenames to search
            top_k: Number of results per document
        """
        logger.info(f"Searching in documents: {doc_identifiers}")
        
        all_results = []
        for doc_id in doc_identifiers:
            # Search with reranking per document
            results = self.doc_store.similarity_search(
                query,
                k=top_k,
                use_reranking=True,
                rerank_multiplier=3,
                filter={"filename": doc_id}  # ChromaDB filter syntax
            )
            all_results.extend(results)

        # If multiple docs, rerank combined results
        if len(doc_identifiers) > 1 and self.reranker and len(all_results) > top_k:
            logger.info(f"Cross-document reranking {len(all_results)} results")
            all_results = self.reranker.rerank(query, all_results, top_k=top_k)

        chunks = []
        for doc in all_results[:top_k]:
            # Support both old and new metadata fields
            page = doc.metadata.get("page_number") or doc.metadata.get("page", "N/A")
            chunk = {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            }
            # Handle bridge chunks
            if doc.metadata.get("is_bridge_chunk"):
                page_2 = doc.metadata.get("page_number_secondary")
                chunk["page"] = f"{chunk['page']}-{page_2}" if page_2 else chunk["page"]
            chunks.append(chunk)

        return {
            "query": query,
            "documents_searched": doc_identifiers,
            "num_results": len(chunks),
            "chunks": chunks,
            "reranked": self.reranker is not None
        }

    def search_specific_page(
        self, doc_identifier: str, page_number: int, query: str = "", top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve content from a specific page with optional query filtering.
        
        Args:
            doc_identifier: Document filename
            page_number: Page number to retrieve
            query: Optional query to filter/rerank page content
            top_k: Number of chunks to return
        """
        logger.info(f"Page search: {doc_identifier} page {page_number}")
        
        # Get all chunks from the page (support both old and new field names)
        all_docs = self.doc_store.load_documents()
        page_chunks = [
            d for d in all_docs
            if d.metadata.get("filename") == doc_identifier
            and (d.metadata.get("page_number") == page_number or d.metadata.get("page") == page_number)
        ]

        if not page_chunks:
            return {
                "error": f"Page {page_number} not found in '{doc_identifier}'",
                "chunks": []
            }

        # If query provided, rerank page chunks
        if query and self.reranker:
            logger.info(f"Reranking {len(page_chunks)} page chunks with query: '{query}'")
            page_chunks = self.reranker.rerank(query, page_chunks, top_k=top_k)
        else:
            page_chunks = page_chunks[:top_k]

        chunks = []
        for doc in page_chunks:
            # Support both old and new metadata fields
            page = doc.metadata.get("page_number") or doc.metadata.get("page", page_number)
            chunk = {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            }
            # Handle bridge chunks
            if doc.metadata.get("is_bridge_chunk"):
                page_2 = doc.metadata.get("page_number_secondary")
                chunk["page"] = f"{chunk['page']}-{page_2}" if page_2 else chunk["page"]
            chunks.append(chunk)

        return {
            "document": doc_identifier,
            "page": page_number,
            "num_results": len(chunks),
            "chunks": chunks,
            "reranked": bool(query and self.reranker)
        }

    def summarize_document(self, filename: str) -> Dict[str, Any]:
        """Generate a summary of the entire document using LLM."""
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        # Concatenate all chunks
        full_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])

        # Truncate if too long (model context limit)
        max_chars = 15000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n...[truncated]"

        prompt = f"Provide a comprehensive summary of this document:\n\n{full_text}"
        summary = self.llm.generate(
            prompt,
            system="You are a document summarization expert.",
            temperature=0.3
        )

        # Format as chunks so it can be added to context
        chunks = [{
            "content": summary,
            "filename": filename,
            "page": "Full Document",
            "chunk_id": "summary",
        }]

        return {
            "filename": filename,
            "summary": summary,
            "total_chunks_processed": len(doc_chunks),
            "chunks": chunks,  # Add this so the graph node can use it
        }

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Advanced multi-query expansion using multiple techniques:
        1. Semantic variations (synonyms, rephrasing)
        2. Perspective shifts (different angles on the same question)
        3. Granularity changes (broader/narrower versions)
        """
        prompt = f"""
Generate query variations to improve document retrieval for: "{query}"

Provide 5-7 alternative queries using these strategies:
1. SYNONYMS: Rephrase using different words with same meaning
2. SPECIFICITY: More specific version (if query is broad)
3. GENERALITY: Broader version (if query is specific)
4. PERSPECTIVE: Different angle or aspect of the same question
5. TECHNICAL: More technical/formal version
6. SIMPLE: Simpler/conversational version

Return ONLY the alternative queries, one per line, without numbering or explanations.
Each query should be substantially different but address the same information need.

Example for "how does photosynthesis work":
- what is the process of photosynthesis
- explain photosynthesis mechanism in plants
- light reaction and Calvin cycle
- how do plants convert sunlight to energy
- photosynthetic pathway in chloroplasts
"""
        expansion = self.llm.generate(
            prompt,
            system="You are a search query expansion expert specializing in information retrieval optimization.",
            temperature=0.8  # Higher temp for more diverse expansions
        )

        expanded_terms = [
            line.strip() 
            for line in expansion.split("\n") 
            if line.strip() and not line.strip().startswith(('-', '•', '*', 'Example'))
        ]
        
        # Deduplicate and limit to top 7
        unique_expansions = []
        seen = set()
        for term in expanded_terms:
            term_lower = term.lower()
            if term_lower not in seen and term_lower != query.lower():
                unique_expansions.append(term)
                seen.add(term_lower)
                if len(unique_expansions) >= 7:
                    break

        logger.info(f"🔍 Expanded '{query}' into {len(unique_expansions)} variations")

        return {
            "original_query": query,
            "expanded_terms": unique_expansions,
            "expansion_count": len(unique_expansions)
        }

    def get_total_pages(self, filename: str) -> Dict[str, Any]:
        """Get total number of pages in a document."""
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        pages = set(d.metadata.get("page", 0) for d in doc_chunks)
        
        return {
            "filename": filename,
            "total_pages": len(pages),
            "pages": sorted(pages)
        }

    def extract_table_of_contents(self, filename: str) -> Dict[str, Any]:
        """Extract section headings that might form a table of contents."""
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        # Concatenate first few chunks (TOC usually at start)
        toc_text = "\n\n".join([chunk.page_content for chunk in doc_chunks[:10]])

        prompt = f"""
Extract the table of contents or main section headings from this document excerpt:

{toc_text}

Return a structured list of headings with their page numbers if available.
"""
        toc = self.llm.generate(
            prompt,
            system="You are a document structure analysis expert.",
            temperature=0.2
        )

        return {
            "filename": filename,
            "table_of_contents": toc,
        }

    def find_figures_and_tables(self, filename: str) -> Dict[str, Any]:
        """Identify references to figures and tables in the document."""
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        # Search for common figure/table markers
        figure_chunks = []
        table_chunks = []

        for doc in doc_chunks:
            content_lower = doc.page_content.lower()
            if any(marker in content_lower for marker in ["figure", "fig.", "fig "]):
                figure_chunks.append({
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content[:200]
                })
            if any(marker in content_lower for marker in ["table", "tab.", "tab "]):
                table_chunks.append({
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content[:200]
                })

        return {
            "filename": filename,
            "figures_found": len(figure_chunks),
            "tables_found": len(table_chunks),
            "figure_references": figure_chunks[:5],  # Limit to first 5
            "table_references": table_chunks[:5],
        }

    def extract_references(self, filename: str) -> Dict[str, Any]:
        """Extract bibliographic references from the document."""
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        # References usually at the end
        reference_chunks = doc_chunks[-10:]
        ref_text = "\n\n".join([chunk.page_content for chunk in reference_chunks])

        prompt = f"""
Extract bibliographic references from this document section:

{ref_text}

Return a structured list of references found.
"""
        references = self.llm.generate(
            prompt,
            system="You are a citation extraction expert.",
            temperature=0.2
        )

        return {
            "filename": filename,
            "references": references,
        }

    def extract_emails(self, filename: str) -> Dict[str, Any]:
        """Extract email addresses or contact information from the document."""
        import re
        
        all_docs = self.doc_store.load_documents()
        doc_chunks = [d for d in all_docs if d.metadata.get("filename") == filename]

        if not doc_chunks:
            return {"error": f"Document '{filename}' not found"}

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        emails_found = set()
        for doc in doc_chunks:
            matches = re.findall(email_pattern, doc.page_content)
            emails_found.update(matches)

        return {
            "filename": filename,
            "emails_found": sorted(emails_found),
            "count": len(emails_found),
        }

    def list_all_chunks_metadata(self, filename: str = None) -> List[Dict[str, Any]]:
        """List metadata for all chunks, optionally filtered by filename.
        
        Args:
            filename: Optional filename to filter chunks
            
        Returns:
            List of metadata dictionaries for each chunk
        """
        all_docs = self.doc_store.load_documents()
        
        if filename:
            all_docs = [d for d in all_docs if d.metadata.get("filename") == filename]
        
        return [d.metadata for d in all_docs]

    def get_chunk_by_metadata(self, filename: str, chunk_id: Any) -> Dict[str, Any]:
        """Retrieve a specific chunk by its filename and chunk_id.
        
        Args:
            filename: The name of the document file
            chunk_id: The chunk_id to retrieve
            
        Returns:
            Dictionary with chunk content and metadata, or error message
        """
        all_docs = self.doc_store.load_documents()
        
        for d in all_docs:
            if d.metadata.get("filename") == filename and d.metadata.get("chunk_id") == chunk_id:
                return {
                    "content": d.page_content,
                    "metadata": d.metadata
                }
        
        return {"error": f"Chunk not found for filename='{filename}' and chunk_id='{chunk_id}'"}
