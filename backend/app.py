"""FastAPI application factory for the Vegah backend."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .agent import AgentState, build_langgraph_agent
from .analytics import AnalyticsStore
from .config import Settings, get_settings
from .models import QueryRequest, QueryResponse, UploadResponse
from .services.document_store import DocumentStore
from .services.embeddings import EmbeddingService
from .services.llm import LLMService
from .services.reranker import RerankerService
from .services.upload import UploadService
from .tools import AgentTools


def _configure_logging(settings: Settings) -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
    )


def create_app() -> FastAPI:
    """Application factory building the FastAPI app and all dependencies."""

    load_dotenv()
    settings = get_settings()
    _configure_logging(settings)
    logger = logging.getLogger(__name__)

    analytics = AnalyticsStore()
    embeddings = EmbeddingService(settings.embedding_model)
    llm = LLMService(settings, analytics)
    reranker = RerankerService(settings.reranker_model if settings.use_reranker else None)
    
    # Pass reranker to document store
    doc_store = DocumentStore(
        settings.chroma_path, 
        embeddings, 
        settings.collection_name,
        reranker_service=reranker  # ← Add this
    )
    
    upload_service = UploadService(doc_store, settings)
    agent_tools = AgentTools(doc_store, reranker, llm)
    langgraph_agent = build_langgraph_agent(agent_tools)

    app = FastAPI(
        title="Vegah Agentic RAG API",
        description="Intelligent agent-based document reasoning powered by DeepSeek-V3.1-Terminus",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.analytics = analytics
    app.state.doc_store = doc_store
    app.state.upload_service = upload_service
    app.state.langgraph_agent = langgraph_agent

    @app.get("/")
    async def root():
        return {
            "service": "Vegah Agentic RAG API",
            "status": "running",
            "agent_type": "Tool-using reasoning agent",
            "llm_provider": "NVIDIA NIM",
            "llm_model": settings.llm_model,
            "documents_indexed": doc_store.count(),
            "capabilities": [
                "Document inspection and reasoning",
                "Strategic retrieval planning",
                "Multi-step tool execution",
                "Llamab powered",
                "Hybrid BM25 + Vector search",
            ],
            "endpoints": {
                "upload": "POST /upload",
                "query": "POST /query",
                "health": "GET /health",
                "documents": "GET /documents",
                "clear": "POST /clear",
                "analytics": "GET /analytics",
            },
        }

    @app.get("/health")
    async def health():
        all_docs = doc_store.load_documents()
        unique_docs = {
            doc.metadata.get("filename", "Unknown")
            for doc in all_docs
        }
        return {
            "status": "healthy",
            "agent_ready": True,
            "llm_provider": "NVIDIA NIM",
            "llm_model": settings.llm_model,
            "documents_indexed": len(unique_docs),
            "pages_indexed": len(all_docs),
            "ready": len(unique_docs) > 0,
        }

    @app.post("/upload", response_model=UploadResponse)
    async def upload_pdfs(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
    ):
        if not files:
            raise HTTPException(400, "No files provided")

        files_data = []
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(400, f"Only PDF files: {file.filename}")
            content = await file.read()
            files_data.append({"filename": file.filename, "content": content})

        job_id = upload_service.create_job([item["filename"] for item in files_data])
        background_tasks.add_task(upload_service.process_upload, job_id, files_data)

        return UploadResponse(
            status="processing",
            job_id=job_id,
            filenames=[item["filename"] for item in files_data],
            message=f"Processing {len(files_data)} PDF(s)",
        )

    @app.get("/upload/status/{job_id}")
    async def upload_status(job_id: str):
        status = upload_service.get_job_status(job_id)
        if status["status"] == "not_found":
            raise HTTPException(404, "Job not found")
        return status
    
    @app.get("/upload/ocr/{job_id}")
    async def get_ocr_results(job_id: str):
        """Get detailed OCR results for testing/debugging.
        
        Returns per-page OCR statistics including:
        - Characters extracted
        - Confidence scores
        - Text previews
        - Success/failure status
        """
        ocr_data = upload_service.get_ocr_results(job_id)
        
        if not ocr_data.get("ocr_used"):
            raise HTTPException(404, "No OCR data available for this job")
        
        return ocr_data

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        if doc_store.count() == 0:
            raise HTTPException(400, "No documents indexed. Upload PDFs first.")

        try:
            # LangGraph needs explicit recursion_limit in config to avoid default limit of 5
            # We set it to a high value to allow multi-step reasoning chains
            recursion_limit = settings.max_agent_iterations * 20  # Extra buffer for review nodes
            
            final_state = langgraph_agent.invoke(
                {
                    "query": request.query,
                    "max_iterations": settings.max_agent_iterations
                },
                config={"recursion_limit": recursion_limit},
            )

            # Extract answer with fallback
            answer = final_state.get("final_answer")
            
            # Handle None case
            if answer is None:
                logger.warning("final_answer is None in final_state")
                answer = "I couldn't generate an answer. Please try rephrasing your question."
            
            # Handle empty string
            if not answer or answer.strip() == "":
                answer = "No answer was generated. The agent may have encountered an issue."

            # Extract sources from context with validation
            sources = []
            for ctx in final_state.get("context", []):
                if isinstance(ctx, dict):
                    # Get metadata for validation
                    metadata = ctx.get("metadata", {})
                    page_num = ctx.get("page") or metadata.get("page_number")
                    total_pages = metadata.get("total_pages")
                    
                    # Validate page number is within bounds
                    if page_num is not None and total_pages is not None:
                        if page_num > total_pages:
                            logger.warning(
                                f"Invalid page {page_num} for document (max: {total_pages}). "
                                f"This indicates a data integrity issue."
                            )
                            page_num = None  # Don't show invalid page
                    
                    source_entry = {
                        "filename": ctx.get("filename", ctx.get("source", metadata.get("filename", "unknown"))),
                        "page": page_num,
                        "chunk_id": ctx.get("chunk_id") or metadata.get("chunk_id"),
                        "total_pages": total_pages,
                        "content": ctx.get("content", "")[:200]  # First 200 chars
                    }
                    sources.append(source_entry)

            return QueryResponse(
                answer=answer,
                sources=sources,
                agent_reasoning=final_state.get("reasoning", []),
                tools_used=final_state.get("tools_used", []),
                confidence_score=final_state.get("confidence_score", 0),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Query failed")
            raise HTTPException(500, f"Query failed: {exc}")

    @app.get("/documents")
    async def list_documents():
        all_docs = doc_store.load_documents()
        docs_summary = {}
        for doc in all_docs:
            filename = doc.metadata.get("filename", "Unknown")
            entry = docs_summary.setdefault(
                filename,
                {"pages": 0, "doc_id": doc.metadata.get("doc_id")},
            )
            entry["pages"] += 1

        return {
            "total_pages": len(all_docs),
            "documents": [
                {"filename": name, "pages": info["pages"], "doc_id": info["doc_id"]}
                for name, info in docs_summary.items()
            ],
        }

    @app.post("/clear")
    async def clear_documents():
        count = doc_store.clear()
        return {"status": "success", "message": f"Cleared {count} pages"}

    @app.get("/analytics")
    async def get_analytics():
        return analytics.snapshot()

    return app
