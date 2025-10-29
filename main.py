"""
Vegah Agentic RAG API - True Agent-Based Document Reasoning
Powered by DeepSeek-V3.1-Terminus via NVIDIA NIM
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import List, Optional, Dict, Any
from functools import lru_cache
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from pypdf import PdfReader
from openai import OpenAI  # Changed from Groq to OpenAI for NVIDIA NIM
import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    # LLM Settings - NVIDIA NIM
    deepseek_api_key: str  # Changed from groq_api_key
    llm_model: str = "meta/llama-3.3-70b-instruct"  # <-- Changed here
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048  # DeepSeek supports longer outputs
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Vector Store Settings
    chroma_path: str = "./chroma_db"
    collection_name: str = "documents"
    
    # Reranker Settings
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Agent Settings
    max_agent_iterations: int = 3
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# =============================================================================
# LOGGING
# =============================================================================

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    doc_ids: Optional[List[str]] = None
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    agent_reasoning: List[str]
    tools_used: List[str]


class UploadResponse(BaseModel):
    status: str
    job_id: str
    filenames: List[str]
    message: str


# =============================================================================
# AGENT TOOLS
# =============================================================================

class AgentTools:
    """Tools available to the agent for document reasoning and retrieval."""
    
    def __init__(self, doc_store: 'DocumentStore', reranker: 'RerankerService', llm: 'LLMService'):
        self.doc_store = doc_store
        self.reranker = reranker
        self.llm = llm  # <-- Pass LLMService instance for analyze/expand
        self.execution_log = []

    def get_tool_definitions(self) -> List[Dict]:
        """Define tools for function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_query",
                    "description": "Analyze query intent, complexity, and information needs. Use FIRST to understand what the user wants.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "User query to analyze"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "expand_query",
                    "description": "Generate alternative phrasings of the query to improve retrieval coverage.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Query to expand"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_all_documents",
                    "description": "Lists all available documents with their metadata (filename, total pages, doc_id). Use this FIRST to see what documents are available.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_specific_documents",
                    "description": "Search within specific documents by filename or doc_id. Use when user asks about specific documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "doc_identifiers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of filenames or doc_ids to search in"
                            },
                            "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                        },
                        "required": ["query", "doc_identifiers"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_specific_page",
                    "description": "Search a specific page number in a document. Use when user asks about 'page X' or 'what's on page Y'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "page_number": {"type": "integer", "description": "Page number to search"},
                            "doc_identifier": {"type": "string", "description": "Optional: specific document to search in"}
                        },
                        "required": ["query", "page_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "broad_search_all_documents",
                    "description": "Search across all documents without restrictions. Use for general questions or when you don't know which document contains the answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document_overview",
                    "description": "Get an overview of a specific document including total pages and metadata. Use to learn about document structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_identifier": {"type": "string", "description": "Filename or doc_id"}
                        },
                        "required": ["doc_identifier"]
                    }
                }
            }
        ]

    def analyze_query(self, query: str) -> Dict:
        """Analyze query complexity, intent, and required information."""
        self.execution_log.append("🔍 Analyzing query intent and complexity")
        
        analysis_prompt = f"""Analyze this query and return JSON:
Query: "{query}"

Provide:
1. intent: (factual, comparison, summarization, explanation, page_specific)
2. complexity: (simple, medium, complex)
3. required_docs: estimated number of documents needed
4. key_entities: list of important entities/topics
5. time_scope: (specific_point, range, all_time, not_applicable)

Return only valid JSON."""
        result = self.llm.generate(analysis_prompt, "You are a query analysis expert.")
        try:
            analysis = json.loads(result)
            self.execution_log.append(f"✅ Query intent: {analysis.get('intent')}, complexity: {analysis.get('complexity')}")
            return {"analysis": analysis}
        except Exception:
            return {"analysis": {"intent": "unknown", "complexity": "medium"}}

    def expand_query(self, query: str) -> Dict:
        """Generate alternative phrasings and related queries."""
        self.execution_log.append("🔄 Expanding query with synonyms and variations")
        
        expansion_prompt = f"""Generate 3 alternative phrasings for: "{query}"
Return as JSON array: {{"alternatives": ["phrase1", "phrase2", "phrase3"]}}"""
        result = self.llm.generate(expansion_prompt, "You are a query expansion expert.")
        try:
            expansions = json.loads(result)
            return {"expanded_queries": expansions.get("alternatives", [query])}
        except Exception:
            return {"expanded_queries": [query]}

    def list_all_documents(self) -> Dict:
        """List all available documents."""
        self.execution_log.append("🔍 Agent inspecting available documents")
        
        all_docs = self.doc_store.load_documents()
        docs_summary = defaultdict(lambda: {"pages": set(), "doc_id": None, "total_pages": 0})
        
        for doc in all_docs:
            filename = doc.metadata.get("filename", "Unknown")
            doc_id = doc.metadata.get("doc_id")
            page = doc.metadata.get("page_number")
            total = doc.metadata.get("total_pages", 0)
            
            docs_summary[filename]["pages"].add(page)
            docs_summary[filename]["doc_id"] = doc_id
            docs_summary[filename]["total_pages"] = max(docs_summary[filename]["total_pages"], total)
        
        result = {
            "total_documents": len(docs_summary),
            "total_pages": len(all_docs),
            "documents": [
                {
                    "filename": name,
                    "doc_id": info["doc_id"],
                    "total_pages": info["total_pages"],
                    "indexed_pages": len(info["pages"])
                }
                for name, info in docs_summary.items()
            ]
        }
        
        self.execution_log.append(f"📚 Found {len(docs_summary)} documents with {len(all_docs)} total pages")
        return result
    
    def search_specific_documents(self, query: str, doc_identifiers: List[str], top_k: int = 5) -> Dict:
        """Search within specific documents."""
        self.execution_log.append(f"🎯 Agent searching in specific documents: {doc_identifiers}")
        top_k = int(top_k) if top_k is not None else 5
        all_docs = self.doc_store.load_documents()
        
        allowed_ids = set()
        for doc in all_docs:
            doc_id = doc.metadata.get("doc_id")
            filename = doc.metadata.get("filename", "")
            if doc_id in doc_identifiers or filename in doc_identifiers or any(ident in filename for ident in doc_identifiers):
                allowed_ids.add(doc_id)
        
        if not allowed_ids:
            return {"error": "No matching documents found", "chunks": []}
        
        candidates = [d for d in all_docs if d.metadata.get("doc_id") in allowed_ids]
        retrieved = self._hybrid_retrieve(query, candidates, top_k)
        
        self.execution_log.append(f"✅ Retrieved {len(retrieved)} relevant chunks")
        return self._format_results(retrieved)
    
    def search_specific_page(self, query: str, page_number: int, doc_identifier: str = None) -> Dict:
        """Search a specific page."""
        self.execution_log.append(f"📄 Agent searching page {page_number}" + (f" in {doc_identifier}" if doc_identifier else ""))
        
        all_docs = self.doc_store.load_documents()
        candidates = [d for d in all_docs if d.metadata.get("page_number") == page_number]
        
        if doc_identifier:
            candidates = [
                d for d in candidates
                if d.metadata.get("doc_id") == doc_identifier or doc_identifier in d.metadata.get("filename", "")
            ]
        
        if not candidates:
            return {"error": f"Page {page_number} not found", "chunks": []}
        
        retrieved = candidates[:1]
        self.execution_log.append(f"✅ Found page {page_number}")
        return self._format_results(retrieved)
    
    def broad_search_all_documents(self, query: str, top_k: int = 5) -> Dict:
        """Search across all documents."""
        self.execution_log.append(f"🌐 Agent performing broad search across all documents")
        top_k = int(top_k) if top_k is not None else 5
        all_docs = self.doc_store.load_documents()
        if not all_docs:
            return {"error": "No documents available", "chunks": []}
        retrieved = self._hybrid_retrieve(query, all_docs, top_k)
        self.execution_log.append(f"✅ Retrieved {len(retrieved)} chunks from multiple documents")
        return self._format_results(retrieved)
    
    def get_document_overview(self, doc_identifier: str) -> Dict:
        """Get overview of a specific document."""
        self.execution_log.append(f"📋 Agent inspecting document: {doc_identifier}")
        
        all_docs = self.doc_store.load_documents()
        matching_docs = [
            d for d in all_docs
            if d.metadata.get("doc_id") == doc_identifier or doc_identifier in d.metadata.get("filename", "")
        ]
        
        if not matching_docs:
            return {"error": "Document not found"}
        
        pages = set(d.metadata.get("page_number") for d in matching_docs)
        filename = matching_docs[0].metadata.get("filename")
        doc_id = matching_docs[0].metadata.get("doc_id")
        total_pages = matching_docs[0].metadata.get("total_pages", 0)
        
        return {
            "filename": filename,
            "doc_id": doc_id,
            "total_pages": total_pages,
            "indexed_pages": len(pages),
            "pages": sorted(pages)
        }
    
    def _hybrid_retrieve(self, query: str, candidates: List[Document], top_k: int) -> List[Document]:
        """Hybrid BM25 + Vector retrieval."""
        bm25 = BM25Retriever.from_documents(candidates)
        bm25.k = min(30, len(candidates))
        bm25_docs = bm25.get_relevant_documents(query)
        
        vector_retriever = self.doc_store.get_vector_retriever()
        vector_docs = vector_retriever.get_relevant_documents(query)
        
        candidate_ids = {(d.metadata.get("doc_id"), d.metadata.get("page_number")) for d in candidates}
        vector_docs = [d for d in vector_docs if (d.metadata.get("doc_id"), d.metadata.get("page_number")) in candidate_ids]
        
        fused = self._rrf_merge(bm25_docs, vector_docs, top_k * 2)
        return self.reranker.rerank(query, fused, top_k)
    
    def _rrf_merge(self, bm25_docs: List[Document], vector_docs: List[Document], k: int) -> List[Document]:
        """Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        doc_map = {}
        
        for rank, doc in enumerate(bm25_docs):
            key = (doc.metadata.get("doc_id"), doc.metadata.get("page_number"))
            scores[key] += 1.0 / (60 + rank + 1)
            doc_map[key] = doc
        
        for rank, doc in enumerate(vector_docs):
            key = (doc.metadata.get("doc_id"), doc.metadata.get("page_number"))
            scores[key] += 1.0 / (60 + rank + 1)
            doc_map[key] = doc
        
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in sorted_keys[:k] if key in doc_map]
    
    def _format_results(self, docs: List[Document]) -> Dict:
        """Format retrieval results."""
        return {
            "chunks": [
                {
                    "content": doc.page_content,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page_number"),
                    "doc_id": doc.metadata.get("doc_id")
                }
                for doc in docs
            ]
        }
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool by name."""
        tool_map = {
            "list_all_documents": self.list_all_documents,
            "search_specific_documents": self.search_specific_documents,
            "search_specific_page": self.search_specific_page,
            "broad_search_all_documents": self.broad_search_all_documents,
            "get_document_overview": self.get_document_overview
        }
        
        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return tool_map[tool_name](**arguments)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}


# =============================================================================
# AGENTIC RAG ENGINE
# =============================================================================

class AgenticRAG:
    """Agentic RAG system with reasoning and tool use."""
    
    def __init__(self, settings: Settings, tools: AgentTools, llm: 'LLMService'):
        self.settings = settings
        self.tools = tools
        self.llm = llm
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query with agentic reasoning."""
        self.tools.execution_log.clear()
        agent_thoughts = []
        
        logger.info(f"[AGENT] Agent received query: {request.query}")
        agent_thoughts.append(f"Received query: {request.query}")
        
        conversation_history = []
        retrieved_context = []
        tools_used = set()
        final_answer = None
        
        system_prompt = """You are an intelligent document analysis agent powered by DeepSeek-V3.1-Terminus. You MUST use tools to retrieve information before answering.

STRICT RULES:
1. You MUST call list_all_documents first to see available documents
2. You MUST call a search tool to retrieve relevant information
3. You are NOT allowed to answer questions without using tools
4. After using tools and retrieving information, then provide your answer

AVAILABLE TOOLS:
- list_all_documents: See what documents exist
- search_specific_page: For "what's on page X" queries
- search_specific_documents: Search in specific files
- broad_search_all_documents: Search all documents
- get_document_overview: Get document metadata

PROCESS:
1. Use list_all_documents
2. Use appropriate search tool
3. Only then provide answer based on retrieved info"""

        user_message = f"""User Query: "{request.query}"

Step 1: Call list_all_documents to see available documents.
Step 2: Use the appropriate search tool to find relevant information.
Step 3: After retrieving information, provide your answer.

You MUST use tools. Do NOT answer without using tools first."""

        conversation_history.append({"role": "system", "content": system_prompt})
        conversation_history.append({"role": "user", "content": user_message})
        
        # Agentic loop
        for iteration in range(self.settings.max_agent_iterations):
            logger.info(f"[ITER] Agent iteration {iteration + 1}/{self.settings.max_agent_iterations}")
            agent_thoughts.append(f"Starting iteration {iteration + 1}")
            
            try:
                response = self.llm.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=conversation_history,
                    tools=self.tools.get_tool_definitions(),
                    tool_choice="auto",
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens
                )
                
                assistant_message = response.choices[0].message
                
                # Convert to dict for easier handling
                message_dict = {
                    "role": "assistant",
                    "content": assistant_message.content if assistant_message.content else ""
                }
                
                # Handle tool calls
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    agent_thoughts.append(f"Agent decided to use {len(assistant_message.tool_calls)} tool(s)")
                    
                    # Add tool_calls to message dict
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                    
                    conversation_history.append(message_dict)
                    
                    # Execute each tool
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        logger.info(f"[TOOL] Executing tool: {tool_name} with args: {arguments}")
                        agent_thoughts.append(f"Executing: {tool_name}")
                        tools_used.add(tool_name)
                        
                        # Execute tool
                        result = self.tools.execute_tool(tool_name, arguments)
                        
                        # Add tool result to conversation
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, default=str)
                        })
                        
                        # Store retrieved chunks
                        if "chunks" in result and result["chunks"]:
                            retrieved_context.extend(result["chunks"])
                            agent_thoughts.append(f"Retrieved {len(result['chunks'])} chunks")
                        elif "error" in result:
                            agent_thoughts.append(f"Tool error: {result['error']}")
                
                # Handle content without tool calls
                elif assistant_message.content:
                    content = assistant_message.content.strip()
                    
                    if iteration > 0 and retrieved_context:
                        logger.info("[OK] Agent provided final answer")
                        agent_thoughts.append("Generating final answer from retrieved context")
                        final_answer = content
                        break
                    else:
                        agent_thoughts.append("Agent tried to answer without tools, redirecting...")
                        conversation_history.append(message_dict)
                        conversation_history.append({
                            "role": "user",
                            "content": "You MUST use the tools to retrieve information before answering. Please use list_all_documents and then search for relevant information."
                        })
                
                else:
                    agent_thoughts.append("Agent produced no output, prompting to use tools...")
                    conversation_history.append({
                        "role": "user",
                        "content": "Please use the available tools to retrieve information. Start with list_all_documents."
                    })
            
            except Exception as e:
                logger.error(f"[ERROR] Agent iteration error: {e}", exc_info=True)
                agent_thoughts.append(f"Error in iteration {iteration + 1}: {str(e)}")
                break
        
        # Generate final answer if not done yet
        if not final_answer:
            if retrieved_context:
                logger.info("[TIME] Max iterations reached, generating answer")
                agent_thoughts.append("Max iterations reached, synthesizing answer")
                final_answer = self._generate_final_answer(request.query, retrieved_context)
            else:
                logger.warning("[WARN] No context retrieved")
                agent_thoughts.append("No relevant information found")
                final_answer = "I couldn't find relevant information in the available documents. Please upload documents and try again."
        
        # Combine reasoning
        all_reasoning = agent_thoughts + self.tools.execution_log
        sources = self._build_sources(retrieved_context)
        
        logger.info(f"[OUT] Query complete. Reasoning: {len(all_reasoning)}, Tools: {list(tools_used)}, Sources: {len(sources)}")
        
        return QueryResponse(
            answer=final_answer,
            sources=sources,
            agent_reasoning=all_reasoning,
            tools_used=list(tools_used)
        )

    def _generate_final_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate final answer from retrieved chunks."""
        if not chunks:
            return "I couldn't find relevant information in the available documents."
        
        context = "\n\n".join([
            f"[{chunk['filename']} - Page {chunk['page']}]\n{chunk['content'][:1000]}"
            for chunk in chunks[:5]
        ])
        
        prompt = f"""Based on the document excerpts below, provide a comprehensive answer.

Question: {query}

Retrieved Information:
{context}

Instructions:
- Answer directly and comprehensively
- Use only the provided context
- Cite specific pages when relevant
- If context doesn't fully answer, acknowledge what's available

Answer:"""

        try:
            answer = self.llm.generate(
                prompt,
                "You are a helpful assistant that provides accurate answers based on document content.",
                max_tokens=1024
            )
            return answer
        except Exception as e:
            logger.error(f"Failed to generate final answer: {e}")
            return f"I found relevant information but encountered an error: {str(e)}"
    
    def _build_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Build sources from chunks."""
        seen = set()
        sources = []
        
        for chunk in chunks:
            key = (chunk.get("filename"), chunk.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": chunk.get("filename", "Unknown"),
                    "page": chunk.get("page"),
                    "doc_id": chunk.get("doc_id"),
                    "preview": chunk.get("content", "")[:200] + "..."
                })
        
        return sources


# =============================================================================
# SUPPORTING SERVICES
# =============================================================================

class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = None
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info(f"Loading embeddings: {self.settings.embedding_model}")
            self._embeddings = SentenceTransformerEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 32},
            )
        return self._embeddings


class LLMService:
    """LLM Service using NVIDIA NIM API with OpenAI compatibility."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        # Initialize OpenAI client with NVIDIA NIM endpoint
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.nvidia_base_url
        )
        logger.info(f"✅ NVIDIA NIM client initialized with model: {settings.llm_model}")
    
    def generate(self, prompt: str, system_prompt: str, max_tokens: int = None) -> str:
        """Generate text using DeepSeek via NVIDIA NIM."""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise HTTPException(500, f"LLM error: {str(e)}")


class RerankerService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
    
    @property
    def model(self):
        if not self.settings.use_reranker or CrossEncoder is None:
            return None
        if self._model is None:
            try:
                logger.info(f"Loading reranker: {self.settings.reranker_model}")
                self._model = CrossEncoder(self.settings.reranker_model, device="cpu")
            except Exception as e:
                logger.warning(f"Reranker load failed: {e}")
        return self._model
    
    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if self.model is None or not docs:
            return docs[:top_k]
        try:
            scores = self.model.predict([(query, d.page_content) for d in docs], convert_to_numpy=True)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [d for _, d in ranked[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return docs[:top_k]


class DocumentStore:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        self.settings = settings
        self.embedding_service = embedding_service
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._doc_cache = []
        self._cache_valid = False
    
    def get_collection(self):
        try:
            return self._client.get_collection(self.settings.collection_name)
        except:
            return self._client.create_collection(self.settings.collection_name)
    
    def count(self) -> int:
        try:
            return self.get_collection().count()
        except:
            return 0
    
    def load_documents(self) -> List[Document]:
        if self._cache_valid and self._doc_cache:
            return self._doc_cache
        
        try:
            collection = self.get_collection()
            results = collection.get()
            
            docs = []
            for doc_id, text, metadata in zip(results.get("ids", []), results.get("documents", []), results.get("metadatas", [])):
                docs.append(Document(page_content=text or "", metadata=metadata or {}))
            
            self._doc_cache = docs
            self._cache_valid = True
            logger.info(f"Loaded {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        try:
            # First create/get the Chroma instance
            vectorstore = Chroma(
                client=self._client,
                collection_name=self.settings.collection_name,
                embedding_function=self.embedding_service.embeddings,
            )
            
            # Then add documents to it
            vectorstore.add_documents(documents)
            
            self._cache_valid = False
            logger.info(f"Added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def clear(self) -> int:
        count = self.count()
        self._client.delete_collection(self.settings.collection_name)
        self._client.create_collection(self.settings.collection_name)
        self._doc_cache.clear()
        self._cache_valid = False
        return count
    
    def get_vector_retriever(self):
        return Chroma(
            client=self._client,
            collection_name=self.settings.collection_name,
            embedding_function=self.embedding_service.embeddings,
        ).as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5})


class UploadService:
    def __init__(self, doc_store: DocumentStore):
        self.doc_store = doc_store
        self.jobs = {}
    
    def create_job(self, filenames: List[str]) -> str:
        job_id = str(uuid4())
        self.jobs[job_id] = {
            "status": "processing",
            "filenames": filenames,
            "created_at": datetime.now().isoformat()
        }
        return job_id
    
    def process_upload(self, job_id: str, files_data: List[Dict]):
        try:
            all_docs = []
            
            for file_data in files_data:
                filename = file_data["filename"]
                content = file_data["content"]
                doc_id = str(uuid4())
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                
                try:
                    reader = PdfReader(tmp_path)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if not text.strip():
                            continue
                        
                        doc = Document(
                            page_content=text,
                            metadata={
                                "filename": filename,
                                "doc_id": doc_id,
                                "page_number": i + 1,
                                "total_pages": len(reader.pages)
                            }
                        )
                        all_docs.append(doc)
                finally:
                    tmp_path.unlink()
            
            if not all_docs:
                raise ValueError("No valid content extracted")
            
            self.doc_store.add_documents(all_docs)
            
            self.jobs[job_id].update({
                "status": "completed",
                "pages_processed": len(all_docs),
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"Job {job_id}: processed {len(all_docs)} pages")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.jobs[job_id].update({"status": "failed", "error": str(e)})
    
    def get_job_status(self, job_id: str) -> Dict:
        return self.jobs.get(job_id, {"status": "not_found"})


# =============================================================================
# INITIALIZE SERVICES
# =============================================================================

load_dotenv()
settings = get_settings()

embedding_service = EmbeddingService(settings)
llm_service = LLMService(settings)
reranker_service = RerankerService(settings)
doc_store = DocumentStore(settings, embedding_service)
agent_tools = AgentTools(doc_store, reranker_service, llm_service)
agentic_rag = AgenticRAG(settings, agent_tools, llm_service)
upload_service = UploadService(doc_store)

logger.info("🤖 Agentic RAG system initialized with DeepSeek-V3.1-Terminus via NVIDIA NIM")
doc_store.load_documents()
logger.info(f"📚 Ready with {doc_store.count()} pages indexed")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Vegah Agentic RAG API",
    description="Intelligent agent-based document reasoning powered by DeepSeek-V3.1-Terminus",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            "DeepSeek-V3.1-Terminus powered",
            "Hybrid BM25 + Vector search"
        ],
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "health": "GET /health",
            "documents": "GET /documents",
            "clear": "POST /clear"
        }
    }


@app.get("/health")
async def health():
    count = doc_store.count()
    return {
        "status": "healthy",
        "agent_ready": True,
        "llm_provider": "NVIDIA NIM",
        "llm_model": settings.llm_model,
        "documents_indexed": count,
        "ready": count > 0
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided")
    
    files_data = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, f"Only PDF files: {file.filename}")
        
        content = await file.read()
        files_data.append({"filename": file.filename, "content": content})
    
    job_id = upload_service.create_job([f["filename"] for f in files_data])
    background_tasks.add_task(upload_service.process_upload, job_id, files_data)
    
    return UploadResponse(
        status="processing",
        job_id=job_id,
        filenames=[f["filename"] for f in files_data],
        message=f"Processing {len(files_data)} PDF(s)"
    )


@app.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    status = upload_service.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(404, "Job not found")
    return status


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query documents using agentic RAG powered by DeepSeek."""
    if doc_store.count() == 0:
        raise HTTPException(400, "No documents indexed. Upload PDFs first.")
    
    try:
        logger.info(f"📨 Received query: {request.query}")
        result = agentic_rag.process_query(request)
        
        logger.info(f"📤 Returning: {len(result.agent_reasoning)} reasoning steps, {len(result.tools_used)} tools, {len(result.sources)} sources")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    all_docs = doc_store.load_documents()
    docs_summary = defaultdict(lambda: {"pages": 0, "doc_id": None})
    
    for doc in all_docs:
        filename = doc.metadata.get("filename", "Unknown")
        doc_id = doc.metadata.get("doc_id")
        docs_summary[filename]["pages"] += 1
        docs_summary[filename]["doc_id"] = doc_id
    
    return {
        "total_pages": len(all_docs),
        "documents": [
            {"filename": name, "pages": info["pages"], "doc_id": info["doc_id"]}
            for name, info in docs_summary.items()
        ]
    }


@app.post("/clear")
async def clear_documents():
    count = doc_store.clear()
    return {"status": "success", "message": f"Cleared {count} pages"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
