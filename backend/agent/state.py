"""Pydantic model describing the agent state passed through LangGraph."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    query: str  # Original user query
    refined_query: Optional[str] = None  # Agent's refined search query
    top_k: int = Field(default=10)  # Number of chunks to retrieve per search (agent can adjust)
    doc_ids: List[str] = Field(default_factory=list)
    intent: Optional[str] = None
    key_entities: List[str] = Field(default_factory=list)
    context: List[Dict] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    final_answer: Optional[str] = None  # Final answer after synthesis and review
    next_tool: Optional[str] = None
    is_complete: bool = False
    iterations: int = 0
    max_iterations: int = 50  # Default, can be overridden in invoke
    expanded_queries: List[str] = Field(default_factory=list)
    node_call_counts: Dict[str, int] = Field(default_factory=dict)
    confidence_score: int = 0  # Answer confidence (0-100) from review agent
    max_context_tokens: int = Field(default=15000)  # Max tokens for context (LLM limit)
    current_context_tokens: int = Field(default=0)  # Estimated current context size
    """Track how many times each node was called."""
