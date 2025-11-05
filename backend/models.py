"""Shared Pydantic models for the API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    doc_ids: Optional[List[str]] = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Query must not be empty")
        return value


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    agent_reasoning: List[str]
    tools_used: List[str]
    confidence_score: Optional[int] = 0  # Answer confidence (0-100)


class UploadResponse(BaseModel):
    status: str
    job_id: str
    filenames: List[str]
    message: str
