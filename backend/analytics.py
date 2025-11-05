"""Analytics and basic observability primitives."""
from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, List


class AnalyticsStore:
    """Thread-safe store capturing usage metrics for the agent."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_queries = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_response_time = 0.0
        self._recent_queries: List[Dict] = []

    def log_query(
        self,
        query: str,
        input_tokens: int,
        output_tokens: int,
        response_time: float,
        docs_retrieved: int,
    ) -> None:
        """Record analytics for a completed query."""

        with self._lock:
            self._total_queries += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_response_time += response_time
            self._recent_queries = (
                [
                    {
                        "query": query,
                        "timestamp": datetime.utcnow().isoformat(),
                        "responseTime": int(response_time * 1000),
                        "documentsRetrieved": docs_retrieved,
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                    }
                ]
                + self._recent_queries
            )[:20]

    def update_latest_documents(self, docs_retrieved: int) -> None:
        """Update the most recent query record with document retrieval stats."""

        with self._lock:
            if self._recent_queries:
                self._recent_queries[0]["documentsRetrieved"] = docs_retrieved

    def snapshot(self) -> Dict:
        """Return a snapshot of the current analytics metrics."""

        with self._lock:
            avg_response = (
                self._total_response_time / self._total_queries
                if self._total_queries
                else 0.0
            )
            total_tokens = self._total_input_tokens + self._total_output_tokens

            return {
                "totalQueries": self._total_queries,
                "totalInputTokens": self._total_input_tokens,
                "totalOutputTokens": self._total_output_tokens,
                "totalTokens": total_tokens,
                "avgResponseTime": int(avg_response * 1000),
                "recentQueries": list(self._recent_queries),
                "sambanovaCost": round(total_tokens / 121_600_000 * 81.20, 2),
            }
