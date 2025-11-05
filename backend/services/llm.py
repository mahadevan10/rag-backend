"""LLM client abstraction built on the OpenAI-compatible API."""
from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from ..analytics import AnalyticsStore
from ..config import Settings


class LLMService:
    """Wrapper for generating completions via the configured LLM provider."""

    def __init__(
        self,
        settings: Settings,
        analytics: AnalyticsStore,
        *,
        model: Optional[str] = None,
    ) -> None:
        self._client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.nvidia_base_url)
        self._model = model or settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._analytics = analytics

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a completion while recording analytics.
        
        Args:
            prompt: User prompt
            system: System message
            max_tokens: Max output tokens
            temperature: Override default temperature for this call
        """

        start = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens or self._max_tokens,
        )

        content = response.choices[0].message.content.strip()
        elapsed = time.time() - start

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        self._analytics.log_query(prompt, input_tokens, output_tokens, elapsed, 0)
        return content
