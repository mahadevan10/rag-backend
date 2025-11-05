"""Application configuration and environment settings."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_api_key: str
    llm_model: str = "meta/llama-3.3-70b-instruct"
    llm_temperature: float = 0.1  # Lower temp for more deterministic tool selection
    llm_max_tokens: int = 2048
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    
    # Agentic behavior settings
    agent_temperature: float = 0.1  # Tool selection (deterministic)
    reasoning_temperature: float = 0.3  # Planning and evaluation (balanced)
    answer_temperature: float = 0.7  # Final answer generation (creative)
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "./chroma_db"
    collection_name: str = "documents"
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_agent_iterations: int = 50  # Increased for complex multi-part queries
    log_level: str = "INFO"

    # Optional integrations
    hf_token: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    imgbb_api_key: Optional[str] = Field(default=None, validation_alias="IMGBB_API_KEY")

    # Optional OCR support
    enable_ocr: bool = False
    poppler_path: Optional[str] = None
    tesseract_cmd: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
