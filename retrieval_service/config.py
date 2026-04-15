from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "retrieval_docs")
    qdrant_timeout_sec: float = float(os.getenv("QDRANT_TIMEOUT_SEC", "10"))

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "2560"))
    embedding_api_url: str = os.getenv("EMBEDDING_API_URL", "").strip()
    embedding_api_format: str = os.getenv("EMBEDDING_API_FORMAT", "ollama").strip().lower()
    embedding_api_timeout_sec: float = float(os.getenv("EMBEDDING_API_TIMEOUT_SEC", "20"))
    embedding_api_auth_token: str = os.getenv("EMBEDDING_API_AUTH_TOKEN", "").strip()
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "").strip()
    embedding_api_key_header: str = os.getenv("EMBEDDING_API_KEY_HEADER", "X-API-Key").strip()

    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "20"))
    min_retrieve_score: float = float(os.getenv("MIN_RETRIEVE_SCORE", "0.68"))
    low_confidence_empty_results: bool = (
        os.getenv("LOW_CONFIDENCE_EMPTY_RESULTS", "false").strip().lower() in {"1", "true", "yes", "on"}
    )

    model_warm_enabled: bool = _env_bool("MODEL_WARM_ENABLED", "true")
    model_warm_interval_sec: float = float(os.getenv("MODEL_WARM_INTERVAL_SEC", "120"))
    model_warm_initial_delay_sec: float = float(os.getenv("MODEL_WARM_INITIAL_DELAY_SEC", "15"))
    embedding_warm_text: str = os.getenv("EMBEDDING_WARM_TEXT", "warmup retrieval embedding").strip()

    llm_warm_enabled: bool = _env_bool("LLM_WARM_ENABLED", "false")
    llm_warm_api_url: str = os.getenv("LLM_WARM_API_URL", "").strip()
    llm_warm_model: str = os.getenv("LLM_WARM_MODEL", "").strip()
    llm_warm_prompt: str = os.getenv("LLM_WARM_PROMPT", "warmup ping").strip()
    llm_warm_timeout_sec: float = float(os.getenv("LLM_WARM_TIMEOUT_SEC", "20"))
    llm_warm_keep_alive: str = os.getenv("LLM_WARM_KEEP_ALIVE", "").strip()

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


def get_settings() -> Settings:
    return Settings()
