from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "retrieval_docs")
    qdrant_timeout_sec: float = float(os.getenv("QDRANT_TIMEOUT_SEC", "10"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "20"))
    min_retrieve_score: float = float(os.getenv("MIN_RETRIEVE_SCORE", "0.68"))
    low_confidence_empty_results: bool = (
        os.getenv("LOW_CONFIDENCE_EMPTY_RESULTS", "false").strip().lower() in {"1", "true", "yes", "on"}
    )

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


def get_settings() -> Settings:
    return Settings()
