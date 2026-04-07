import os
from functools import lru_cache
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # LLM
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: str = ""
    llm_api_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # Embedding
    embedding_provider: str = ""
    embedding_model: str = "embeddinggemma:300m"
    embedding_dim: int = 768

    # Storage URLs — assembled from individual vars below if empty
    postgres_url: str = ""
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = ""

    # Individual Postgres vars (read from .env)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_db: str = "noble_rag"

    # Individual Redis vars (read from .env)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_session_ttl: int = 86400

    # Haystack / retrieval storage
    knowledge_collection_name: str = "knowledge_chunks"
    rag_workspace: str = "default"
    rag_working_dir: str = "./rag_db"

    # Chunking — reads CHUNK_SIZE; see get_settings() for CHUNK_TOKEN_SIZE fallback
    chunk_size: int = 1024
    chunk_overlap: int = 20

    # Query
    query_timeout_sec: float = 45.0
    subquery_max_concurrency: int = 4
    router_low_confidence_threshold: float = 0.60
    router_skip_kb_probe_confidence: float = 0.90
    search_fallback_on_empty_rag: bool = True
    enable_kb_chat_refactor: bool = False
    enable_cosine_search_fallback: bool = True
    enable_human_loop_search_approval: bool = True
    human_loop_search_cosine_threshold: float = 0.4
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_candidate_limit: int = 20
    reranker_weight_default: float = 0.35
    reranker_weight_compare_quant: float = 0.60
    enable_unified_chat_response: bool = False
    enable_legacy_router: bool = True
    kb_search_score_threshold: float = 0.5

    # Embedding runtime
    embedding_func_max_async: int = 2
    embedding_timeout: int = 120
    embedding_batch_size: int = 4
    max_token_size: int = 8192

    # Tools
    tavily_api_key: str = ""

    # App
    log_level: str = "INFO"
    rag_service_port: int = 8001
    rag_service_host: str = "0.0.0.0"
    session_export_dir: str = "./exports/sessions"
    vision_service_url: str = "http://127.0.0.1:8020"
    vision_session_sync_enabled: bool = True
    vision_session_lookup_timeout_sec: float = 2.0
    cors_origins: str = "*"
    machine_b_base_url: str = ""
    machine_b_timeout_sec: float = 8.0
    machine_b_api_key: str = ""
    machine_b_force_update: bool = True
    machine_b_presence_latest_path: str = "/api/v1/vision/presence/{camera}"
    machine_b_presence_status_path: str = "/api/v1/vision/status"
    machine_b_presence_camera_path: str = "/api/v1/vision/presence/{camera}"
    machine_b_presence_camera: str = "cam01"
    machine_b_presence_poll_interval_sec: float = 2.0
    machine_a_ingest_secret: str = ""
    vision_face_delete_token: str = ""
    chat_history_purge_token: str = ""
    # Khi false: purge-all chỉ xóa Redis/snapshot, không gọi Máy B xóa embedding (ghi đè query notify_machine_b).
    purge_all_notify_machine_b: bool = True

    @model_validator(mode="after")
    def _assemble_derived_fields(self) -> "Settings":
        # Build postgres_url from individual vars if not already set
        if not self.postgres_url:
            self.postgres_url = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

        # Build redis_url from individual vars if not already set
        if not self.redis_url:
            if self.redis_password:
                self.redis_url = (
                    f"redis://:{self.redis_password}"
                    f"@{self.redis_host}:{self.redis_port}/{self.redis_db}"
                )
            else:
                self.redis_url = (
                    f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
                )

        # Fallback embedding_provider → llm_provider
        if not self.embedding_provider:
            self.embedding_provider = self.llm_provider

        return self


def _apply_alt_env_names(s: Settings) -> Settings:
    """Support alternative env var names used in the existing .env:
    - CHUNK_TOKEN_SIZE  → chunk_size
    - CHUNK_OVERLAP_TOKEN_SIZE → chunk_overlap
    - LLM_GEMINI_* → generic LLM_* fields when provider=gemini
    """
    provider = (os.getenv("LLM_PROVIDER") or s.llm_provider or "").lower()

    if (val := os.getenv("CHUNK_TOKEN_SIZE")) and s.chunk_size == 1024:
        try:
            s.chunk_size = int(val)
        except ValueError:
            pass

    if (val := os.getenv("CHUNK_OVERLAP_TOKEN_SIZE")) and s.chunk_overlap == 20:
        try:
            s.chunk_overlap = int(val)
        except ValueError:
            pass

    if provider == "gemini":
        if (val := os.getenv("LLM_GEMINI_MODEL")) and s.llm_model == "gemini-2.5-flash":
            s.llm_model = val
        if (val := os.getenv("LLM_GEMINI_API_KEY")) and not s.llm_api_key:
            s.llm_api_key = val
        if (val := os.getenv("LLM_GEMINI_API_URL")) and s.llm_api_url == "https://generativelanguage.googleapis.com/v1beta/openai":
            s.llm_api_url = val

    return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _apply_alt_env_names(Settings())
