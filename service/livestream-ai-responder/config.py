"""
Configuration for the Livestream AI Responder service.
Reads settings from environment variables / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Redis (livestream-tiktok-service writes comments here) ────────────────
    # Kết nối tới Redis của noble_rag qua Docker network nội bộ
    redis_url: str = "redis://redis:6379/0"
    redis_queue_key: str = "livestream:comments"
    # Giây chờ BRPOP nếu queue rỗng (0 = non-blocking)
    pop_timeout: int = 5

    # ── RAG Sales Agent ───────────────────────────────────────────────────────
    # URL của rag-service trong Docker network
    rag_service_url: str = "http://rag-service:8000"
    # Tên session dùng cho tất cả comment livestream (có thể override)
    livestream_session_id: str = "livestream_session"
    # Timeout (giây) khi gọi RAG Sales Agent
    rag_request_timeout: float = 60.0

    # ── Behaviour ─────────────────────────────────────────────────────────────
    # Số lần retry khi RAG agent lỗi trước khi bỏ qua comment
    max_rag_retries: int = 2
    # Delay (giây) giữa các retry
    retry_delay_sec: float = 2.0
    # Log mức độ
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
