from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    retrieval_service_url: str = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8011")
    retrieval_timeout_sec: float = float(os.getenv("RETRIEVAL_TIMEOUT_SEC", "15"))
    default_top_k: int = int(os.getenv("ORCHESTRATOR_DEFAULT_TOP_K", "5"))

    host: str = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    port: int = int(os.getenv("ORCHESTRATOR_PORT", "8021"))


def get_settings() -> Settings:
    return Settings()
