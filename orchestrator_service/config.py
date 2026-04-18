from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    retrieval_service_url: str = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8011")
    retrieval_timeout_sec: float = float(os.getenv("RETRIEVAL_TIMEOUT_SEC", "15"))
    default_top_k: int = int(os.getenv("ORCHESTRATOR_DEFAULT_TOP_K", "5"))
    decider_enabled: bool = _env_bool("ORCHESTRATOR_DECIDER_ENABLED", "false")
    decider_api_format: str = os.getenv("ORCHESTRATOR_DECIDER_API_FORMAT", "ollama").strip().lower()
    decider_api_url: str = os.getenv("ORCHESTRATOR_DECIDER_API_URL", "http://host.docker.internal:11434/api/generate")
    decider_api_key: str = os.getenv("ORCHESTRATOR_DECIDER_API_KEY", "").strip()
    decider_api_key_header: str = os.getenv("ORCHESTRATOR_DECIDER_API_KEY_HEADER", "Authorization").strip()
    decider_model: str = os.getenv("ORCHESTRATOR_DECIDER_MODEL", "gemma4:latest")
    decider_timeout_sec: float = float(os.getenv("ORCHESTRATOR_DECIDER_TIMEOUT_SEC", "35"))
    decider_temperature: float = float(os.getenv("ORCHESTRATOR_DECIDER_TEMPERATURE", "0.2"))
    decider_keep_alive: str = os.getenv("ORCHESTRATOR_DECIDER_KEEP_ALIVE", "30m").strip()
    decider_history_turns: int = int(os.getenv("ORCHESTRATOR_DECIDER_HISTORY_TURNS", "4"))
    synthesis_history_turns: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_HISTORY_TURNS", "3"))
    synthesis_state_topic_limit: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_STATE_TOPIC_LIMIT", "4"))
    grounded_card_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_CARD_LIMIT", "1"))
    grounded_trait_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_TRAIT_LIMIT", "3"))
    grounded_proximity_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_PROXIMITY_LIMIT", "3"))
    grounded_evidence_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_EVIDENCE_LIMIT", "2"))
    langfuse_enabled: bool = _env_bool("LANGFUSE_ENABLED", "false")
    langfuse_flush_at_request_end: bool = _env_bool("LANGFUSE_FLUSH_AT_REQUEST_END", "false")

    host: str = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    port: int = int(os.getenv("ORCHESTRATOR_PORT", "8021"))


def get_settings() -> Settings:
    return Settings()
