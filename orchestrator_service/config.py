from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


_GROQ_OPENAI_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def _env_text(name: str) -> str:
    return os.getenv(name, "").strip()


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = _env_text(name) or default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = _env_text(name)
        if value:
            return value
    return default


def _use_llm_decider_alias() -> bool:
    return any(
        _env_text(name)
        for name in (
            "LLM_DECIDER_NAME",
            "LLM_DECIDER_KEY",
            "LLM_DECIDER_API_URL",
            "LLM_DECIDER_API_FORMAT",
        )
    )


def _decider_api_format() -> str:
    if _use_llm_decider_alias():
        return _env_first("LLM_DECIDER_API_FORMAT", "ORCHESTRATOR_DECIDER_API_FORMAT", default="openai").lower()
    return _env_first("ORCHESTRATOR_DECIDER_API_FORMAT", default="ollama").lower()


def _decider_api_url() -> str:
    if _use_llm_decider_alias():
        return _env_first(
            "LLM_DECIDER_API_URL",
            "ORCHESTRATOR_DECIDER_API_URL",
            default=_GROQ_OPENAI_CHAT_COMPLETIONS_URL,
        )
    return _env_first("ORCHESTRATOR_DECIDER_API_URL", default="http://ollama:11434/api/generate")


def _decider_api_key() -> str:
    return _env_first("LLM_DECIDER_KEY", "ORCHESTRATOR_DECIDER_API_KEY", default="")


def _decider_api_key_header() -> str:
    return _env_first("LLM_DECIDER_API_KEY_HEADER", "ORCHESTRATOR_DECIDER_API_KEY_HEADER", default="Authorization")


def _decider_model() -> str:
    return _env_first("LLM_DECIDER_NAME", "ORCHESTRATOR_DECIDER_MODEL", default="gemma4:latest")


def _synthesis_api_format() -> str:
    return _env_first("ORCHESTRATOR_SYNTHESIS_API_FORMAT", "ORCHESTRATOR_DECIDER_API_FORMAT", default=_decider_api_format()).lower()


def _synthesis_api_url() -> str:
    return _env_first("ORCHESTRATOR_SYNTHESIS_API_URL", "ORCHESTRATOR_DECIDER_API_URL", default=_decider_api_url())


def _synthesis_api_key() -> str:
    return _env_first("ORCHESTRATOR_SYNTHESIS_API_KEY", "ORCHESTRATOR_DECIDER_API_KEY", default=_decider_api_key())


def _synthesis_api_key_header() -> str:
    return _env_first(
        "ORCHESTRATOR_SYNTHESIS_API_KEY_HEADER",
        "ORCHESTRATOR_DECIDER_API_KEY_HEADER",
        default=_decider_api_key_header(),
    )


def _synthesis_model() -> str:
    return _env_first("ORCHESTRATOR_SYNTHESIS_MODEL", "ORCHESTRATOR_DECIDER_MODEL", default=_decider_model())


@dataclass(frozen=True)
class Settings:
    retrieval_service_url: str = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8111")
    retrieval_timeout_sec: float = float(os.getenv("RETRIEVAL_TIMEOUT_SEC", "15"))
    vision_enabled: bool = _env_bool("VISION_ENABLED", "true")
    vision_service_url: str = os.getenv("VISION_SERVICE_URL", "http://127.0.0.1:8031").strip()
    vision_timeout_sec: float = float(os.getenv("VISION_TIMEOUT_SEC", "12"))
    vision_greeting_enabled: bool = _env_bool("VISION_GREETING_ENABLED", "true")
    redis_url: str = os.getenv("REDIS_URL", "").strip()
    redis_namespace: str = os.getenv("REDIS_NAMESPACE", "noble_rag").strip() or "noble_rag"
    session_known_ttl_sec: int = int(os.getenv("SESSION_KNOWN_TTL_SEC", "3600"))
    session_guest_ttl_sec: int = int(os.getenv("SESSION_GUEST_TTL_SEC", "300"))
    cors_allow_origins: tuple[str, ...] = _env_csv(
        "ORCHESTRATOR_CORS_ALLOW_ORIGINS",
        default="http://127.0.0.1:8011,http://localhost:8011",
    )
    default_top_k: int = int(os.getenv("ORCHESTRATOR_DEFAULT_TOP_K", "5"))
    decider_enabled: bool = _env_bool("ORCHESTRATOR_DECIDER_ENABLED", "true" if _use_llm_decider_alias() else "false")
    decider_api_format: str = _decider_api_format()
    decider_api_url: str = _decider_api_url()
    decider_api_key: str = _decider_api_key()
    decider_api_key_header: str = _decider_api_key_header()
    decider_model: str = _decider_model()
    decider_timeout_sec: float = float(os.getenv("ORCHESTRATOR_DECIDER_TIMEOUT_SEC", "35"))
    decider_temperature: float = float(os.getenv("ORCHESTRATOR_DECIDER_TEMPERATURE", "0.2"))
    decider_keep_alive: str = os.getenv("ORCHESTRATOR_DECIDER_KEEP_ALIVE", "30m").strip()
    decider_output_max_tokens: int = int(os.getenv("ORCHESTRATOR_DECIDER_OUTPUT_MAX_TOKENS", "140"))
    synthesis_api_format: str = _synthesis_api_format()
    synthesis_api_url: str = _synthesis_api_url()
    synthesis_api_key: str = _synthesis_api_key()
    synthesis_api_key_header: str = _synthesis_api_key_header()
    synthesis_model: str = _synthesis_model()
    synthesis_timeout_sec: float = float(
        os.getenv("ORCHESTRATOR_SYNTHESIS_TIMEOUT_SEC", os.getenv("ORCHESTRATOR_DECIDER_TIMEOUT_SEC", "35"))
    )
    synthesis_temperature: float = float(
        os.getenv("ORCHESTRATOR_SYNTHESIS_TEMPERATURE", os.getenv("ORCHESTRATOR_DECIDER_TEMPERATURE", "0.38"))
    )
    synthesis_keep_alive: str = os.getenv(
        "ORCHESTRATOR_SYNTHESIS_KEEP_ALIVE",
        os.getenv("ORCHESTRATOR_DECIDER_KEEP_ALIVE", "30m"),
    ).strip()
    decider_history_turns: int = int(os.getenv("ORCHESTRATOR_DECIDER_HISTORY_TURNS", "4"))
    synthesis_history_turns: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_HISTORY_TURNS", "3"))
    synthesis_state_topic_limit: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_STATE_TOPIC_LIMIT", "4"))
    grounded_card_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_CARD_LIMIT", "1"))
    grounded_trait_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_TRAIT_LIMIT", "3"))
    grounded_proximity_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_PROXIMITY_LIMIT", "3"))
    grounded_evidence_limit: int = int(os.getenv("ORCHESTRATOR_GROUNDED_EVIDENCE_LIMIT", "2"))
    
    # ===== Optimization Config =====
    # Synthesis latency optimization flags
    synthesis_cache_enabled: bool = _env_bool("ORCHESTRATOR_SYNTHESIS_CACHE_ENABLED", "true")
    synthesis_cache_max_entries: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_CACHE_MAX_ENTRIES", "1000"))
    synthesis_cache_ttl_seconds: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_CACHE_TTL_SECONDS", "3600"))
    
    # Streaming enables progressive response return (faster perceived latency)
    synthesis_enable_streaming: bool = _env_bool("ORCHESTRATOR_SYNTHESIS_ENABLE_STREAMING", "false")

    # Fast response lane for greeting only to keep latency low.
    quick_intent_fast_response_enabled: bool = _env_bool("ORCHESTRATOR_QUICK_INTENT_FAST_RESPONSE_ENABLED", "true")
    quick_intent_response_max_words: int = int(os.getenv("ORCHESTRATOR_QUICK_INTENT_RESPONSE_MAX_WORDS", "48"))
    
    # Periodic model warmup to avoid cold-start latency spikes.
    model_warmup_enabled: bool = _env_bool("ORCHESTRATOR_MODEL_WARMUP_ENABLED", "true")
    model_warmup_interval_sec: float = float(os.getenv("ORCHESTRATOR_MODEL_WARMUP_INTERVAL_SEC", "60"))
    model_warmup_decider_enabled: bool = _env_bool("ORCHESTRATOR_MODEL_WARMUP_DECIDER_ENABLED", "true")
    model_warmup_synthesis_enabled: bool = _env_bool("ORCHESTRATOR_MODEL_WARMUP_SYNTHESIS_ENABLED", "false")
    
    # Compression mode: "disabled" (default), "moderate" (reduce history+context), "aggressive" (max reduction)
    synthesis_compression_mode: str = os.getenv("ORCHESTRATOR_SYNTHESIS_COMPRESSION_MODE", "moderate").strip().lower()
    
    # Max output tokens constraint (helps Kimi respond faster with concise answers)
    synthesis_output_max_tokens: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_OUTPUT_MAX_TOKENS", "300"))
    
    # Timing instrumentation
    enable_timing_instrumentation: bool = _env_bool("ORCHESTRATOR_ENABLE_TIMING_INSTRUMENTATION", "true")
    
    # Optimized history depth (reduce token count)
    # Can be overridden per-use via compression_mode
    synthesis_optimized_history_turns: int = int(os.getenv("ORCHESTRATOR_SYNTHESIS_OPTIMIZED_HISTORY_TURNS", "2"))
    
    langfuse_enabled: bool = _env_bool("LANGFUSE_ENABLED", "false")
    langfuse_flush_at_request_end: bool = _env_bool("LANGFUSE_FLUSH_AT_REQUEST_END", "false")

    host: str = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
    port: int = int(os.getenv("ORCHESTRATOR_PORT", "8021"))


def get_settings() -> Settings:
    return Settings()
