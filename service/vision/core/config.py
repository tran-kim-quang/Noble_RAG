import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class HostRuntimeConfig:
    model: str
    ollama_base_url: str
    timeout_seconds: float


@dataclass(frozen=True)
class VisionSettings:
    default_model: str
    default_ollama_base_url: str
    default_timeout_seconds: float
    host_configs: dict[str, HostRuntimeConfig]

    def resolve(self, host_name: str | None) -> HostRuntimeConfig:
        if host_name:
            host_key = host_name.strip().lower()
            if host_key in self.host_configs:
                return self.host_configs[host_key]

        return HostRuntimeConfig(
            model=self.default_model,
            ollama_base_url=self.default_ollama_base_url,
            timeout_seconds=self.default_timeout_seconds,
        )


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _load_host_configs(raw_value: str) -> dict[str, HostRuntimeConfig]:
    if not raw_value:
        return {}

    loaded = json.loads(raw_value)
    if not isinstance(loaded, dict):
        raise ValueError("VISION_HOST_CONFIG must be a JSON object")

    host_configs: dict[str, HostRuntimeConfig] = {}
    for host_name, host_data in loaded.items():
        if not isinstance(host_name, str) or not isinstance(host_data, dict):
            continue

        model = str(host_data.get("model", "")).strip()
        if not model:
            continue

        base_url = str(
            host_data.get("ollama_base_url", "http://127.0.0.1:11434")
        ).strip()
        timeout = _safe_float(host_data.get("timeout_seconds", 120), 120.0)
        host_configs[host_name.strip().lower()] = HostRuntimeConfig(
            model=model,
            ollama_base_url=base_url,
            timeout_seconds=timeout,
        )

    return host_configs


@lru_cache(maxsize=1)
def get_settings() -> VisionSettings:
    default_model = (
        os.getenv("VISION_DEFAULT_MODEL")
        or os.getenv("MODEL_UNDERSTAND_VISION")
        or "gemma4:latest"
    )
    default_ollama_base_url = os.getenv(
        "VISION_DEFAULT_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
    )
    default_timeout_seconds = _safe_float(
        os.getenv("VISION_DEFAULT_TIMEOUT_SECONDS", "120"), 120.0
    )
    host_configs = _load_host_configs(os.getenv("VISION_HOST_CONFIG", ""))

    return VisionSettings(
        default_model=default_model,
        default_ollama_base_url=default_ollama_base_url,
        default_timeout_seconds=default_timeout_seconds,
        host_configs=host_configs,
    )
