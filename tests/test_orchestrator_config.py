from __future__ import annotations

from pathlib import Path
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODULE_NAME = "orchestrator_service.config"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


def _reload_config_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_settings_prefer_llm_decider_env_for_groq(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_ENABLED", "true")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_API_FORMAT", "openai")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_API_URL", "https://api.moonshot.ai/v1/chat/completions")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_API_KEY", "legacy-key")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_MODEL", "kimi-k2.5")
    monkeypatch.setenv("LLM_DECIDER_API_FORMAT", "openai")
    monkeypatch.setenv("LLM_DECIDER_API_URL", GROQ_CHAT_COMPLETIONS_URL)
    monkeypatch.setenv("LLM_DECIDER_KEY", "groq-key")
    monkeypatch.setenv("LLM_DECIDER_NAME", "qwen/qwen3-32b")
    monkeypatch.setenv("ORCHESTRATOR_SYNTHESIS_API_FORMAT", "openai")
    monkeypatch.setenv("ORCHESTRATOR_SYNTHESIS_API_URL", "https://api.moonshot.ai/v1/chat/completions")
    monkeypatch.setenv("ORCHESTRATOR_SYNTHESIS_API_KEY", "kimi-key")
    monkeypatch.setenv("ORCHESTRATOR_SYNTHESIS_MODEL", "kimi-k2.5")

    config = _reload_config_module()
    settings = config.Settings()

    assert settings.decider_enabled is True
    assert settings.decider_api_format == "openai"
    assert settings.decider_api_url == GROQ_CHAT_COMPLETIONS_URL
    assert settings.decider_api_key == "groq-key"
    assert settings.decider_model == "qwen/qwen3-32b"
    assert settings.synthesis_api_format == "openai"
    assert settings.synthesis_api_url == "https://api.moonshot.ai/v1/chat/completions"
    assert settings.synthesis_api_key == "kimi-key"
    assert settings.synthesis_model == "kimi-k2.5"


def test_settings_keep_legacy_decider_env_without_llm_alias(monkeypatch):
    monkeypatch.delenv("LLM_DECIDER_API_FORMAT", raising=False)
    monkeypatch.delenv("LLM_DECIDER_API_URL", raising=False)
    monkeypatch.delenv("LLM_DECIDER_KEY", raising=False)
    monkeypatch.delenv("LLM_DECIDER_NAME", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_SYNTHESIS_API_FORMAT", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_SYNTHESIS_API_URL", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_SYNTHESIS_MODEL", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_ENABLED", "true")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_API_FORMAT", "openai")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_API_URL", "https://api.moonshot.ai/v1/chat/completions")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_API_KEY", "legacy-key")
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_MODEL", "kimi-k2.5")

    config = _reload_config_module()
    settings = config.Settings()

    assert settings.decider_enabled is True
    assert settings.decider_api_format == "openai"
    assert settings.decider_api_url == "https://api.moonshot.ai/v1/chat/completions"
    assert settings.decider_api_key == "legacy-key"
    assert settings.decider_model == "kimi-k2.5"
    assert settings.synthesis_api_format == "openai"
    assert settings.synthesis_api_url == "https://api.moonshot.ai/v1/chat/completions"
    assert settings.synthesis_api_key == "legacy-key"
    assert settings.synthesis_model == "kimi-k2.5"
