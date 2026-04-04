"""LLM helper module for Haystack-based RAG runtime."""

from .choose_llm import (
    HaystackLLMSelector,
    LightRAGLLMSelector,  # backward-compat alias
    create_haystack_selector_with_provider,
    create_lightrag_with_provider,  # legacy symbol
)

__all__ = [
    "HaystackLLMSelector",
    "LightRAGLLMSelector",
    "create_haystack_selector_with_provider",
    "create_lightrag_with_provider",
]
