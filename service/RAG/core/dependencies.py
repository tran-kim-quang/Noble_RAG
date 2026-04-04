"""Singleton dependencies for RAG service.

This module now boots a Haystack-based adapter (instead of LightRAG).
"""

from __future__ import annotations

import logging as _logging
import os
from typing import Any, List

import httpx
import numpy as np
from openai import AsyncOpenAI

from core.config import get_settings
from rag.haystack_adapter import HaystackRAGAdapter, HaystackRAGSettings

settings = get_settings()

_logging.basicConfig(
    level=getattr(_logging, settings.log_level.upper(), _logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = _logging.getLogger("rag-service")


def _openai_client(base_url: str | None = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.llm_api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url or settings.llm_api_url or None,
    )


async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    history = history_messages or []
    response_format = kwargs.pop("response_format", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    provider = settings.llm_provider.lower().strip()

    messages: List[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history[-8:]:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if content:
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    if provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{ollama_host}/api/chat",
                    json={
                        "model": settings.llm_model,
                        "messages": messages,
                        "stream": False,
                    },
                )
            resp.raise_for_status()
            payload = resp.json()
            return str((payload.get("message") or {}).get("content") or "")
        except Exception as e:
            log.error("Ollama LLM request failed: %s", e)
            return ""

    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": kwargs.pop("temperature", 0.2),
    }
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        payload["response_format"] = {"type": "json_object"}

    client = _openai_client()
    try:
        resp = await client.chat.completions.create(**payload)
    except Exception as e:
        log.error("LLM request failed: %s", e)
        return ""
    if not resp.choices:
        return ""
    return str(resp.choices[0].message.content or "")


async def _embed_with_openai(texts: List[str]) -> np.ndarray:
    client = _openai_client()
    resp = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


async def _embed_with_ollama(texts: List[str]) -> np.ndarray:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = settings.embedding_model
    vectors: List[List[float]] = []

    async with httpx.AsyncClient(timeout=max(30.0, float(settings.embedding_timeout))) as client:
        for text in texts:
            # `/api/embeddings` works across Ollama versions.
            response = await client.post(
                f"{host}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            vec = data.get("embedding") or []
            vectors.append([float(v) for v in vec])
    return np.array(vectors, dtype=np.float32)


async def embedding_func(texts: List[str]) -> np.ndarray:
    provider = (settings.embedding_provider or settings.llm_provider).lower()
    if provider == "ollama":
        return await _embed_with_ollama(texts)
    try:
        return await _embed_with_openai(texts)
    except Exception as e:
        log.warning("OpenAI-compatible embedding failed (%s), fallback to ollama if available", e)
        return await _embed_with_ollama(texts)


rag = HaystackRAGAdapter(
    settings=HaystackRAGSettings(
        postgres_url=settings.postgres_url,
        qdrant_url=settings.qdrant_url,
        collection_name=settings.knowledge_collection_name,
        workspace=settings.rag_workspace,
        embedding_dim=settings.embedding_dim,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ),
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
)
