"""Singleton dependencies for RAG service.

This module now boots a Haystack-based adapter (instead of LightRAG).
"""

from __future__ import annotations

import logging as _logging
import os
import json
from datetime import datetime
from typing import Any, AsyncIterator, List
from zoneinfo import ZoneInfo

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
_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _runtime_system_prompt() -> str:
    now = datetime.now(_VN_TZ)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "Bạn tên là Sunny, trợ lý bất động sản cho Noble Place Tây Thăng Long. "
        "Luôn trả lời rõ ràng, đúng trọng tâm, lịch sự, tiếng Việt tự nhiên.\n"
        "Khi tin nhắn người dùng có khối nhận diện từ Máy B và phần xưng hô bắt buộc, "
        "bạn phải chào và xưng hô đúng giới (chỉ anh hoặc chỉ chị), "
        'không được dùng "Anh/Chị" hay "Chào anh/chị" nếu giới đã rõ.\n'
        f"Thời gian hệ thống hiện tại (Asia/Ho_Chi_Minh): {timestamp}."
    )


def _openai_client(base_url: str | None = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.llm_api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url or settings.llm_api_url or None,
    )


def _build_chat_messages(
    *,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
) -> List[dict[str, str]]:
    history = history_messages or []
    messages: List[dict[str, str]] = []
    base_system_prompt = _runtime_system_prompt()
    if system_prompt:
        messages.append({"role": "system", "content": f"{base_system_prompt}\n\n{system_prompt}"})
    else:
        messages.append({"role": "system", "content": base_system_prompt})
    for item in history[-8:]:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    response_format = kwargs.pop("response_format", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    provider = settings.llm_provider.lower().strip()

    messages = _build_chat_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
    )

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


async def llm_model_stream_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    kwargs.pop("response_format", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    provider = settings.llm_provider.lower().strip()
    messages = _build_chat_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
    )

    if provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{ollama_host}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    raw = (line or "").strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    delta = str((data.get("message") or {}).get("content") or "")
                    if delta:
                        yield delta
        return

    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": kwargs.pop("temperature", 0.2),
        "stream": True,
    }
    client = _openai_client()
    stream = await client.chat.completions.create(**payload)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = str(chunk.choices[0].delta.content or "")
        if delta:
            yield delta


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
    max_words = max(64, int(os.getenv("EMBEDDING_MAX_WORDS", "320")))

    def _truncate_text(value: str, limit_words: int) -> str:
        words = (value or "").split()
        if len(words) <= limit_words:
            return value
        return " ".join(words[:limit_words])

    async with httpx.AsyncClient(timeout=max(30.0, float(settings.embedding_timeout))) as client:
        for text in texts:
            candidate = _truncate_text(text, max_words)
            dynamic_limit = max_words
            response = None

            # `/api/embeddings` works across Ollama versions.
            for _ in range(3):
                response = await client.post(
                    f"{host}/api/embeddings",
                    json={"model": model, "prompt": candidate},
                )
                if response.status_code < 400:
                    break

                body = (response.text or "").lower()
                if (
                    response.status_code >= 500
                    and "input length exceeds the context length" in body
                    and dynamic_limit > 64
                ):
                    dynamic_limit = max(64, dynamic_limit // 2)
                    candidate = _truncate_text(candidate, dynamic_limit)
                    log.warning(
                        "Embedding input too long for Ollama; retrying with %s words",
                        dynamic_limit,
                    )
                    continue
                response.raise_for_status()

            assert response is not None
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
    llm_stream_model_func=llm_model_stream_func,
    embedding_func=embedding_func,
)
