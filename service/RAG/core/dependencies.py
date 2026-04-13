"""Singleton dependencies for RAG service.

This module now boots a Haystack-based adapter (instead of LightRAG).
"""

from __future__ import annotations

import logging as _logging
import asyncio
import os
import json
import random
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
_LLM_CALL_SEMAPHORE = asyncio.Semaphore(max(1, int(settings.llm_max_concurrency)))


def _resolve_temperature(requested: Any, *, kimi_thinking_disabled: bool = False) -> float:
    """Normalize temperature per provider/model compatibility."""
    try:
        value = float(requested)
    except Exception:
        value = 0.2

    provider = (settings.llm_provider or "").strip().lower()
    model = (settings.llm_model or "").strip().lower()
    # Moonshot Kimi:
    # - Thinking mode: temperature must be 1.
    # - Instant mode (thinking disabled): temperature must be 0.6.
    if provider == "kimi" or "kimi-k2.5" in model:
        if kimi_thinking_disabled:
            return float(os.getenv("LLM_KIMI_INSTANT_TEMPERATURE", "0.6"))
        return 1.0
    return value


def _kimi_disable_thinking() -> bool:
    provider = (settings.llm_provider or "").strip().lower()
    model = (settings.llm_model or "").strip().lower()
    if not (provider == "kimi" or "kimi" in model):
        return False
    return (os.getenv("LLM_KIMI_DISABLE_THINKING", "true").strip().lower() in {"1", "true", "yes", "on"})


def _runtime_system_prompt() -> str:
    now = datetime.now(_VN_TZ)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return (
        "Bạn tên là Sunny, trợ lý bất động sản cho Noble Place Tây Thăng Long. "
        "Luôn trả lời rõ ràng, đúng trọng tâm, lịch sự, tiếng Việt tự nhiên.\n"
        "Quy tắc xưng hô bắt buộc: luôn xưng là Sunny và luôn gọi người dùng là bạn.\n"
        "Không dùng các cách gọi anh, chị, anh/chị trong câu trả lời.\n"
        f"Thời gian hệ thống hiện tại (Asia/Ho_Chi_Minh): {timestamp}."
    )


def _openai_client(base_url: str | None = None, timeout_sec: float | None = None) -> AsyncOpenAI:
    resolved_timeout = float(timeout_sec) if timeout_sec is not None else float(settings.llm_request_timeout_sec)
    return AsyncOpenAI(
        api_key=settings.llm_api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url=base_url or settings.llm_api_url or None,
        timeout=max(5.0, resolved_timeout),
        max_retries=max(0, int(settings.llm_sdk_max_retries)),
    )


def _is_transient_llm_error(error: Exception) -> bool:
    text = str(error).lower()
    transient_markers = (
        "429",
        "rate limit",
        "timed out",
        "timeout",
        "connection",
        "service unavailable",
        "502",
        "503",
        "504",
    )
    return any(marker in text for marker in transient_markers)


def _retry_delay_sec(attempt: int) -> float:
    base = max(0.05, float(settings.llm_retry_base_delay_sec))
    cap = max(base, float(settings.llm_retry_max_delay_sec))
    jitter = max(0.0, float(settings.llm_retry_jitter_sec))
    delay = min(cap, base * (2 ** max(0, attempt)))
    return delay + random.uniform(0.0, jitter)


def _trim_prompt(prompt: str) -> tuple[str, bool]:
    text = str(prompt or "")
    limit = max(1024, int(settings.llm_max_prompt_chars))
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.55)
    tail = max(256, limit - head - 64)
    trimmed = text[:head] + "\n\n[...prompt truncated for latency...]\n\n" + text[-tail:]
    return trimmed, True


def _dynamic_timeout_sec(prompt_len: int) -> float:
    base = max(5.0, float(settings.llm_request_timeout_sec))
    cap = max(base, float(settings.llm_request_timeout_max_sec))
    # Increase timeout with prompt length while capping upper bound.
    adaptive = base + (max(0, prompt_len - 1200) / 700.0)
    return min(cap, adaptive)


def _max_output_tokens_for_prompt(prompt_len: int) -> int:
    threshold = max(512, int(settings.llm_long_prompt_threshold_chars))
    if prompt_len >= threshold:
        tokens = max(64, int(settings.llm_long_prompt_max_tokens))
    else:
        tokens = max(64, int(settings.llm_default_max_tokens))

    provider = (settings.llm_provider or "").strip().lower()
    model = (settings.llm_model or "").strip().lower()
    if (provider == "kimi" or "kimi" in model) and (not _kimi_disable_thinking()):
        kimi_floor = int(os.getenv("LLM_KIMI_MIN_OUTPUT_TOKENS", "480"))
        tokens = max(tokens, kimi_floor)
    return tokens


def _choice_content(choice: Any) -> str:
    message = getattr(choice, "message", None)
    content = str(getattr(message, "content", "") or "").strip()
    return content


async def _recover_empty_completion_once(
    *,
    client: AsyncOpenAI,
    payload: dict[str, Any],
    timeout_sec: float,
) -> str:
    recovery_payload = dict(payload)
    recovery_messages = list(recovery_payload.get("messages") or [])
    recovery_messages.append(
        {
            "role": "user",
            "content": (
                "Trả lời ngay bằng nội dung cuối cùng trong message.content. "
                "Không phân tích từng bước. Không để trống câu trả lời."
            ),
        }
    )
    recovery_payload["messages"] = recovery_messages
    recovery_payload["max_tokens"] = max(
        int(recovery_payload.get("max_tokens") or 0),
        int(os.getenv("LLM_EMPTY_CONTENT_RECOVERY_MAX_TOKENS", "640")),
    )
    log.warning(
        "LLM empty-content recovery: provider=%s model=%s timeout=%.1fs max_tokens=%s",
        settings.llm_provider,
        settings.llm_model,
        timeout_sec,
        recovery_payload["max_tokens"],
    )
    async with _LLM_CALL_SEMAPHORE:
        recovered = await client.chat.completions.create(**recovery_payload)
    if not getattr(recovered, "choices", None):
        return ""
    return _choice_content(recovered.choices[0])


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
    prompt, was_trimmed = _trim_prompt(prompt)
    prompt_len = len(prompt or "")
    timeout_sec = _dynamic_timeout_sec(prompt_len)
    max_tokens = _max_output_tokens_for_prompt(prompt_len)
    response_format = kwargs.pop("response_format", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    provider = settings.llm_provider.lower().strip()
    kimi_thinking_disabled = _kimi_disable_thinking()

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
        "temperature": _resolve_temperature(
            kwargs.pop("temperature", 0.2),
            kimi_thinking_disabled=kimi_thinking_disabled,
        ),
        "max_tokens": max_tokens,
    }
    if provider == "kimi" and kimi_thinking_disabled:
        payload["extra_body"] = {"thinking": {"type": "disabled"}}
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        payload["response_format"] = {"type": "json_object"}

    client = _openai_client(timeout_sec=timeout_sec)
    retries = max(0, int(settings.llm_request_retries))
    resp = None
    for attempt in range(retries + 1):
        try:
            log.info(
                "LLM outbound call: provider=%s model=%s base_url=%s stream=%s prompt_len=%s trimmed=%s timeout=%.1fs max_tokens=%s attempt=%s/%s",
                settings.llm_provider,
                settings.llm_model,
                settings.llm_api_url,
                False,
                prompt_len,
                was_trimmed,
                timeout_sec,
                max_tokens,
                attempt + 1,
                retries + 1,
            )
            if provider == "kimi":
                log.info("LLM kimi mode: thinking_disabled=%s", kimi_thinking_disabled)
            async with _LLM_CALL_SEMAPHORE:
                resp = await client.chat.completions.create(**payload)
            log.info(
                "LLM outbound success: provider=%s model=%s base_url=%s response_id=%s",
                settings.llm_provider,
                settings.llm_model,
                settings.llm_api_url,
                getattr(resp, "id", ""),
            )
            break
        except Exception as e:
            should_retry = attempt < retries and _is_transient_llm_error(e)
            if should_retry:
                delay = _retry_delay_sec(attempt)
                log.warning(
                    "LLM transient error (attempt %s/%s), retry in %.2fs: %s",
                    attempt + 1,
                    retries + 1,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            log.error("LLM request failed: %s", e)
            return ""
    if not resp.choices:
        return ""
    choice = resp.choices[0]
    content = _choice_content(choice)
    if content:
        return content

    finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
    message = getattr(choice, "message", None)
    has_reasoning = bool(str(getattr(message, "reasoning_content", "") or "").strip())
    log.warning(
        "LLM empty content: provider=%s model=%s finish_reason=%s has_reasoning=%s prompt_len=%s max_tokens=%s",
        settings.llm_provider,
        settings.llm_model,
        finish_reason,
        has_reasoning,
        prompt_len,
        max_tokens,
    )

    provider = settings.llm_provider.lower().strip()
    if provider == "kimi" and finish_reason == "length":
        try:
            recovered_text = await _recover_empty_completion_once(
                client=client,
                payload=payload,
                timeout_sec=timeout_sec,
            )
            if recovered_text:
                return recovered_text
        except Exception as e:
            log.warning("LLM empty-content recovery failed: %s", e)
    return ""


async def llm_model_stream_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    prompt, was_trimmed = _trim_prompt(prompt)
    prompt_len = len(prompt or "")
    timeout_sec = _dynamic_timeout_sec(prompt_len)
    max_tokens = _max_output_tokens_for_prompt(prompt_len)
    kwargs.pop("response_format", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)
    provider = settings.llm_provider.lower().strip()
    kimi_thinking_disabled = _kimi_disable_thinking()
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
        "temperature": _resolve_temperature(
            kwargs.pop("temperature", 0.2),
            kimi_thinking_disabled=kimi_thinking_disabled,
        ),
        "stream": True,
        "max_tokens": max_tokens,
    }
    if provider == "kimi" and kimi_thinking_disabled:
        payload["extra_body"] = {"thinking": {"type": "disabled"}}
    client = _openai_client(timeout_sec=timeout_sec)
    retries = max(0, int(settings.llm_request_retries))
    for attempt in range(retries + 1):
        try:
            log.info(
                "LLM outbound call: provider=%s model=%s base_url=%s stream=%s prompt_len=%s trimmed=%s timeout=%.1fs max_tokens=%s attempt=%s/%s",
                settings.llm_provider,
                settings.llm_model,
                settings.llm_api_url,
                True,
                prompt_len,
                was_trimmed,
                timeout_sec,
                max_tokens,
                attempt + 1,
                retries + 1,
            )
            emitted_any = False
            async with _LLM_CALL_SEMAPHORE:
                stream = await client.chat.completions.create(**payload)
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = str(chunk.choices[0].delta.content or "")
                    if delta:
                        emitted_any = True
                        yield delta
            if not emitted_any:
                log.warning("LLM stream empty output, fallback to non-stream call")
                fallback = await llm_model_func(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                )
                if fallback:
                    yield fallback
            return
        except Exception as e:
            should_retry = attempt < retries and _is_transient_llm_error(e)
            if should_retry:
                delay = _retry_delay_sec(attempt)
                log.warning(
                    "LLM stream transient error (attempt %s/%s), retry in %.2fs: %s",
                    attempt + 1,
                    retries + 1,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            log.error("LLM stream request failed: %s", e)
            return


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
