"""LLM selector utilities for Haystack-based RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI


def _openai_client(api_key: str, base_url: Optional[str]) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url or None)


@dataclass
class HaystackLLMSelector:
    provider: str
    api_key: Optional[str] = None
    llm_model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-3-large"
    embedding_dim: int = 1536
    base_url: Optional[str] = None
    ollama_host: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        self.provider = self.provider.lower().strip()
        if self.provider not in {"openai", "deepseek", "ollama", "gemini"}:
            raise ValueError("Unsupported provider for HaystackLLMSelector")
        if not self.api_key:
            self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

    def get_llm_model_func(self) -> Callable[..., Awaitable[str]]:
        async def llm_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[list[dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> str:
            history = history_messages or []
            if self.provider == "ollama":
                return await self._call_ollama_chat(prompt, system_prompt, history)
            client = _openai_client(self.api_key or "", self.base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for item in history[-8:]:
                role = item.get("role") or "user"
                if role not in {"system", "user", "assistant"}:
                    role = "user"
                messages.append({"role": role, "content": str(item.get("content") or "")})
            messages.append({"role": "user", "content": prompt})
            resp = await client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.2),
            )
            if not resp.choices:
                return ""
            return str(resp.choices[0].message.content or "")

        return llm_func

    def get_embedding_func(self) -> Callable[[list[str]], Awaitable[np.ndarray]]:
        async def embedding_func(texts: list[str]) -> np.ndarray:
            if self.provider == "ollama":
                vectors = []
                async with httpx.AsyncClient(timeout=60.0) as client:
                    for text in texts:
                        resp = await client.post(
                            self.ollama_host.rstrip("/") + "/api/embeddings",
                            json={"model": self.embedding_model_name, "prompt": text},
                        )
                        resp.raise_for_status()
                        vectors.append(resp.json().get("embedding") or [])
                return np.array(vectors, dtype=np.float32)
            client = _openai_client(self.api_key or "", self.base_url)
            resp = await client.embeddings.create(
                model=self.embedding_model_name,
                input=texts,
            )
            return np.array([item.embedding for item in resp.data], dtype=np.float32)

        return embedding_func

    async def _call_ollama_chat(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history_messages: list[dict[str, Any]],
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in history_messages[-8:]:
            role = item.get("role") or "user"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": str(item.get("content") or "")})
        messages.append({"role": "user", "content": prompt})
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.ollama_host.rstrip("/") + "/api/chat",
                json={
                    "model": self.llm_model_name,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        return str((payload.get("message") or {}).get("content") or "")


# Backward-compatible aliases for old imports.
LightRAGLLMSelector = HaystackLLMSelector


def create_haystack_selector_with_provider(
    *,
    provider: str,
    api_key: Optional[str] = None,
    llm_model_name: str = "gpt-4o-mini",
    embedding_model_name: str = "text-embedding-3-large",
    embedding_dim: int = 1536,
    base_url: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
) -> HaystackLLMSelector:
    return HaystackLLMSelector(
        provider=provider,
        api_key=api_key,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        base_url=base_url,
        ollama_host=ollama_host,
    )


def create_lightrag_with_provider(*args: Any, **kwargs: Any):  # legacy symbol
    raise RuntimeError(
        "LightRAG is no longer used. Please migrate to create_haystack_selector_with_provider()."
    )
