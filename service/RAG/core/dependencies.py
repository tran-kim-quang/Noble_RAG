"""
Singletons: LightRAG instance, llm_model_func, embedding_func.
Initialised once at import time so all modules share the same objects.
"""

import asyncio
import os
import time
from functools import partial
from urllib.parse import urlparse

from lightrag import LightRAG
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed,
    wrap_embedding_func_with_attrs,
)

from core.config import get_settings

settings = get_settings()

import logging as _logging
_logging.basicConfig(
    level=getattr(_logging, settings.log_level.upper(), _logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = _logging.getLogger("rag-service")


# ── Map compose-style URLs to LightRAG backend env vars ──────────────────
def _configure_storage_env() -> None:
    parsed_pg = urlparse(settings.postgres_url)
    if parsed_pg.scheme.startswith("postgres"):
        if parsed_pg.hostname and not os.getenv("POSTGRES_HOST"):
            os.environ["POSTGRES_HOST"] = parsed_pg.hostname
        if parsed_pg.port and not os.getenv("POSTGRES_PORT"):
            os.environ["POSTGRES_PORT"] = str(parsed_pg.port)
        if parsed_pg.username and not os.getenv("POSTGRES_USER"):
            os.environ["POSTGRES_USER"] = parsed_pg.username
        if parsed_pg.password and not os.getenv("POSTGRES_PASSWORD"):
            os.environ["POSTGRES_PASSWORD"] = parsed_pg.password
        db_name = parsed_pg.path.lstrip("/")
        if db_name and not os.getenv("POSTGRES_DATABASE"):
            os.environ["POSTGRES_DATABASE"] = db_name

    if settings.qdrant_url and not os.getenv("QDRANT_URL"):
        os.environ["QDRANT_URL"] = settings.qdrant_url
    if settings.redis_url and not os.getenv("REDIS_URI"):
        os.environ["REDIS_URI"] = settings.redis_url


# ── LLM factory ──────────────────────────────────────────────────────────
def _init_llm():
    provider = settings.llm_provider.lower()
    log.info("Initialising LLM provider='%s' model='%s'", provider, settings.llm_model)

    if provider in {"openai", "deepseek"}:
        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            if history_messages is None:
                history_messages = []
            force_json = False
            if kwargs.get("keyword_extraction"):
                kwargs["keyword_extraction"] = False
                force_json = True
            if "response_format" in kwargs:
                rf = kwargs["response_format"]
                if not isinstance(rf, dict) or rf.get("type") != "json_object":
                    force_json = True
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
                prompt += "\n\nIMPORTANT: Return strictly a valid JSON object. No additional text."

            cot = kwargs.pop("enable_cot", True)
            result = await openai_complete_if_cache(
                settings.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_url or None,
                enable_cot=cot,
                **kwargs,
            )
            if result is None:
                log.error("LLM returned None for model=%s", settings.llm_model)
                return ""
            return result

        return llm_func

    if provider == "ollama":
        from lightrag.llm.ollama import ollama_model_complete
        return partial(
            ollama_model_complete,
            model=settings.llm_model,
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

    if provider == "gemini":
        from lightrag.llm.gemini import gemini_model_complete

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            if history_messages is None:
                history_messages = []
            force_json = False
            if kwargs.get("keyword_extraction"):
                kwargs["keyword_extraction"] = False
                force_json = True
            if "response_format" in kwargs:
                rf = kwargs["response_format"]
                if not isinstance(rf, dict) or rf.get("type") != "json_object":
                    force_json = True
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
                prompt += "\n\nIMPORTANT: Return strictly a valid JSON object. No additional text."

            result = None
            last_error = None
            for attempt in range(3):
                try:
                    result = await gemini_model_complete(
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=settings.llm_api_key,
                        model_name=settings.llm_model,
                        **kwargs,
                    )
                    break
                except Exception as e:
                    last_error = e
                    if "429" not in str(e) or attempt == 2:
                        raise
                    wait_sec = float(attempt + 1)
                    log.warning(
                        "Gemini rate limited for model=%s; retrying in %.1fs (attempt %d/3)",
                        settings.llm_model,
                        wait_sec,
                        attempt + 1,
                    )
                    await asyncio.sleep(wait_sec)
            if result is None:
                if last_error is not None:
                    log.error("LLM failed for model=%s error=%s", settings.llm_model, last_error)
                log.error("LLM returned None for model=%s", settings.llm_model)
                return ""
            return result

        return llm_func

    raise ValueError(
        f"Unsupported LLM provider: {provider}. Use 'deepseek', 'openai', 'ollama', or 'gemini'."
    )


# ── Embedding factory ─────────────────────────────────────────────────────
def _init_embedding():
    embed_provider = (settings.embedding_provider or settings.llm_provider).lower()
    log.info(
        "Initialising embedding provider='%s' model='%s'",
        embed_provider,
        settings.embedding_model,
    )

    if embed_provider == "ollama":
        from lightrag.llm.ollama import ollama_embed

        @wrap_embedding_func_with_attrs(
            embedding_dim=settings.embedding_dim,
            max_token_size=settings.max_token_size,
            model_name=settings.embedding_model,
        )
        async def embedding_func(texts: list[str]):
            import numpy as np
            batch_size = max(1, settings.embedding_batch_size)
            if len(texts) <= batch_size:
                return await ollama_embed.func(
                    texts,
                    embed_model=settings.embedding_model,
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                    api_key=os.getenv("OLLAMA_API_KEY") or None,
                )
            vectors = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start: start + batch_size]
                batch_vectors = await ollama_embed.func(
                    batch,
                    embed_model=settings.embedding_model,
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                    api_key=os.getenv("OLLAMA_API_KEY") or None,
                )
                vectors.append(batch_vectors)
            return np.concatenate(vectors, axis=0)

        return embedding_func

    if embed_provider == "openai":
        @wrap_embedding_func_with_attrs(
            embedding_dim=settings.embedding_dim,
            max_token_size=settings.max_token_size,
            model_name=settings.embedding_model,
        )
        async def embedding_func(texts: list[str]):
            return await openai_embed.func(
                texts,
                model=settings.embedding_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_api_url or None,
            )

        return embedding_func

    raise ValueError(f"Unsupported embedding provider: {embed_provider}.")


# ── Bootstrap ──────────────────────────────────────────────────────────────
log.info("Bootstrapping RAG service components...")
_t0 = time.perf_counter()

_configure_storage_env()
llm_model_func = _init_llm()
embedding_func = _init_embedding()

rag = LightRAG(
    working_dir=settings.rag_working_dir,
    kv_storage=settings.kv_storage,
    vector_storage=settings.vector_storage,
    graph_storage=settings.graph_storage,
    doc_status_storage=settings.doc_status_storage,
    workspace=settings.rag_workspace,
    llm_model_func=llm_model_func,
    llm_model_name=settings.llm_model,
    embedding_func=embedding_func,
    embedding_func_max_async=settings.embedding_func_max_async,
    default_embedding_timeout=settings.embedding_timeout,
    chunk_token_size=settings.chunk_size,
    chunk_overlap_token_size=settings.chunk_overlap,
)

log.info("RAG components bootstrapped in %.2fs", time.perf_counter() - _t0)
