from __future__ import annotations

import json
import logging
import re
import socket
import threading
from typing import Any
import urllib.error
import urllib.request

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request

from retrieval_service.config import get_settings
from retrieval_service.observability import flush_observability
from retrieval_service.observability import propagate_context
from retrieval_service.observability import start_observation
from retrieval_service.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ProjectGroundedRetrieveRequest,
    ProjectGroundedRetrieveResponse,
    RetrieveRequest,
    RetrieveResponse,
)

log = logging.getLogger("retrieval-service")


def _extract_langfuse_trace_context(request: Request) -> dict[str, str] | None:
    trace_id = str(request.headers.get("X-Langfuse-Trace-Id", "")).strip().lower()
    if not trace_id:
        return None
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        return None
    return {"trace_id": trace_id}


def _extract_langfuse_session_id(request: Request) -> str | None:
    value = str(request.headers.get("X-Langfuse-Session-Id", "")).strip()
    return value[:128] if value else None


def _extract_langfuse_user_id(request: Request) -> str | None:
    value = str(request.headers.get("X-Langfuse-User-Id", "")).strip()
    return value[:128] if value else None


def _post_json(url: str, payload: dict[str, Any], timeout_sec: float, settings) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.embedding_api_key:
        key_header = settings.embedding_api_key_header or "X-API-Key"
        headers[key_header] = settings.embedding_api_key
    if settings.embedding_api_auth_token:
        headers["Authorization"] = f"Bearer {settings.embedding_api_auth_token}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("warm endpoint returned non-object JSON payload")
    return parsed


def _warm_embedding_model(svc, settings) -> None:
    warm_text = (settings.embedding_warm_text or "warmup retrieval embedding").strip()
    if not warm_text:
        warm_text = "warmup retrieval embedding"
    embedding = svc._embed_remote_text(warm_text)
    log.info("model warm embedding ok dim=%s", len(embedding))


def _warm_llm_model(settings) -> None:
    if not settings.llm_warm_api_url or not settings.llm_warm_model:
        return

    payload: dict[str, Any] = {
        "model": settings.llm_warm_model,
        "prompt": settings.llm_warm_prompt,
        "stream": False,
    }
    if settings.llm_warm_keep_alive:
        payload["keep_alive"] = settings.llm_warm_keep_alive

    parsed = _post_json(
        url=settings.llm_warm_api_url,
        payload=payload,
        timeout_sec=settings.llm_warm_timeout_sec,
        settings=settings,
    )
    _ = parsed.get("done")
    log.info("model warm llm ok model=%s", settings.llm_warm_model)


def _run_model_warmer_loop(stop_event: threading.Event, get_service_instance, settings) -> None:
    initial_delay = max(0.0, float(settings.model_warm_initial_delay_sec))
    if stop_event.wait(initial_delay):
        return

    interval_sec = max(5.0, float(settings.model_warm_interval_sec))
    while not stop_event.is_set():
        try:
            svc = get_service_instance()
            if settings.embedding_backend == "remote":
                _warm_embedding_model(svc, settings)
            if settings.llm_warm_enabled and settings.llm_warm_api_url and settings.llm_warm_model:
                _warm_llm_model(settings)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            log.warning("model warm HTTP %s: %s", exc.code, raw[:200])
        except urllib.error.URLError as exc:
            log.warning("model warm unreachable: %s", exc.reason)
        except socket.timeout:
            log.warning("model warm timeout")
        except Exception as exc:
            log.warning("model warm failed: %s", exc)

        if stop_event.wait(interval_sec):
            return


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="Minimal Retrieval Service", version="0.1.0")
    holder = {"service": service}
    warmer_holder: dict[str, Any] = {"thread": None, "stop_event": None}
    settings = get_settings()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    def get_service_instance():
        if holder["service"] is None:
            from retrieval_service.pipeline import RetrievalService

            holder["service"] = RetrievalService(settings)
        return holder["service"]

    @app.on_event("startup")
    def startup() -> None:
        try:
            svc = get_service_instance()
            log.info(
                "startup host=%s port=%s collection=%s",
                settings.host,
                settings.port,
                svc.settings.collection_name,
            )
            llm_warm_ready = bool(settings.llm_warm_api_url and settings.llm_warm_model)
            if settings.llm_warm_enabled and not llm_warm_ready:
                log.warning("llm warm requested but missing LLM_WARM_API_URL or LLM_WARM_MODEL; llm warm disabled")

            warm_targets = []
            if settings.embedding_backend == "remote":
                warm_targets.append("embedding")
            if settings.llm_warm_enabled and llm_warm_ready:
                warm_targets.append("llm")

            if settings.model_warm_enabled and warm_targets:
                stop_event = threading.Event()
                thread = threading.Thread(
                    target=_run_model_warmer_loop,
                    args=(stop_event, get_service_instance, settings),
                    name="model-warmer",
                    daemon=True,
                )
                thread.start()
                warmer_holder["thread"] = thread
                warmer_holder["stop_event"] = stop_event
                log.info(
                    "model warm enabled targets=%s interval=%ss initial_delay=%ss",
                    ",".join(warm_targets),
                    settings.model_warm_interval_sec,
                    settings.model_warm_initial_delay_sec,
                )
            else:
                log.info("model warm disabled")
        except Exception as exc:
            log.exception("startup failed: %s", exc)
            raise

    @app.on_event("shutdown")
    def shutdown() -> None:
        stop_event = warmer_holder.get("stop_event")
        thread = warmer_holder.get("thread")
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=3)
        warmer_holder["thread"] = None
        warmer_holder["stop_event"] = None

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc = get_service_instance()
        payload = HealthResponse(**svc.health())
        log.info("health status=%s collection=%s", payload.status, payload.collection)
        return payload

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
        svc = get_service_instance()
        try:
            with start_observation(
                settings.langfuse_enabled,
                name="retrieval.ingest",
                as_type="span",
                input={"documents": len(payload.documents)},
                metadata={"service": "retrieval-service", "endpoint": "/ingest"},
                trace_context=_extract_langfuse_trace_context(request),
            ) as obs:
                with propagate_context(
                    settings.langfuse_enabled,
                    session_id=_extract_langfuse_session_id(request),
                    user_id=_extract_langfuse_user_id(request),
                    metadata={"service": "retrieval-service"},
                ):
                    count = svc.ingest(payload.documents)
                    response = IngestResponse(ingested=count, collection=svc.settings.collection_name)
                    obs.update(output={"ingested": response.ingested, "collection": response.collection})
                    log.info("ingest request_docs=%s ingested=%s", len(payload.documents), response.ingested)
                    return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if settings.langfuse_flush_at_request_end:
                flush_observability(settings.langfuse_enabled)

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
        svc = get_service_instance()
        try:
            with start_observation(
                settings.langfuse_enabled,
                name="retrieval.retrieve",
                as_type="span",
                input={"query": payload.query[:500], "top_k": payload.top_k},
                metadata={"service": "retrieval-service", "endpoint": "/retrieve"},
                trace_context=_extract_langfuse_trace_context(request),
            ) as obs:
                with propagate_context(
                    settings.langfuse_enabled,
                    session_id=_extract_langfuse_session_id(request),
                    user_id=_extract_langfuse_user_id(request),
                    metadata={"service": "retrieval-service"},
                ):
                    raw = svc.retrieve(payload.query, payload.top_k)
                    if isinstance(raw, dict):
                        results = raw.get("results", []) or []
                        confidence = float(raw.get("confidence", 0.0))
                        low_confidence = bool(raw.get("low_confidence", False))
                    else:
                        # Backward compatibility for simple test doubles that return list
                        results = raw or []
                        confidence = float(results[0]["score"]) if results else 0.0
                        threshold = getattr(getattr(svc, "settings", object()), "min_retrieve_score", 0.0)
                        low_confidence = confidence < float(threshold)
                    response = RetrieveResponse(
                        results=results,
                        confidence=confidence,
                        low_confidence=low_confidence,
                    )
                    obs.update(
                        output={
                            "results": len(response.results),
                            "confidence": response.confidence,
                            "low_confidence": response.low_confidence,
                        }
                    )
                    log.info(
                        "retrieve query_len=%s top_k=%s results=%s confidence=%.4f low_confidence=%s",
                        len(payload.query),
                        payload.top_k or svc.settings.default_top_k,
                        len(response.results),
                        response.confidence or 0.0,
                        response.low_confidence,
                    )
                    return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if settings.langfuse_flush_at_request_end:
                flush_observability(settings.langfuse_enabled)

    @app.post("/retrieve/project-grounded", response_model=ProjectGroundedRetrieveResponse)
    def retrieve_project_grounded(payload: ProjectGroundedRetrieveRequest, request: Request) -> ProjectGroundedRetrieveResponse:
        svc = get_service_instance()
        try:
            with start_observation(
                settings.langfuse_enabled,
                name="retrieval.project_grounded",
                as_type="span",
                input={"query": payload.query[:500], "top_k": payload.top_k},
                metadata={"service": "retrieval-service", "endpoint": "/retrieve/project-grounded"},
                trace_context=_extract_langfuse_trace_context(request),
            ) as obs:
                with propagate_context(
                    settings.langfuse_enabled,
                    session_id=_extract_langfuse_session_id(request),
                    user_id=_extract_langfuse_user_id(request),
                    metadata={"service": "retrieval-service"},
                ):
                    retrieval_intent = payload.retrieval_intent
                    if hasattr(retrieval_intent, "model_dump"):
                        retrieval_intent = retrieval_intent.model_dump()
                    raw = svc.retrieve_project_grounded(
                        query=payload.query,
                        retrieval_intent=retrieval_intent,
                        top_k=payload.top_k,
                    )
                    response = ProjectGroundedRetrieveResponse(
                        project_cards=raw.get("project_cards", []) or [],
                        trait_tags=raw.get("trait_tags", []) or [],
                        proximity_facts=raw.get("proximity_facts", []) or [],
                        evidence_chunks=raw.get("evidence_chunks", []) or [],
                        confidence=raw.get("confidence"),
                        low_confidence=bool(raw.get("low_confidence", False)),
                    )
                    obs.update(
                        output={
                            "project_cards": len(response.project_cards),
                            "trait_tags": len(response.trait_tags),
                            "proximity_facts": len(response.proximity_facts),
                            "evidence_chunks": len(response.evidence_chunks),
                            "confidence": response.confidence,
                            "low_confidence": response.low_confidence,
                        }
                    )
                    log.info(
                        (
                            "project-grounded retrieve query_len=%s top_k=%s project_cards=%s "
                            "trait_tags=%s proximity_facts=%s evidence_chunks=%s confidence=%.4f low_confidence=%s"
                        ),
                        len(payload.query),
                        payload.top_k or getattr(getattr(svc, "settings", object()), "default_top_k", 5),
                        len(response.project_cards),
                        len(response.trait_tags),
                        len(response.proximity_facts),
                        len(response.evidence_chunks),
                        response.confidence or 0.0,
                        response.low_confidence,
                    )
                    return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if settings.langfuse_flush_at_request_end:
                flush_observability(settings.langfuse_enabled)

    return app


app = create_app()
