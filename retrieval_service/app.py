from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi import HTTPException

from retrieval_service.config import get_settings
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


def create_app(service=None) -> FastAPI:
    app = FastAPI(title="Minimal Retrieval Service", version="0.1.0")
    holder = {"service": service}
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
        except Exception as exc:
            log.exception("startup failed: %s", exc)
            raise

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc = get_service_instance()
        payload = HealthResponse(**svc.health())
        log.info("health status=%s collection=%s", payload.status, payload.collection)
        return payload

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest) -> IngestResponse:
        svc = get_service_instance()
        try:
            count = svc.ingest(payload.documents)
            response = IngestResponse(ingested=count, collection=svc.settings.collection_name)
            log.info("ingest request_docs=%s ingested=%s", len(payload.documents), response.ingested)
            return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
        svc = get_service_instance()
        try:
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

    @app.post("/retrieve/project-grounded", response_model=ProjectGroundedRetrieveResponse)
    def retrieve_project_grounded(payload: ProjectGroundedRetrieveRequest) -> ProjectGroundedRetrieveResponse:
        svc = get_service_instance()
        try:
            raw = svc.retrieve_project_grounded(
                query=payload.query,
                retrieval_intent=payload.retrieval_intent,
                top_k=payload.top_k,
            )
            response = ProjectGroundedRetrieveResponse(
                project_cards=raw.get("project_cards", []) or [],
                trait_tags=raw.get("trait_tags", []) or [],
                evidence_chunks=raw.get("evidence_chunks", []) or [],
                confidence=raw.get("confidence"),
                low_confidence=bool(raw.get("low_confidence", False)),
            )
            log.info(
                (
                    "project-grounded retrieve query_len=%s top_k=%s project_cards=%s "
                    "trait_tags=%s evidence_chunks=%s confidence=%.4f low_confidence=%s"
                ),
                len(payload.query),
                payload.top_k or getattr(getattr(svc, "settings", object()), "default_top_k", 5),
                len(response.project_cards),
                len(response.trait_tags),
                len(response.evidence_chunks),
                response.confidence or 0.0,
                response.low_confidence,
            )
            return response
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
