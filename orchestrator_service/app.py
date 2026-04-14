from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException

from orchestrator_service.config import get_settings
from orchestrator_service.retrieval_client import RetrievalClient
from orchestrator_service.schemas import QueryRequest, QueryResponse, RetrievalPayload

log = logging.getLogger("sales-orchestrator")

_RETRIEVAL_KEYWORDS = {
    "gia",
    "giá",
    "dien tich",
    "diện tích",
    "ho tay",
    "hồ tây",
    "phap ly",
    "pháp lý",
    "ngan hang",
    "ngân hàng",
    "thanh toan",
    "thanh toán",
    "ban giao",
    "bàn giao",
    "tien ich",
    "tiện ích",
}


def _needs_retrieval_heuristic(message: str) -> bool:
    text = (message or "").strip().lower()
    return any(key in text for key in _RETRIEVAL_KEYWORDS)


def create_app(retrieval_fetcher=None) -> FastAPI:
    app = FastAPI(title="Minimal Sales Orchestrator", version="0.1.0")
    settings = get_settings()

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    client = RetrievalClient(
        base_url=settings.retrieval_service_url,
        timeout_sec=settings.retrieval_timeout_sec,
    )

    if retrieval_fetcher is None:
        retrieval_fetcher = client.retrieve

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "sales-orchestrator"}

    @app.post("/sales/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        message = payload.message.strip()
        need_retrieval = payload.need_retrieval

        if need_retrieval is None:
            need_retrieval = _needs_retrieval_heuristic(message)

        if not need_retrieval:
            response = QueryResponse(
                route="no_retrieval_needed",
                action="proceed_without_retrieval",
                answer="Mình đã ghi nhận. Bạn muốn làm rõ thêm tiêu chí nào để mình tư vấn chính xác hơn?",
                decision_reason="need_retrieval=false",
                retrieval=None,
            )
            log.info("decision route=%s action=%s reason=%s", response.route, response.action, response.decision_reason)
            return response

        top_k = payload.top_k or settings.default_top_k
        try:
            retrieval_raw: dict[str, Any] = retrieval_fetcher(message, top_k)
        except Exception as exc:
            log.exception("retrieval call failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"retrieval service error: {exc}") from exc

        retrieval_payload = RetrievalPayload(
            results=retrieval_raw.get("results", []) or [],
            confidence=float(retrieval_raw.get("confidence", 0.0) or 0.0),
            low_confidence=bool(retrieval_raw.get("low_confidence", False)),
        )

        has_results = len(retrieval_payload.results) > 0
        if retrieval_payload.low_confidence or not has_results:
            response = QueryResponse(
                route="retrieval_low_confidence",
                action="ask_clarifying_question",
                answer=(
                    "Mình chưa đủ dữ liệu chắc chắn để tư vấn ngay. "
                    "Bạn có thể cho mình thêm tiêu chí cụ thể (ngân sách, khu vực, diện tích) không?"
                ),
                decision_reason="retrieval low_confidence=true or empty results",
                retrieval=retrieval_payload,
            )
            log.info("decision route=%s action=%s reason=%s", response.route, response.action, response.decision_reason)
            return response

        top_context = retrieval_payload.results[:3]
        context_summary = " | ".join(
            f"{item.source}:{item.doc_id} ({item.score:.2f})" for item in top_context
        )
        response = QueryResponse(
            route="retrieval_confident",
            action="use_retrieved_context",
            answer=(
                "Mình đã tìm được ngữ cảnh phù hợp và có thể tiếp tục nhánh tư vấn. "
                f"Nguồn chính: {context_summary}"
            ),
            decision_reason="retrieval confident and has results",
            retrieval=retrieval_payload,
        )
        log.info("decision route=%s action=%s reason=%s", response.route, response.action, response.decision_reason)
        return response

    return app


app = create_app()
