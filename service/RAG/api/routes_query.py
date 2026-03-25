"""Streaming query endpoint backed by the sales agent flow."""

import json
import logging
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import get_settings
from models.api_models import QueryRequest
from sales.graph import sales_graph

router = APIRouter(tags=["query"])
settings = get_settings()
log = logging.getLogger("rag-service")


@router.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    user_text = (request.query or request.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def generate():
        t_total = time.perf_counter()
        session_id = (request.session_id or "").strip() or f"sales_stream_{uuid.uuid4().hex}"

        yield json.dumps(
            {"chunk": "  \n*(Em đang phân tích nhu cầu của Anh/Chị...)*", "done": False},
            ensure_ascii=False,
        ) + "\n"

        try:
            result = await sales_graph.ainvoke(
                {
                    "session_id": session_id,
                    "user_text": user_text,
                    "errors": [],
                }
            )
        except Exception as e:
            log.error("sales_graph stream error: %s", e)
            fallback = "Xin lỗi, em gặp lỗi khi xử lý yêu cầu. Anh/Chị thử lại giúp em nhé."
            yield json.dumps({"chunk": fallback, "done": False}, ensure_ascii=False) + "\n"
            yield json.dumps(
                {"chunk": "", "done": True, "model": settings.llm_model, "session_id": session_id},
                ensure_ascii=False,
            ) + "\n"
            return

        full_answer = (result.get("final_response") or "").strip()
        if not full_answer:
            full_answer = "Xin lỗi, em chưa có đủ dữ liệu để tư vấn. Anh/Chị chia sẻ thêm giúp em nhé."

        for sent in re.split(r"(?<=[.!?\n])\s*", full_answer):
            if sent.strip():
                yield json.dumps({"chunk": sent.strip(), "done": False}, ensure_ascii=False) + "\n"

        yield json.dumps(
            {
                "chunk": "",
                "done": True,
                "model": settings.llm_model,
                "session_id": session_id,
                "sales_state": result.get("next_sales_state"),
                "missing_slots": result.get("missing_slots"),
                "lead_profile": result.get("lead_profile"),
            },
            ensure_ascii=False,
        ) + "\n"
        log.info("Timing[stream.total]: %.2fs", time.perf_counter() - t_total)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
