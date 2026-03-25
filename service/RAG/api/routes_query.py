"""Streaming query endpoint backed by the sales agent flow."""

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.config import get_settings
from memory.chat_history_store import load_chat_history
from rag.retriever import route_query, summarize_search_answer
from sales.edges import route_after_action_resolution
from models.api_models import QueryRequest
from sales.nodes.build_response import build_response
from sales.nodes.classify_and_extract import classify_and_extract
from sales.nodes.decide_response_action import decide_response_action_node
from sales.nodes.finalize_output import finalize_output
from sales.nodes.ingest_user_turn import ingest_user_turn
from sales.nodes.persist_turn import persist_turn
from sales.nodes.render_response_from_template import render_response_from_template
from sales.nodes.resolve_sales_state import resolve_sales_state_node
from sales.nodes.resolve_script_step import resolve_script_step_node
from sales.nodes.retrieve_context import retrieve_context
from sales.nodes.update_lead_profile import update_lead_profile
from sales.nodes.validate_response import validate_response_node
from utils.text import iter_stream_chunks

router = APIRouter(tags=["query"])
settings = get_settings()
log = logging.getLogger("rag-service")
_SEARCH_PERSONA = (
    "Bạn là trợ lý tư vấn của Noble. "
    "Với câu hỏi ngoài kho tri thức dự án, hãy trả lời trực tiếp, tự nhiên, ngắn gọn nhưng hữu ích. "
    "Nếu thông tin phụ thuộc thời gian, hãy ưu tiên dữ liệu vừa tìm được."
)


async def _run_sales_flow_streaming(session_id: str, user_text: str):
    state: dict[str, Any] = {
        "session_id": session_id,
        "user_text": user_text,
        "errors": [],
    }

    yield {"chunk": "", "done": False, "phase": "ingest"}
    state.update(await ingest_user_turn(state))

    yield {"chunk": "", "done": False, "phase": "classify"}
    state.update(await classify_and_extract(state))
    state.update(update_lead_profile(state))
    state.update(resolve_sales_state_node(state))
    state.update(resolve_script_step_node(state))
    state.update(decide_response_action_node(state))

    route = route_after_action_resolution(state)
    if route == "render_response_from_template":
        yield {"chunk": "", "done": False, "phase": "template"}
        state.update(render_response_from_template(state))
    else:
        yield {"chunk": "", "done": False, "phase": "retrieve"}
        state.update(await retrieve_context(state))
        yield {"chunk": "", "done": False, "phase": "respond"}
        state.update(await build_response(state))

    state.update(validate_response_node(state))
    state.update(await persist_turn(state))
    state.update(finalize_output(state))
    yield {"result": state}


@router.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    user_text = (request.query or request.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def generate():
        t_total = time.perf_counter()
        session_id = (request.session_id or "").strip() or f"sales_stream_{uuid.uuid4().hex}"

        try:
            history = await load_chat_history(session_id)
            yield json.dumps({"chunk": "", "done": False, "phase": "route"}, ensure_ascii=False) + "\n"
            category, routed_query = await route_query(user_text, history)

            if category == "SEARCH":
                yield json.dumps({"chunk": "", "done": False, "phase": "search"}, ensure_ascii=False) + "\n"
                search_answer = await summarize_search_answer(
                    user_query=user_text,
                    search_query=routed_query or user_text,
                    history=history,
                    system_persona=_SEARCH_PERSONA,
                    max_sentences=4,
                )
                for chunk in iter_stream_chunks(search_answer):
                    yield json.dumps({"chunk": chunk, "done": False}, ensure_ascii=False) + "\n"
                yield json.dumps(
                    {
                        "chunk": "",
                        "done": True,
                        "model": settings.llm_model,
                        "session_id": session_id,
                        "route_category": category,
                    },
                    ensure_ascii=False,
                ) + "\n"
                log.info("Timing[stream.total]: %.2fs", time.perf_counter() - t_total)
                return

            result = None
            async for event in _run_sales_flow_streaming(session_id, user_text):
                if "result" in event:
                    result = event["result"]
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            log.error("sales_graph stream error: %s", e)
            fallback = "Xin lỗi, em gặp lỗi khi xử lý yêu cầu. Anh/Chị thử lại giúp em nhé."
            yield json.dumps({"chunk": fallback, "done": False}, ensure_ascii=False) + "\n"
            yield json.dumps(
                {"chunk": "", "done": True, "model": settings.llm_model, "session_id": session_id},
                ensure_ascii=False,
            ) + "\n"
            return

        result = result or {}
        full_answer = (result.get("final_response") or "").strip()
        if not full_answer:
            full_answer = "Xin lỗi, em chưa có đủ dữ liệu để tư vấn. Anh/Chị chia sẻ thêm giúp em nhé."

        for chunk in iter_stream_chunks(full_answer):
            yield json.dumps({"chunk": chunk, "done": False}, ensure_ascii=False) + "\n"

        yield json.dumps(
            {
                "chunk": "",
                "done": True,
                "model": settings.llm_model,
                "session_id": session_id,
                "route_category": "RAG",
                "sales_state": result.get("next_sales_state"),
                "missing_slots": result.get("missing_slots"),
                "lead_profile": result.get("lead_profile"),
            },
            ensure_ascii=False,
        ) + "\n"
        log.info("Timing[stream.total]: %.2fs", time.perf_counter() - t_total)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
