"""Sales agent API endpoints."""

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from integrations.camera_identity import maybe_enrich_identity
from models.api_models import LeadUpdateRequest, SalesChatRequest, SalesChatResponse
from memory.chat_history_store import delete_chat_history, load_chat_history
from memory.lead_profile_store import (
    delete_lead_profile_cache,
    ensure_sales_schema,
    load_lead_profile,
    save_lead_profile,
)
from memory.local_snapshot_store import (
    get_local_snapshot_path,
    load_local_session_snapshot,
    read_local_session_snapshot_text,
)
from memory.session_store import delete_session_context, load_session_context
from rag.retriever import route_query, summarize_search_answer
from sales.graph import sales_graph
from sales.session_export import export_session_to_txt
from utils.text import iter_stream_chunks

router = APIRouter(prefix="/sales", tags=["sales"])
log = logging.getLogger("rag-service")
_THINKING_ACK_MESSAGES = [
    "Em đã nhận được thông tin rồi ạ, Anh/Chị chờ em một chút để em kiểm tra nhanh nhé.",
    "Em nhận yêu cầu của Anh/Chị rồi, cho em ít giây để em xử lý và phản hồi chuẩn nhất nhé.",
    "Em đang tiếp nhận nội dung của Anh/Chị, em rà nhanh dữ liệu rồi trả lời ngay ạ.",
    "Em đã ghi nhận câu hỏi, Anh/Chị đợi em một lát để em đối chiếu thông tin cho chính xác nhé.",
    "Em nhận được rồi ạ, em đang xử lý nhanh để gửi lại câu trả lời ngắn gọn cho Anh/Chị.",
]
_SEARCH_PERSONA = (
    "Bạn là trợ lý tư vấn của Noble. "
    "Với câu hỏi ngoài kho tri thức dự án, hãy trả lời trực tiếp, tự nhiên, ngắn gọn nhưng hữu ích. "
    "Nếu thông tin phụ thuộc thời gian, hãy ưu tiên dữ liệu vừa tìm được."
)


async def _run_sales_or_search(
    *,
    session_id: str,
    user_text: str,
    raw_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    history = await load_chat_history(session_id)
    category, routed_query = await route_query(user_text, history)

    if category == "SEARCH":
        answer = await summarize_search_answer(
            user_query=user_text,
            search_query=routed_query or user_text,
            history=history,
            system_persona=_SEARCH_PERSONA,
            max_sentences=4,
        )
        return {
            "route_category": "SEARCH",
            "final_response": answer,
            "next_sales_state": None,
            "lead_profile": await load_lead_profile(session_id),
            "missing_slots": None,
        }

    initial_state: Dict[str, Any] = {
        "session_id": session_id,
        "user_text": user_text,
        "raw_transcript": raw_transcript,
        "errors": [],
    }
    result = await sales_graph.ainvoke(initial_state)
    result["route_category"] = "SALES"
    return result


@router.post("/chat", response_model=SalesChatResponse)
async def sales_chat(request: SalesChatRequest):
    """
    Main sales agent endpoint.
    Invokes the LangGraph sales orchestrator and returns the AI response.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await ensure_sales_schema()
    await maybe_enrich_identity(request.session_id)

    try:
        result = await _run_sales_or_search(
            session_id=request.session_id,
            user_text=request.message.strip(),
            raw_transcript=request.raw_transcript,
        )
    except Exception as e:
        log.error("sales_graph.ainvoke error: %s", e)
        raise HTTPException(status_code=500, detail=f"Sales agent error: {e}")

    return SalesChatResponse(
        session_id=request.session_id,
        response=result.get("final_response") or "",
        route_category=result.get("route_category"),
        sales_state=result.get("next_sales_state"),
        lead_profile=result.get("lead_profile"),
        missing_slots=result.get("missing_slots"),
    )


@router.post("/chat/stream")
async def sales_chat_stream(request: SalesChatRequest):
    """Streaming sales chat with immediate thinking_ack for better UX."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await ensure_sales_schema()
    await maybe_enrich_identity(request.session_id)

    async def generate():
        t_total = time.perf_counter()
        yield json.dumps(
            {
                "chunk": random.choice(_THINKING_ACK_MESSAGES),
                "done": False,
                "phase": "thinking_ack",
                "session_id": request.session_id,
            },
            ensure_ascii=False,
        ) + "\n"

        try:
            task = asyncio.create_task(
                _run_sales_or_search(
                    session_id=request.session_id,
                    user_text=request.message.strip(),
                    raw_transcript=request.raw_transcript,
                )
            )
            result = await task
        except Exception as e:
            log.error("sales_graph.ainvoke(stream) error: %s", e)
            fallback = "Xin lỗi, em gặp lỗi khi xử lý yêu cầu. Anh/Chị thử lại giúp em nhé."
            yield json.dumps(
                {
                    "chunk": fallback,
                    "done": False,
                    "phase": "error",
                    "session_id": request.session_id,
                },
                ensure_ascii=False,
            ) + "\n"
            yield json.dumps(
                {"chunk": "", "done": True, "phase": "complete", "session_id": request.session_id},
                ensure_ascii=False,
            ) + "\n"
            return

        full_answer = (result.get("final_response") or "").strip()
        if not full_answer:
            full_answer = "Xin lỗi, em chưa có đủ dữ liệu để tư vấn. Anh/Chị chia sẻ thêm giúp em nhé."

        for chunk in iter_stream_chunks(full_answer):
            yield json.dumps(
                {
                    "chunk": chunk,
                    "done": False,
                    "phase": "response",
                    "session_id": request.session_id,
                },
                ensure_ascii=False,
            ) + "\n"
            await asyncio.sleep(0)

        yield json.dumps(
            {
                "chunk": "",
                "done": True,
                "phase": "complete",
                "session_id": request.session_id,
                "route_category": result.get("route_category"),
                "sales_state": result.get("next_sales_state"),
                "missing_slots": result.get("missing_slots"),
                "lead_profile": result.get("lead_profile"),
                "latency_sec": round(time.perf_counter() - t_total, 3),
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/lead/{session_id}")
async def get_lead(session_id: str):
    """Get lead profile for a session."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")
    return JSONResponse({"session_id": session_id, "lead_profile": profile})


@router.patch("/lead/{session_id}")
async def update_lead(session_id: str, body: LeadUpdateRequest):
    """Manually update lead profile fields."""
    profile = await load_lead_profile(session_id) or {"lead_id": session_id, "current_state": "greeting"}
    profile.update(body.updates)
    await save_lead_profile(session_id, profile)
    return JSONResponse({"session_id": session_id, "lead_profile": profile})


@router.get("/state/{session_id}")
async def get_sales_state(session_id: str):
    """Get current sales state for a session."""
    ctx = await load_session_context(session_id)
    profile = await load_lead_profile(session_id) or {}
    return JSONResponse(
        {
            "session_id": session_id,
            "current_state": ctx.get("current_state", "greeting"),
            "previous_state": ctx.get("previous_state"),
            "conversation_turn_count": ctx.get("conversation_turn_count", 0),
            "lead_temperature": profile.get("lead_temperature", "cold"),
        }
    )


@router.get("/session/{session_id}/snapshot")
async def get_session_snapshot(session_id: str):
    """Read the local txt snapshot for a session."""
    snapshot = load_local_session_snapshot(session_id)
    raw_text = read_local_session_snapshot_text(session_id)
    if not snapshot and not raw_text:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return JSONResponse(
        {
            "session_id": session_id,
            "snapshot_path": get_local_snapshot_path(session_id),
            "updated_at": snapshot.get("updated_at"),
            "lead_profile": snapshot.get("lead_profile"),
            "session_context": snapshot.get("session_context"),
            "chat_history": snapshot.get("chat_history"),
            "raw_text": raw_text,
        }
    )


@router.post("/recommendations/refresh")
async def refresh_recommendations(session_id: str):
    """Trigger a product_matching cycle for the session."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")

    initial_state: Dict[str, Any] = {
        "session_id": session_id,
        "user_text": "Anh/Chị muốn xem lại các sản phẩm phù hợp.",
        "errors": [],
    }
    try:
        result = await sales_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh error: {e}")

    return JSONResponse(
        {
            "session_id": session_id,
            "response": result.get("final_response", ""),
            "sales_state": result.get("next_sales_state"),
        }
    )


@router.post("/followup/generate")
async def generate_followup(session_id: str):
    """Generate a follow-up message for a lead."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")

    initial_state: Dict[str, Any] = {
        "session_id": session_id,
        "user_text": "Anh/Chị có cần em hỗ trợ thêm thông tin gì không?",
        "errors": [],
    }
    try:
        result = await sales_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up error: {e}")

    return JSONResponse(
        {
            "session_id": session_id,
            "response": result.get("final_response", ""),
        }
    )


@router.post("/session/{session_id}/close")
async def close_session(session_id: str):
    """Export session data to txt and clear ephemeral session caches."""
    await ensure_sales_schema()
    profile = await load_lead_profile(session_id) or {}
    ctx = await load_session_context(session_id)
    history = await load_chat_history(session_id)

    if not profile and not history:
        raise HTTPException(status_code=404, detail="Session not found")

    export_path = export_session_to_txt(
        session_id=session_id,
        lead_profile=profile,
        session_context=ctx,
        chat_history=history,
    )

    await delete_chat_history(session_id)
    await delete_session_context(session_id)
    await delete_lead_profile_cache(session_id)

    return JSONResponse(
        {
            "session_id": session_id,
            "status": "closed",
            "export_path": export_path,
            "messages_exported": len(history),
        }
    )


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get all chat messages (human and AI) for a specific session.
    Useful for frontend to poll or display session conversation history.
    """
    history = await load_chat_history(session_id)
    if not history:
        # Don't 404, just return empty list as session might be new
        return JSONResponse({"session_id": session_id, "history": []})

    formatted_history = []
    for msg in history:
        # history returned from store usually contains langchain objects or dicts
        # format according to your storage implementation pattern
        formatted_history.append({
            "role": msg.type if hasattr(msg, "type") else msg.get("role", "unknown"),
            "content": msg.content if hasattr(msg, "content") else msg.get("content", ""),
            "timestamp": msg.additional_kwargs.get("timestamp") if hasattr(msg, "additional_kwargs") else msg.get("timestamp")
        })

    return JSONResponse(
        {
            "session_id": session_id,
            "history": formatted_history
        }
    )
