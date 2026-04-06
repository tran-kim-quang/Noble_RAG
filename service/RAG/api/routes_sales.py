"""Sales agent API endpoints."""

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from chat.service import run_chat_route
from integrations.camera_identity import maybe_enrich_identity
from knowledge_base.service import resolve_knowledge
from models.api_models import LeadUpdateRequest, SalesChatRequest, SalesChatResponse
from core.config import get_settings
from memory.chat_history_store import (
    append_turn,
    delete_chat_history,
    load_chat_history,
    purge_all_chat_history,
)
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
from integrations.machine_b_face_client import notify_machine_b_customer_removed
from sales.session_export import export_session_to_txt
from sales.response_templates import _apply_customer_pronoun
from utils.text import iter_stream_chunks

router = APIRouter(prefix="/sales", tags=["sales"])
log = logging.getLogger("rag-service")


def _require_chat_purge_token(x_chat_history_purge_token: Optional[str]) -> None:
    expected = (get_settings().chat_history_purge_token or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="CHAT_HISTORY_PURGE_TOKEN is not configured on server",
        )
    if (x_chat_history_purge_token or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid X-Chat-History-Purge-Token")
_THINKING_ACK_MESSAGES = [
    "Em đã nhận được thông tin rồi ạ, Anh/Chị chờ em một chút để em kiểm tra nhanh nhé.",
    "Em nhận yêu cầu của Anh/Chị rồi, cho em ít giây để em xử lý và phản hồi chuẩn nhất nhé.",
    "Em đang tiếp nhận nội dung của Anh/Chị, em rà nhanh dữ liệu rồi trả lời ngay ạ.",
    "Em đã ghi nhận câu hỏi, Anh/Chị đợi em một lát để em đối chiếu thông tin cho chính xác nhé.",
    "Em nhận được rồi ạ, em đang xử lý nhanh để gửi lại câu trả lời ngắn gọn cho Anh/Chị.",
]


async def _persist_chat_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    user = (user_text or "").strip()
    assistant = (assistant_text or "").strip()
    if not user or not assistant:
        return
    try:
        await append_turn(session_id, user, assistant)
    except Exception as e:
        log.warning("append_turn failed: session=%s error=%s", session_id, e)


def _stream_meta_from_flow_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Các khóa final_meta cho /chat/stream sau khi chạy _run_*_flow."""
    return {
        "route_category": result.get("route_category"),
        "sales_state": result.get("next_sales_state"),
        "missing_slots": result.get("missing_slots"),
        "lead_profile": result.get("lead_profile"),
        "knowledge_used": bool(result.get("knowledge_used")),
        "search_used": bool(result.get("search_used")),
        "kb_top_score": result.get("kb_top_score"),
    }


def _ndjson_response_line(session_id: str, chunk: str) -> str:
    """Một dòng NDJSON phase=response (dùng cho stream token hoặc chunk)."""
    return json.dumps(
        {
            "chunk": chunk,
            "done": False,
            "phase": "response",
            "session_id": session_id,
        },
        ensure_ascii=False,
    ) + "\n"


async def _yield_ndjson_response_chunks(session_id: str, full_text: str):
    """Sinh các dòng NDJSON (phase=response) từ nội dung đã có sẵn."""
    for chunk in iter_stream_chunks((full_text or "").strip()):
        yield _ndjson_response_line(session_id, chunk)
        await asyncio.sleep(0)


async def _run_sales_flow(
    *,
    session_id: str,
    user_text: str,
    raw_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    history = await load_chat_history(session_id)
    session_context = await load_session_context(session_id)
    lead_profile = await load_lead_profile(session_id)

    knowledge_payload = await resolve_knowledge(
        query=user_text,
        history=history,
        session_context=session_context,
        top_k=6,
    )
    result = await run_chat_route(
        session_id=session_id,
        message=user_text,
        history=history,
        lead_profile=lead_profile or {},
        session_context=session_context or {},
        knowledge_payload=knowledge_payload,
        raw_transcript=raw_transcript,
    )

    patched_answer = _apply_customer_pronoun(
        str(result.get("final_response") or ""),
        {"lead_profile": lead_profile, "session_context": session_context},
    )
    result["final_response"] = patched_answer
    result["lead_profile"] = lead_profile

    await _persist_chat_turn(session_id, user_text, patched_answer)
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
        result = await _run_sales_flow(
            session_id=request.session_id,
            user_text=request.message.strip(),
            raw_transcript=request.raw_transcript,
        )
    except Exception as e:
        log.error("sales chat pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Sales agent error: {e}")

    return SalesChatResponse(
        session_id=request.session_id,
        response=result.get("final_response") or "",
        route_category=result.get("route_category"),
        sales_state=result.get("next_sales_state"),
        lead_profile=result.get("lead_profile"),
        missing_slots=result.get("missing_slots"),
        knowledge_used=bool(result.get("knowledge_used")),
        search_used=bool(result.get("search_used")),
        kb_top_score=result.get("kb_top_score"),
    )


@router.post("/chat/stream")
async def sales_chat_stream(request: SalesChatRequest):
    """Streaming sales chat through the unified chat route."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await ensure_sales_schema()
    await maybe_enrich_identity(request.session_id)

    async def generate():
        t_total = time.perf_counter()
        user_text = request.message.strip()
        final_meta: Dict[str, Any] = {
            "route_category": None,
            "sales_state": None,
            "missing_slots": None,
            "lead_profile": None,
            "knowledge_used": False,
            "search_used": False,
            "kb_top_score": None,
        }
        yield json.dumps(
            {
                "chunk": _apply_customer_pronoun(random.choice(_THINKING_ACK_MESSAGES), {"lead_profile": await load_lead_profile(request.session_id), "session_context": await load_session_context(request.session_id)}),
                "done": False,
                "phase": "thinking_ack",
                "session_id": request.session_id,
            },
            ensure_ascii=False,
        ) + "\n"

        try:
            result = await _run_sales_flow(
                session_id=request.session_id,
                user_text=user_text,
                raw_transcript=request.raw_transcript,
            )
            async for line in _yield_ndjson_response_chunks(
                request.session_id,
                str(result.get("final_response") or ""),
            ):
                yield line
            final_meta.update(_stream_meta_from_flow_result(result))

            if not final_meta.get("route_category"):
                final_meta["route_category"] = "CHAT"
            if final_meta.get("lead_profile") is None:
                final_meta["lead_profile"] = await load_lead_profile(request.session_id)

        except Exception as e:
            log.error("sales chat stream pipeline error: %s", e)
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

        yield json.dumps(
            {
                "chunk": "",
                "done": True,
                "phase": "complete",
                "session_id": request.session_id,
                "route_category": final_meta.get("route_category"),
                "sales_state": final_meta.get("sales_state"),
                "missing_slots": final_meta.get("missing_slots"),
                "lead_profile": final_meta.get("lead_profile"),
                "knowledge_used": bool(final_meta.get("knowledge_used")),
                "search_used": bool(final_meta.get("search_used")),
                "kb_top_score": final_meta.get("kb_top_score"),
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
    """Generate refreshed recommendations using natural LLM consulting."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")

    try:
        result = await _run_sales_flow(
            session_id=session_id,
            user_text="Anh/Chị muốn xem lại các sản phẩm phù hợp.",
        )
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
    """Generate a follow-up message using natural LLM consulting."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")

    try:
        result = await _run_sales_flow(
            session_id=session_id,
            user_text="Anh/Chị có cần em hỗ trợ thêm thông tin gì không?",
        )
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

    # Đồng bộ xóa face embedding bên Machine B nếu session này có customer_id.
    customer_id = str(profile.get("customer_id") or "").strip()
    if not customer_id and isinstance(ctx, dict):
        customer_id = str(ctx.get("customer_id") or "").strip()
    machine_b_deleted: bool = False
    if customer_id:
        result = await notify_machine_b_customer_removed(customer_id)
        machine_b_deleted = bool(result and result.get("deleted"))

    return JSONResponse(
        {
            "session_id": session_id,
            "status": "closed",
            "export_path": export_path,
            "messages_exported": len(history),
            "machine_b_face_deleted": machine_b_deleted,
        }
    )


@router.post("/customer/{customer_id}/machine-b-face-removal")
async def notify_machine_b_after_customer_record_removed(
    customer_id: str,
    x_chat_history_purge_token: Optional[str] = Header(
        None, alias="X-Chat-History-Purge-Token"
    ),
):
    """Sau khi đã xóa / vô hiệu hóa khách trong DB (khóa `customer_id`), báo Máy B xóa embedding tương ứng.

    Gọi từ job/worker phía A ngay sau khi commit xóa bản ghi Postgres (hoặc tương đương).
    Cùng header bảo vệ với `purge-all`: `X-Chat-History-Purge-Token`.
    """
    _require_chat_purge_token(x_chat_history_purge_token)
    cid = (customer_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required")
    result = await notify_machine_b_customer_removed(cid)
    if result is None:
        raise HTTPException(
            status_code=502,
            detail="Could not notify Machine B (check MACHINE_B_BASE_URL, network, VISION_FACE_DELETE_TOKEN)",
        )
    return JSONResponse({"ok": True, "customer_id": cid, "machine_b": result})


@router.post("/chat-history/purge-all")
async def purge_all_chat_history_endpoint(
    x_chat_history_purge_token: Optional[str] = Header(
        None, alias="X-Chat-History-Purge-Token"
    ),
    notify_machine_b: bool = Query(
        True,
        description=(
            "True: trước khi xóa Redis, gom customer_id từ session_context + lead_profile_cache; "
            "sau purge gọi Máy B DELETE /v1/face/{id} cho từng id (cần MACHINE_B_BASE_URL + VISION_FACE_DELETE_TOKEN). "
            "False: chỉ xóa cache/snapshot, không gọi B."
        ),
    ),
):
    """Xóa sạch phiên sales: Redis, Postgres (`lead_profiles` + `customer_sessions` và CASCADE),
    toàn bộ file trong `live_snapshots/*.txt` (không chỉ xóa chat trong file).

    Gộp với nhu cầu "quên phiên": không chỉ xóa tin nhắn mà xóa luôn context/lead cache Redis
    để lần sau không đọc nhầm `customer_id` cũ.

    Response gồm `postgres_*_deleted`, `redis_session_context_keys_deleted`, `live_snapshot_files_deleted`,
    `customer_ids_collected_before_purge`, `session_ids_seen_in_redis_before_purge`, và `machine_b`.

    Tắt gọi B toàn cục: `.env` `PURGE_ALL_NOTIFY_MACHINE_B=false` (ghi đè ý định gọi API).

    Cần CHAT_HISTORY_PURGE_TOKEN; gửi header X-Chat-History-Purge-Token.
    """
    _require_chat_purge_token(x_chat_history_purge_token)
    stats = await purge_all_chat_history(notify_machine_b=notify_machine_b)
    return JSONResponse({"ok": True, **stats})


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
