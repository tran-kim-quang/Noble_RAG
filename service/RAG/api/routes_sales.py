"""Sales agent API endpoints."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

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
from sales.graph import sales_graph
from sales.session_export import export_session_to_txt

router = APIRouter(prefix="/sales", tags=["sales"])
log = logging.getLogger("rag-service")


@router.post("/chat", response_model=SalesChatResponse)
async def sales_chat(request: SalesChatRequest):
    """
    Main sales agent endpoint.
    Invokes the LangGraph sales orchestrator and returns the AI response.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await ensure_sales_schema()

    initial_state: Dict[str, Any] = {
        "session_id": request.session_id,
        "user_text": request.message.strip(),
        "raw_transcript": request.raw_transcript,
        "errors": [],
    }

    try:
        result = await sales_graph.ainvoke(initial_state)
    except Exception as e:
        log.error("sales_graph.ainvoke error: %s", e)
        raise HTTPException(status_code=500, detail=f"Sales agent error: {e}")

    return SalesChatResponse(
        session_id=request.session_id,
        response=result.get("final_response") or "",
        sales_state=result.get("next_sales_state"),
        lead_profile=result.get("lead_profile"),
        missing_slots=result.get("missing_slots"),
    )


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
