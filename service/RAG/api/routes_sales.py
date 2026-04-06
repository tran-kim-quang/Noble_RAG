"""Sales agent API endpoints."""

import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from core.dependencies import llm_model_func
from integrations.camera_identity import maybe_enrich_identity
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
from rag.retriever import (
    build_rag_fact_constraints,
    plan_query_adaptive,
    query_rag,
    query_rag_stream,
    summarize_search_answer,
)
from integrations.machine_b_face_client import notify_machine_b_customer_removed
from sales.session_export import export_session_to_txt
from sales.response_templates import _apply_customer_pronoun
from sales.prompt_builder import build_prompt
from utils.text import iter_stream_chunks
from utils.time import now_vietnam_str

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
_SEARCH_PERSONA = (
    "Bạn là trợ lý tư vấn của Noble. "
    "Với câu hỏi ngoài kho tri thức dự án, hãy trả lời trực tiếp, tự nhiên, ngắn gọn nhưng hữu ích. "
    "Nếu thông tin phụ thuộc thời gian, hãy ưu tiên dữ liệu vừa tìm được."
)


async def _persist_chat_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    user = (user_text or "").strip()
    assistant = (assistant_text or "").strip()
    if not user or not assistant:
        return
    try:
        await append_turn(session_id, user, assistant)
    except Exception as e:
        log.warning("append_turn failed: session=%s error=%s", session_id, e)


async def _run_search_flow(
    *,
    session_id: str,
    user_text: str,
    search_query: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    answer = await summarize_search_answer(
        user_query=user_text,
        search_query=search_query or user_text,
        history=history,
        system_persona=_SEARCH_PERSONA,
        max_sentences=4,
    )
    # Áp dụng xưng hô (Anh/Chị/Nam/Nữ) dựa trên dữ liệu Máy B
    context = await load_session_context(session_id)
    profile = await load_lead_profile(session_id)
    state_for_pronoun = {"lead_profile": profile, "session_context": context}
    patched_answer = _apply_customer_pronoun(answer, state_for_pronoun)

    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "SEARCH",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


def _summarize_history_for_other(history: List[Dict[str, Any]], max_turns: int = 6) -> str:
    if not history:
        return "Chưa có lịch sử hội thoại trước đó."
    lines: List[str] = []
    for item in history[-(max_turns * 2):]:
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        speaker = "Người dùng" if role == "user" else ("Trợ lý" if role == "assistant" else role)
        lines.append(f"- {speaker}: {content}")
    return "\n".join(lines) if lines else "Chưa có lịch sử hội thoại trước đó."


async def _run_other_llm_flow(
    *,
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    history_summary = _summarize_history_for_other(history)
    # Chuẩn bị state tối thiểu để build_prompt có đủ dữ liệu xưng hô
    context = await load_session_context(session_id)
    profile = await load_lead_profile(session_id)
    state = {
        "session_id": session_id,
        "user_text": user_text,
        "chat_history": history,
        "lead_profile": profile,
        "session_context": context,
        "next_sales_state": "natural_consult",
    }
    
    # Lấy system prompt chuẩn từ prompt_builder (có xưng hô Máy B)
    system_prompt = build_prompt(state)
    
    prompt = f"""
{system_prompt}

YÊU CẦU BỔ SUNG CHO LUỒNG TỰ DO:
- Đã tóm tắt lịch sử hội thoại: {history_summary}
- Luôn trả lời tự nhiên, lịch sự, ngắn gọn, đúng ngữ cảnh.
- Ưu tiên dùng dữ liệu khách từ vision_context/lead_profile trước khi hỏi thêm.
- Nếu đã có độ tuổi, tên, mô tả/tính cách thì cá nhân hoá tư vấn và gợi ý sản phẩm phù hợp với hồ sơ đó.
- Không hỏi theo mẫu cứng kiểu "Gia đình có mấy người?", "Có mấy con nhỏ?".
- Nếu câu hỏi quá chung chung thì hỏi lại 1 câu làm rõ, không dùng kịch bản sales cứng.
""".strip()

    answer = ""
    try:
        answer = str(
            await llm_model_func(
                prompt,
                history_messages=history[-6:],
                temperature=0.2,
            )
            or ""
        ).strip()
    except Exception as e:
        log.warning("OTHER llm flow failed: %s", e)

    if not answer:
        answer = "Em đang ở đây để hỗ trợ mình. Anh/Chị muốn em tư vấn thông tin nào cụ thể hơn ạ?"

    # Áp dụng xưng hô (Anh/Chị/Nam/Nữ) dựa trên dữ liệu Máy B
    context = await load_session_context(session_id)
    profile = await load_lead_profile(session_id)
    state_for_pronoun = {"lead_profile": profile, "session_context": context}
    patched_answer = _apply_customer_pronoun(answer, state_for_pronoun)

    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "OTHER",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


def _stream_meta_from_flow_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Các khóa final_meta cho /chat/stream sau khi chạy _run_*_flow."""
    return {
        "route_category": result.get("route_category"),
        "sales_state": result.get("next_sales_state"),
        "missing_slots": result.get("missing_slots"),
        "lead_profile": result.get("lead_profile"),
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
    return await _run_other_llm_flow(
        session_id=session_id,
        user_text=user_text,
        history=history,
    )


def _normalize_plan(user_text: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    route = str(plan.get("route") or "RAG").upper().strip()
    if route not in {"RAG", "SEARCH", "OTHER"}:
        route = "RAG"

    refined = str(plan.get("refined_query") or user_text).strip() or user_text
    raw_subqueries = plan.get("subqueries")
    subqueries: List[Dict[str, str]] = []
    if isinstance(raw_subqueries, list):
        for item in raw_subqueries[:2]:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            sub_route = str(item.get("route") or route).upper().strip()
            if not query:
                continue
            if sub_route not in {"RAG", "SEARCH", "OTHER"}:
                sub_route = route
            subqueries.append({"query": query, "route": sub_route})

    multi_intent = bool(plan.get("multi_intent")) and len(subqueries) > 1
    if not multi_intent:
        subqueries = []

    return {
        "route": route,
        "refined_query": refined,
        "multi_intent": multi_intent,
        "subqueries": subqueries,
    }


async def _run_rag_flow(
    *,
    session_id: str,
    user_text: str,
    rag_query: str,
    history: List[Dict[str, Any]],
    raw_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    fact_context = await build_rag_fact_constraints(rag_query or user_text)
    rag_answer = await query_rag(
        text=rag_query or user_text,
        top_k=6,
        history=history,
        system_context=fact_context,
    )
    if rag_answer.strip():
        # Áp dụng xưng hô (Anh/Chị/Nam/Nữ) dựa trên dữ liệu Máy B
        context = await load_session_context(session_id)
        profile = await load_lead_profile(session_id)
        state_for_pronoun = {"lead_profile": profile, "session_context": context}
        patched_answer = _apply_customer_pronoun(rag_answer, state_for_pronoun)

        await _persist_chat_turn(session_id, user_text, patched_answer)
        return {
            "route_category": "RAG",
            "final_response": patched_answer,
            "next_sales_state": None,
            "lead_profile": profile,
            "missing_slots": None,
        }

    return await _run_sales_flow(
        session_id=session_id,
        user_text=user_text,
        raw_transcript=raw_transcript,
    )


async def _answer_subquery_non_sales(
    *,
    subquery: str,
    route: str,
    history: List[Dict[str, Any]],
) -> str:
    if route == "SEARCH":
        return await summarize_search_answer(
            user_query=subquery,
            search_query=subquery,
            history=history,
            system_persona=_SEARCH_PERSONA,
            max_sentences=3,
        )
    if route == "RAG":
        fact_context = await build_rag_fact_constraints(subquery)
        return await query_rag(
            text=subquery,
            top_k=6,
            history=history,
            system_context=fact_context,
        )
    return ""


async def _run_sales_or_search(
    *,
    session_id: str,
    user_text: str,
    raw_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    history = await load_chat_history(session_id)
    plan = _normalize_plan(user_text, await plan_query_adaptive(user_text, history))

    if plan["multi_intent"] and plan["subqueries"]:
        subqueries = list(plan["subqueries"])
        if any(item["route"] == "OTHER" for item in subqueries):
            return await _run_other_llm_flow(
                session_id=session_id,
                user_text=user_text,
                history=history,
            )

        tasks = [
            _answer_subquery_non_sales(
                subquery=item["query"],
                route=item["route"],
                history=history,
            )
            for item in subqueries
        ]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        ordered_lines: List[str] = []
        for idx, answer in enumerate(answers, start=1):
            if isinstance(answer, Exception):
                log.warning("subquery[%s] failed: %s", idx, answer)
                continue
            text = str(answer or "").strip()
            if text:
                ordered_lines.append(f"{idx}) {text}")

        final_response = "\n\n".join(ordered_lines).strip()
        if final_response:
            await _persist_chat_turn(session_id, user_text, final_response)
            return {
                "route_category": "MIXED",
                "final_response": final_response,
                "next_sales_state": None,
                "lead_profile": await load_lead_profile(session_id),
                "missing_slots": None,
            }

        return await _run_sales_flow(
            session_id=session_id,
            user_text=user_text,
            raw_transcript=raw_transcript,
        )

    category = str(plan["route"])
    routed_query = str(plan["refined_query"] or user_text)

    if category == "SEARCH":
        return await _run_search_flow(
            session_id=session_id,
            user_text=user_text,
            search_query=routed_query or user_text,
            history=history,
        )
    if category == "RAG":
        return await _run_rag_flow(
            session_id=session_id,
            user_text=user_text,
            rag_query=routed_query or user_text,
            history=history,
            raw_transcript=raw_transcript,
        )

    return await _run_other_llm_flow(
        session_id=session_id,
        user_text=user_text,
        history=history,
    )


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
        log.error("sales chat pipeline error: %s", e)
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
    """Streaming sales chat with true token streaming when route is RAG."""
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
            history = await load_chat_history(request.session_id)
            plan = _normalize_plan(user_text, await plan_query_adaptive(user_text, history))
            if plan["multi_intent"] and plan["subqueries"]:
                if any(item["route"] == "OTHER" for item in plan["subqueries"]):
                    result = await _run_other_llm_flow(
                        session_id=request.session_id,
                        user_text=user_text,
                        history=history,
                    )
                    async for line in _yield_ndjson_response_chunks(
                        request.session_id,
                        str(result.get("final_response") or ""),
                    ):
                        yield line
                    final_meta.update(_stream_meta_from_flow_result(result))
                else:
                    final_meta["route_category"] = "MIXED"
                    parts: List[str] = []
                    for idx, item in enumerate(plan["subqueries"], start=1):
                        answer = await _answer_subquery_non_sales(
                            subquery=item["query"],
                            route=item["route"],
                            history=history,
                        )
                        normalized = str(answer or "").strip()
                        if not normalized:
                            continue
                        numbered = f"{idx}) {normalized}"
                        parts.append(numbered)
                        async for line in _yield_ndjson_response_chunks(request.session_id, numbered):
                            yield line

                    full_answer = "\n\n".join(parts).strip()
                    if full_answer:
                        await _persist_chat_turn(request.session_id, user_text, full_answer)
                        final_meta["lead_profile"] = await load_lead_profile(request.session_id)
                    else:
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
            else:
                category = str(plan["route"])
                routed_query = str(plan["refined_query"] or user_text)
                if category == "RAG":
                    final_meta["route_category"] = "RAG"
                    fact_context = await build_rag_fact_constraints(routed_query or user_text)
                    parts: List[str] = []
                    async for delta in query_rag_stream(
                        text=routed_query or user_text,
                        top_k=6,
                        history=history,
                        system_context=fact_context,
                    ):
                        chunk = str(delta or "")
                        if not chunk:
                            continue
                        parts.append(chunk)
                        yield _ndjson_response_line(request.session_id, chunk)
                        await asyncio.sleep(0)

                    full_answer = "".join(parts).strip()
                    if full_answer:
                        await _persist_chat_turn(request.session_id, user_text, full_answer)
                        final_meta["lead_profile"] = await load_lead_profile(request.session_id)
                    else:
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
                elif category == "SEARCH":
                    result = await _run_search_flow(
                        session_id=request.session_id,
                        user_text=user_text,
                        search_query=routed_query or user_text,
                        history=history,
                    )
                    async for line in _yield_ndjson_response_chunks(
                        request.session_id,
                        str(result.get("final_response") or ""),
                    ):
                        yield line
                    final_meta.update(_stream_meta_from_flow_result(result))
                else:
                    result = await _run_other_llm_flow(
                        session_id=request.session_id,
                        user_text=user_text,
                        history=history,
                    )
                    async for line in _yield_ndjson_response_chunks(
                        request.session_id,
                        str(result.get("final_response") or ""),
                    ):
                        yield line
                    final_meta.update(_stream_meta_from_flow_result(result))

            if not final_meta.get("route_category"):
                final_meta["route_category"] = "SALES"
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

    history = await load_chat_history(session_id)
    try:
        result = await _run_other_llm_flow(
            session_id=session_id,
            user_text="Anh/Chị muốn xem lại các sản phẩm phù hợp.",
            history=history,
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

    history = await load_chat_history(session_id)
    try:
        result = await _run_other_llm_flow(
            session_id=session_id,
            user_text="Anh/Chị có cần em hỗ trợ thêm thông tin gì không?",
            history=history,
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
