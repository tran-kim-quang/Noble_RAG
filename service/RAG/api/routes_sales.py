"""Sales agent API endpoints."""

import asyncio
import ast
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from chat.service import run_chat_route
from core.dependencies import llm_model_func
from integrations.camera_identity import maybe_enrich_identity
from knowledge_base.service import resolve_knowledge
from models.api_models import LeadUpdateRequest, SalesChatRequest, SalesChatResponse
from utils.json_extract import extract_first_json_object
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
from utils.text import iter_stream_chunks, normalize_user_text

router = APIRouter(prefix="/sales", tags=["sales"])
log = logging.getLogger("rag-service")

_SEARCH_APPROVAL_PROMPT_VI = "có muốn em thực hiện tìm kiếm thông tin bên ngoài"


async def _classify_turn_route(user_text: str, history: List[Dict[str, Any]]) -> str:
    text = (user_text or "").strip()
    if not text:
        return "PROJECT"

    history_lines: List[str] = []
    for item in (history or [])[-6:]:
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "(empty)"

    prompt = f"""
Bạn là bộ phân loại route cho trợ lý bất động sản.
Hãy phân loại user turn thành đúng 1 loại:
- SMALLTALK: xã giao/chat vui (chào hỏi, cảm ơn, tạm biệt, phản hồi lịch sự, nói chuyện nhẹ nhàng), không cần truy xuất tri thức dự án.
- PROJECT: có ý định hỏi thông tin dự án/sản phẩm/chính sách/so sánh/pháp lý/giá/tiến độ hoặc cần tư vấn bất động sản cụ thể.

Chỉ trả về JSON object duy nhất:
{{"route": "SMALLTALK|PROJECT"}}

Quy tắc:
- Nếu mơ hồ, chọn PROJECT (an toàn).
- Không giải thích thêm.

Ví dụ bắt buộc:
- "xin chào" -> SMALLTALK
- "xin chào và hẹn gặp lại" -> SMALLTALK
- "cảm ơn em nhé" -> SMALLTALK
- "cho anh bảng giá dự án" -> PROJECT
- "pháp lý dự án thế nào" -> PROJECT

History:
{history_text}

User:
{text}
""".strip()

    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=40,
        )
        route = ""
        if isinstance(raw, dict):
            route = str(raw.get("route") or "").strip().upper()
        else:
            raw_text = str(raw or "")
            payload = extract_first_json_object(raw_text)
            if payload:
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        route = str(data.get("route") or "").strip().upper()
                except Exception:
                    try:
                        data = ast.literal_eval(payload)
                        if isinstance(data, dict):
                            route = str(data.get("route") or "").strip().upper()
                    except Exception:
                        route = ""
            if not route:
                upper = raw_text.strip().upper()
                if "SMALLTALK" in upper:
                    route = "SMALLTALK"
                elif "PROJECT" in upper:
                    route = "PROJECT"
        if route in {"SMALLTALK", "PROJECT"}:
            log.info("turn_route classifier: route=%s user_text=%r", route, text)
            return route

        try:
            normalize_prompt = f"""
Phân loại câu user sau thành đúng 1 nhãn và CHỈ trả về một từ duy nhất:
- SMALLTALK: chat xã giao, không cần tri thức dự án.
- PROJECT: có ý định hỏi/tư vấn thông tin dự án bất động sản.

User: {text}
""".strip()
            normalized = await llm_model_func(
                normalize_prompt,
                enable_cot=False,
                temperature=0.0,
                max_tokens=8,
            )
            label = str(normalized or "").strip().upper()
            if label.startswith("SMALLTALK"):
                log.info("turn_route classifier normalized: route=SMALLTALK user_text=%r", text)
                return "SMALLTALK"
            if label.startswith("PROJECT"):
                log.info("turn_route classifier normalized: route=PROJECT user_text=%r", text)
                return "PROJECT"
        except Exception as exc:
            log.warning("turn_route normalize failed, fallback PROJECT: %s", exc)
    except Exception as exc:
        log.warning("turn_route classifier failed, fallback PROJECT: %s", exc)

    log.info("turn_route classifier fallback: route=PROJECT user_text=%r", text)
    return "PROJECT"


def _parse_boolean_json(raw: str, key: str) -> Optional[bool]:
    text = (raw or "").strip()
    if not text:
        return None
    candidate = extract_first_json_object(text) or text
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if isinstance(value, bool) else None


def _is_external_search_approval_fallback(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    deny_patterns = [
        r"\b(khong|không|ko)\b",
        r"\b(khong can|không cần|khong dong y|không đồng ý)\b",
        r"\b(khoi|khỏi)\b",
    ]
    if any(re.search(pattern, text) for pattern in deny_patterns):
        return False

    approve_patterns = [
        r"\b(dong y|đồng ý|ok|oke|yes|duoc|được)\b",
        r"\b(tim ngoai|tìm ngoài|search)\b",
        r"\b(cu tra|cứ tra|tra giup|tra giúp|nho ban tra|nhờ bạn tra)\b",
    ]
    return any(re.search(pattern, text) for pattern in approve_patterns)


async def _is_external_search_approval(user_text: str, history: List[Dict[str, Any]]) -> bool:
    text = (user_text or "").strip()
    if not text:
        return False

    last_assistant = ""
    for item in reversed(history):
        if str(item.get("role") or "").strip().lower() != "assistant":
            continue
        candidate = str(item.get("content") or "").strip()
        if candidate:
            last_assistant = candidate
            break

    prompt = f"""
Bạn là bộ phân loại intent trong hội thoại sales.
Nhiệm vụ: xác định câu người dùng có phải là ĐỒNG Ý cho phép trợ lý tìm kiếm thông tin bên ngoài hay không.

Chỉ trả về JSON object duy nhất:
{{"approve_external_search": true|false}}

Quy tắc:
- true nếu người dùng thể hiện đồng ý/cho phép tiếp tục tìm kiếm ngoài.
- false nếu người dùng từ chối, phủ định, hoặc câu không rõ là chấp thuận.
- Nếu mơ hồ, trả về false.

Assistant previous message: {last_assistant}
User message: {text}
""".strip()

    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=60,
        )
        parsed = _parse_boolean_json(str(raw), "approve_external_search")
        if parsed is not None:
            log.info("external_search_approval classifier: parsed=%s user_text=%r", parsed, text)
            return parsed
    except Exception as exc:
        log.warning("external search approval classifier failed, using fallback: %s", exc)

    fallback = _is_external_search_approval_fallback(text)
    log.info("external_search_approval classifier fallback: parsed=%s user_text=%r", fallback, text)
    return fallback


def _assistant_requested_external_search(history: List[Dict[str, Any]]) -> bool:
    if not history:
        return False
    for item in reversed(history):
        if str(item.get("role") or "").strip().lower() != "assistant":
            continue
        content = str(item.get("content") or "").strip().lower()
        if _SEARCH_APPROVAL_PROMPT_VI in content:
            return True
        return False
    return False


def _last_user_query_before_current_turn(history: List[Dict[str, Any]]) -> str:
    for item in reversed(history):
        if str(item.get("role") or "").strip().lower() != "user":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            return content
    return ""


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


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"\D+", "", value or "")
    if digits.startswith("84") and len(digits) >= 10:
        digits = "0" + digits[2:]
    return digits


def _extract_name(text: str) -> str:
    patterns = [
        r"(?:m(?:i|ì|ình)nh|tôi|toi|em|anh|chị|chi)\s+t(?:ê|e)n\s+l(?:à|a)\s+([A-Za-zÀ-ỹ\s]{2,60})",
        r"t(?:ê|e)n\s+c(?:ủa|ua)\s+(?:m(?:i|ì|ình)nh|tôi|toi)\s+l(?:à|a)\s+([A-Za-zÀ-ỹ\s]{2,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")
        if len(name) >= 2:
            return name
    return ""


def _extract_location_preference(text: str) -> str:
    patterns = [
        r"(?:ở|o|khu vực|khu v(?:ự|u)c|muốn ở|muon o)\s+([A-Za-zÀ-ỹ0-9\s,./-]{3,90})",
        r"(?:quận|quan|huyện|huyen|phường|phuong|tỉnh|tinh|thành phố|thanh pho)\s+([A-Za-zÀ-ỹ0-9\s,./-]{2,80})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")
        if len(value) >= 3:
            return value
    return ""


def _extract_apartment_preference(text: str) -> str:
    match = re.search(r"(?:gu\s*căn hộ|gu\s*can\s*ho)\s*:?\s*([^\n]{2,120})", text, flags=re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip(" .,:;-")

    if re.search(r"(căn hộ|can ho|chung cư|chung cu|studio|penthouse|2pn|3pn|1pn)", text, flags=re.IGNORECASE):
        return re.sub(r"\s+", " ", text).strip()[:140]
    return ""


def _extract_customer_profile_updates(user_text: str) -> Dict[str, Any]:
    text = (user_text or "").strip()
    updates: Dict[str, Any] = {}
    if not text:
        return updates

    phone_match = re.search(r"(?:(?:\+84|84|0)(?:[\s.\-]?\d){8,10})", text)
    if phone_match:
        normalized_phone = _normalize_phone(phone_match.group(0))
        if 9 <= len(normalized_phone) <= 11:
            updates["phone_number"] = normalized_phone

    name = _extract_name(text)
    if name:
        updates["customer_name"] = name

    location = _extract_location_preference(text)
    if location:
        updates["location_preference"] = location

    apartment_pref = _extract_apartment_preference(text)
    if apartment_pref:
        updates["apartment_preference"] = apartment_pref
        updates["gu_can_ho"] = apartment_pref
    elif location:
        # "Gu hoặc nơi ở" được quy chiếu về một tiêu chí gu căn hộ.
        updates["apartment_preference"] = location
        updates["gu_can_ho"] = location

    return updates


def _desktop_customer_profile_dir() -> Path:
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    target = desktop / "Noble_RAG_customer_profiles"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_customer_profile_to_desktop(session_id: str, profile: Dict[str, Any]) -> None:
    sid = (session_id or "unknown").strip() or "unknown"
    safe_sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    customer_name = str(profile.get("customer_name") or "").strip()
    phone_number = str(profile.get("phone_number") or "").strip()
    gu_can_ho = str(profile.get("gu_can_ho") or profile.get("apartment_preference") or "").strip()

    payload = {
        "session_id": sid,
        "saved_at": now,
        "customer_name": customer_name,
        "phone_number": phone_number,
        "gu_can_ho": gu_can_ho,
        "raw_profile": profile,
    }

    base_dir = _desktop_customer_profile_dir()
    json_path = base_dir / f"{safe_sid}_latest.json"
    txt_path = base_dir / f"{safe_sid}_latest.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(
        (
            "Tên khách hàng:\n"
            f"{customer_name}\n"
            "Số điện thoại:\n"
            f"{phone_number}\n"
            "Gu căn hộ:\n"
            f"{gu_can_ho}\n"
        ),
        encoding="utf-8",
    )


def _build_sunny_follow_up(lead_profile: Dict[str, Any]) -> str:
    profile = lead_profile or {}
    has_pref = bool(
        str(
            profile.get("gu_can_ho")
            or profile.get("apartment_preference")
            or profile.get("location_preference")
            or ""
        ).strip()
    )
    has_name = bool(str(profile.get("customer_name") or "").strip())
    has_phone = bool(str(profile.get("phone_number") or "").strip())
    criteria = {
        "Tên khách hàng": has_name,
        "Số điện thoại": has_phone,
        "Gu căn hộ": has_pref,
    }
    missing = [key for key, ok in criteria.items() if not ok]

    if not missing:
        return (
            "Sunny đã lưu thông tin của bạn rồi nè. "
            "Mình cùng khám phá thêm các điểm thú vị của Noble Palace Tây Thăng Long nhé, còn nhiều điều hay lắm."
        )

    random.shuffle(missing)
    missing_block = "\n".join(f"{field}:" for field in missing)
    return f"Sunny cần thêm một vài thông tin để lưu danh sách khách hàng nhé:\n{missing_block}"


def _append_follow_up_if_missing(base_answer: str, follow_up: str) -> str:
    answer = (base_answer or "").strip()
    hint = (follow_up or "").strip()
    if not hint:
        return answer
    if hint.lower() in answer.lower():
        return answer
    if not answer:
        return hint
    return f"{answer}\n\n{hint}"


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
    stream_callback=None,
    stage_timings_ms: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    t_flow_start = time.perf_counter()
    timings: Dict[str, int] = {}

    user_text = normalize_user_text(user_text)

    t0 = time.perf_counter()
    history, session_context, lead_profile = await asyncio.gather(
        load_chat_history(session_id),
        load_session_context(session_id),
        load_lead_profile(session_id),
    )
    lead_profile = lead_profile or {"lead_id": session_id}
    profile_updates = _extract_customer_profile_updates(user_text)
    if profile_updates:
        lead_profile.update(profile_updates)
        try:
            await save_lead_profile(session_id, lead_profile)
            _save_customer_profile_to_desktop(session_id, lead_profile)
        except Exception as exc:
            log.warning("save_lead_profile pre-chat failed: session=%s err=%s", session_id, exc)
    timings["load_context"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    turn_route = await _classify_turn_route(user_text, history)
    timings["route_classify"] = int((time.perf_counter() - t0) * 1000)
    if turn_route == "SMALLTALK":
        log.info("smalltalk_bypass: session=%s user_text=%r", session_id, user_text)
        t0 = time.perf_counter()
        result = await run_chat_route(
            session_id=session_id,
            message=user_text,
            history=history,
            lead_profile=lead_profile or {},
            session_context=session_context or {},
            knowledge_payload=None,
            raw_transcript=raw_transcript,
            stream_callback=stream_callback,
        )
        timings["chat_generation"] = int((time.perf_counter() - t0) * 1000)

        patched_answer = _apply_customer_pronoun(
            str(result.get("final_response") or ""),
            {"lead_profile": lead_profile, "session_context": session_context},
        )
        patched_answer = _append_follow_up_if_missing(
            patched_answer,
            _build_sunny_follow_up(lead_profile),
        )
        result["final_response"] = patched_answer
        result["lead_profile"] = lead_profile

        t0 = time.perf_counter()
        await _persist_chat_turn(session_id, user_text, patched_answer)
        timings["persist_turn"] = int((time.perf_counter() - t0) * 1000)
        timings["total"] = int((time.perf_counter() - t_flow_start) * 1000)
        if stage_timings_ms is not None:
            stage_timings_ms.update(timings)
        log.info("sales_flow_timing session=%s route=%s timings_ms=%s", session_id, turn_route, timings)
        return result

    settings = get_settings()
    force_external_search = False
    effective_query = user_text

    if settings.enable_human_loop_search_approval:
        if _assistant_requested_external_search(history) and await _is_external_search_approval(user_text, history):
            previous_query = _last_user_query_before_current_turn(history)
            if previous_query:
                effective_query = previous_query
                force_external_search = True
                log.info(
                    "human_loop_search: approval detected session=%s using_previous_query=%r",
                    session_id,
                    effective_query,
                )

    t0 = time.perf_counter()
    knowledge_payload = await resolve_knowledge(
        query=effective_query,
        history=history,
        session_context=session_context,
        top_k=6,
        force_external_search=force_external_search,
        human_loop_threshold=settings.human_loop_search_cosine_threshold,
    )
    timings["resolve_knowledge"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    result = await run_chat_route(
        session_id=session_id,
        message=effective_query if force_external_search else user_text,
        history=history,
        lead_profile=lead_profile or {},
        session_context=session_context or {},
        knowledge_payload=knowledge_payload,
        raw_transcript=raw_transcript,
        stream_callback=stream_callback,
    )
    timings["chat_generation"] = int((time.perf_counter() - t0) * 1000)

    patched_answer = _apply_customer_pronoun(
        str(result.get("final_response") or ""),
        {"lead_profile": lead_profile, "session_context": session_context},
    )
    patched_answer = _append_follow_up_if_missing(
        patched_answer,
        _build_sunny_follow_up(lead_profile),
    )
    result["final_response"] = patched_answer
    result["lead_profile"] = lead_profile

    t0 = time.perf_counter()
    await _persist_chat_turn(session_id, user_text, patched_answer)
    timings["persist_turn"] = int((time.perf_counter() - t0) * 1000)
    timings["total"] = int((time.perf_counter() - t_flow_start) * 1000)
    if stage_timings_ms is not None:
        stage_timings_ms.update(timings)
    log.info("sales_flow_timing session=%s route=%s timings_ms=%s", session_id, turn_route, timings)
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
            stage_timings_ms: Dict[str, int] = {}
            stream_queue: "asyncio.Queue[str]" = asyncio.Queue()

            async def _on_stream_delta(delta: str) -> None:
                await stream_queue.put(delta)

            flow_task = asyncio.create_task(
                _run_sales_flow(
                    session_id=request.session_id,
                    user_text=user_text,
                    raw_transcript=request.raw_transcript,
                    stream_callback=_on_stream_delta,
                    stage_timings_ms=stage_timings_ms,
                )
            )

            while True:
                if flow_task.done() and stream_queue.empty():
                    break
                try:
                    delta = await asyncio.wait_for(stream_queue.get(), timeout=0.15)
                except asyncio.TimeoutError:
                    continue
                if delta:
                    yield _ndjson_response_line(request.session_id, delta)

            result = await flow_task
            final_meta.update(_stream_meta_from_flow_result(result))
            final_meta["stage_timings_ms"] = stage_timings_ms

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
                "stage_timings_ms": final_meta.get("stage_timings_ms") or {},
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
