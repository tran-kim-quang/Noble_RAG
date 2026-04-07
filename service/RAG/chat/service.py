from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.dependencies import llm_model_func, llm_model_stream_func
from models.api_models import KnowledgeDecisionPayload
from sales.nodes.decide_response_action import decide_response_action_node
from sales.prompt_builder import build_prompt
from sales.state_machine import resolve_next_state


_FALLBACK_RESPONSE = (
    "Em đang ở đây để hỗ trợ Anh/Chị. "
    "Anh/Chị có thể nói rõ hơn nhu cầu ưu tiên của mình để em tư vấn sát hơn ạ?"
)

_EXTERNAL_SEARCH_APPROVAL_PROMPT = (
    "Dựa trên dữ liệu nội bộ hiện có, em đã trả lời trong phạm vi tốt nhất. "
    "Anh/Chị có muốn em thực hiện tìm kiếm thông tin bên ngoài để bổ sung thêm không ạ?"
)

_OUT_OF_SCOPE_RESPONSE = (
    "Em chỉ hỗ trợ thông tin liên quan đến dự án Noble Place Tây Thăng Long. "
    "Em chưa thể hỗ trợ tra cứu các chủ đề ngoài phạm vi dự án. "
    "Anh/Chị muốn em tư vấn phần nào của dự án ạ: pháp lý, bảng giá, mặt bằng hay chính sách bán hàng?"
)


def _extract_target_unit_code(message: str) -> Optional[str]:
    match = re.search(r"\b([NS]\d{1,3})\b", str(message or "").upper())
    if not match:
        return None
    return match.group(1)


def _extract_lot_area_rows_from_kb(kb_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    # Parse markdown rows matching:
    # | TT | Mẫu nhà | ... | Diện tích lô đất (m2) | ...
    row_pattern = re.compile(
        r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*[^|]*\|\s*[^|]*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|",
        re.MULTILINE,
    )

    for item in kb_items:
        content = str(item.get("content") or "")
        if not content:
            continue

        for match in row_pattern.finditer(content):
            model_field = re.sub(r"\s+", " ", match.group(1).strip())
            try:
                lot_area = float(match.group(2))
            except ValueError:
                continue

            if not model_field:
                continue
            dedupe_key = (model_field.upper(), lot_area)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append({"model": model_field, "lot_area": lot_area})

    return rows


def _models_contain_code(model_field: str, unit_code: str) -> bool:
    cleaned = str(model_field or "").upper().replace(" ", "")
    parts = [p for p in cleaned.split(",") if p]
    return unit_code in parts


def _is_lot_area_gap_question(message: str) -> bool:
    text = str(message or "").lower()
    has_area = any(token in text for token in ["diện tích", "dien tich", "mét vuông", "m2", "m²"])
    has_gap = any(token in text for token in ["cách nhau", "chênh lệch", "bao nhiêu"])
    has_minmax = any(token in text for token in ["nhỏ nhất", "bé nhất", "lớn nhất"])
    return has_area and has_gap and has_minmax


def _build_lot_area_gap_answer(message: str, kb_items: List[Dict[str, Any]]) -> Optional[str]:
    if not _is_lot_area_gap_question(message):
        return None

    rows = _extract_lot_area_rows_from_kb(kb_items)
    if not rows:
        return None

    unit_code = _extract_target_unit_code(message)
    scoped_rows = rows
    if unit_code:
        matched_rows = [row for row in rows if _models_contain_code(str(row.get("model") or ""), unit_code)]
        if matched_rows:
            scoped_rows = matched_rows
        else:
            all_areas = [float(row["lot_area"]) for row in rows]
            if not all_areas:
                return None
            min_area = min(all_areas)
            max_area = max(all_areas)
            gap = max_area - min_area
            return (
                f"Hiện trong dữ liệu nội bộ chưa thấy mã căn {unit_code} trong bảng DT_SHOPHOUSE để tính riêng cho căn này. "
                f"Nếu xét toàn bộ các mẫu shophouse đang có, diện tích lô đất nhỏ nhất là {min_area:g} m2, "
                f"lớn nhất là {max_area:g} m2, chênh lệch {gap:g} m2."
            )

    areas = [float(row["lot_area"]) for row in scoped_rows]
    if not areas:
        return None

    min_area = min(areas)
    max_area = max(areas)
    gap = max_area - min_area

    min_models = sorted({str(row["model"]) for row in scoped_rows if float(row["lot_area"]) == min_area})
    max_models = sorted({str(row["model"]) for row in scoped_rows if float(row["lot_area"]) == max_area})

    scope_note = f" cho nhóm {unit_code}" if unit_code else ""
    return (
        f"Theo bảng DT_SHOPHOUSE{scope_note}, diện tích lô đất nhỏ nhất là {min_area:g} m2 "
        f"({'; '.join(min_models)}), lớn nhất là {max_area:g} m2 ({'; '.join(max_models)}), "
        f"nên chênh lệch là {gap:g} m2."
    )


def _summarize_history(history: List[Dict[str, Any]], max_turns: int = 6) -> str:
    if not history:
        return "Chưa có lịch sử hội thoại trước đó."

    lines: List[str] = []
    for item in history[-(max_turns * 2) :]:
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        speaker = "Người dùng" if role == "user" else ("Trợ lý" if role == "assistant" else role)
        lines.append(f"- {speaker}: {content}")
    return "\n".join(lines) if lines else "Chưa có lịch sử hội thoại trước đó."


def _format_evidence_block(items: List[Dict[str, Any]], *, title: str, max_items: int = 3) -> str:
    if not items:
        return f"{title}: (không có)"

    lines = [f"{title}:"]
    for idx, item in enumerate(items[:max_items], start=1):
        content = str(item.get("content") or "").strip()
        source_name = str(item.get("source_name") or item.get("source") or "").strip() or "unknown"
        score = item.get("score")

        if not content:
            continue

        content = content[:700]
        score_text = "n/a" if score is None else f"{float(score):.3f}"
        lines.append(f"{idx}) score={score_text}; source={source_name}; content={content}")

    return "\n".join(lines)


def _to_payload(knowledge_payload: KnowledgeDecisionPayload | Dict[str, Any] | None) -> Optional[KnowledgeDecisionPayload]:
    if knowledge_payload is None:
        return None
    if isinstance(knowledge_payload, KnowledgeDecisionPayload):
        return knowledge_payload
    try:
        return KnowledgeDecisionPayload.model_validate(knowledge_payload)
    except Exception:
        return None


def _infer_detected_intent(message: str, payload: Optional[KnowledgeDecisionPayload]) -> str:
    text = (message or "").lower()
    query_kind = ""
    if payload and isinstance(payload.planner_notes, dict):
        query_kind = str(payload.planner_notes.get("query_kind") or "").strip().lower()

    if payload and payload.multi_intent:
        return "comparison"
    if any(token in text for token in ["so sanh", "so sánh", "khac nhau", "khác nhau"]):
        return "comparison"
    if any(token in text for token in ["khong", "không", "lo", "băn khoăn", "phan doi", "phản đối"]):
        return "objection"
    if any(token in text for token in ["dat coc", "đặt cọc", "tham quan", "hen lich", "hẹn lịch", "chot", "chốt"]):
        return "buy_signal"
    if any(token in text for token in ["tu van", "tư vấn", "goi y", "gợi ý", "phu hop", "phù hợp"]):
        return "ask_recommendation"
    if query_kind in {"comparison", "compare"}:
        return "comparison"
    if query_kind in {"project_info", "fact", "project_qa"}:
        return "project_info"
    if query_kind in {"recommend", "recommendation", "matching"}:
        return "ask_recommendation"
    return "follow_up"


def _compute_missing_slots(lead_profile: Dict[str, Any]) -> List[str]:
    required = ["family_member_count", "children_count", "purpose", "location_preference"]
    missing: List[str] = []
    for key in required:
        value = lead_profile.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
            continue
        if isinstance(value, (list, tuple, set)) and len(value) == 0:
            missing.append(key)
            continue
    return missing


def _should_ask_external_search_approval(payload: Optional[KnowledgeDecisionPayload], search_items: List[Dict[str, Any]]) -> bool:
    if payload is None:
        return False
    if search_items:
        return False
    if _is_out_of_scope_non_project(payload):
        return False
    notes = payload.planner_notes if isinstance(payload.planner_notes, dict) else {}
    if bool(notes.get("human_loop_search_forced")):
        return False
    return bool(notes.get("human_loop_search_approval_needed"))


def _is_out_of_scope_non_project(payload: Optional[KnowledgeDecisionPayload]) -> bool:
    if payload is None:
        return False
    if str(payload.decision_reason or "").strip().lower() == "out_of_scope_non_project":
        return True
    notes = payload.planner_notes if isinstance(payload.planner_notes, dict) else {}
    return bool(notes.get("out_of_scope_non_project"))


def _build_free_chat_prompt(
    *,
    session_id: str,
    message: str,
    raw_transcript: Optional[str],
    history_summary: str,
) -> str:
    return f"""
Bạn là Sunny, trợ lý hội thoại tự nhiên cho dự án bất động sản.

BỐI CẢNH:
- session_id: {session_id}
- user_message: {message}
- raw_transcript: {raw_transcript or ""}
- history_summary:\n{history_summary}

YÊU CẦU:
- Trả lời tự nhiên, linh hoạt, không bị đóng khung theo mẫu cố định.
- Ưu tiên đúng ý người dùng ở lượt hiện tại.
- Nếu người dùng chỉ xã giao (chào hỏi/cảm ơn/tạm biệt), trả lời ngắn gọn và thân thiện.
- Không bịa thông tin dự án cụ thể khi chưa có dữ liệu xác thực.

Trả lời:
""".strip()


def _build_project_route_prompt(
    *,
    system_prompt: str,
    session_id: str,
    message: str,
    raw_transcript: Optional[str],
    history_summary: str,
    decision_reason: str,
    unresolved: bool,
    top_score: Optional[float],
    payload: Optional[KnowledgeDecisionPayload],
    orchestration_state: Dict[str, Any],
    evidence_kb: str,
    evidence_search: str,
    search_items: List[Dict[str, Any]],
) -> str:
    return f"""
{system_prompt}

BỐI CẢNH HỘI THOẠI:
- session_id: {session_id}
- user_message: {message}
- raw_transcript: {raw_transcript or ""}
- history_summary:\n{history_summary}

KNOWLEDGE PAYLOAD:
- decision_reason: {decision_reason}
- unresolved: {str(unresolved).lower()}
- kb_top_score: {top_score}
- search_used: {str(bool(search_items)).lower()}
- multi_intent: {str(bool(payload.multi_intent) if payload else False).lower()}
- subqueries: {payload.subqueries if payload else []}

SALES ORCHESTRATION (deterministic):
- sales_state: {orchestration_state.get("next_sales_state")}
- response_action: {orchestration_state.get("response_action")}
- missing_slots: {orchestration_state.get("missing_slots")}

{evidence_kb}

{evidence_search}

YÊU CẦU TRẢ LỜI:
- Luôn trả lời tự nhiên, ấm áp, ngắn gọn và rõ ràng.
- Nếu có cả KB và search, ưu tiên facts từ KB cho thông tin dự án, dùng search để bổ sung phần thị trường/thời gian thực.
- Nếu unresolved=true hoặc evidence yếu, hỏi đúng 1 câu làm rõ (không hỏi dồn dập).
- Không bịa thông tin nếu evidence không đủ; nêu giới hạn ngắn gọn và dẫn hướng câu hỏi tiếp theo.
- Tránh dạng trả lời dạng script cứng; giữ giọng tư vấn sales tự nhiên.
""".strip()


def _build_orchestration_state(
    *,
    session_id: str,
    message: str,
    history: List[Dict[str, Any]],
    lead_profile: Dict[str, Any],
    session_context: Dict[str, Any],
    payload: Optional[KnowledgeDecisionPayload],
) -> Dict[str, Any]:
    detected_intent = _infer_detected_intent(message, payload)
    missing_slots = _compute_missing_slots(lead_profile)

    knowledge_payload_dict: Optional[Dict[str, Any]] = None
    if payload is not None:
        knowledge_payload_dict = payload.model_dump(mode="python")

    state: Dict[str, Any] = {
        "session_id": session_id,
        "user_text": message,
        "chat_history": history,
        "lead_profile": lead_profile,
        "session_context": session_context,
        "current_sales_state": str(session_context.get("current_state") or lead_profile.get("current_state") or "greeting"),
        "detected_intent": detected_intent,
        "missing_slots": missing_slots,
        "extracted_slots": {},
        "retrieval_goal": "none",
        "turn_role": "user",
        "knowledge_payload": knowledge_payload_dict or {},
        "knowledge_decision_reason": str(payload.decision_reason or "") if payload else None,
        "kb_top_score": payload.top_score if payload else None,
        "search_used": bool(payload.search_evidence) if payload else False,
    }
    state["next_sales_state"] = resolve_next_state(state)
    state.update(decide_response_action_node(state))
    return state


async def run_chat_route(
    *,
    session_id: str,
    message: str,
    history: List[Dict[str, Any]],
    lead_profile: Dict[str, Any],
    session_context: Dict[str, Any],
    knowledge_payload: KnowledgeDecisionPayload | Dict[str, Any] | None,
    raw_transcript: str | None = None,
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    payload = _to_payload(knowledge_payload)
    orchestration_state = _build_orchestration_state(
        session_id=session_id,
        message=message,
        history=history,
        lead_profile=lead_profile,
        session_context=session_context,
        payload=payload,
    )

    state = {
        "session_id": session_id,
        "user_text": message,
        "chat_history": history,
        "lead_profile": lead_profile,
        "session_context": session_context,
        "next_sales_state": orchestration_state.get("next_sales_state"),
    }
    system_prompt = build_prompt(state)

    history_summary = _summarize_history(history)

    kb_items: List[Dict[str, Any]] = []
    search_items: List[Dict[str, Any]] = []
    decision_reason = ""
    unresolved = False
    top_score = None

    if payload is not None:
        kb_items = [item.model_dump(mode="python") for item in payload.kb_evidence]
        search_items = [item.model_dump(mode="python") for item in payload.search_evidence]
        decision_reason = str(payload.decision_reason or "")
        unresolved = bool(payload.unresolved)
        top_score = payload.top_score

    evidence_kb = _format_evidence_block(kb_items, title="KB evidence (internal)")
    evidence_search = _format_evidence_block(search_items, title="Search evidence (external)")

    is_project_route = payload is not None
    if is_project_route:
        prompt = _build_project_route_prompt(
            system_prompt=system_prompt,
            session_id=session_id,
            message=message,
            raw_transcript=raw_transcript,
            history_summary=history_summary,
            decision_reason=decision_reason,
            unresolved=unresolved,
            top_score=top_score,
            payload=payload,
            orchestration_state=orchestration_state,
            evidence_kb=evidence_kb,
            evidence_search=evidence_search,
            search_items=search_items,
        )
    else:
        prompt = _build_free_chat_prompt(
            session_id=session_id,
            message=message,
            raw_transcript=raw_transcript,
            history_summary=history_summary,
        )

    answer = ""
    structured_answer = _build_lot_area_gap_answer(message, kb_items)
    if structured_answer:
        answer = structured_answer
    try:
        if not answer:
            if stream_callback is None:
                answer = str(
                    await llm_model_func(
                        prompt,
                        history_messages=history[-8:],
                        temperature=0.2,
                    )
                    or ""
                ).strip()
            else:
                chunks: List[str] = []
                async for delta in llm_model_stream_func(
                    prompt,
                    history_messages=history[-8:],
                    temperature=0.2,
                ):
                    if not delta:
                        continue
                    chunks.append(delta)
                    await stream_callback(delta)
                answer = "".join(chunks).strip()
    except Exception:
        if not answer:
            answer = ""

    if not answer:
        answer = _FALLBACK_RESPONSE

    if _is_out_of_scope_non_project(payload):
        answer = _OUT_OF_SCOPE_RESPONSE

    if _should_ask_external_search_approval(payload, search_items):
        lower_answer = answer.lower()
        if "có muốn em thực hiện tìm kiếm thông tin bên ngoài" not in lower_answer:
            answer = f"{answer.rstrip()}\n\n{_EXTERNAL_SEARCH_APPROVAL_PROMPT}"

    return {
        "route_category": "CHAT",
        "final_response": answer,
        "next_sales_state": orchestration_state.get("next_sales_state"),
        "lead_profile": lead_profile,
        "missing_slots": orchestration_state.get("missing_slots"),
        "response_action": orchestration_state.get("response_action"),
        "knowledge_used": bool(payload),
        "search_used": bool(search_items),
        "kb_top_score": top_score,
    }
