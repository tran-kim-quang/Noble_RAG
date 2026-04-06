from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.dependencies import llm_model_func
from models.api_models import KnowledgeDecisionPayload
from sales.nodes.decide_response_action import decide_response_action_node
from sales.prompt_builder import build_prompt
from sales.state_machine import resolve_next_state


_FALLBACK_RESPONSE = (
    "Em đang ở đây để hỗ trợ Anh/Chị. "
    "Anh/Chị có thể nói rõ hơn nhu cầu ưu tiên của mình để em tư vấn sát hơn ạ?"
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

    prompt = f"""
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

    answer = ""
    try:
        answer = str(
            await llm_model_func(
                prompt,
                history_messages=history[-8:],
                temperature=0.2,
            )
            or ""
        ).strip()
    except Exception:
        answer = ""

    if not answer:
        answer = _FALLBACK_RESPONSE

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
