"""Context-aware turn understanding with thin rules for latency-sensitive cases."""

import json
import logging
import re
from typing import Any, Dict, Optional

from core.dependencies import llm_model_func
from sales.graph_state import SalesAgentState
from utils.json_extract import extract_first_json_object

log = logging.getLogger("rag-service")

_PROJECT_NAME_PATTERN = re.compile(r"(Noble[^\n,.;:!?()]*)", re.IGNORECASE)
_LOCATION_PATTERN = re.compile(
    r"\b(?:quanh|gan|gần|o|ở|tai|tại)\b\s+([A-Za-zÀ-ỹ0-9\-\s]{2,40})",
    re.IGNORECASE,
)
_FAMILY_PATTERN = re.compile(r"\b(\d{1,2})\s*(?:nguoi|người)\b", re.IGNORECASE)
_CHILDREN_PATTERN = re.compile(r"\b(\d{1,2})\s*(?:con|be|bé)\b", re.IGNORECASE)
_SHORT_SLOT_MAX_WORDS = 8
_INTENT_CLASSES = {
    "greeting",
    "ask_recommendation",
    "project_info",
    "comparison",
    "objection",
    "buy_signal",
    "follow_up",
    "out_of_scope",
    "other",
}
_TURN_ROLES = {
    "answer_previous_question",
    "ask_catalog_overview",
    "ask_project_info",
    "ask_comparison",
    "raise_objection",
    "show_buy_signal",
    "ask_recommendation",
    "continue_previous_topic",
    "greeting",
    "other",
}
_RETRIEVAL_GOALS = {
    "none",
    "shortlist",
    "project_qa",
    "comparison",
    "objection_support",
    "closing_next_step",
}
_SLOT_FIELDS = {
    "family_member_count",
    "children_count",
    "purpose",
    "property_type",
    "budget_text",
    "budget_min",
    "budget_max",
    "location_preference",
    "timeline",
    "financing_need",
    "key_concerns",
    "contact_phone",
}
_INVALID_LOCATION_TOKENS = (
    "dự án",
    "du an",
    "noble",
    "pháp lý",
    "phap ly",
    "so sánh",
    "so sanh",
    "giá",
    "gia",
    "tiện ích",
    "tien ich",
    "chính sách",
    "chinh sach",
    "nào",
    "nao",
    "?",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_project_name(text: str) -> Optional[str]:
    match = _PROJECT_NAME_PATTERN.search(text or "")
    if not match:
        return None
    candidate = _normalize_text(match.group(1))
    if re.search(r"\b(nao|nào|gi|gì|the nao|thế nào|co nhung|có những|hien co|hiện có)\b", candidate, re.IGNORECASE):
        return None
    return candidate


def _normalize_slot_value(key: str, value: Any) -> Any:
    if value in (None, "", []):
        return None
    if key in {"family_member_count", "children_count"}:
        try:
            return int(value)
        except Exception:
            return None
    if key in {"budget_min", "budget_max"}:
        try:
            return float(value)
        except Exception:
            return None
    if key == "location_preference":
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            clean_items = []
            seen = set()
            for item in value:
                text = _normalize_text(str(item))
                if not text:
                    continue
                lower = text.lower()
                if lower in seen:
                    continue
                seen.add(lower)
                clean_items.append(text)
            return clean_items or None
        return None
    if key == "key_concerns":
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            clean_items = [_normalize_text(str(item)) for item in value if _normalize_text(str(item))]
            return clean_items or None
        return None
    if key == "financing_need":
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"true", "1", "yes"}:
            return True
        if str(value).strip().lower() in {"false", "0", "no"}:
            return False
        return None
    return _normalize_text(str(value))


def _sanitize_slot_updates(payload: Dict[str, Any]) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    for key in _SLOT_FIELDS:
        normalized = _normalize_slot_value(key, payload.get(key))
        if normalized in (None, "", []):
            continue
        slots[key] = normalized
    return slots


def _parse_family_slots(text_lower: str) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}

    family_match = _FAMILY_PATTERN.search(text_lower)
    if family_match:
        slots["family_member_count"] = int(family_match.group(1))

    children_match = _CHILDREN_PATTERN.search(text_lower)
    if children_match:
        slots["children_count"] = int(children_match.group(1))

    if any(phrase in text_lower for phrase in ("song 1 minh", "sống 1 mình", "độc thân", "doc than")):
        slots.setdefault("family_member_count", 1)
        slots.setdefault("children_count", 0)

    if any(phrase in text_lower for phrase in ("vo chong", "vợ chồng", "hai vo chong", "hai vợ chồng")):
        slots.setdefault("family_member_count", 2)

    if any(phrase in text_lower for phrase in ("chua co con", "chưa có con")):
        slots["children_count"] = 0

    return slots


def _parse_purpose_slot(text_lower: str) -> Dict[str, Any]:
    if any(token in text_lower for token in ("de o", "để ở", "mua o", "mua ở")):
        return {"purpose": "mua_o"}
    if any(token in text_lower for token in ("kinh doanh", "mo cua hang", "mở cửa hàng", "buon ban", "buôn bán")):
        return {"purpose": "kinh_doanh"}
    if any(token in text_lower for token in ("cho thue", "cho thuê")):
        return {"purpose": "dau_tu_cho_thue"}
    if any(token in text_lower for token in ("dau tu", "đầu tư", "tang gia", "tăng giá")):
        return {"purpose": "dau_tu_tang_gia"}
    return {}


def _parse_location_slot(text: str) -> Dict[str, Any]:
    matches = []
    for raw in _LOCATION_PATTERN.findall(text):
        clean = re.sub(r"\s+", " ", raw).strip(" .,!?:;")
        if len(clean) < 2:
            continue
        lower = clean.lower()
        if any(token in lower for token in _INVALID_LOCATION_TOKENS):
            continue
        if len(clean.split()) > 4:
            continue
        matches.append(clean)

    deduped = []
    seen = set()
    for location in matches:
        key = location.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(location)

    if deduped:
        return {"location_preference": deduped[:2]}
    return {}


def _parse_contextual_slot_reply(text: str, state: SalesAgentState) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    text_lower = normalized.lower()
    words = [token for token in re.split(r"\s+", normalized) if token]
    if not words:
        return {}

    current_state = state.get("current_sales_state") or "greeting"
    current_step = state.get("current_script_step") or ""
    missing_slots = set(state.get("missing_slots") or [])

    if current_state != "need_discovery":
        return {}
    if "?" in normalized:
        return {}

    if "location_preference" in missing_slots and current_step.endswith("ask_location"):
        if not _extract_project_name(normalized) and len(words) <= 4:
            return {"location_preference": [normalized]}

    if "family_member_count" in missing_slots and current_step.endswith("ask_family_size"):
        if re.fullmatch(r"\d{1,2}", text_lower):
            return {"family_member_count": int(text_lower)}

    if "children_count" in missing_slots and current_step.endswith("ask_children"):
        if re.fullmatch(r"\d{1,2}", text_lower):
            return {"children_count": int(text_lower)}
        if text_lower in {"khong", "không", "chua co", "chưa có", "khong co", "0"}:
            return {"children_count": 0}

    return {}


def _infer_turn_from_slots(state: SalesAgentState, slots: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not slots:
        return None

    current_state = state.get("current_sales_state") or "greeting"
    if current_state == "greeting":
        return {
            "turn_role": "answer_previous_question",
            "intent": "other",
            "confidence": 0.55,
            "should_retrieve": False,
            "retrieval_goal": "none",
            "buy_signal": False,
            "objection_type": None,
        }
    if current_state == "need_discovery":
        missing_slots = set(state.get("missing_slots") or [])
        should_retrieve = bool(missing_slots) and set(slots).issuperset(missing_slots)
        return {
            "turn_role": "answer_previous_question",
            "intent": "follow_up",
            "confidence": 0.6,
            "should_retrieve": should_retrieve,
            "retrieval_goal": "shortlist" if should_retrieve else "none",
            "buy_signal": False,
            "objection_type": None,
        }
    return {
        "turn_role": "continue_previous_topic",
        "intent": "other",
        "confidence": 0.5,
        "should_retrieve": False,
        "retrieval_goal": "none",
        "buy_signal": False,
        "objection_type": None,
    }


def _recent_history_text(chat_history: list[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{(item.get('role') or 'unknown')}: {((item.get('content') or '').strip())[:160]}"
        for item in chat_history[-4:]
        if isinstance(item, dict)
    ) or "(chưa có)"


def _working_memory_summary(state: SalesAgentState) -> str:
    lead = state.get("lead_profile") or {}
    session_context = state.get("session_context") or {}
    pieces = [
        f"state={state.get('current_sales_state') or 'greeting'}",
        f"step={state.get('current_script_step') or 'S1_opening'}",
        f"missing={','.join(state.get('missing_slots') or []) or 'none'}",
        f"family={lead.get('family_member_count') or 'unknown'}",
        f"children={lead.get('children_count') if lead.get('children_count') is not None else 'unknown'}",
        f"purpose={lead.get('purpose') or 'unknown'}",
        f"location={','.join(lead.get('location_preference') or []) or 'unknown'}",
        f"last_action={session_context.get('last_agent_action') or 'unknown'}",
        f"last_agent_message={(session_context.get('last_agent_message') or '')[:160] or 'unknown'}",
    ]
    return " | ".join(pieces)


async def _micro_understand_turn_with_llm(
    text: str,
    state: SalesAgentState,
) -> Dict[str, Any]:
    current_state = state.get("current_sales_state") or "greeting"
    current_step = state.get("current_script_step") or "S1_opening"
    missing_slots = ", ".join(state.get("missing_slots") or []) or "không có"
    working_memory = _working_memory_summary(state)
    slot_hints = state.get("_slot_hints") or {}

    prompt = f"""Bạn là bộ hiểu nhanh một lượt hội thoại sales bất động sản.
Mục tiêu: dựa trên working memory rất ngắn, xác định xem khách đang trả lời câu hỏi trước hay mở chủ đề mới, và điền slot nếu thấy rõ.

Chỉ trả về JSON hợp lệ duy nhất theo schema:
{{
  "turn_role": "answer_previous_question" | "ask_catalog_overview" | "ask_project_info" | "ask_comparison" | "raise_objection" | "show_buy_signal" | "ask_recommendation" | "continue_previous_topic" | "greeting" | "other",
  "intent": "greeting" | "ask_recommendation" | "project_info" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "confidence": 0.0,
  "should_escalate": false,
  "should_retrieve": false,
  "retrieval_goal": "none" | "shortlist" | "project_qa" | "comparison" | "objection_support" | "closing_next_step",
  "resolved_project_name": null,
  "slot_updates": {{
    "family_member_count": null,
    "children_count": null,
    "purpose": null,
    "location_preference": []
  }}
}}

Quy tắc:
- Ưu tiên dùng slot_hints nếu chúng phù hợp với câu hiện tại.
- Nếu working memory + user_text chưa đủ để kết luận chắc, đặt should_escalate=true.
- Chỉ bật retrieval nếu khách đang hỏi danh sách dự án, hỏi thông tin dự án, hoặc hỏi so sánh/phản đối/mua ngay.
- Không cần đọc toàn bộ lịch sử xa.

Current sales state: {current_state}
Current script step: {current_step}
Missing slots: {missing_slots}
Working memory:
{working_memory}
Slot hints:
{slot_hints}

Tin nhắn khách:
\"\"\"{text}\"\"\""""

    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
            max_tokens=220,
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            raise ValueError("No JSON in micro understanding response")
        data = json.loads(payload)
    except Exception as e:
        log.warning("fast_parse micro understanding failed: %s", e)
        return {
            "turn_role": "other",
            "intent": "other",
            "confidence": 0.2,
            "should_escalate": True,
            "should_retrieve": False,
            "retrieval_goal": "none",
            "resolved_project_name": None,
            "slot_updates": {},
        }

    turn_role = str(data.get("turn_role") or "other")
    if turn_role not in _TURN_ROLES:
        turn_role = "other"

    intent = str(data.get("intent") or "other")
    if intent not in _INTENT_CLASSES:
        intent = "other"

    try:
        confidence = float(data.get("confidence", 0.4))
    except Exception:
        confidence = 0.4
    confidence = min(1.0, max(0.0, confidence))

    retrieval_goal = str(data.get("retrieval_goal") or "none")
    if retrieval_goal not in _RETRIEVAL_GOALS:
        retrieval_goal = "none"

    resolved_project_name = _normalize_text(str(data.get("resolved_project_name") or "")) or None
    if resolved_project_name and re.search(r"\b(nao|nào|gi|gì|the nao|thế nào|co nhung|có những)\b", resolved_project_name, re.IGNORECASE):
        resolved_project_name = None

    slots = _sanitize_slot_updates(data.get("slot_updates") or {})
    return {
        "turn_role": turn_role,
        "intent": intent,
        "confidence": confidence,
        "should_escalate": bool(data.get("should_escalate", confidence < 0.72)),
        "should_retrieve": bool(data.get("should_retrieve", retrieval_goal != "none")),
        "retrieval_goal": retrieval_goal,
        "resolved_project_name": resolved_project_name,
        "slot_updates": slots,
    }


async def _deep_understand_turn_with_llm(
    text: str,
    state: SalesAgentState,
) -> Dict[str, Any]:
    chat_history = state.get("chat_history") or []
    history_str = _recent_history_text(chat_history)
    working_memory = _working_memory_summary(state)

    prompt = f"""Bạn là bộ hiểu hội thoại sales bất động sản ở mức sâu hơn khi lượt nói còn mơ hồ.
Chỉ trả về JSON hợp lệ duy nhất:
{{
  "turn_role": "answer_previous_question" | "ask_catalog_overview" | "ask_project_info" | "ask_comparison" | "raise_objection" | "show_buy_signal" | "ask_recommendation" | "continue_previous_topic" | "greeting" | "other",
  "intent": "greeting" | "ask_recommendation" | "project_info" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "confidence": 0.0,
  "should_retrieve": false,
  "retrieval_goal": "none" | "shortlist" | "project_qa" | "comparison" | "objection_support" | "closing_next_step",
  "buy_signal": false,
  "objection_type": null,
  "resolved_project_name": null,
  "slot_updates": {{}}
}}

Working memory:
{working_memory}

Lịch sử gần đây:
{history_str}

Tin nhắn khách:
\"\"\"{text}\"\"\""""
    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
            max_tokens=350,
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            raise ValueError("No JSON in deep understanding response")
        data = json.loads(payload)
    except Exception as e:
        log.warning("fast_parse deep understanding failed: %s", e)
        return {
            "turn_role": "other",
            "intent": "other",
            "confidence": 0.2,
            "should_retrieve": False,
            "retrieval_goal": "none",
            "buy_signal": False,
            "objection_type": None,
            "resolved_project_name": None,
            "slot_updates": {},
        }

    turn_role = str(data.get("turn_role") or "other")
    if turn_role not in _TURN_ROLES:
        turn_role = "other"
    intent = str(data.get("intent") or "other")
    if intent not in _INTENT_CLASSES:
        intent = "other"
    try:
        confidence = float(data.get("confidence", 0.4))
    except Exception:
        confidence = 0.4
    confidence = min(1.0, max(0.0, confidence))
    retrieval_goal = str(data.get("retrieval_goal") or "none")
    if retrieval_goal not in _RETRIEVAL_GOALS:
        retrieval_goal = "none"
    objection_type = data.get("objection_type")
    if objection_type not in {"gia_cao", "phap_ly", "vi_tri", "chua_du_tien", "suy_nghi_them", "khac", None}:
        objection_type = None
    resolved_project_name = _normalize_text(str(data.get("resolved_project_name") or "")) or None
    slots = _sanitize_slot_updates(data.get("slot_updates") or {})
    return {
        "turn_role": turn_role,
        "intent": intent,
        "confidence": confidence,
        "should_retrieve": bool(data.get("should_retrieve", retrieval_goal != "none")),
        "retrieval_goal": retrieval_goal,
        "buy_signal": bool(data.get("buy_signal", intent == "buy_signal")),
        "objection_type": objection_type,
        "resolved_project_name": resolved_project_name,
        "slot_updates": slots,
    }


def _is_short_slot_reply(text: str, slots: Dict[str, Any], intent: str) -> bool:
    if intent not in {"other", "follow_up", "greeting"}:
        return False
    if not slots:
        return False
    words = [token for token in re.split(r"\s+", text.strip()) if token]
    return len(words) <= _SHORT_SLOT_MAX_WORDS


def _estimate_confidence(
    text: str,
    intent: str,
    slots: Dict[str, Any],
    project_name: Optional[str],
    llm_confidence: float,
) -> float:
    score = llm_confidence * 0.7 + 0.15
    if intent != "other":
        score += 0.08
    if slots:
        score += min(0.12, 0.05 * len(slots))
    if project_name:
        score += 0.1
    if len(text.split()) <= 8 and slots:
        score += 0.05
    return min(0.95, max(0.0, score))


def _route_lane(state: SalesAgentState, intent: str, slots: Dict[str, Any], confidence: float) -> str:
    current_state = state.get("current_sales_state") or "greeting"

    if intent == "greeting":
        return "lane_a"

    if _is_short_slot_reply(state.get("user_text") or "", slots, intent):
        return "lane_a"

    if intent in {"project_info", "comparison", "objection", "buy_signal"} and confidence >= 0.55:
        return "lane_b"

    if current_state == "need_discovery" and slots and confidence >= 0.55:
        return "lane_a"

    return "lane_c"


async def fast_parse_user_turn(state: SalesAgentState) -> Dict[str, Any]:
    text = _normalize_text(state.get("user_text") or "")

    slots: Dict[str, Any] = {}
    slots.update(_parse_family_slots(text.lower()))
    slots.update(_parse_purpose_slot(text.lower()))
    slots.update(_parse_location_slot(text))
    slots.update(_parse_contextual_slot_reply(text, state))

    project_name = _extract_project_name(text)
    fallback = _infer_turn_from_slots(state, slots) or {
        "turn_role": "other",
        "intent": "other",
        "confidence": 0.2,
        "should_retrieve": False,
        "retrieval_goal": "none",
        "buy_signal": False,
        "objection_type": None,
        "resolved_project_name": project_name,
        "slot_updates": slots,
    }
    state_for_llm = dict(state)
    state_for_llm["_slot_hints"] = dict(slots)
    micro = await _micro_understand_turn_with_llm(text=text, state=state_for_llm)
    slots.update(micro.get("slot_updates") or {})
    contextual = micro
    if bool(micro.get("should_escalate")):
        deep = await _deep_understand_turn_with_llm(text=text, state=state_for_llm)
        contextual = deep
        slots.update(deep.get("slot_updates") or {})

    intent_result = {
        "intent": contextual.get("intent") or fallback["intent"],
        "confidence": contextual.get("confidence", fallback["confidence"]),
        "buy_signal": contextual.get("buy_signal", fallback["buy_signal"]),
        "objection_type": contextual.get("objection_type", fallback["objection_type"]),
    }
    if intent_result["intent"] not in _INTENT_CLASSES:
        intent_result = {
            "intent": fallback["intent"],
            "confidence": fallback["confidence"],
            "buy_signal": fallback["buy_signal"],
            "objection_type": fallback["objection_type"],
        }

    intent = intent_result["intent"]
    turn_role = contextual.get("turn_role") or fallback["turn_role"]
    if turn_role == "answer_previous_question" and slots:
        intent = "other"
    if turn_role == "ask_catalog_overview":
        intent = "ask_recommendation"
    retrieval_goal = contextual.get("retrieval_goal") or fallback["retrieval_goal"]
    should_retrieve = bool(contextual.get("should_retrieve", fallback["should_retrieve"]))
    if retrieval_goal == "shortlist" and intent in {"other", "follow_up"}:
        intent = "ask_recommendation"
    if retrieval_goal == "project_qa" and intent == "other":
        intent = "project_info"
    if retrieval_goal == "comparison" and intent == "other":
        intent = "comparison"
    if retrieval_goal == "objection_support" and intent == "other":
        intent = "objection"
    if retrieval_goal == "closing_next_step" and intent == "other":
        intent = "buy_signal"

    confidence = _estimate_confidence(
        text=text,
        intent=intent,
        slots=slots,
        project_name=contextual.get("resolved_project_name") or fallback.get("resolved_project_name") or project_name,
        llm_confidence=float(intent_result.get("confidence") or 0.0),
    )
    lane = _route_lane(state, intent, slots, confidence)

    objection_type = intent_result.get("objection_type") if intent == "objection" else None
    buy_signal = bool(intent_result.get("buy_signal", intent == "buy_signal"))

    retrieval_mode = "lite" if lane == "lane_b" else "full"

    log.info(
        "fast_parse_user_turn: lane=%s intent=%s confidence=%.2f slots=%d escalated=%s",
        lane,
        intent,
        confidence,
        len(slots),
        bool(micro.get("should_escalate")),
    )

    return {
        "user_text": text,
        "detected_intent": intent,
        "extracted_slots": slots,
        "resolved_project_name": contextual.get("resolved_project_name") or fallback.get("resolved_project_name") or project_name,
        "objection_type": objection_type,
        "buy_signal": buy_signal,
        "turn_role": turn_role,
        "should_retrieve": should_retrieve,
        "retrieval_goal": retrieval_goal,
        "fast_path_confidence": confidence,
        "fast_lane": lane,
        "retrieval_mode": retrieval_mode,
    }
