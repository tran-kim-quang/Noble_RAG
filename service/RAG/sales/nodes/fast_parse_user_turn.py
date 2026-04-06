"""Context-aware turn understanding with thin rules for latency-sensitive cases."""

import json
import logging
import re
import unicodedata
from typing import Any, Dict, Optional

from core.dependencies import llm_model_func
from sales.graph_state import SalesAgentState
from utils.json_extract import extract_first_json_object

log = logging.getLogger("rag-service")

_PROJECT_NAME_PATTERN = re.compile(r"(Noble[^\n,.;:!?()]*)", re.IGNORECASE)
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
_SEMANTIC_MOVES = {
    "answer_slot",
    "request_shortlist",
    "request_project_fact",
    "raise_objection",
    "advance_purchase",
    "general_follow_up",
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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _fold_text(text: str) -> str:
    raw = _normalize_text(text).lower()
    folded = unicodedata.normalize("NFD", raw)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def _quick_opening_route(
    *,
    text: str,
    current_state: str,
    slots: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if slots:
        return None
    if current_state not in {"greeting", "need_discovery"}:
        return None

    folded = _fold_text(text)
    if not folded:
        return None
    words = [token for token in folded.split(" ") if token]
    if not words:
        return None

    greeting_patterns = (
        "xin chao",
        "chao",
        "hello",
        "hi",
    )
    advisory_patterns = (
        "tu van",
        "ho tro",
        "gioi thieu",
        "tim hieu",
        "giup minh chon",
        "nen mua",
        "muon mua",
    )
    fact_markers = (
        "phap ly",
        "gia",
        "bao nhieu",
        "may can",
        "so can",
        "tien do",
        "dien tich",
        "chinh sach",
        "thanh toan",
    )

    if len(words) <= 6 and any(folded == p or folded.startswith(p + " ") for p in greeting_patterns):
        return {
            "detected_intent": "greeting",
            "turn_role": "greeting",
            "should_retrieve": False,
            "retrieval_goal": "none",
            "buy_signal": False,
            "objection_type": None,
            "fast_path_confidence": 0.9,
            "fast_lane": "lane_a",
            "retrieval_mode": "lite",
            "semantic_parse_done": False,
            "semantic_parse_confidence": 0.0,
        }

    if any(pattern in folded for pattern in advisory_patterns) and not any(marker in folded for marker in fact_markers):
        return {
            "detected_intent": "ask_recommendation",
            "turn_role": "ask_recommendation",
            "should_retrieve": False,
            "retrieval_goal": "none",
            "buy_signal": False,
            "objection_type": None,
            "fast_path_confidence": 0.78,
            "fast_lane": "lane_a",
            "retrieval_mode": "lite",
            "semantic_parse_done": False,
            "semantic_parse_confidence": 0.0,
        }

    return None


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


def _parse_budget_slot(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    slots: Dict[str, Any] = {}
    if match := re.search(r"(\d+(?:[.,]\d+)?)\s*(?:-|den|đến)\s*(\d+(?:[.,]\d+)?)\s*t(?:y|ỷ)", text_lower):
        low = float(match.group(1).replace(",", "."))
        high = float(match.group(2).replace(",", "."))
        slots["budget_min"] = min(low, high)
        slots["budget_max"] = max(low, high)
        slots["budget_text"] = _normalize_text(match.group(0))
        return slots
    if match := re.search(r"(?:duoi|dưới)\s*(\d+(?:[.,]\d+)?)\s*t(?:y|ỷ)", text_lower):
        high = float(match.group(1).replace(",", "."))
        slots["budget_max"] = high
        slots["budget_text"] = _normalize_text(match.group(0))
        return slots
    if match := re.search(r"(\d+(?:[.,]\d+)?)\s*t(?:y|ỷ)", text_lower):
        value = float(match.group(1).replace(",", "."))
        slots["budget_max"] = value
        slots["budget_text"] = _normalize_text(match.group(0))
    return slots


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
        return {
            "turn_role": "answer_previous_question",
            "intent": "other",
            "confidence": 0.6,
            "should_retrieve": False,
            "retrieval_goal": "none",
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
  "semantic_move": "answer_slot" | "request_shortlist" | "request_project_fact" | "raise_objection" | "advance_purchase" | "general_follow_up" | "greeting" | "other",
  "intent": "greeting" | "ask_recommendation" | "project_info" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "confidence": 0.0,
  "should_escalate": false,
  "should_retrieve": false,
  "retrieval_goal": "none" | "shortlist" | "project_qa" | "comparison" | "objection_support" | "closing_next_step",
  "buy_signal": false,
  "objection_type": null,
  "resolved_project_name": null,
  "slot_updates": {{
    "family_member_count": null,
    "children_count": null,
    "purpose": null,
    "budget_text": null,
    "budget_min": null,
    "budget_max": null,
    "location_preference": [],
    "key_concerns": []
  }}
}}

Quy tắc:
- `slot_hints` chỉ là gợi ý mỏng từ parser rule-based, không phải ground truth. Nếu ngữ nghĩa câu hiện tại mâu thuẫn, hãy bỏ qua chúng.
- Hãy hiểu location/purpose/objection/buy signal theo ngữ nghĩa của câu và working memory, không dựa vào pattern hardcode.
- Nếu working memory + user_text chưa đủ để kết luận chắc, đặt should_escalate=true.
- Chỉ bật retrieval nếu khách đang hỏi danh sách dự án, hỏi thông tin dự án, hoặc hỏi so sánh/phản đối/mua ngay.
- Không cần đọc toàn bộ lịch sử xa.
- Nếu khách đang bổ sung thêm tiêu chí như ngân sách, khu vực, mục đích, số người, số con để tiếp tục tư vấn, ưu tiên `semantic_move="answer_slot"` và thường `intent="follow_up"`.
- Nếu khách yêu cầu agent lọc/gợi ý/xem danh sách phương án phù hợp nhất theo nhu cầu đã nói, ưu tiên `semantic_move="request_shortlist"` và `intent="ask_recommendation"`, kể cả khi vẫn còn thiếu vài slot discovery.
- Nếu khách nghi ngại về giá, khả năng chi trả, hoặc chất vấn xem có phương án rẻ hơn/phù hợp ngân sách hơn không, ưu tiên `semantic_move="raise_objection"` và `intent="objection"`.
- Chỉ dùng `intent="project_info"` khi khách thật sự đang hỏi facts/thông tin dự án cụ thể cần trả lời trực tiếp.

Ví dụ ngắn:
- "ngân sách khoảng 4 đến 6 tỷ" sau khi agent đang hỏi thêm nhu cầu
  => semantic_move=answer_slot, intent=follow_up, slot_updates có budget_min/budget_max
- "giá vậy có cao quá không, dưới 5 tỷ được không"
  => semantic_move=raise_objection, intent=objection, objection_type=gia_cao
- "ok cho tôi xem danh sách căn phù hợp nhất"
  => semantic_move=request_shortlist, intent=ask_recommendation, retrieval_goal=shortlist

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
            "semantic_move": "other",
            "intent": "other",
            "confidence": 0.2,
            "should_escalate": False,
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
    semantic_move = str(data.get("semantic_move") or "other")
    if semantic_move not in _SEMANTIC_MOVES:
        semantic_move = "other"

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
    if resolved_project_name and re.search(r"\b(nao|nào|gi|gì|the nao|thế nào|co nhung|có những)\b", resolved_project_name, re.IGNORECASE):
        resolved_project_name = None

    slots = _sanitize_slot_updates(data.get("slot_updates") or {})
    return {
        "turn_role": turn_role,
        "semantic_move": semantic_move,
        "intent": intent,
        "confidence": confidence,
        "should_escalate": bool(data.get("should_escalate", confidence < 0.72)),
        "should_retrieve": bool(data.get("should_retrieve", retrieval_goal != "none")),
        "retrieval_goal": retrieval_goal,
        "buy_signal": bool(data.get("buy_signal", intent == "buy_signal")),
        "objection_type": objection_type,
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
  "semantic_move": "answer_slot" | "request_shortlist" | "request_project_fact" | "raise_objection" | "advance_purchase" | "general_follow_up" | "greeting" | "other",
  "intent": "greeting" | "ask_recommendation" | "project_info" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "confidence": 0.0,
  "should_retrieve": false,
  "retrieval_goal": "none" | "shortlist" | "project_qa" | "comparison" | "objection_support" | "closing_next_step",
  "buy_signal": false,
  "objection_type": null,
  "resolved_project_name": null,
  "slot_updates": {{
    "family_member_count": null,
    "children_count": null,
    "purpose": null,
    "budget_text": null,
    "budget_min": null,
    "budget_max": null,
    "location_preference": [],
    "key_concerns": []
  }}
}}

Quy tắc:
- Ưu tiên semantic understanding của user_text + lịch sử gần đây.
- `slot_updates` chỉ điền khi thật sự chắc; nếu không chắc thì để null/[] và dùng confidence thấp hơn.
- Nếu khách đang tiếp tục discovery bằng cách bổ sung ngân sách, khu vực, mục đích, số người, số con thì ưu tiên turn_role=`answer_previous_question` hoặc `continue_previous_topic`, không đẩy sang `project_info` trừ khi họ thực sự hỏi thông tin dự án cụ thể.
- Nếu khách chuyển từ hỏi/trao đổi sang yêu cầu agent lọc/gợi ý/xem danh sách phương án phù hợp, ưu tiên `semantic_move="request_shortlist"` và `intent="ask_recommendation"`.
- Nếu khách thể hiện băn khoăn về giá/tài chính/sự phù hợp và đang muốn phản biện lại đề xuất, ưu tiên `semantic_move="raise_objection"` và `intent="objection"`.

Ví dụ:
- "giá vậy có cao quá không, dưới 5 tỷ được không" => semantic_move=raise_objection, intent=objection, objection_type=gia_cao
- "ok cho tôi xem danh sách căn phù hợp nhất" => semantic_move=request_shortlist, intent=ask_recommendation, retrieval_goal=shortlist
- "quận 7" sau khi agent hỏi vị trí => semantic_move=answer_slot, intent=follow_up, slot_updates có location_preference

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
            "semantic_move": "other",
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
    semantic_move = str(data.get("semantic_move") or "other")
    if semantic_move not in _SEMANTIC_MOVES:
        semantic_move = "other"
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
        "semantic_move": semantic_move,
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


def _rule_only_context(
    state: SalesAgentState,
    fallback: Dict[str, Any],
    slots: Dict[str, Any],
    project_name: Optional[str],
) -> Dict[str, Any]:
    intent = str(fallback.get("intent") or "other")
    semantic_move = "answer_slot" if slots else "general_follow_up"
    retrieval_goal = "none"
    should_retrieve = False
    confidence = _estimate_confidence(
        text=str(state.get("user_text") or ""),
        intent=intent,
        slots=slots,
        project_name=project_name,
        llm_confidence=float(fallback.get("confidence") or 0.55),
    )
    return {
        "turn_role": fallback.get("turn_role") or "answer_previous_question",
        "semantic_move": semantic_move,
        "intent": intent,
        "confidence": confidence,
        "should_escalate": False,
        "should_retrieve": should_retrieve,
        "retrieval_goal": retrieval_goal,
        "buy_signal": bool(fallback.get("buy_signal")),
        "objection_type": fallback.get("objection_type"),
        "resolved_project_name": project_name,
        "slot_updates": dict(slots),
    }


async def fast_parse_user_turn(state: SalesAgentState) -> Dict[str, Any]:
    text = _normalize_text(state.get("user_text") or "")

    slots: Dict[str, Any] = {}
    slots.update(_parse_family_slots(text.lower()))
    slots.update(_parse_purpose_slot(text.lower()))
    slots.update(_parse_budget_slot(text))
    slots.update(_parse_contextual_slot_reply(text, state))

    project_name = _extract_project_name(text)
    quick_route = _quick_opening_route(
        text=text,
        current_state=state.get("current_sales_state") or "greeting",
        slots=slots,
    )
    if quick_route:
        log.info(
            "fast_parse_user_turn: quick route applied intent=%s state=%s",
            quick_route["detected_intent"],
            state.get("current_sales_state") or "greeting",
        )
        return {
            "user_text": text,
            "detected_intent": quick_route["detected_intent"],
            "extracted_slots": slots,
            "resolved_project_name": project_name,
            "objection_type": quick_route["objection_type"],
            "buy_signal": quick_route["buy_signal"],
            "turn_role": quick_route["turn_role"],
            "should_retrieve": quick_route["should_retrieve"],
            "retrieval_goal": quick_route["retrieval_goal"],
            "fast_path_confidence": quick_route["fast_path_confidence"],
            "fast_lane": quick_route["fast_lane"],
            "retrieval_mode": quick_route["retrieval_mode"],
            "semantic_parse_done": quick_route["semantic_parse_done"],
            "semantic_parse_confidence": quick_route["semantic_parse_confidence"],
        }
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
    semantic_parse_done = False
    semantic_parse_confidence = 0.0
    skip_micro = state.get("current_sales_state") == "need_discovery" and _is_short_slot_reply(
        text,
        slots,
        str(fallback.get("intent") or "other"),
    )
    if skip_micro:
        micro = {"should_escalate": False}
        contextual = _rule_only_context(state, fallback, slots, project_name)
    else:
        micro = await _micro_understand_turn_with_llm(text=text, state=state_for_llm)
        slots.update(micro.get("slot_updates") or {})
        contextual = micro
        if bool(micro.get("should_escalate")):
            deep = await _deep_understand_turn_with_llm(text=text, state=state_for_llm)
            contextual = deep
            slots.update(deep.get("slot_updates") or {})
            semantic_parse_done = True
            try:
                semantic_parse_confidence = float(deep.get("confidence") or 0.0)
            except Exception:
                semantic_parse_confidence = 0.0
    semantic_move = contextual.get("semantic_move") or "other"

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
    if semantic_move == "request_shortlist" and intent in {"other", "follow_up"}:
        intent = "ask_recommendation"
        should_retrieve = True
        if retrieval_goal == "none":
            retrieval_goal = "shortlist"
    if semantic_move == "raise_objection" and intent == "other":
        intent = "objection"
        if retrieval_goal == "none":
            retrieval_goal = "objection_support"
    if semantic_move == "advance_purchase" and intent == "other":
        intent = "buy_signal"
        should_retrieve = True
        if retrieval_goal == "none":
            retrieval_goal = "closing_next_step"
    if semantic_move == "answer_slot" and slots:
        should_retrieve = False
        if retrieval_goal == "project_qa":
            retrieval_goal = "none"

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
        "fast_parse_user_turn: lane=%s intent=%s confidence=%.2f slots=%d escalated=%s rule_only=%s semantic_done=%s",
        lane,
        intent,
        confidence,
        len(slots),
        bool(micro.get("should_escalate")),
        skip_micro,
        semantic_parse_done,
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
        "semantic_parse_done": semantic_parse_done,
        "semantic_parse_confidence": semantic_parse_confidence,
    }
