"""Fast rule-based parser for first-turn latency optimization."""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from lightrag.llm.ollama import ollama_model_complete
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
_FAST_INTENT_MODEL = os.getenv("FAST_INTENT_MODEL", "ollama2.5:7b")
_FAST_INTENT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _extract_project_name(text: str) -> Optional[str]:
    match = _PROJECT_NAME_PATTERN.search(text or "")
    return match.group(1).strip() if match else None


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


async def _detect_intent_with_ollama(
    text: str,
    current_state: str,
    chat_history: list[Dict[str, Any]],
) -> Dict[str, Any]:
    history_str = "\n".join(
        f"{(item.get('role') or 'unknown')}: {(item.get('content') or '').strip()}"
        for item in chat_history[-3:]
        if isinstance(item, dict)
    ) or "(chưa có)"

    prompt = f"""Bạn là bộ phân loại intent hội thoại sales bất động sản.
Chỉ trả về JSON hợp lệ duy nhất theo schema:
{{
  "intent": "greeting" | "ask_recommendation" | "project_info" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "confidence": 0.0,
  "buy_signal": false,
  "objection_type": null
}}

Quy tắc:
- Không giải thích.
- Nếu không chắc, dùng intent="other" và confidence thấp.
- objection_type chỉ nhận: "gia_cao" | "phap_ly" | "vi_tri" | "chua_du_tien" | "suy_nghi_them" | "khac" | null

Current sales state: {current_state}
Lịch sử gần đây:
{history_str}

Tin nhắn khách:
\"\"\"{text}\"\"\""""

    try:
        raw = await ollama_model_complete(
            prompt,
            model=_FAST_INTENT_MODEL,
            host=_FAST_INTENT_HOST,
            options={"temperature": 0, "num_predict": 180},
        )
        payload = extract_first_json_object(str(raw))
        if not payload:
            raise ValueError("No JSON in Ollama intent response")
        data = json.loads(payload)
    except Exception as e:
        log.warning("fast_parse intent classify failed: %s", e)
        return {
            "intent": "other",
            "confidence": 0.2,
            "buy_signal": False,
            "objection_type": None,
        }

    intent = str(data.get("intent") or "other")
    if intent not in _INTENT_CLASSES:
        intent = "other"

    try:
        confidence = float(data.get("confidence", 0.4))
    except Exception:
        confidence = 0.4
    confidence = min(1.0, max(0.0, confidence))

    objection_type = data.get("objection_type")
    if objection_type not in {"gia_cao", "phap_ly", "vi_tri", "chua_du_tien", "suy_nghi_them", "khac", None}:
        objection_type = None

    return {
        "intent": intent,
        "confidence": confidence,
        "buy_signal": bool(data.get("buy_signal", intent == "buy_signal")),
        "objection_type": objection_type,
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
    text_lower = text.lower()

    slots: Dict[str, Any] = {}
    slots.update(_parse_family_slots(text_lower))
    slots.update(_parse_purpose_slot(text_lower))
    slots.update(_parse_location_slot(text))

    project_name = _extract_project_name(text)
    intent_result = await _detect_intent_with_ollama(
        text=text,
        current_state=state.get("current_sales_state") or "greeting",
        chat_history=state.get("chat_history") or [],
    )
    intent = intent_result["intent"]
    confidence = _estimate_confidence(
        text=text,
        intent=intent,
        slots=slots,
        project_name=project_name,
        llm_confidence=float(intent_result.get("confidence") or 0.0),
    )
    lane = _route_lane(state, intent, slots, confidence)

    objection_type = intent_result.get("objection_type") if intent == "objection" else None
    buy_signal = bool(intent_result.get("buy_signal", intent == "buy_signal"))

    retrieval_mode = "lite" if lane == "lane_b" else "full"

    log.info(
        "fast_parse_user_turn: lane=%s intent=%s confidence=%.2f slots=%d",
        lane,
        intent,
        confidence,
        len(slots),
    )

    return {
        "user_text": text,
        "detected_intent": intent,
        "extracted_slots": slots,
        "resolved_project_name": project_name,
        "objection_type": objection_type,
        "buy_signal": buy_signal,
        "fast_path_confidence": confidence,
        "fast_lane": lane,
        "retrieval_mode": retrieval_mode,
    }
