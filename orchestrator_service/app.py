from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
import socket
from typing import Any, Callable
import urllib.error
import urllib.request

from fastapi import FastAPI
from fastapi import HTTPException

from orchestrator_service.config import Settings
from orchestrator_service.config import get_settings
from orchestrator_service.observability import flush_observability
from orchestrator_service.observability import propagate_context
from orchestrator_service.observability import start_observation
from orchestrator_service.retrieval_client import RetrievalClient
from orchestrator_service.schemas import (
    DecisionTrace,
    HistoryTurn,
    LeadState,
    NeedPainpointDelta,
    NeedPainpointState,
    QueryRequest,
    QueryResponse,
    RoutingSignal,
    TopicWeight,
)

log = logging.getLogger("sales-orchestrator")
_MODEL_HTTP_USER_AGENT = "Noble-RAG-Orchestrator/1.0"

_QUERY_TYPES = {"advisory_strategy", "project_matching", "project_specific", "clarification"}
_READINESS_VALUES = {"not_ready", "soft_ready", "ready"}
_PROJECT_QUERY_TYPES = {"project_matching", "project_specific"}
_POI_TYPE_VALUES = {"hospital", "school", "park", "mall"}
_ENGAGEMENT_STATES = {"cold", "warm", "interested", "ready"}
_SALES_STATES = {
    "unknown",
    "exploring",
    "need_identified",
    "qualified",
    "interested",
    "appointment_ready",
    "nurture",
    "handoff",
}
_CONVERSATION_GOALS = {
    "build_trust",
    "discover_need",
    "surface_priority",
    "show_fit",
    "handle_concern",
    "invite_next_step",
    "nurture_lead",
    "handoff_to_human",
    "capture_contact",
    "confirm_followup",
}
_NEXT_BEST_ACTIONS = {
    "continue_discovery",
    "show_project_fit",
    "handle_concern",
    "invite_brochure",
    "invite_call",
    "invite_site_visit",
    "handoff_human",
    "ask_name",
    "ask_phone",
    "ask_name_and_phone",
    "schedule_followup",
}
_RESPONSE_MODES = {
    "warm_welcome",
    "value_teaser",
    "discover_need",
    "consultive_recommendation",
    "grounded_recommendation",
    "handle_concern",
    "soft_next_step",
    "nurture_followup",
    "meeting_invite",
    "contact_capture",
    "followup_confirm",
}
_ASK_POLICIES = {"avoid_question", "allow_question", "must_clarify"}
_LEGACY_RESPONSE_MODE_MAP = {
    "recommendation": "grounded_recommendation",
    "handle_objection": "handle_concern",
    "next_step_invite": "soft_next_step",
}
_FASTPATH_CONSULT_RESPONSE_MODES = {
    "warm_welcome",
    "value_teaser",
    "discover_need",
    "consultive_recommendation",
    "followup_confirm",
}
_SYNTHESIS_REPLY_MAX_WORDS = 180
_DECIDER_CONSULT_REPLY_MAX_WORDS = 140


@dataclass(frozen=True)
class TurnAnalysis:
    route: str
    decision_reason: str
    need_update: NeedPainpointDelta
    painpoint_update: NeedPainpointDelta
    routing_signal: RoutingSignal
    consult_reply: str
    query_type: str = "clarification"
    retrieval_readiness: str = "not_ready"
    route_source: str = "unknown"
    engagement_state_after_hint: str | None = None
    sales_state_after_hint: str | None = None
    conversation_goal_hint: str | None = None
    extracted_name: str | None = None
    extracted_phone: str | None = None


@dataclass(frozen=True)
class ReplyPlan:
    response_mode: str
    ask_policy: str
    focus: str
    question_focus: str
    conversation_goal: str
    next_best_action: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_keep_order(values: list[str], max_items: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _normalize_route(route_raw: Any) -> str:
    route_str = str(route_raw or "").strip().lower()
    if route_str == "project_grounded":
        return "project_grounded"
    return "consult_discovery"


def _normalize_query_type(raw: Any, fallback: str = "clarification") -> str:
    value = str(raw or "").strip().lower()
    if value in _QUERY_TYPES:
        return value
    return fallback if fallback in _QUERY_TYPES else "clarification"


def _normalize_retrieval_readiness(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in _READINESS_VALUES:
        return value
    return "not_ready"


def _extract_name(message: str) -> str | None:
    patterns = [
        r"(?:mình|toi|tôi|em|anh|chi|chị)\s+(?:là|la)\s+([a-zA-ZÀ-ỹ][a-zA-ZÀ-ỹ\s]{1,30})",
        r"(?:tên|ten)\s+(?:mình|toi|tôi|em|anh|chi|chị)\s+(?:là|la)\s+([a-zA-ZÀ-ỹ][a-zA-ZÀ-ỹ\s]{1,30})",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return " ".join(match.group(1).strip().split())[:64]
    return None


def _extract_phone_contact(message: str) -> str | None:
    match = re.search(r"(?:(?:\+?84)|0)\d{8,10}", message)
    if not match: 
        return None
    phone = re.sub(r"\s+", "", match.group(0))
    return phone


def _coerce_extracted_name(raw: Any) -> str | None:
    cleaned = " ".join(str(raw or "").strip().split())
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "n/a"}:
        return None
    if not re.search(r"[a-zA-ZÀ-ỹ]", cleaned):
        return None
    return cleaned[:64]


def _coerce_extracted_phone(raw: Any) -> str | None:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "n/a"}:
        return None
    return _extract_phone_contact(cleaned)


def _merge_summaries(old_summary: str, summary_delta: str) -> str:
    parts = _dedupe_keep_order([old_summary, summary_delta], max_items=3)
    return " ".join(parts).strip()


def _merge_topics(old_topics: list[TopicWeight], new_topics: list[TopicWeight]) -> list[TopicWeight]:
    by_label: dict[str, float] = {item.label: float(item.weight) for item in old_topics}
    for topic in new_topics:
        if topic.label in by_label:
            by_label[topic.label] = min(1.0, round(by_label[topic.label] * 0.65 + float(topic.weight) * 0.35, 4))
        else:
            by_label[topic.label] = float(topic.weight)
    merged = [TopicWeight(label=label, weight=weight) for label, weight in by_label.items()]
    merged.sort(key=lambda item: item.weight, reverse=True)
    return merged[:10]


def _merge_evidence(old: list[str], new: list[str]) -> list[str]:
    return _dedupe_keep_order(old + new, max_items=10)


def _merge_block(state: NeedPainpointState, delta: NeedPainpointDelta) -> NeedPainpointState:
    changed = bool(delta.summary_delta.strip() or delta.topics or delta.evidence)
    return NeedPainpointState(
        summary=_merge_summaries(state.summary, delta.summary_delta),
        topics=_merge_topics(state.topics, delta.topics),
        evidence=_merge_evidence(state.evidence, delta.evidence),
        last_updated_at=_now_iso() if changed else state.last_updated_at,
    )


def merge_lead_state(
    lead_state: LeadState,
    need_update: NeedPainpointDelta,
    painpoint_update: NeedPainpointDelta,
    extracted_name: str | None,
    extracted_phone: str | None,
    ) -> LeadState:
    return LeadState(
        name=extracted_name or lead_state.name,
        phone_contact=extracted_phone or lead_state.phone_contact,
        need=_merge_block(lead_state.need, need_update),
        painpoint=_merge_block(lead_state.painpoint, painpoint_update),
        engagement_state=lead_state.engagement_state,
        engagement_confidence=lead_state.engagement_confidence,
        sales_state=lead_state.sales_state,
        lead_level=lead_state.lead_level,
        last_conversation_goal=lead_state.last_conversation_goal,
        next_best_action=lead_state.next_best_action,
        contact_capture_status=lead_state.contact_capture_status,
    )


def _derive_observability_session_id(payload: QueryRequest, lead_state: LeadState) -> str | None:
    explicit = str(payload.session_id or "").strip()
    if explicit:
        return explicit[:128]
    if lead_state.phone_contact:
        return f"phone:{lead_state.phone_contact}"[:128]
    if lead_state.name:
        normalized = re.sub(r"\s+", "-", lead_state.name.strip().lower())
        normalized = re.sub(r"[^a-z0-9_\-]", "", normalized)
        if normalized:
            return f"name:{normalized}"[:128]
    return None


def _derive_observability_user_id(lead_state: LeadState) -> str | None:
    phone = str(lead_state.phone_contact or "").strip()
    if phone:
        return phone[:128]
    name = str(lead_state.name or "").strip()
    if name:
        return name[:128]
    return None


def _build_retrieval_intent(
    message: str,
    lead_state: LeadState,
    query_type: str,
    retrieval_readiness: str,
) -> dict[str, Any]:
    need_topics = [topic.label for topic in lead_state.need.topics[:4]]
    pain_topics = [topic.label for topic in lead_state.painpoint.topics[:4]]
    semantic_focus = need_topics + pain_topics
    persona_hint: list[str] = []
    poi_types: list[str] = []

    intent_goal = query_type if query_type in _PROJECT_QUERY_TYPES else "project_matching"
    return {
        "goal": intent_goal,
        "semantic_focus": _dedupe_keep_order(semantic_focus, max_items=8),
        "persona_hint": _dedupe_keep_order(persona_hint, max_items=4),
        "poi_types": [item for item in poi_types if item in _POI_TYPE_VALUES],
        "filters": {},
        "retrieval_readiness": retrieval_readiness,
        "raw_user_query": message.strip(),
    }


def _call_project_grounded_fetcher(
    fetcher,
    message: str,
    intent: dict[str, Any],
    top_k: int,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    try:
        if trace_id or session_id or user_id:
            return fetcher(
                message,
                intent,
                top_k,
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
            )
        return fetcher(message, intent, top_k)
    except TypeError:
        try:
            if trace_id or session_id or user_id:
                return fetcher(
                    message,
                    json.dumps(intent, ensure_ascii=False),
                    top_k,
                    trace_id=trace_id,
                    session_id=session_id,
                    user_id=user_id,
                )
            return fetcher(message, json.dumps(intent, ensure_ascii=False), top_k)
        except TypeError:
            legacy = fetcher(message, top_k)
            return {
                "project_cards": [],
                "trait_tags": [],
                "proximity_facts": [],
                "evidence_chunks": legacy.get("results", []) if isinstance(legacy, dict) else [],
                "confidence": legacy.get("confidence", 0.0) if isinstance(legacy, dict) else 0.0,
                "low_confidence": bool(legacy.get("low_confidence", False)) if isinstance(legacy, dict) else True,
            }


def _coerce_topics(raw: Any, max_items: int = 8) -> list[TopicWeight]:
    if not isinstance(raw, list):
        return []
    topics: list[TopicWeight] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label or label in seen:
            continue
        try:
            weight = float(item.get("weight", 0.65))
        except (TypeError, ValueError):
            weight = 0.65
        weight = max(0.0, min(1.0, weight))
        topics.append(TopicWeight(label=label, weight=weight))
        seen.add(label)
        if len(topics) >= max_items:
            break
    return topics


def _coerce_evidence(raw: Any, fallback_message: str) -> list[str]:
    evidence: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                evidence.append(text)
            if len(evidence) >= 5:
                break
    if not evidence and fallback_message.strip():
        evidence = [fallback_message.strip()]
    return evidence


def _coerce_delta(raw: Any, fallback_message: str) -> NeedPainpointDelta:
    if not isinstance(raw, dict):
        return NeedPainpointDelta(
            summary_delta="",
            topics=[],
            evidence=_coerce_evidence([], fallback_message),
        )
    summary_delta = str(raw.get("summary_delta", "")).strip()
    return NeedPainpointDelta(
        summary_delta=summary_delta,
        topics=_coerce_topics(raw.get("topics")),
        evidence=_coerce_evidence(raw.get("evidence"), fallback_message),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response does not contain JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed


def _compact_text(text: str, max_words: int) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    words = cleaned.split(" ")
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "..."


def _friendly_project_name(project_id: str) -> str:
    cleaned = (project_id or "").strip()
    if not cleaned:
        return ""
    if "_" in cleaned:
        cleaned = cleaned.replace("_", " ")
    return cleaned.title()


def _is_single_project_mode(project_cards: list[dict], proximity_facts: list[dict]) -> bool:
    ids: set[str] = set()
    for card in project_cards:
        project_id = str(card.get("project_id", "")).strip()
        if project_id:
            ids.add(project_id)
    for fact in proximity_facts:
        project_id = str(fact.get("project_id", "")).strip()
        if project_id:
            ids.add(project_id)
    return len(ids) == 1


def _normalize_engagement_state(raw: Any, fallback: str = "cold") -> str:
    value = str(raw or "").strip().lower()
    if value in _ENGAGEMENT_STATES:
        return value
    return fallback if fallback in _ENGAGEMENT_STATES else "cold"


def _normalize_response_mode(raw: Any, fallback: str = "warm_welcome") -> str:
    value = str(raw or "").strip().lower()
    if value in _LEGACY_RESPONSE_MODE_MAP:
        value = _LEGACY_RESPONSE_MODE_MAP[value]
    if value in _RESPONSE_MODES:
        return value
    return fallback if fallback in _RESPONSE_MODES else "warm_welcome"


def _normalize_ask_policy(raw: Any, fallback: str = "avoid_question") -> str:
    value = str(raw or "").strip().lower()
    if value in _ASK_POLICIES:
        return value
    return fallback if fallback in _ASK_POLICIES else "avoid_question"


def _normalize_sales_state(raw: Any, fallback: str = "unknown") -> str:
    value = str(raw or "").strip().lower()
    if value in _SALES_STATES:
        return value
    return fallback if fallback in _SALES_STATES else "unknown"


def _normalize_conversation_goal(raw: Any, fallback: str = "build_trust") -> str:
    value = str(raw or "").strip().lower()
    if value in _CONVERSATION_GOALS:
        return value
    return fallback if fallback in _CONVERSATION_GOALS else "build_trust"


def _normalize_next_best_action(raw: Any, fallback: str = "continue_discovery") -> str:
    value = str(raw or "").strip().lower()
    if value in _NEXT_BEST_ACTIONS:
        return value
    return fallback if fallback in _NEXT_BEST_ACTIONS else "continue_discovery"


def _derive_lead_level(sales_state: str) -> str:
    state = _normalize_sales_state(sales_state)
    if state in {"unknown", "exploring"}:
        return "exploratory"
    if state in {"need_identified", "nurture"}:
        return "interested"
    if state in {"qualified", "interested"}:
        return "qualified"
    return "hot"


def _has_structured_need(lead_state: LeadState) -> bool:
    return bool(
        lead_state.need.summary.strip()
        or lead_state.painpoint.summary.strip()
        or lead_state.need.topics
        or lead_state.painpoint.topics
    )


def _has_grounded_fit(grounded_result: dict[str, Any] | None) -> bool:
    if not grounded_result:
        return False
    cards = grounded_result.get("project_cards", []) or []
    return bool(cards and not bool(grounded_result.get("low_confidence", False)))


def _missing_contact_fields(lead_state: LeadState) -> tuple[bool, bool]:
    missing_name = not bool((lead_state.name or "").strip())
    missing_phone = not bool((lead_state.phone_contact or "").strip())
    return missing_name, missing_phone


def _message_requests_followup(message: str) -> bool:
    lowered = (message or "").strip().lower()
    signals = [
        "đi xem",
        "xem thực tế",
        "gặp trực tiếp",
        "hẹn",
        "gọi lại",
        "liên hệ",
        "tư vấn kỹ hơn",
        "tư vấn sâu hơn",
        "gửi thông tin",
        "gửi bảng giá",
        "đặt lịch",
        "trao đổi thêm",
    ]
    return any(token in lowered for token in signals)


def _promote_engagement_state(current: str, candidate: str) -> str:
    ranking = {"cold": 0, "warm": 1, "interested": 2, "ready": 3}
    current_norm = _normalize_engagement_state(current)
    candidate_norm = _normalize_engagement_state(candidate, fallback=current_norm)
    if ranking[candidate_norm] >= ranking[current_norm]:
        return candidate_norm
    return current_norm


def update_engagement_state(
    previous_state: str,
    query_type: str,
    lead_state: LeadState,
    grounded_result: dict[str, Any] | None,
    recent_history: list[HistoryTurn],
    routed_to_project: bool,
    suggested_state: str | None = None,
) -> str:
    state = _normalize_engagement_state(previous_state)
    suggestion = _normalize_engagement_state(suggested_state, fallback=state) if suggested_state else None
    grounded_fit = _has_grounded_fit(grounded_result)
    low_confidence = bool(grounded_result.get("low_confidence", False)) if grounded_result else False
    has_state_context = _has_structured_need(lead_state)

    if query_type == "advisory_strategy":
        if has_state_context or len(recent_history) >= 1:
            state = _promote_engagement_state(state, "warm")
    elif query_type == "project_matching":
        state = _promote_engagement_state(state, "warm")
        if routed_to_project or grounded_fit:
            state = _promote_engagement_state(state, "interested")
    elif query_type == "project_specific":
        state = _promote_engagement_state(state, "interested")
    elif query_type == "clarification":
        if has_state_context:
            state = _promote_engagement_state(state, "warm")

    if grounded_fit and query_type in _PROJECT_QUERY_TYPES:
        state = _promote_engagement_state(state, "interested")
    if has_state_context and len(recent_history) >= 3:
        state = _promote_engagement_state(state, "interested")
    if lead_state.sales_state in {"appointment_ready", "handoff"}:
        state = _promote_engagement_state(state, "ready")
    if lead_state.next_best_action in {"invite_call", "invite_site_visit", "handoff_human"}:
        state = _promote_engagement_state(state, "ready")

    if low_confidence and routed_to_project and state == "ready":
        state = "interested"

    if suggestion:
        state = _promote_engagement_state(state, suggestion)
    return _normalize_engagement_state(state)


def update_sales_state(
    previous_state: str,
    message: str,
    query_type: str,
    lead_state: LeadState,
    grounded_result: dict[str, Any] | None,
    recent_history: list[HistoryTurn],
    routed_to_project: bool,
    suggested_state: str | None = None,
) -> str:
    state = _normalize_sales_state(previous_state)
    suggestion = _normalize_sales_state(suggested_state, fallback=state) if suggested_state else None
    low_confidence = bool(grounded_result.get("low_confidence", False)) if grounded_result else False
    grounded_fit = _has_grounded_fit(grounded_result)

    if suggestion == "handoff":
        return "handoff"
    if routed_to_project and low_confidence and query_type in _PROJECT_QUERY_TYPES:
        return "handoff"
    if query_type == "project_specific" and routed_to_project and not grounded_fit:
        return "handoff"

    if query_type == "advisory_strategy":
        if state == "unknown":
            state = "exploring"
        if _has_structured_need(lead_state):
            state = "need_identified"
    elif query_type == "project_matching":
        if state in {"unknown", "exploring"}:
            state = "need_identified"
        if routed_to_project and grounded_fit:
            state = "qualified"
    elif query_type == "project_specific":
        if state in {"unknown", "exploring"}:
            state = "need_identified"
        if routed_to_project:
            state = "interested" if grounded_fit else "qualified"
    elif query_type == "clarification":
        if state == "unknown":
            state = "exploring"

    if state == "qualified" and grounded_fit and len(recent_history) >= 2:
        state = "interested"
    if state == "interested":
        if _message_requests_followup(message):
            state = "appointment_ready"
        elif lead_state.next_best_action in {"invite_call", "invite_site_visit", "handoff_human"}:
            state = "appointment_ready"
    if state in {"exploring", "need_identified"} and len(recent_history) >= 4 and not _has_grounded_fit(grounded_result):
        state = "nurture"

    if suggestion and suggestion in {"appointment_ready", "handoff"}:
        state = suggestion
    return _normalize_sales_state(state)


def select_conversation_goal(
    sales_state: str,
    engagement_state: str,
    query_type: str,
    grounded_result: dict[str, Any] | None,
    lead_state: LeadState,
    message: str,
    suggested_goal: str | None = None,
) -> str:
    state = _normalize_sales_state(sales_state)
    engagement = _normalize_engagement_state(engagement_state)
    suggestion = _normalize_conversation_goal(suggested_goal, fallback="build_trust") if suggested_goal else None
    missing_name, missing_phone = _missing_contact_fields(lead_state)

    if state in {"appointment_ready", "handoff"} or _message_requests_followup(message):
        if missing_name or missing_phone:
            return "capture_contact"
        return "confirm_followup"
    if suggestion == "handoff_to_human":
        return "handoff_to_human"

    if engagement == "ready":
        return "invite_next_step"
    if engagement == "interested" and query_type in _PROJECT_QUERY_TYPES:
        return "invite_next_step" if _has_grounded_fit(grounded_result) else "handle_concern"

    if state == "unknown":
        goal = "build_trust"
    elif state == "exploring":
        goal = "discover_need"
    elif state == "need_identified":
        goal = "show_fit" if query_type in _PROJECT_QUERY_TYPES else "surface_priority"
    elif state == "qualified":
        goal = "show_fit"
    elif state == "interested":
        goal = "invite_next_step" if _has_grounded_fit(grounded_result) else "handle_concern"
    elif state == "appointment_ready":
        goal = "invite_next_step"
    elif state == "nurture":
        goal = "nurture_lead"
    else:
        goal = "handoff_to_human"
    return _normalize_conversation_goal(goal)


def select_next_best_action(
    sales_state: str,
    engagement_state: str,
    conversation_goal: str,
    grounded_result: dict[str, Any] | None,
    lead_state: LeadState,
) -> str:
    state = _normalize_sales_state(sales_state)
    engagement = _normalize_engagement_state(engagement_state)
    goal = _normalize_conversation_goal(conversation_goal)
    missing_name, missing_phone = _missing_contact_fields(lead_state)

    if goal == "capture_contact":
        if missing_name and missing_phone:
            return "ask_name_and_phone"
        if missing_name:
            return "ask_name"
        if missing_phone:
            return "ask_phone"
    if goal == "confirm_followup":
        return "schedule_followup"

    if state == "handoff" or goal == "handoff_to_human":
        return "handoff_human"
    if engagement == "ready":
        return "invite_site_visit"
    if state == "appointment_ready":
        return "invite_site_visit"
    if goal == "invite_next_step":
        return "invite_call"
    if goal == "nurture_lead":
        return "invite_brochure"
    if goal == "handle_concern":
        return "handle_concern"
    if goal == "show_fit":
        return "show_project_fit"
    if _has_grounded_fit(grounded_result):
        return "show_project_fit"
    return "continue_discovery"


def _ends_with_question(text: str) -> bool:
    return bool(re.search(r"\?\s*$", (text or "").strip()))


def _recent_assistant_question_count(recent_history: list[HistoryTurn], take_last_assistant_turns: int = 2) -> int:
    count = 0
    seen_assistant = 0
    for turn in reversed(recent_history):
        if turn.role != "assistant":
            continue
        seen_assistant += 1
        if _ends_with_question(turn.message):
            count += 1
        if seen_assistant >= take_last_assistant_turns:
            break
    return count


def _is_short_user_reply_after_assistant_question(message: str, recent_history: list[HistoryTurn]) -> bool:
    cleaned = (message or "").strip()
    if not cleaned:
        return False
    if len(cleaned.split()) > 6:
        return False
    if not recent_history:
        return False
    last_turn = recent_history[-1]
    if last_turn.role != "assistant":
        return False
    return _ends_with_question(last_turn.message)


def _select_question_focus(conversation_goal: str, next_best_action: str, query_type: str) -> str:
    goal = _normalize_conversation_goal(conversation_goal)
    action = _normalize_next_best_action(next_best_action)
    if action == "ask_name":
        return "tên xưng hô thuận tiện"
    if action == "ask_phone":
        return "số điện thoại liên hệ"
    if action == "ask_name_and_phone":
        return "tên và số điện thoại liên hệ"
    if action == "schedule_followup":
        return "thời điểm tiện để mình liên hệ lại"
    if goal == "discover_need":
        return "mục tiêu sử dụng"
    if goal == "surface_priority":
        return "ưu tiên quan trọng nhất"
    if goal == "invite_next_step":
        if action == "invite_site_visit":
            return "thời điểm đi xem thực tế"
        if action == "invite_call":
            return "khung thời gian trao đổi"
        return "mức sẵn sàng bước tiếp theo"
    if goal == "handle_concern":
        return "băn khoăn lớn nhất"
    if query_type == "project_specific":
        return "điểm muốn làm rõ sâu hơn"
    return "điểm ưu tiên chính"


def _select_response_mode(
    query_type: str,
    engagement_state: str,
    conversation_goal: str,
    grounded_result: dict[str, Any] | None,
) -> str:
    qtype = _normalize_query_type(query_type)
    engagement = _normalize_engagement_state(engagement_state)
    goal = _normalize_conversation_goal(conversation_goal)
    has_grounded_result = _has_grounded_fit(grounded_result)
    low_confidence = bool(grounded_result.get("low_confidence", False)) if grounded_result else False

    if goal == "capture_contact":
        return "contact_capture"
    if goal == "confirm_followup":
        return "followup_confirm"
    if goal == "handoff_to_human":
        return "meeting_invite"

    if qtype == "clarification":
        if goal == "invite_next_step":
            return "meeting_invite" if engagement == "ready" else "soft_next_step"
        if goal == "handle_concern":
            return "handle_concern"
        if goal == "nurture_lead":
            return "nurture_followup"
        if has_grounded_result:
            return "grounded_recommendation"
        if engagement == "cold":
            return "warm_welcome"
        if engagement == "warm":
            return "discover_need"
        if engagement == "interested":
            return "consultive_recommendation"
        return "soft_next_step"

    if qtype == "advisory_strategy":
        if engagement == "cold":
            return "warm_welcome"
        if engagement == "warm":
            return "discover_need"
        if engagement == "interested":
            return "consultive_recommendation"
        return "soft_next_step"

    if qtype == "project_matching":
        if engagement == "cold":
            return "value_teaser"
        if engagement == "warm":
            return "consultive_recommendation"
        if engagement == "interested":
            return "grounded_recommendation" if has_grounded_result else "consultive_recommendation"
        return "soft_next_step"

    # project_specific
    if engagement == "cold":
        return "value_teaser"
    if engagement == "warm":
        return "grounded_recommendation" if has_grounded_result else "consultive_recommendation"
    if engagement == "interested":
        if low_confidence:
            return "handle_concern"
        return "grounded_recommendation" if has_grounded_result else "consultive_recommendation"
    if goal == "nurture_lead":
        return "nurture_followup"
    return "meeting_invite"


def _build_reply_focus(
    message: str,
    lead_state: LeadState,
    grounded_result: dict[str, Any] | None,
) -> str:
    if grounded_result:
        cards = grounded_result.get("project_cards", []) or []
        if cards:
            first = cards[0]
            project_name = _friendly_project_name(str(first.get("project_id", "Noble Palace Tây Thăng Long")))
            strengths = [str(item).strip() for item in (first.get("strengths") or []) if str(item).strip()]
            if strengths:
                return _compact_text(f"{project_name}: {', '.join(strengths[:2])}", max_words=18)
            summary = str(first.get("summary", "")).strip()
            if summary:
                return _compact_text(summary, max_words=18)
    if lead_state.need.summary.strip():
        return _compact_text(lead_state.need.summary, max_words=18)
    if lead_state.painpoint.summary.strip():
        return _compact_text(lead_state.painpoint.summary, max_words=18)
    return _compact_text(message, max_words=18)


def build_reply_plan(
    message: str,
    recent_history: list[HistoryTurn],
    lead_state: LeadState,
    analysis: TurnAnalysis,
    conversation_goal: str,
    next_best_action: str,
    grounded_result: dict[str, Any] | None,
) -> ReplyPlan:
    sales_state = _normalize_sales_state(lead_state.sales_state)
    engagement_state = _normalize_engagement_state(lead_state.engagement_state)
    query_type = _normalize_query_type(analysis.query_type)
    goal = _normalize_conversation_goal(conversation_goal)
    action = _normalize_next_best_action(next_best_action)
    response_mode = _normalize_response_mode(
        _select_response_mode(
            query_type=query_type,
            engagement_state=engagement_state,
            conversation_goal=goal,
            grounded_result=grounded_result,
        )
    )
    has_grounded_result = _has_grounded_fit(grounded_result=grounded_result)
    has_state_context = _has_structured_need(lead_state)

    if response_mode == "contact_capture":
        return ReplyPlan(
            response_mode=response_mode,
            ask_policy="must_clarify",
            focus=_build_reply_focus(message=message, lead_state=lead_state, grounded_result=grounded_result),
            question_focus=_select_question_focus(
                conversation_goal=goal,
                next_best_action=action,
                query_type=analysis.query_type,
            ),
            conversation_goal=goal,
            next_best_action=action,
        )
    if response_mode == "followup_confirm":
        return ReplyPlan(
            response_mode=response_mode,
            ask_policy="allow_question",
            focus=_build_reply_focus(message=message, lead_state=lead_state, grounded_result=grounded_result),
            question_focus=_select_question_focus(
                conversation_goal=goal,
                next_best_action=action,
                query_type=analysis.query_type,
            ),
            conversation_goal=goal,
            next_best_action=action,
        )

    if engagement_state == "ready":
        ask_policy = "avoid_question"
    elif engagement_state == "interested":
        ask_policy = "avoid_question" if has_grounded_result else "allow_question"
    elif engagement_state == "warm":
        ask_policy = "allow_question"
    else:
        ask_policy = "avoid_question"

    if response_mode in {"meeting_invite", "soft_next_step", "grounded_recommendation", "value_teaser", "warm_welcome"}:
        ask_policy = "avoid_question"
    elif response_mode in {
        "discover_need",
        "consultive_recommendation",
        "nurture_followup",
        "handle_concern",
        "followup_confirm",
    }:
        ask_policy = "allow_question"

    if response_mode == "discover_need" and not has_state_context and engagement_state in {"cold", "warm"}:
        ask_policy = "must_clarify"
    if has_grounded_result and response_mode in {"grounded_recommendation", "soft_next_step"}:
        ask_policy = "avoid_question"
    if _recent_assistant_question_count(recent_history=recent_history, take_last_assistant_turns=2) >= 1 and ask_policy != "must_clarify":
        ask_policy = "avoid_question"
    if _is_short_user_reply_after_assistant_question(message=message, recent_history=recent_history) and ask_policy != "must_clarify":
        ask_policy = "avoid_question"
    if engagement_state == "ready":
        ask_policy = "avoid_question"
    return ReplyPlan(
        response_mode=response_mode,
        ask_policy=_normalize_ask_policy(ask_policy),
        focus=_build_reply_focus(message=message, lead_state=lead_state, grounded_result=grounded_result),
        question_focus=_select_question_focus(
            conversation_goal=goal,
            next_best_action=action,
            query_type=analysis.query_type,
        ),
        conversation_goal=goal,
        next_best_action=action,
    )


def _looks_unaccented_vietnamese(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    letters = [ch for ch in content if ch.isalpha()]
    if not letters:
        return False
    return all(ord(ch) < 128 for ch in letters)


def _has_money_signal(text: str) -> bool:
    content = str(text or "").strip().lower()
    if not content:
        return False
    return bool(re.search(r"\b\d+(?:[.,]\d+)?\s*(?:tỷ|ty|tỉ|ti|triệu|trieu)\b", content))


def _has_personal_budget_phrase(text: str) -> bool:
    content = str(text or "").strip().lower()
    if not content:
        return False
    patterns = [
        r"\bvới\s+\d+(?:[.,]\d+)?\s*(?:tỷ|ty|tỉ|ti|triệu|trieu)\b",
        r"\bngân\s*sách\s+\d+(?:[.,]\d+)?\s*(?:tỷ|ty|tỉ|ti|triệu|trieu)\b",
        r"\btầm\s+\d+(?:[.,]\d+)?\s*(?:tỷ|ty|tỉ|ti|triệu|trieu)\b",
    ]
    return any(re.search(pattern, content) for pattern in patterns)


def _user_context_has_budget_signal(message: str, recent_history: list[HistoryTurn]) -> bool:
    if _has_money_signal(message):
        return True
    for turn in recent_history[-6:]:
        if str(turn.role).strip().lower() != "user":
            continue
        if _has_money_signal(turn.message):
            return True
    return False


def _build_grounded_snapshot(grounded_result: dict[str, Any] | None, settings: Settings) -> dict[str, Any]:
    if not grounded_result:
        return {}
    project_cards_raw = grounded_result.get("project_cards", []) or []
    trait_tags_raw = grounded_result.get("trait_tags", []) or []
    proximity_facts_raw = grounded_result.get("proximity_facts", []) or []
    evidence_chunks_raw = grounded_result.get("evidence_chunks", []) or []
    snapshot_cards: list[dict[str, Any]] = []
    for card in project_cards_raw[: settings.grounded_card_limit]:
        snapshot_cards.append(
            {
                "project_id": card.get("project_id"),
                "summary": card.get("summary"),
                "strengths": (card.get("strengths") or [])[:3],
                "tradeoffs": (card.get("tradeoffs") or [])[:2],
                "key_pois": (card.get("key_pois") or [])[:3],
            }
        )
    snapshot_traits = [
        {"tag": item.get("tag"), "reason": item.get("reason"), "weight": item.get("weight")}
        for item in trait_tags_raw[: settings.grounded_trait_limit]
    ]
    snapshot_proximity = [
        {
            "project_id": item.get("project_id"),
            "poi_type": item.get("poi_type"),
            "poi_name": item.get("poi_name"),
            "proximity_text": item.get("proximity_text"),
        }
        for item in proximity_facts_raw[: settings.grounded_proximity_limit]
    ]
    snapshot_evidence = [
        {"text": item.get("text"), "source": item.get("source"), "topic": item.get("topic")}
        for item in evidence_chunks_raw[: settings.grounded_evidence_limit]
    ]
    return {
        "single_project_mode": _is_single_project_mode(
            project_cards=project_cards_raw,
            proximity_facts=proximity_facts_raw,
        ),
        "used_projects": grounded_result.get("used_projects", []),
        "project_cards": snapshot_cards,
        "trait_tags": snapshot_traits,
        "proximity_facts": snapshot_proximity,
        "evidence_chunks": snapshot_evidence,
        "low_confidence": bool(grounded_result.get("low_confidence", False)),
        "confidence": grounded_result.get("confidence"),
    }


def _build_reply_synthesis_prompt(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    final_route: str,
    query_type: str,
    decision_reason: str,
    sales_state: str,
    engagement_state: str,
    conversation_goal: str,
    next_best_action: str,
    response_mode: str,
    ask_policy: str,
    focus: str,
    question_focus: str,
    grounded_result: dict[str, Any] | None,
    settings: Settings,
) -> str:
    history = [
        {"role": turn.role, "message": turn.message}
        for turn in recent_history[-settings.synthesis_history_turns :]
    ]
    state_payload = {
        "need_summary": lead_state.need.summary,
        "need_topics": [
            {"label": t.label, "weight": t.weight}
            for t in lead_state.need.topics[: settings.synthesis_state_topic_limit]
        ],
        "painpoint_summary": lead_state.painpoint.summary,
        "painpoint_topics": [
            {"label": t.label, "weight": t.weight}
            for t in lead_state.painpoint.topics[: settings.synthesis_state_topic_limit]
        ],
        "name": lead_state.name,
        "phone_contact": lead_state.phone_contact,
        "sales_state": lead_state.sales_state,
        "engagement_state": lead_state.engagement_state,
        "engagement_confidence": lead_state.engagement_confidence,
        "lead_level": lead_state.lead_level,
        "last_conversation_goal": lead_state.last_conversation_goal,
        "next_best_action": lead_state.next_best_action,
        "contact_capture_status": lead_state.contact_capture_status,
    }
    grounded_snapshot = _build_grounded_snapshot(grounded_result=grounded_result, settings=settings)
    return (
        "Bạn là chuyên viên tư vấn bất động sản.\n"
        "Nhiệm vụ duy nhất: trò chuyện tư vấn và giới thiệu dự án cho khách hàng bằng ngôn ngữ đời thường.\n"
        "Collection hiện tại chỉ có 1 dự án chính: Noble Palace Tây Thăng Long.\n"
        "Hãy trả về DUY NHẤT 1 JSON object theo schema:\n"
        '{ "assistant_reply": "string" }\n'
        "Quy tắc bắt buộc:\n"
        "- Dùng tiếng Việt có dấu, rõ ràng, tự nhiên, không máy móc.\n"
        "- Không dùng cụm từ kỹ thuật như route/retrieval/metadata/payload/confidence/vector/schema.\n"
        "- Độ dài linh hoạt theo nhu cầu câu hỏi: mặc định ngắn gọn (thường 1-3 câu), chỉ dài hơn khi user cần chi tiết.\n"
        "- Nếu có dữ liệu phù hợp thì nêu nhận định cụ thể; nếu chưa đủ dữ liệu thì nói rõ còn thiếu gì.\n"
        "- Không tự suy diễn ngân sách cá nhân cụ thể (ví dụ 'với 5 tỷ...') nếu người dùng chưa nêu ngân sách.\n"
        "- Khi nhắc con số tài chính, chỉ dùng số đã có trong grounded_context hoặc user_message/recent_history.\n"
        "- Nếu người dùng cần phân tích quá sâu (tài chính chi tiết, pháp lý sâu, phương án căn cụ thể), đề xuất 1 buổi hẹn trực tiếp.\n"
        "- Không trả lời theo mẫu form/checklist; không hỏi dồn nhiều câu.\n"
        "- Không bịa thông tin ngoài dữ liệu cung cấp.\n"
        "- Nếu dữ liệu hiện tại chưa đủ chi tiết để kết luận sâu, nói ngắn gọn và đề xuất hẹn gặp trực tiếp.\n"
        "- Nếu grounded_context.single_project_mode=true: chỉ nói về Noble Palace Tây Thăng Long, không nói 'nhiều lựa chọn' hay 'nhiều dự án'.\n"
        "- Nếu grounded_context.single_project_mode=true: không hỏi khu vực/quận.\n"
        "- Không dùng câu hỏi nhị phân theo mẫu 'ở hay đầu tư'.\n"
        "Ràng buộc do orchestrator cung cấp:\n"
        "- Bạn KHÔNG tự quyết hỏi hay không, phải làm theo ask_policy.\n"
        "- Mỗi lượt ưu tiên reflect và insight; invite chỉ dùng khi phù hợp với response_mode và conversation_goal.\n"
        "- Nếu user vừa bộc lộ need/painpoint thì phải phản chiếu ngắn trước khi đưa insight.\n"
        "- Mỗi lượt phải mở ra 1 góc tư vấn mới, không lặp intro brochure qua nhiều lượt.\n"
        "- engagement_state=cold/warm thì không CTA mạnh.\n"
        "- engagement_state=interested/ready mới được mời bước tiếp theo rõ hơn.\n"
        "- response_mode=warm_welcome: tạo thiện cảm tự nhiên, không ép chốt.\n"
        "- response_mode=value_teaser: nêu 1-2 điểm hợp nổi bật để tạo hứng thú tìm hiểu tiếp.\n"
        "- response_mode=discover_need: tư vấn trước, sau đó mới làm rõ nhẹ nếu cần.\n"
        "- response_mode=consultive_recommendation: tư vấn như consultant, nêu logic vì sao phù hợp.\n"
        "- response_mode=grounded_recommendation: nêu điểm phù hợp dựa trên grounded_context, ngắn và chắc.\n"
        "- response_mode=handle_concern: giải tỏa băn khoăn nhẹ, không tranh cãi.\n"
        "- response_mode=soft_next_step: mời bước tiếp theo mềm, không gây áp lực.\n"
        "- response_mode=nurture_followup: nuôi lead, không gây áp lực.\n"
        "- response_mode=meeting_invite: tư vấn ngắn gọn và đề xuất buổi hẹn trực tiếp.\n"
        "- response_mode=contact_capture: khi khách đã muốn đi tiếp, xin thông tin liên hệ trực tiếp nhưng lịch sự.\n"
        "- Nếu thiếu cả name và phone_contact, có thể hỏi gọn cả hai trong cùng một câu.\n"
        "- Khi xin thông tin liên hệ, luôn nêu lý do rõ: để sắp xếp tư vấn sâu hơn, gửi tài liệu phù hợp hoặc đặt lịch hẹn.\n"
        "- response_mode=followup_confirm: xác nhận bước tiếp theo ngắn gọn, không hỏi lan man.\n"
        "- ask_policy=avoid_question: kết thúc KHÔNG có dấu hỏi.\n"
        "- ask_policy=allow_question: có thể không hỏi, hoặc hỏi tối đa 1 câu mở.\n"
        "- ask_policy=must_clarify: hỏi đúng 1 câu ngắn về question_focus sau khi đã có nhận định.\n"
        "- Nếu có câu hỏi: chỉ 1 câu, không hỏi form, không lặp mẫu câu hỏi gần đây.\n\n"
        f"user_message={json.dumps(message, ensure_ascii=False)}\n"
        f"final_route={json.dumps(final_route, ensure_ascii=False)}\n"
        f"query_type={json.dumps(query_type, ensure_ascii=False)}\n"
        f"decision_reason={json.dumps(decision_reason, ensure_ascii=False)}\n"
        f"sales_state={json.dumps(_normalize_sales_state(sales_state), ensure_ascii=False)}\n"
        f"engagement_state={json.dumps(_normalize_engagement_state(engagement_state), ensure_ascii=False)}\n"
        f"conversation_goal={json.dumps(_normalize_conversation_goal(conversation_goal), ensure_ascii=False)}\n"
        f"next_best_action={json.dumps(_normalize_next_best_action(next_best_action), ensure_ascii=False)}\n"
        f"focus={json.dumps(focus, ensure_ascii=False)}\n"
        f"response_mode={json.dumps(_normalize_response_mode(response_mode), ensure_ascii=False)}\n"
        f"ask_policy={json.dumps(_normalize_ask_policy(ask_policy), ensure_ascii=False)}\n"
        f"question_focus={json.dumps(question_focus, ensure_ascii=False)}\n"
        f"lead_state={json.dumps(state_payload, ensure_ascii=False)}\n"
        f"recent_history={json.dumps(history, ensure_ascii=False)}\n"
        f"grounded_context={json.dumps(grounded_snapshot, ensure_ascii=False)}\n"
    )


def _call_model_generate(
    *,
    api_format: str,
    api_url: str,
    api_key: str,
    api_key_header: str,
    model: str,
    timeout_sec: float,
    keep_alive: str,
    prompt: str,
    temperature: float,
    response_format: str | None = "json",
) -> Any:
    api_format = str(api_format or "ollama").strip().lower()
    model_name = str(model or "").strip().lower()
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": _MODEL_HTTP_USER_AGENT,
    }
    if api_key:
        header_name = api_key_header or "Authorization"
        if header_name.lower() == "authorization":
            headers[header_name] = f"Bearer {api_key}"
        else:
            headers[header_name] = api_key

    if api_format == "openai":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        # Kimi k2.5 supports thinking/non-thinking modes; disable thinking for faster routing/synthesis turns.
        if model_name.startswith("kimi-k2.5"):
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["temperature"] = temperature
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if response_format:
            payload["format"] = response_format
        if keep_alive:
            payload["keep_alive"] = keep_alive

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("model response is not a JSON object")
    if api_format == "openai":
        choices = parsed.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("openai response missing choices")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
            return "".join(parts)
        return ""
    return parsed.get("response", "")


def _normalize_reply_to_accented_vietnamese(reply: str, settings: Settings) -> str:
    _ = settings
    return (reply or "").strip()


def _reply_needs_retry(reply: str, single_project_mode: bool, ask_policy: str) -> bool:
    cleaned = (reply or "").strip()
    if not cleaned:
        return True

    normalized_ask_policy = _normalize_ask_policy(ask_policy)
    question_marks = cleaned.count("?")

    if normalized_ask_policy == "avoid_question" and question_marks > 0:
        return True
    if normalized_ask_policy == "must_clarify" and question_marks == 0:
        return True
    if question_marks > 1:
        return True

    lowered = cleaned.lower()
    technical_terms = ["route", "retrieval", "metadata", "payload", "confidence", "schema", "vector"]
    if any(term in lowered for term in technical_terms):
        return True

    if single_project_mode and re.search(r"\bnhiều\s+(lựa chọn|dự án|căn hộ)\b", lowered):
        return True
    if single_project_mode and re.search(r"\bkhu\s*vực\b|\bquận\b", lowered):
        return True

    return False


def _rewrite_reply_by_policy(reply: str, single_project_mode: bool, ask_policy: str, settings: Settings) -> str:
    _ = single_project_mode
    _ = ask_policy
    _ = settings
    return (reply or "").strip()


def _sanitize_reply_for_policy(
    reply: str,
    ask_policy: str,
    single_project_mode: bool,
    question_focus: str,
) -> str:
    cleaned = re.sub(r"\s+", " ", (reply or "").strip())
    if not cleaned:
        return ""

    if single_project_mode:
        cleaned = re.sub(r"\bnhiều\s+(lựa chọn|dự án|căn hộ)\b", "dự án này", cleaned, flags=re.IGNORECASE)

    normalized_ask_policy = _normalize_ask_policy(ask_policy)
    if normalized_ask_policy == "avoid_question":
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", cleaned) if item.strip()]
        non_questions = [item for item in sentences if "?" not in item]
        if non_questions:
            cleaned = " ".join(non_questions).strip()
        cleaned = cleaned.replace("?", ".")
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned).strip()
        if cleaned and cleaned[-1] not in ".!":
            cleaned += "."
        return cleaned

    if normalized_ask_policy == "must_clarify" and "?" not in cleaned:
        cleaned = (
            f"{cleaned} Bạn có thể chia sẻ thêm về {question_focus} để mình tư vấn sát hơn không?"
            if cleaned
            else f"Bạn có thể chia sẻ thêm về {question_focus} để mình tư vấn sát hơn không?"
        )

    if cleaned.count("?") > 1:
        first_q = cleaned.find("?")
        cleaned = cleaned[: first_q + 1].strip()

    return cleaned


def _extract_assistant_reply(response_payload: Any) -> str:
    if isinstance(response_payload, dict):
        return str(response_payload.get("assistant_reply", "")).strip()
    parsed_obj = _extract_json_object(str(response_payload))
    return str(parsed_obj.get("assistant_reply", "")).strip()


def _build_reply_repair_prompt(
    original_prompt: str,
    invalid_reply: str,
    invalid_reason: str,
) -> str:
    return (
        "Ban vua tao assistant_reply chua dat yeu cau cho tro ly tu van bat dong san.\n"
        "Hay viet lai va CHI tra ve 1 JSON object hop le theo schema {\"assistant_reply\":\"...\"}.\n"
        "Khong duoc giai thich them, khong duoc them markdown, khong duoc bo trong assistant_reply.\n"
        f"invalid_reason={json.dumps(invalid_reason, ensure_ascii=False)}\n"
        f"invalid_reply={json.dumps(invalid_reply, ensure_ascii=False)}\n"
        f"original_prompt={json.dumps(original_prompt, ensure_ascii=False)}\n"
    )


def synthesize_assistant_reply(
    message: str,
    recent_history: list[HistoryTurn],
    lead_state: LeadState,
    analysis: TurnAnalysis,
    final_route: str,
    decision_reason: str,
    reply_plan: ReplyPlan,
    grounded_result: dict[str, Any] | None,
    settings: Settings,
) -> str:
    user_has_budget_context = _user_context_has_budget_signal(message=message, recent_history=recent_history)
    primary_prompt = _build_reply_synthesis_prompt(
        message=message,
        lead_state=lead_state,
        recent_history=recent_history,
        final_route=final_route,
        query_type=analysis.query_type,
        decision_reason=decision_reason,
        sales_state=lead_state.sales_state,
        engagement_state=lead_state.engagement_state,
        conversation_goal=reply_plan.conversation_goal,
        next_best_action=reply_plan.next_best_action,
        response_mode=reply_plan.response_mode,
        ask_policy=reply_plan.ask_policy,
        focus=reply_plan.focus,
        question_focus=reply_plan.question_focus,
        grounded_result=grounded_result,
        settings=settings,
    )
    single_project_mode = False
    if grounded_result:
        single_project_mode = _is_single_project_mode(
            project_cards=grounded_result.get("project_cards", []) or [],
            proximity_facts=grounded_result.get("proximity_facts", []) or [],
        )
    attempt_prompt = primary_prompt
    last_failure_reason = "reply_synthesis_not_attempted"
    for attempt_index in range(2):
        try:
            response_payload = _call_model_generate(
                api_format=settings.synthesis_api_format,
                api_url=settings.synthesis_api_url,
                api_key=settings.synthesis_api_key,
                api_key_header=settings.synthesis_api_key_header,
                model=settings.synthesis_model,
                timeout_sec=settings.synthesis_timeout_sec,
                keep_alive=settings.synthesis_keep_alive,
                prompt=attempt_prompt,
                temperature=max(0.0, min(1.0, settings.synthesis_temperature)),
                response_format="json",
            )
            reply = _sanitize_reply_for_policy(
                reply=_extract_assistant_reply(response_payload),
                ask_policy=reply_plan.ask_policy,
                single_project_mode=single_project_mode,
                question_focus=reply_plan.question_focus,
            )
            if _has_personal_budget_phrase(reply) and not user_has_budget_context:
                log.info("reply synthesis rejected unsupported personal budget phrase")
                last_failure_reason = "unsupported_personal_budget_phrase"
            elif _reply_needs_retry(
                reply=reply,
                single_project_mode=single_project_mode,
                ask_policy=reply_plan.ask_policy,
            ):
                last_failure_reason = "reply_failed_policy_or_quality_gate"
            elif reply:
                return _compact_text(reply, max_words=_SYNTHESIS_REPLY_MAX_WORDS)
            else:
                last_failure_reason = "empty_reply_after_sanitize"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:240]
            log.warning("reply synthesis HTTP %s: %s", exc.code, detail)
            raise RuntimeError(f"reply synthesis HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            log.warning("reply synthesis unreachable: %s", exc.reason)
            raise RuntimeError(f"reply synthesis unreachable: {exc.reason}") from exc
        except socket.timeout as exc:
            log.warning("reply synthesis timeout after %ss", settings.synthesis_timeout_sec)
            raise RuntimeError(f"reply synthesis timeout after {settings.synthesis_timeout_sec}s") from exc
        except Exception as exc:
            log.warning("reply synthesis failed: %s", exc)
            raise RuntimeError(f"reply synthesis failed: {exc}") from exc

        if attempt_index == 0:
            attempt_prompt = _build_reply_repair_prompt(
                original_prompt=primary_prompt,
                invalid_reply=reply,
                invalid_reason=last_failure_reason,
            )

    raise RuntimeError(f"reply synthesis returned unusable output after retries: {last_failure_reason}")


def _build_decider_prompt(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    settings: Settings,
) -> str:
    compact_state = {
        "name": lead_state.name,
        "phone_contact": lead_state.phone_contact,
        "need": {
            "summary": lead_state.need.summary,
            "topics": [{"label": t.label, "weight": t.weight} for t in lead_state.need.topics[:6]],
            "evidence": lead_state.need.evidence[:5],
        },
        "painpoint": {
            "summary": lead_state.painpoint.summary,
            "topics": [{"label": t.label, "weight": t.weight} for t in lead_state.painpoint.topics[:6]],
            "evidence": lead_state.painpoint.evidence[:5],
        },
        "sales_state": lead_state.sales_state,
        "engagement_state": lead_state.engagement_state,
        "engagement_confidence": lead_state.engagement_confidence,
        "lead_level": lead_state.lead_level,
        "last_conversation_goal": lead_state.last_conversation_goal,
        "next_best_action": lead_state.next_best_action,
    }
    compact_history = [
        {"role": turn.role, "message": turn.message}
        for turn in recent_history[-settings.decider_history_turns :]
    ]
    return (
        "Ban la bo phan decider route cho tro ly tu van bat dong san.\n"
        "Nhiem vu duy nhat cua agent: tro chuyen tu van va gioi thieu du an cho khach hang.\n"
        "Ngoai route, ban phai de xuat sales_state_after va conversation_goal cho luot hien tai.\n"
        "Khi nguoi dung can phan tich qua chi tiet (tai chinh, phap ly, phuong an can cu the), huong dan de xuat buoi hen gap truc tiep.\n"
        "Data scope hien tai: collection dang co 1 du an chinh la Noble Palace Tay Thang Long.\n"
        "Vi vay, consult_reply khong duoc dat cau hoi kieu form nhu 'quan nao/khu vuc nao' de bat user dien thong tin.\n"
        "Hay uu tien phan tich va goi y tren du lieu hien co truoc, sau do moi hoi 1 cau mo de mo rong trao doi.\n"
        "Ban phai route theo query_type, khong route theo kieu thieu slot.\n"
        "4 query_type bat buoc: advisory_strategy, project_matching, project_specific, clarification.\n"
        "retrieval_readiness: not_ready | soft_ready | ready.\n"
        "Quy tac route bat buoc:\n"
        "- advisory_strategy -> start_route=consult_discovery.\n"
        "- project_specific -> start_route=project_grounded, should_route_project=true.\n"
        "- project_matching -> uu tien project_grounded. Neu ban chon start_route=consult_discovery thi van phai should_route_project=true de runtime chain cung request.\n"
        "- clarification -> dua vao history/state; neu thuc chat la tiep noi cho project query thi can should_route_project=true.\n"
        "- KHONG duoc de should_route_project=false chi vi thieu budget/khu vuc/timeline.\n"
        "- KHONG duoc route theo hard keyword rules; phai suy luan tu message + state + recent_history.\n\n"
        "Few-shot huong dan:\n"
        "1) 'Mua de dau tu thi nen chon nhu the nao?' => query_type=advisory_strategy, start_route=consult_discovery, should_route_project=false.\n"
        "2) 'Co can ho nao gan benh vien khong?' => query_type=project_matching, start_route=project_grounded hoac consult_discovery, should_route_project=true.\n"
        "3) 'Du an A phap ly sao?' => query_type=project_specific, start_route=project_grounded, should_route_project=true.\n"
        "4) 'U minh thien ve an toan hon' => query_type=clarification, route theo history.\n\n"
        "Kiem tra tinh nhat quan truoc khi tra ve:\n"
        "- Neu query_type in {project_matching, project_specific} thi should_route_project bat buoc true.\n"
        "- Neu start_route=project_grounded thi should_route_project bat buoc true.\n"
        "- decision_reason phai ngan, ro, va giai thich duoc tai sao chon route.\n\n"
        "Quy tac consult_reply:\n"
        "- Giong consultant, khong giong form checklist.\n"
        "- Dung tieng Viet tu nhien, uu tien co dau, khong dung cum ky thuat.\n"
        "- consult_reply phai du dung nhu cau tra loi cuoi cho consult turn don gian.\n"
        "- Voi advisory_strategy hoac clarification nhe, consult_reply can usable ngay khong can rewrite.\n"
        "- Khong tu suy dien ngan sach ca nhan cu the (vi du 'voi 5 ty...') neu user chua neu ngan sach.\n"
        "- Voi consultive_recommendation, consult_reply phai usable ngay nhu cau tra loi cuoi cho consult-only turn.\n"
        "- Neu khong can grounding du an hoac khong can xin contact/hen gap, consult_reply phai du de tra thang cho user.\n"
        "- Tranh noi dung mo ho phu thuoc vao grounded_context hoac buoc reply_synthesis phia sau.\n"
        "- Bat buoc co it nhat 1 nhan dinh huu ich truoc.\n"
        "- Chi dat cau hoi khi thieu 1 thong tin quan trong; khong mac dinh ket thuc bang cau hoi.\n"
        "- Do dai consult_reply linh hoat theo message; uu tien ngan gon, khong ep so cau co dinh; neu hoi thi toi da 1 cau hoi.\n"
        "- Khong dat cau hoi dang thu thap form nhu 'ban o quan nao', 'ban muon khu vuc nao'.\n"
        "- Khong hoi lai thong tin user vua noi.\n"
        "- Neu can phan tich sau hon, uu tien de xuat buoi hen gap truc tiep thay vi co gang phan tich qua sau trong chat.\n"
        "- Neu query la project_matching, consult_reply chi la cau bridge ngan de chain sang project route, khong dong request o consult.\n"
        "- Khong lap lai brochure intro qua nhieu luot lien tiep.\n\n"
        "Quy tac trich xuat thong tin lien he:\n"
        "- extracted_name: ten khach hang neu message/history vua neu ro rang; neu khong chac chan thi null.\n"
        "- extracted_phone: so dien thoai neu user neu ro rang; neu khong chac chan thi null.\n"
        "- Khong duoc suy dien ten/so dien thoai neu user chua noi.\n\n"
        "Bat buoc tra ve 1 JSON object theo schema:\n"
        "{\n"
        '  "query_type": "advisory_strategy|project_matching|project_specific|clarification",\n'
        '  "retrieval_readiness": "not_ready|soft_ready|ready",\n'
        '  "start_route": "consult_discovery|project_grounded",\n'
        '  "route": "consult_discovery|project_grounded",\n'
        '  "engagement_state_after": "cold|warm|interested|ready",\n'
        '  "sales_state_after": "unknown|exploring|need_identified|qualified|interested|appointment_ready|nurture|handoff",\n'
        '  "conversation_goal": "build_trust|discover_need|surface_priority|show_fit|handle_concern|invite_next_step|nurture_lead|handoff_to_human|capture_contact|confirm_followup",\n'
        '  "decision_reason": "string",\n'
        '  "should_route_project": true|false,\n'
        '  "project_query_hint": "string|null",\n'
        '  "need_update": {\n'
        '    "summary_delta": "string",\n'
        '    "topics": [{"label":"string","weight":0.0}],\n'
        '    "evidence": ["string"]\n'
        "  },\n"
        '  "painpoint_update": {\n'
        '    "summary_delta": "string",\n'
        '    "topics": [{"label":"string","weight":0.0}],\n'
        '    "evidence": ["string"]\n'
        "  },\n"
        '  "consult_reply": "string",\n'
        '  "extracted_name": "string|null",\n'
        '  "extracted_phone": "string|null"\n'
        "}\n\n"
        f"message={json.dumps(message, ensure_ascii=False)}\n"
        f"lead_state={json.dumps(compact_state, ensure_ascii=False)}\n"
        f"recent_history={json.dumps(compact_history, ensure_ascii=False)}\n"
    )


def _analyze_turn_with_model(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    settings: Settings,
) -> TurnAnalysis:
    response_payload = _call_model_generate(
        api_format=settings.decider_api_format,
        api_url=settings.decider_api_url,
        api_key=settings.decider_api_key,
        api_key_header=settings.decider_api_key_header,
        model=settings.decider_model,
        timeout_sec=settings.decider_timeout_sec,
        keep_alive=settings.decider_keep_alive,
        prompt=_build_decider_prompt(
            message=message,
            lead_state=lead_state,
            recent_history=recent_history,
            settings=settings,
        ),
        temperature=settings.decider_temperature,
        response_format="json",
    )
    if isinstance(response_payload, dict):
        result_obj = response_payload
    else:
        result_obj = _extract_json_object(str(response_payload))

    query_type = _normalize_query_type(result_obj.get("query_type"))
    retrieval_readiness = _normalize_retrieval_readiness(result_obj.get("retrieval_readiness"))
    start_route = _normalize_route(result_obj.get("start_route", result_obj.get("route", "consult_discovery")))

    need_update = _coerce_delta(result_obj.get("need_update"), message)
    painpoint_update = _coerce_delta(result_obj.get("painpoint_update"), message)
    should_route_default = start_route == "project_grounded"
    should_route_project = bool(result_obj.get("should_route_project", should_route_default))
    if start_route == "project_grounded":
        should_route_project = True

    project_query_hint_raw = str(result_obj.get("project_query_hint", "")).strip()
    if project_query_hint_raw.lower() in {"none", "null", "n/a"}:
        project_query_hint_raw = ""
    project_query_hint = project_query_hint_raw or (message.strip() if should_route_project else None)
    routing_reason = str(result_obj.get("decision_reason", "")).strip() or "llm_decider"
    engagement_state_hint = _normalize_engagement_state(
        result_obj.get("engagement_state_after"),
        fallback=lead_state.engagement_state,
    )
    sales_state_hint = _normalize_sales_state(result_obj.get("sales_state_after"), fallback=lead_state.sales_state)
    conversation_goal_hint = _normalize_conversation_goal(result_obj.get("conversation_goal"), fallback="build_trust")
    extracted_name = _coerce_extracted_name(result_obj.get("extracted_name"))
    extracted_phone = _coerce_extracted_phone(result_obj.get("extracted_phone"))

    consult_reply = _compact_text(
        str(result_obj.get("consult_reply", "")).strip(),
        max_words=_DECIDER_CONSULT_REPLY_MAX_WORDS,
    )
    return TurnAnalysis(
        route=start_route,
        decision_reason=routing_reason,
        need_update=need_update,
        painpoint_update=painpoint_update,
        routing_signal=RoutingSignal(
            should_route_project=should_route_project,
            project_query_hint=project_query_hint,
            reason=routing_reason,
        ),
        consult_reply=consult_reply,
        query_type=query_type,
        retrieval_readiness=retrieval_readiness,
        route_source="llm_decider",
        engagement_state_after_hint=engagement_state_hint,
        sales_state_after_hint=sales_state_hint,
        conversation_goal_hint=conversation_goal_hint,
        extracted_name=extracted_name,
        extracted_phone=extracted_phone,
    )


def _analyze_turn_fallback(message: str, lead_state: LeadState, recent_history: list[HistoryTurn]) -> TurnAnalysis:
    _ = lead_state
    _ = recent_history
    need_update = NeedPainpointDelta(
        summary_delta="Decider unavailable trong turn nay.",
        topics=[],
        evidence=[message.strip()] if message.strip() else [],
    )
    painpoint_update = NeedPainpointDelta(
        summary_delta="Can du lieu decider on dinh de route theo query_type.",
        topics=[],
        evidence=[message.strip()] if message.strip() else [],
    )
    focus = _compact_text(message, max_words=16)
    consult_reply = (
        f"Mình đã ghi nhận nhu cầu bạn đang chia sẻ: {focus}. "
        "Mình sẽ tư vấn theo dự án hiện có và nếu bạn muốn phân tích sâu hơn thì mình đề xuất một buổi hẹn trực tiếp."
        if focus
        else (
            "Mình sẽ tư vấn ngắn gọn theo dự án hiện có. "
            "Nếu bạn cần phân tích sâu theo phương án cụ thể, mình đề xuất một buổi hẹn trực tiếp."
        )
    )
    reason = "fallback_no_decider"
    return TurnAnalysis(
        route="consult_discovery",
        decision_reason=reason,
        need_update=need_update,
        painpoint_update=painpoint_update,
        routing_signal=RoutingSignal(
            should_route_project=False,
            project_query_hint=None,
            reason=reason,
        ),
        consult_reply=consult_reply,
        query_type="clarification",
        retrieval_readiness="not_ready",
        route_source="fallback",
        engagement_state_after_hint=lead_state.engagement_state,
        sales_state_after_hint=lead_state.sales_state,
        conversation_goal_hint=lead_state.last_conversation_goal or "build_trust",
        extracted_name=_extract_name(message),
        extracted_phone=_extract_phone_contact(message),
    )


def _build_contact_capture_fastpath_analysis(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
) -> TurnAnalysis | None:
    _ = recent_history
    extracted_name = _extract_name(message)
    extracted_phone = _extract_phone_contact(message)
    if not extracted_name and not extracted_phone:
        return None

    if lead_state.contact_capture_status not in {"requested", "partial"} and lead_state.sales_state not in {
        "interested",
        "appointment_ready",
    }:
        return None

    merged_name = extracted_name or lead_state.name
    merged_phone = extracted_phone or lead_state.phone_contact
    has_complete_contact = bool(merged_name and merged_phone)

    consult_reply_parts: list[str] = []
    if extracted_name:
        consult_reply_parts.append(f"Em cảm ơn anh/chị {extracted_name}.")
    elif extracted_phone:
        consult_reply_parts.append("Em đã ghi nhận thông tin liên hệ của anh/chị.")
    if extracted_phone:
        consult_reply_parts.append(f"Em đã lưu số {extracted_phone}.")
    consult_reply_parts.append(
        "Em sẽ dùng thông tin này để sắp xếp tư vấn sâu hơn và chốt khung thời gian phù hợp."
        if has_complete_contact
        else "Em đã ghi nhận trước thông tin này để tiếp tục hỗ trợ anh/chị ở bước kế tiếp."
    )

    evidence = [message.strip()] if message.strip() else []
    if has_complete_contact:
        summary_delta = "Khach da cung cap du thong tin lien he de di tiep sang buoc tu van hoac hen lich."
        topics = [TopicWeight(label="da_cung_cap_du_thong_tin_lien_he", weight=0.95)]
        engagement_state_after_hint = "ready"
        sales_state_after_hint = "appointment_ready"
        conversation_goal_hint = "confirm_followup"
    else:
        summary_delta = "Khach da bat dau cung cap thong tin lien he de tiep tuc tu van."
        topics = [TopicWeight(label="da_cung_cap_mot_phan_thong_tin_lien_he", weight=0.88)]
        engagement_state_after_hint = "interested"
        sales_state_after_hint = "interested"
        conversation_goal_hint = "capture_contact"

    return TurnAnalysis(
        route="consult_discovery",
        decision_reason="deterministic_contact_capture_fastpath",
        need_update=NeedPainpointDelta(
            summary_delta=summary_delta,
            topics=topics,
            evidence=evidence,
        ),
        painpoint_update=NeedPainpointDelta(summary_delta="", topics=[], evidence=evidence),
        routing_signal=RoutingSignal(
            should_route_project=False,
            project_query_hint=None,
            reason="deterministic_contact_capture_fastpath",
        ),
        consult_reply=" ".join(consult_reply_parts),
        query_type="clarification",
        retrieval_readiness="not_ready",
        route_source="deterministic_fastpath",
        engagement_state_after_hint=engagement_state_after_hint,
        sales_state_after_hint=sales_state_after_hint,
        conversation_goal_hint=conversation_goal_hint,
        extracted_name=merged_name,
        extracted_phone=merged_phone,
    )


def analyze_turn(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    settings: Settings,
) -> TurnAnalysis:
    fastpath = _build_contact_capture_fastpath_analysis(
        message=message,
        lead_state=lead_state,
        recent_history=recent_history,
    )
    if fastpath is not None:
        return fastpath
    if not settings.decider_enabled:
        return _analyze_turn_fallback(message=message, lead_state=lead_state, recent_history=recent_history)
    try:
        return _analyze_turn_with_model(
            message=message,
            lead_state=lead_state,
            recent_history=recent_history,
            settings=settings,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:240]
        log.warning("decider HTTP %s: %s", exc.code, detail)
    except urllib.error.URLError as exc:
        log.warning("decider unreachable: %s", exc.reason)
    except socket.timeout:
        log.warning("decider timeout after %ss", settings.decider_timeout_sec)
    except Exception as exc:
        log.warning("decider failed: %s", exc)
    return _analyze_turn_fallback(message=message, lead_state=lead_state, recent_history=recent_history)


def classify_route(message: str, lead_state: LeadState) -> tuple[str, str]:
    analysis = _analyze_turn_fallback(message=message, lead_state=lead_state, recent_history=[])
    return analysis.route, analysis.decision_reason


def run_consult_discovery(message: str, lead_state: LeadState, analysis: TurnAnalysis) -> dict[str, Any]:
    _ = message
    _ = lead_state
    return {
        "assistant_reply": analysis.consult_reply,
        "need_update": analysis.need_update,
        "painpoint_update": analysis.painpoint_update,
        "routing_signal": analysis.routing_signal,
    }


def run_project_grounded(
    message: str,
    lead_state: LeadState,
    query_type: str,
    retrieval_readiness: str,
    top_k: int,
    retrieval_fetcher,
    need_update: NeedPainpointDelta,
    painpoint_update: NeedPainpointDelta,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    retrieval_intent = _build_retrieval_intent(
        message=message,
        lead_state=lead_state,
        query_type=query_type,
        retrieval_readiness=retrieval_readiness,
    )
    retrieval_raw = _call_project_grounded_fetcher(
        retrieval_fetcher,
        message,
        retrieval_intent,
        top_k,
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
    )
    project_cards = retrieval_raw.get("project_cards", []) or []
    trait_tags = retrieval_raw.get("trait_tags", []) or []
    proximity_facts = retrieval_raw.get("proximity_facts", []) or []
    evidence_chunks = retrieval_raw.get("evidence_chunks", []) or []
    low_confidence = bool(retrieval_raw.get("low_confidence", False))
    used_projects = [str(card.get("project_id")) for card in project_cards if card.get("project_id")]
    return {
        "used_projects": _dedupe_keep_order(used_projects, max_items=5),
        "project_cards": project_cards,
        "trait_tags": trait_tags,
        "proximity_facts": proximity_facts,
        "evidence_chunks": evidence_chunks,
        "confidence": retrieval_raw.get("confidence", 0.0),
        "low_confidence": low_confidence,
        "retrieval_intent": retrieval_intent,
        "need_update": need_update,
        "painpoint_update": painpoint_update,
    }


def create_app(
    project_grounded_fetcher=None,
    turn_analyzer: Callable[[str, LeadState, list[HistoryTurn]], TurnAnalysis] | None = None,
) -> FastAPI:
    app = FastAPI(title="Sales Orchestrator 2 Routes", version="1.0.0")
    settings = get_settings()

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    client = RetrievalClient(
        base_url=settings.retrieval_service_url,
        timeout_sec=settings.retrieval_timeout_sec,
    )
    if project_grounded_fetcher is None:
        project_grounded_fetcher = client.retrieve_project_grounded

    if turn_analyzer is None:

        def _default_turn_analyzer(message: str, lead_state: LeadState, history: list[HistoryTurn]) -> TurnAnalysis:
            return analyze_turn(
                message=message,
                lead_state=lead_state,
                recent_history=history,
                settings=settings,
            )

        turn_analyzer = _default_turn_analyzer

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "sales-orchestrator-2-routes"}

    @app.post("/sales/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        message = payload.message.strip()
        lead_state = payload.lead_state or LeadState()
        top_k = payload.top_k or settings.default_top_k
        langfuse_enabled = bool(settings.langfuse_enabled)
        session_id = _derive_observability_session_id(payload=payload, lead_state=lead_state)
        user_id = _derive_observability_user_id(lead_state=lead_state)
        trace_metadata = {
            "service": "sales-orchestrator",
            "endpoint": "/sales/query",
        }
        try:
            with start_observation(
                langfuse_enabled,
                name="orchestrator.sales.query",
                as_type="span",
                input={
                    "message": message[:500],
                    "top_k": top_k,
                    "force_route": payload.force_route,
                    "history_turns": len(payload.recent_history),
                },
                metadata=trace_metadata,
            ) as request_obs:
                trace_id = str(getattr(request_obs, "trace_id", "") or "")
                trace_id_for_header = trace_id if trace_id else None
                with propagate_context(
                    langfuse_enabled,
                    session_id=session_id,
                    user_id=user_id,
                    metadata=trace_metadata,
                ):
                    with start_observation(
                        langfuse_enabled,
                        name="orchestrator.analyze_turn",
                        as_type="generation",
                        model=settings.decider_model,
                        input={
                            "message": message[:500],
                            "lead_state_sales": lead_state.sales_state,
                            "history_turns": len(payload.recent_history),
                        },
                    ) as analyze_obs:
                        analysis = turn_analyzer(message, lead_state, payload.recent_history)
                        analyze_obs.update(
                            output={
                                "start_route": analysis.route,
                                "query_type": analysis.query_type,
                                "retrieval_readiness": analysis.retrieval_readiness,
                                "analysis_source": analysis.route_source,
                                "should_route_project": bool(analysis.routing_signal.should_route_project),
                                "route_source": analysis.route_source,
                                "extracted_name_from_decider": bool(analysis.extracted_name),
                                "extracted_phone_from_decider": bool(analysis.extracted_phone),
                            }
                        )

                    start_route = analysis.route
                    reason = analysis.decision_reason
                    route_source = analysis.route_source or "turn_analyzer"
                    extracted_name_fallback = _extract_name(message)
                    extracted_phone_fallback = _extract_phone_contact(message)
                    extracted_name = analysis.extracted_name or extracted_name_fallback
                    extracted_phone = analysis.extracted_phone or extracted_phone_fallback

                    if payload.force_route is not None:
                        start_route = payload.force_route
                        reason = f"force_route={payload.force_route}"
                        route_source = "force_route_debug"

                    consult_result: dict[str, Any] | None = None
                    grounded_result: dict[str, Any] | None = None
                    final_state: LeadState
                    final_route = start_route
                    chained_from_consult = False
                    active_need_update = analysis.need_update
                    active_painpoint_update = analysis.painpoint_update
                    routing_signal: RoutingSignal | None = None

                    if start_route == "consult_discovery":
                        with start_observation(
                            langfuse_enabled,
                            name="orchestrator.consult_discovery",
                            as_type="span",
                            input={"query_type": analysis.query_type},
                        ) as consult_obs:
                            consult_result = run_consult_discovery(message=message, lead_state=lead_state, analysis=analysis)
                            routing_signal = consult_result.get("routing_signal")
                            after_consult_state = merge_lead_state(
                                lead_state=lead_state,
                                need_update=consult_result["need_update"],
                                painpoint_update=consult_result["painpoint_update"],
                                extracted_name=extracted_name,
                                extracted_phone=extracted_phone,
                            )
                            active_need_update = consult_result["need_update"]
                            active_painpoint_update = consult_result["painpoint_update"]

                            should_chain_project = bool(routing_signal and routing_signal.should_route_project)
                            if payload.force_route == "consult_discovery":
                                should_chain_project = False
                            consult_obs.update(
                                output={
                                    "should_chain_project": should_chain_project,
                                    "project_query_hint": (
                                        str(routing_signal.project_query_hint)[:200] if routing_signal else None
                                    ),
                                }
                            )

                        if should_chain_project:
                            chained_from_consult = True
                            final_route = "project_grounded"
                            project_message = str(routing_signal.project_query_hint or message).strip() or message
                            try:
                                with start_observation(
                                    langfuse_enabled,
                                    name="orchestrator.project_grounded.chain",
                                    as_type="span",
                                    input={"query": project_message[:500], "top_k": top_k},
                                ) as grounded_obs:
                                    grounded_result = run_project_grounded(
                                        message=project_message,
                                        lead_state=after_consult_state,
                                        query_type=analysis.query_type,
                                        retrieval_readiness=analysis.retrieval_readiness,
                                        top_k=top_k,
                                        retrieval_fetcher=project_grounded_fetcher,
                                        need_update=active_need_update,
                                        painpoint_update=active_painpoint_update,
                                        trace_id=trace_id_for_header,
                                        session_id=session_id,
                                        user_id=user_id,
                                    )
                                    grounded_obs.update(
                                        output={
                                            "project_cards": len(grounded_result.get("project_cards", []) or []),
                                            "trait_tags": len(grounded_result.get("trait_tags", []) or []),
                                            "evidence_chunks": len(grounded_result.get("evidence_chunks", []) or []),
                                            "confidence": grounded_result.get("confidence"),
                                            "low_confidence": bool(grounded_result.get("low_confidence", False)),
                                        }
                                    )
                            except Exception as exc:
                                log.exception("project_grounded retrieval failed after consult chain: %s", exc)
                                raise HTTPException(status_code=502, detail=f"retrieval service error: {exc}") from exc
                            active_need_update = grounded_result["need_update"]
                            active_painpoint_update = grounded_result["painpoint_update"]
                            final_state = merge_lead_state(
                                lead_state=after_consult_state,
                                need_update=active_need_update,
                                painpoint_update=active_painpoint_update,
                                extracted_name=extracted_name,
                                extracted_phone=extracted_phone,
                            )
                        else:
                            final_state = after_consult_state
                    else:
                        final_route = "project_grounded"
                        try:
                            with start_observation(
                                langfuse_enabled,
                                name="orchestrator.project_grounded.direct",
                                as_type="span",
                                input={"query": message[:500], "top_k": top_k},
                            ) as grounded_obs:
                                grounded_result = run_project_grounded(
                                    message=message,
                                    lead_state=lead_state,
                                    query_type=analysis.query_type,
                                    retrieval_readiness=analysis.retrieval_readiness,
                                    top_k=top_k,
                                    retrieval_fetcher=project_grounded_fetcher,
                                    need_update=analysis.need_update,
                                    painpoint_update=analysis.painpoint_update,
                                    trace_id=trace_id_for_header,
                                    session_id=session_id,
                                    user_id=user_id,
                                )
                                grounded_obs.update(
                                    output={
                                        "project_cards": len(grounded_result.get("project_cards", []) or []),
                                        "trait_tags": len(grounded_result.get("trait_tags", []) or []),
                                        "evidence_chunks": len(grounded_result.get("evidence_chunks", []) or []),
                                        "confidence": grounded_result.get("confidence"),
                                        "low_confidence": bool(grounded_result.get("low_confidence", False)),
                                    }
                                )
                        except Exception as exc:
                            log.exception("project_grounded retrieval failed: %s", exc)
                            raise HTTPException(status_code=502, detail=f"retrieval service error: {exc}") from exc
                        active_need_update = grounded_result["need_update"]
                        active_painpoint_update = grounded_result["painpoint_update"]
                        final_state = merge_lead_state(
                            lead_state=lead_state,
                            need_update=active_need_update,
                            painpoint_update=active_painpoint_update,
                            extracted_name=extracted_name,
                            extracted_phone=extracted_phone,
                        )

                    with start_observation(
                        langfuse_enabled,
                        name="orchestrator.sales_state_engine",
                        as_type="span",
                        input={
                            "sales_state_before": lead_state.sales_state,
                            "engagement_state_before": lead_state.engagement_state,
                            "query_type": analysis.query_type,
                            "final_route": final_route,
                        },
                    ) as state_obs:
                        sales_state_before = _normalize_sales_state(lead_state.sales_state)
                        engagement_state_before = _normalize_engagement_state(lead_state.engagement_state)
                        engagement_state_after = update_engagement_state(
                            previous_state=engagement_state_before,
                            query_type=_normalize_query_type(analysis.query_type),
                            lead_state=final_state,
                            grounded_result=grounded_result,
                            recent_history=payload.recent_history,
                            routed_to_project=(final_route == "project_grounded"),
                            suggested_state=analysis.engagement_state_after_hint,
                        )
                        sales_state_after = update_sales_state(
                            previous_state=sales_state_before,
                            message=message,
                            query_type=_normalize_query_type(analysis.query_type),
                            lead_state=final_state,
                            grounded_result=grounded_result,
                            recent_history=payload.recent_history,
                            routed_to_project=(final_route == "project_grounded"),
                            suggested_state=analysis.sales_state_after_hint,
                        )
                        conversation_goal = select_conversation_goal(
                            sales_state=sales_state_after,
                            engagement_state=engagement_state_after,
                            query_type=_normalize_query_type(analysis.query_type),
                            grounded_result=grounded_result,
                            lead_state=final_state,
                            message=message,
                            suggested_goal=analysis.conversation_goal_hint,
                        )
                        next_best_action = _normalize_next_best_action(
                            select_next_best_action(
                                sales_state=sales_state_after,
                                engagement_state=engagement_state_after,
                                conversation_goal=conversation_goal,
                                grounded_result=grounded_result,
                                lead_state=final_state,
                            )
                        )
                        final_state = final_state.model_copy(
                            update={
                                "engagement_state": engagement_state_after,
                                "engagement_confidence": 0.75,
                                "sales_state": sales_state_after,
                                "lead_level": _derive_lead_level(sales_state_after),
                                "last_conversation_goal": conversation_goal,
                                "next_best_action": next_best_action,
                                "contact_capture_status": (
                                    "complete"
                                    if final_state.name and final_state.phone_contact
                                    else "partial"
                                    if final_state.name or final_state.phone_contact
                                    else "requested"
                                    if conversation_goal == "capture_contact"
                                    else final_state.contact_capture_status
                                ),
                            }
                        )
                        state_obs.update(
                            output={
                                "engagement_state_after": engagement_state_after,
                                "sales_state_after": sales_state_after,
                                "conversation_goal": conversation_goal,
                                "next_best_action": next_best_action,
                            }
                        )

                    with start_observation(
                        langfuse_enabled,
                        name="orchestrator.reply_planning",
                        as_type="span",
                        input={"sales_state": final_state.sales_state, "conversation_goal": conversation_goal},
                    ) as plan_obs:
                        reply_plan = build_reply_plan(
                            message=message,
                            recent_history=payload.recent_history,
                            lead_state=final_state,
                            analysis=analysis,
                            conversation_goal=conversation_goal,
                            next_best_action=next_best_action,
                            grounded_result=grounded_result,
                        )
                        trace = DecisionTrace(
                            query_type=_normalize_query_type(analysis.query_type),
                            retrieval_readiness=_normalize_retrieval_readiness(analysis.retrieval_readiness),
                            start_route=start_route,
                            final_route=final_route,
                            chained_from_consult=chained_from_consult,
                            route_source=route_source,
                            decision_reason=reason,
                            engagement_state_before=engagement_state_before,
                            engagement_state_after=engagement_state_after,
                            sales_state_before=sales_state_before,
                            sales_state_after=sales_state_after,
                            conversation_goal=conversation_goal,
                            response_mode=reply_plan.response_mode,
                            ask_policy=reply_plan.ask_policy,
                        )
                        plan_obs.update(
                            output={
                                "response_mode": reply_plan.response_mode,
                                "ask_policy": reply_plan.ask_policy,
                                "question_focus": reply_plan.question_focus,
                            }
                        )

                    used_fastpath_consult_reply = False
                    reply_source = "reply_synthesis"
                    fastpath_gate_reason = "forced_model_synthesis"

                    with start_observation(
                        langfuse_enabled,
                        name="orchestrator.reply_synthesis",
                        as_type="generation",
                        model=settings.synthesis_model,
                        input={
                            "final_route": final_route,
                            "response_mode": reply_plan.response_mode,
                            "ask_policy": reply_plan.ask_policy,
                            "fastpath_gate_reason": fastpath_gate_reason,
                        },
                    ) as reply_obs:
                        try:
                            assistant_reply = synthesize_assistant_reply(
                                message=message,
                                recent_history=payload.recent_history,
                                lead_state=final_state,
                                analysis=analysis,
                                final_route=final_route,
                                decision_reason=reason,
                                reply_plan=reply_plan,
                                grounded_result=grounded_result,
                                settings=settings,
                            )
                        except Exception as exc:
                            reply_obs.update(output={"error": str(exc)[:240]})
                            log.exception(
                                "reply synthesis error final_route=%s response_mode=%s ask_policy=%s reason=%s",
                                final_route,
                                reply_plan.response_mode,
                                reply_plan.ask_policy,
                                reason,
                            )
                            raise HTTPException(status_code=502, detail=f"reply synthesis error: {exc}") from exc
                        reply_obs.update(
                            output={
                                "assistant_reply_chars": len(assistant_reply),
                                "assistant_reply_preview": assistant_reply[:240],
                                "used_model_rewrite": False,
                                "used_model_accent_normalize": False,
                            }
                        )

                    if final_route == "project_grounded" and grounded_result is not None:
                        response = QueryResponse(
                            route="project_grounded",
                            assistant_reply=assistant_reply,
                            decision_reason=reason,
                            lead_state=final_state,
                            need_update=active_need_update,
                            painpoint_update=active_painpoint_update,
                            routing_signal=routing_signal,
                            project_grounded_payload={
                                "used_projects": grounded_result["used_projects"],
                                "project_cards": grounded_result["project_cards"],
                                "trait_tags": grounded_result["trait_tags"],
                                "proximity_facts": grounded_result["proximity_facts"],
                                "evidence_chunks": grounded_result["evidence_chunks"],
                                "retrieval_intent": grounded_result["retrieval_intent"],
                                "confidence": grounded_result["confidence"],
                                "low_confidence": grounded_result["low_confidence"],
                            },
                            decision_trace=trace,
                        )
                    else:
                        if consult_result is None:
                            consult_result = run_consult_discovery(message=message, lead_state=lead_state, analysis=analysis)
                        response = QueryResponse(
                            route="consult_discovery",
                            assistant_reply=assistant_reply,
                            decision_reason=reason,
                            lead_state=final_state,
                            need_update=active_need_update,
                            painpoint_update=active_painpoint_update,
                            routing_signal=consult_result.get("routing_signal"),
                            project_grounded_payload=None,
                            decision_trace=trace,
                        )

                    request_obs.update(
                        output={
                            "final_route": final_route,
                            "chained_from_consult": chained_from_consult,
                            "response_mode": reply_plan.response_mode,
                            "ask_policy": reply_plan.ask_policy,
                            "used_fastpath_consult_reply": used_fastpath_consult_reply,
                            "reply_source": reply_source,
                            "fastpath_gate_reason": fastpath_gate_reason,
                            "name_extract_source": (
                                "decider"
                                if analysis.extracted_name
                                else "regex_fallback"
                                if extracted_name_fallback
                                else "none"
                            ),
                            "phone_extract_source": (
                                "decider"
                                if analysis.extracted_phone
                                else "regex_fallback"
                                if extracted_phone_fallback
                                else "none"
                            ),
                            "low_confidence": bool(grounded_result.get("low_confidence", False))
                            if grounded_result
                            else None,
                        }
                    )

                    log.info(
                        (
                            "decision start_route=%s final_route=%s chained_from_consult=%s "
                            "query_type=%s retrieval_readiness=%s route_source=%s "
                            "engagement_before=%s engagement_after=%s "
                            "sales_state_before=%s sales_state_after=%s conversation_goal=%s next_best_action=%s "
                            "response_mode=%s ask_policy=%s reply_source=%s fastpath_gate=%s reason=%s"
                        ),
                        trace.start_route,
                        trace.final_route,
                        trace.chained_from_consult,
                        trace.query_type,
                        trace.retrieval_readiness,
                        trace.route_source,
                        trace.engagement_state_before,
                        trace.engagement_state_after,
                        trace.sales_state_before,
                        trace.sales_state_after,
                        trace.conversation_goal,
                        next_best_action,
                        reply_plan.response_mode,
                        reply_plan.ask_policy,
                        reply_source,
                        fastpath_gate_reason,
                        trace.decision_reason,
                    )
                    return response
        finally:
            if settings.langfuse_flush_at_request_end:
                flush_observability(langfuse_enabled)

    return app


app = create_app()
