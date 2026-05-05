from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
import socket
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from orchestrator_service.config import Settings
from orchestrator_service.config import get_settings
from orchestrator_service.observability import flush_observability
from orchestrator_service.observability import propagate_context
from orchestrator_service.observability import start_observation
from orchestrator_service.prompt_optimization import (
    init_optimization,
    get_synthesis_cache,
    get_timing_instrument,
    PromptCompressor,
)
from orchestrator_service.retrieval_client import RetrievalClient
from orchestrator_service.session_store import SessionStore
from orchestrator_service.schemas import (
    DecisionTrace,
    HistoryTurn,
    LeadState,
    LiveTalkingQueryRequest,
    LiveTalkingQueryResponse,
    LiveTalkingSessionStartRequest,
    LiveTalkingSessionStartResponse,
    LiveTalkingSessionState,
    LiveTalkingStopRequest,
    LiveTalkingStopResponse,
    NeedPainpointDelta,
    NeedPainpointState,
    QueryRequest,
    QueryResponse,
    RoutingSignal,
    TopicWeight,
    VisionContext,
    VisionQueryRequest,
)
from orchestrator_service.vision_client import VisionClient

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


def _warmup_decider_model(settings: Settings) -> None:
    if not settings.decider_enabled or not settings.model_warmup_decider_enabled:
        return
    _call_model_generate(
        api_format=settings.decider_api_format,
        api_url=settings.decider_api_url,
        api_key=settings.decider_api_key,
        api_key_header=settings.decider_api_key_header,
        model=settings.decider_model,
        timeout_sec=max(5.0, settings.decider_timeout_sec),
        keep_alive=settings.decider_keep_alive,
        prompt="Warmup ping. Reply briefly with OK.",
        temperature=0.0,
        response_format=None,
    )


def _warmup_synthesis_model(settings: Settings) -> None:
    if not settings.model_warmup_synthesis_enabled:
        return
    _call_model_generate(
        api_format=settings.synthesis_api_format,
        api_url=settings.synthesis_api_url,
        api_key=settings.synthesis_api_key,
        api_key_header=settings.synthesis_api_key_header,
        model=settings.synthesis_model,
        timeout_sec=max(5.0, settings.synthesis_timeout_sec),
        keep_alive=settings.synthesis_keep_alive,
        prompt="Warmup ping. Reply briefly with OK.",
        temperature=0.0,
        response_format=None,
        max_tokens=16,
        enable_stream=False,
    )


def _run_model_warmup_loop(settings: Settings, stop_event: threading.Event) -> None:
    interval_sec = max(10.0, float(settings.model_warmup_interval_sec))
    log.info(
        "Periodic model warmup started interval_sec=%s decider=%s synthesis=%s",
        interval_sec,
        bool(settings.model_warmup_decider_enabled and settings.decider_enabled),
        bool(settings.model_warmup_synthesis_enabled),
    )
    while not stop_event.is_set():
        t0 = time.time()
        try:
            _warmup_decider_model(settings)
            _warmup_synthesis_model(settings)
            elapsed_ms = (time.time() - t0) * 1000.0
            log.info("Periodic model warmup completed elapsed_ms=%.1f", elapsed_ms)
        except Exception as exc:
            log.warning("Periodic model warmup failed: %s", exc)
        if stop_event.wait(interval_sec):
            break


def _coerce_extracted_phone(raw: Any) -> str | None:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"none", "null", "n/a"}:
        return None
    return _extract_phone_contact(cleaned)


def _is_trivial_ack_message(message: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(message or "").strip().lower())
    if not lowered:
        return True
    normalized = re.sub(r"[!,.?]", "", lowered).strip()
    trivial_tokens = {
        "ok",
        "oke",
        "okay",
        "vay ha",
        "vậy hả",
        "cam on",
        "cảm ơn",
        "thanks",
        "thank you",
        "vang",
        "vâng",
        "da",
        "dạ",
        "roi",
        "rồi",
        "uh",
        "uhm",
        "um",
    }
    return normalized in trivial_tokens


def _derive_retrieval_readiness(
    query_type: str,
    start_route: str,
    should_route_project: bool,
) -> str:
    qtype = _normalize_query_type(query_type)
    route = _normalize_route(start_route)
    if not should_route_project:
        return "not_ready"
    if qtype == "project_specific":
        return "ready"
    if qtype == "project_matching":
        return "ready" if route == "project_grounded" else "soft_ready"
    if qtype == "clarification":
        return "soft_ready"
    return "not_ready"


def _build_lightweight_need_update(message: str, query_type: str) -> NeedPainpointDelta:
    cleaned = _compact_text(message, max_words=18)
    evidence = [message.strip()] if str(message or "").strip() else []
    qtype = _normalize_query_type(query_type)
    if not cleaned or _is_trivial_ack_message(cleaned):
        return NeedPainpointDelta(summary_delta="", topics=[], evidence=[])
    topic_label_map = {
        "advisory_strategy": "tu_van_chien_luoc",
        "project_matching": "tim_du_an_phu_hop",
        "project_specific": "lam_ro_du_an_cu_the",
        "clarification": "lam_ro_nhu_cau",
    }
    topic_weight_map = {
        "advisory_strategy": 0.68,
        "project_matching": 0.74,
        "project_specific": 0.78,
        "clarification": 0.58,
    }
    return NeedPainpointDelta(
        summary_delta=cleaned,
        topics=[
            TopicWeight(
                label=topic_label_map.get(qtype, "lam_ro_nhu_cau"),
                weight=topic_weight_map.get(qtype, 0.58),
            )
        ],
        evidence=evidence,
    )


def _build_lightweight_painpoint_update(message: str) -> NeedPainpointDelta:
    lowered = str(message or "").strip().lower()
    evidence = [message.strip()] if str(message or "").strip() else []
    if not lowered or _is_trivial_ack_message(lowered):
        return NeedPainpointDelta(summary_delta="", topics=[], evidence=[])
    pain_signals = (
        "lo",
        "so",
        "sợ",
        "ban khoan",
        "băn khoăn",
        "phap ly",
        "pháp lý",
        "ngan sach",
        "ngân sách",
        "gia",
        "giá",
        "rui ro",
        "rủi ro",
        "thanh khoan",
        "thanh khoản",
        "tien do",
        "tiến độ",
        "xa",
        "ket xe",
        "kẹt xe",
        "on ao",
        "ồn ào",
    )
    if not any(token in lowered for token in pain_signals):
        return NeedPainpointDelta(summary_delta="", topics=[], evidence=[])
    return NeedPainpointDelta(
        summary_delta=_compact_text(message, max_words=18),
        topics=[TopicWeight(label="painpoint_signal", weight=0.66)],
        evidence=evidence,
    )


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
        customer_profile=lead_state.customer_profile,
        engagement_state=lead_state.engagement_state,
        engagement_confidence=lead_state.engagement_confidence,
        sales_state=lead_state.sales_state,
        lead_level=lead_state.lead_level,
        last_conversation_goal=lead_state.last_conversation_goal,
        next_best_action=lead_state.next_best_action,
        contact_capture_status=lead_state.contact_capture_status,
    )


def _merge_customer_profile(lead_state: LeadState, vision_context: VisionContext | None) -> LeadState:
    if vision_context is None:
        return lead_state

    current = lead_state.customer_profile
    resolved_name = current.name
    if vision_context.recognized and vision_context.name:
        resolved_name = vision_context.name
    resolved_gender = vision_context.gender if vision_context.gender is not None else current.gender
    resolved_age = vision_context.age if vision_context.age is not None else current.age
    resolved_source = current.source
    if vision_context.source and (vision_context.source != "none" or not current.source):
        resolved_source = vision_context.source
    resolved_confidence = current.confidence
    if vision_context.confidence is not None:
        resolved_confidence = vision_context.confidence
    resolved_face_id = vision_context.face_id or current.face_id

    updated_profile = current.model_copy(
        update={
            "recognized": bool(current.recognized or vision_context.recognized),
            "name": resolved_name,
            "age": resolved_age,
            "gender": resolved_gender,
            "source": resolved_source,
            "confidence": resolved_confidence,
            "last_seen_at": _now_iso(),
            "greeted_by_name": current.greeted_by_name if resolved_name == current.name else False,
            "greeted_generic": current.greeted_generic if resolved_gender == current.gender else False,
            "face_id": resolved_face_id,
        }
    )
    return lead_state.model_copy(update={"customer_profile": updated_profile})


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
    if lead_state.customer_profile.name:
        normalized = re.sub(r"\s+", "-", lead_state.customer_profile.name.strip().lower())
        normalized = re.sub(r"[^a-z0-9_\-]", "", normalized)
        if normalized:
            return f"vision:{normalized}"[:128]
    return None


def _derive_observability_user_id(lead_state: LeadState) -> str | None:
    phone = str(lead_state.phone_contact or "").strip()
    if phone:
        return phone[:128]
    name = str(lead_state.name or "").strip()
    if name:
        return name[:128]
    profile_name = str(lead_state.customer_profile.name or "").strip()
    if profile_name:
        return profile_name[:128]
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

    # Common case: markdown fenced JSON.
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Robust fallback: try every balanced {...} candidate and return the first valid JSON object.
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break

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
    missing_name = not bool((lead_state.name or lead_state.customer_profile.name or "").strip())
    missing_phone = not bool((lead_state.phone_contact or "").strip())
    return missing_name, missing_phone


def _slugify_identifier(value: str, fallback: str = "guest") -> str:
    normalized = re.sub(r"\s+", "-", str(value or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9_\-]", "", normalized)
    return normalized or fallback


def _derive_customer_honorific(lead_state: LeadState) -> str | None:
    gender = lead_state.customer_profile.gender
    if gender == "female":
        return "chị"
    if gender == "male":
        return "anh"
    return None


def _build_customer_greeting_prefix(
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    settings: Settings,
) -> str | None:
    if not settings.vision_greeting_enabled:
        return None
    if any(str(turn.role).strip().lower() == "assistant" for turn in recent_history):
        return None

    profile = lead_state.customer_profile
    honorific = _derive_customer_honorific(lead_state)
    if profile.recognized and profile.name and not profile.greeted_by_name:
        if honorific:
            return f"Em chào {honorific} {profile.name} ạ."
        return f"Em chào {profile.name} ạ."
    if (not profile.recognized) and profile.gender and not profile.greeted_generic:
        if honorific:
            return f"Em chào {honorific} ạ."
    return None


def _build_livetalking_greeting(lead_state: LeadState) -> str:
    profile = lead_state.customer_profile
    honorific = _derive_customer_honorific(lead_state)
    if profile.recognized and profile.name:
        if honorific:
            return f"Em chào {honorific} {profile.name} ạ."
        return f"Em chào {profile.name} ạ."
    if honorific:
        return f"Em chào {honorific} ạ."
    return "Em chào anh/chị ạ."


def _reply_has_greeting_prefix(reply: str) -> bool:
    lowered = str(reply or "").strip().lower()
    return lowered.startswith("em chào") or lowered.startswith("xin chào")


def _prepend_customer_greeting(reply: str, prefix: str | None) -> str:
    cleaned_reply = str(reply or "").strip()
    if not prefix:
        return cleaned_reply
    if not cleaned_reply:
        return prefix
    if _reply_has_greeting_prefix(cleaned_reply):
        return cleaned_reply
    return f"{prefix} {cleaned_reply}".strip()


def _mark_customer_greeting_applied(lead_state: LeadState) -> LeadState:
    profile = lead_state.customer_profile
    if profile.recognized and profile.name:
        updated_profile = profile.model_copy(update={"greeted_by_name": True})
        return lead_state.model_copy(update={"customer_profile": updated_profile})
    if profile.gender:
        updated_profile = profile.model_copy(update={"greeted_generic": True})
        return lead_state.model_copy(update={"customer_profile": updated_profile})
    return lead_state


def _build_face_session_binding(
    vision_context: VisionContext | None,
    *,
    fallback_session_id: str | None = None,
) -> tuple[str, str]:
    if vision_context and vision_context.recognized and vision_context.name:
        return "known", _slugify_identifier(vision_context.name, fallback="known-customer")
    if vision_context and vision_context.face_id:
        return "guest", _slugify_identifier(vision_context.face_id, fallback="guest-face")
    if fallback_session_id:
        return "anonymous", _slugify_identifier(fallback_session_id, fallback="anonymous")
    return "anonymous", f"anonymous-{uuid4().hex[:12]}"


def _build_face_session_key(customer_kind: str, face_id: str) -> str:
    return f"{customer_kind}:{face_id}"


def _build_session_ttl_sec(customer_kind: str, settings: Settings) -> int:
    if customer_kind == "known":
        return max(1, settings.session_known_ttl_sec)
    return max(1, settings.session_guest_ttl_sec)


def _serialize_history(history: list[HistoryTurn]) -> list[dict[str, str]]:
    return [{"role": turn.role, "message": turn.message} for turn in history]


def _deserialize_history(raw: list[dict[str, Any]] | None) -> list[HistoryTurn]:
    out: list[HistoryTurn] = []
    for item in raw or []:
        try:
            out.append(HistoryTurn.model_validate(item))
        except Exception:
            continue
    return out


def _load_livetalking_session_record(
    record: dict[str, Any] | None,
) -> tuple[LeadState, list[HistoryTurn], VisionContext | None]:
    if not isinstance(record, dict):
        return LeadState(), [], None
    lead_state = LeadState.model_validate(record.get("lead_state") or {})
    recent_history = _deserialize_history(record.get("recent_history") or [])
    raw_vision = record.get("vision_context")
    vision_context = VisionContext.model_validate(raw_vision) if isinstance(raw_vision, dict) else None
    return lead_state, recent_history, vision_context


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


def _is_basic_consult_greeting(message: str) -> bool:
    lowered = (message or "").strip().lower()
    if not lowered:
        return False
    greeting_patterns = (
        r"^\s*(xin\s+)?ch(?:ao|ào)(?:\s+(anh|chị|chi|em|bạn|ban))?(?:\s+(ạ|a|nhe|nhé|nha|nhá|ơi|oi))*\s*[!,.?]*\s*$",
        r"^\s*hello(?:\s+(anh|chị|chi|em|bạn|ban))?(?:\s+(ạ|a|nhe|nhé|nha|nhá|ơi|oi))*\s*[!,.?]*\s*$",
        r"^\s*hi(?:\s+(anh|chị|chi|em|bạn|ban))?(?:\s+(ạ|a|nhe|nhé|nha|nhá|ơi|oi))*\s*[!,.?]*\s*$",
        r"^\s*alo+\s*[!,.?]*\s*$",
    )
    # Keep fastpath reserved for pure greeting turns so substantive requests fall back to LLM synthesis.
    short_turn = len(lowered) <= 40
    return short_turn and any(re.match(pattern, lowered) for pattern in greeting_patterns)


def _build_quick_intent_response(message: str, ask_policy: str) -> str | None:
    lowered = (message or "").strip().lower()
    if not _is_basic_consult_greeting(lowered):
        return None
    normalized_ask_policy = _normalize_ask_policy(ask_policy)
    if normalized_ask_policy == "avoid_question":
        return (
            "Chào anh/chị, em sẵn sàng tư vấn bất động sản theo nhu cầu thực tế của mình. "
            "Hiện em có thể hỗ trợ nhanh theo toàn bộ dữ liệu dự án đang có và đề xuất hướng phù hợp cho anh/chị."
        )
    return (
        "Chào anh/chị, em sẵn sàng tư vấn bất động sản theo nhu cầu thực tế của mình. "
        "Anh/chị đang ưu tiên nhu cầu ở thực hay đầu tư để em tư vấn sát hơn?"
    )


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
            project_name = _friendly_project_name(str(first.get("project_id", ""))) or "Dự án đang được nhắc tới"
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
        "customer_profile": {
            "recognized": lead_state.customer_profile.recognized,
            "name": lead_state.customer_profile.name,
            "age": lead_state.customer_profile.age,
            "gender": lead_state.customer_profile.gender,
            "source": lead_state.customer_profile.source,
            "confidence": lead_state.customer_profile.confidence,
            "greeted_by_name": lead_state.customer_profile.greeted_by_name,
            "greeted_generic": lead_state.customer_profile.greeted_generic,
            "face_id": lead_state.customer_profile.face_id,
        },
    }
    grounded_snapshot = _build_grounded_snapshot(grounded_result=grounded_result, settings=settings)
    
    # Apply compression based on settings
    is_aggressive = settings.synthesis_compression_mode == "aggressive"
    is_moderate = settings.synthesis_compression_mode == "moderate"
    
    if is_moderate or is_aggressive:
        # Use optimized history depth and apply compression
        history_turns_limit = settings.synthesis_optimized_history_turns if is_moderate else 1
        history = [
            {"role": turn.role, "message": turn.message}
            for turn in recent_history[-history_turns_limit:]
        ]
        # Compress state payload
        state_payload = PromptCompressor.compress_state_payload(state_payload, aggressive=is_aggressive)
        # Compress grounded context
        grounded_snapshot = PromptCompressor.compress_grounded_context(grounded_snapshot, aggressive=is_aggressive)
    
    output_constraint = (
        f"Giữ câu trả lời TÓM TẮT, dưới {settings.synthesis_output_max_tokens // 4} từ "
        f"(khoảng {settings.synthesis_output_max_tokens // 5}-{settings.synthesis_output_max_tokens // 4} từ).\n"
    )
    
    return (
        "Bạn là chuyên viên tư vấn bất động sản.\n"
        "Nhiệm vụ duy nhất: trò chuyện tư vấn và giới thiệu dự án cho khách hàng bằng ngôn ngữ đời thường.\n"
        "Collection hiện tại có thể có nhiều dự án trong Qdrant.\n"
        "Hãy trả về DUY NHẤT 1 JSON object theo schema:\n"
        '{ "assistant_reply": "string" }\n'
        "Quy tắc bắt buộc:\n"
        f"- {output_constraint}"
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
        "- Nếu grounded_context.single_project_mode=true: chỉ nói về dự án hiện diện trong grounded_context, không nói 'nhiều lựa chọn' hay 'nhiều dự án'.\n"
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
        "- Nếu có câu hỏi: chỉ 1 câu, không hỏi form, không lặp mẫu câu hỏi gần đây.\n"
        "- Nếu customer_profile.recognized=true và có customer_profile.name, có thể xưng hô tự nhiên với tên khách hàng khi chào hỏi.\n"
        "- Nếu customer_profile.recognized=false nhưng customer_profile.gender đã biết, có thể dùng anh/chị cho lời chào mở đầu.\n\n"
        "- Luôn ưu tiên tham chiếu customer_profile từ vision nếu có (name, gender, age).\n"
        "- Nếu customer_profile.gender đã biết: chọn đại từ xưng hô phù hợp và nhất quán trong toàn bộ câu trả lời.\n"
        "- Nếu customer_profile.age='trẻ': ưu tiên văn phong năng động, ngắn gọn, nhấn mạnh tính linh hoạt, tiềm năng tăng giá, tiện ích sống hiện đại.\n"
        "- Nếu customer_profile.age='trung niên': ưu tiên văn phong điềm tĩnh, rõ ràng, nhấn mạnh pháp lý, an toàn, vận hành ổn định và giá trị sử dụng lâu dài.\n"
        "- Không nêu trực tiếp kiểu 'vì anh/chị thuộc nhóm tuổi ...'; chỉ điều chỉnh cách tư vấn một cách tự nhiên.\n\n"
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
    max_tokens: int | None = None,
    enable_stream: bool = False,
) -> Any:
    api_format = str(api_format or "ollama").strip().lower()
    model_name = str(model or "").strip().lower()
    api_url = str(api_url or "").strip()

    # Ollama Cloud serves OpenAI-compatible endpoints on ollama.com.
    # If env is configured as `ollama` format against ollama.com root,
    # auto-upgrade to OpenAI format to avoid HTML/non-JSON responses.
    if api_format == "ollama" and "ollama.com" in api_url:
        api_format = "openai"
        if "/v1/" not in api_url:
            api_url = api_url.rstrip("/") + "/v1/chat/completions"
        elif not api_url.rstrip("/").endswith("/chat/completions"):
            api_url = api_url.rstrip("/") + "/chat/completions"
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
            "stream": bool(enable_stream),
        }
        # Kimi k2.5 supports thinking/non-thinking modes; disable thinking for faster routing/synthesis turns.
        if model_name.startswith("kimi-k2.5"):
            payload["thinking"] = {"type": "disabled"}
            # Add max_tokens constraint to optimize synthesis latency (Kimi responds faster with token limit)
            if max_tokens:
                payload["max_tokens"] = max_tokens
        else:
            payload["temperature"] = temperature
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": bool(enable_stream),
            "options": {"temperature": temperature},
        }
        if response_format:
            payload["format"] = response_format
            # Some reasoning-capable Ollama models return only `thinking` with empty
            # `response` for JSON tasks unless reasoning is explicitly disabled.
            payload["think"] = False
        if keep_alive:
            payload["keep_alive"] = keep_alive

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    if enable_stream:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if api_format == "openai":
                parts: list[str] = []
                while True:
                    line_bytes = resp.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except Exception:
                        continue
                    choices = event.get("choices", [])
                    if not isinstance(choices, list) or not choices:
                        continue
                    first = choices[0] if isinstance(choices[0], dict) else {}
                    delta = first.get("delta") if isinstance(first, dict) else {}
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        parts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text")
                                if isinstance(text, str) and text:
                                    parts.append(text)
                return "".join(parts)

            # Ollama streaming: newline-delimited JSON objects with `response` chunks.
            parts = []
            while True:
                line_bytes = resp.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                chunk = event.get("response")
                if isinstance(chunk, str) and chunk:
                    parts.append(chunk)
            return "".join(parts)

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
    response_text = parsed.get("response", "")
    if isinstance(response_text, str) and response_text.strip():
        return response_text
    thinking_text = parsed.get("thinking", "")
    if isinstance(thinking_text, str):
        return thinking_text
    return ""


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


def _build_synthesis_cache_key(
    *,
    message: str,
    final_route: str,
    analysis: TurnAnalysis,
    reply_plan: ReplyPlan,
    grounded_result: dict[str, Any] | None,
) -> str:
    used_projects = []
    if grounded_result:
        used_projects = list(grounded_result.get("used_projects", []) or [])[:2]
    payload = {
        "message": re.sub(r"\s+", " ", (message or "").strip().lower()),
        "final_route": _normalize_route(final_route),
        "query_type": _normalize_query_type(analysis.query_type),
        "response_mode": _normalize_response_mode(reply_plan.response_mode),
        "ask_policy": _normalize_ask_policy(reply_plan.ask_policy),
        "conversation_goal": _normalize_conversation_goal(reply_plan.conversation_goal),
        "next_best_action": _normalize_next_best_action(reply_plan.next_best_action),
        "projects": used_projects,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
    
    # Try to retrieve from cache (before retry loop)
    synthesis_cache = get_synthesis_cache()
    timing_inst = get_timing_instrument()
    cached_response = None
    cache_key = _build_synthesis_cache_key(
        message=message,
        final_route=final_route,
        analysis=analysis,
        reply_plan=reply_plan,
        grounded_result=grounded_result,
    )
    if synthesis_cache:
        cached_response = synthesis_cache.get(cache_key)
        if cached_response:
            log.info("synthesis cache hit (stable-key)")
            return _compact_text(cached_response, max_words=_SYNTHESIS_REPLY_MAX_WORDS)
    
    attempt_prompt = primary_prompt
    last_failure_reason = "reply_synthesis_not_attempted"
    for attempt_index in range(2):
        try:
            # Record synthesis timing
            if timing_inst:
                timing_inst.start("synthesis_call")
            
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
                max_tokens=settings.synthesis_output_max_tokens,
                enable_stream=settings.synthesis_enable_streaming,
            )
            
            if timing_inst:
                elapsed_ms = timing_inst.end("synthesis_call")
                log.debug(f"synthesis stage latency: {elapsed_ms:.1f}ms")
            
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
                # Cache the successful response
                if synthesis_cache and attempt_index == 0:
                    synthesis_cache.set(
                        cache_key,
                        reply,
                        metadata={"route": final_route, "query_type": analysis.query_type}
                    )
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
        "need_summary": lead_state.need.summary,
        "painpoint_summary": lead_state.painpoint.summary,
        "sales_state": lead_state.sales_state,
        "engagement_state": lead_state.engagement_state,
        "last_conversation_goal": lead_state.last_conversation_goal,
        "next_best_action": lead_state.next_best_action,
    }
    compact_history = [
        {"role": turn.role, "message": turn.message}
        for turn in recent_history[-settings.decider_history_turns :]
    ]
    return (
        "Ban la bo phan decider route cho tro ly tu van bat dong san.\n"
        "Nhiem vu duy nhat: phan loai query va quyet dinh co can route sang retrieval du an hay khong.\n"
        "Khong tra loi tu van. Khong cap nhat CRM state. Khong viet consult_reply.\n"
        "Data scope hien tai: collection co the co nhieu du an trong Qdrant, khong mac dinh 1 du an co dinh.\n"
        "4 query_type bat buoc: advisory_strategy, project_matching, project_specific, clarification.\n"
        "Quy tac route bat buoc:\n"
        "- advisory_strategy: hoi cach chon, chien luoc, so sanh tong quan -> start_route=consult_discovery, should_route_project=false.\n"
        "- project_matching: tim du an/can phu hop theo tieu chi -> should_route_project=true.\n"
        "- project_specific: hoi thong tin cu the ve du an, can, phap ly, gia, tien ich, gan POI -> start_route=project_grounded, should_route_project=true.\n"
        "- clarification: tiep noi ngan theo history/state; neu dang lam ro cho project query thi should_route_project=true.\n"
        "- Khong duoc de should_route_project=false chi vi thieu budget, khu vuc, timeline.\n"
        "- Neu query_type in {project_matching, project_specific} thi should_route_project bat buoc true.\n"
        "- Neu start_route=project_grounded thi should_route_project bat buoc true.\n"
        "- project_query_hint chi can khi should_route_project=true; viet thanh 1 truy van ngan gon, giu lai tieu chi quan trong. Neu khong can thi null.\n\n"
        "Bat buoc tra ve 1 JSON object theo schema:\n"
        "{\n"
        '  "query_type": "advisory_strategy|project_matching|project_specific|clarification",\n'
        '  "start_route": "consult_discovery|project_grounded",\n'
        '  "should_route_project": true|false,\n'
        '  "project_query_hint": "string|null"\n'
        "}\n\n"
        "QUY DINH DINH DANG CUNG:\n"
        "- Chi tra ve DUY NHAT JSON object.\n"
        "- Khong markdown, khong ```json, khong giai thich, khong text truoc/sau JSON.\n"
        "- project_query_hint: neu should_route_project=false thi bat buoc la null.\n\n"
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
        max_tokens=settings.decider_output_max_tokens,
    )
    if isinstance(response_payload, dict):
        result_obj = response_payload
    else:
        result_obj = _extract_json_object(str(response_payload))

    query_type = _normalize_query_type(result_obj.get("query_type"))
    start_route = _normalize_route(result_obj.get("start_route", result_obj.get("route", "consult_discovery")))
    should_route_default = start_route == "project_grounded"
    should_route_project = bool(result_obj.get("should_route_project", should_route_default))
    if start_route == "project_grounded":
        should_route_project = True
    retrieval_readiness = _derive_retrieval_readiness(
        query_type=query_type,
        start_route=start_route,
        should_route_project=should_route_project,
    )

    project_query_hint_raw = str(result_obj.get("project_query_hint", "")).strip()
    if project_query_hint_raw.lower() in {"none", "null", "n/a"}:
        project_query_hint_raw = ""
    project_query_hint = project_query_hint_raw or (message.strip() if should_route_project else None)
    routing_reason = str(result_obj.get("decision_reason", "")).strip() or f"llm_decider_minimal:{query_type}:{start_route}"
    need_update = (
        _coerce_delta(result_obj.get("need_update"), message)
        if "need_update" in result_obj
        else _build_lightweight_need_update(message=message, query_type=query_type)
    )
    painpoint_update = (
        _coerce_delta(result_obj.get("painpoint_update"), message)
        if "painpoint_update" in result_obj
        else _build_lightweight_painpoint_update(message=message)
    )
    engagement_state_hint = (
        _normalize_engagement_state(result_obj.get("engagement_state_after"), fallback=lead_state.engagement_state)
        if "engagement_state_after" in result_obj
        else None
    )
    sales_state_hint = (
        _normalize_sales_state(result_obj.get("sales_state_after"), fallback=lead_state.sales_state)
        if "sales_state_after" in result_obj
        else None
    )
    conversation_goal_hint = (
        _normalize_conversation_goal(result_obj.get("conversation_goal"), fallback="build_trust")
        if "conversation_goal" in result_obj
        else None
    )
    extracted_name = _coerce_extracted_name(result_obj.get("extracted_name"))
    extracted_phone = _coerce_extracted_phone(result_obj.get("extracted_phone"))

    consult_reply = ""
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


def _build_basic_consult_greeting_fastpath_analysis(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
) -> TurnAnalysis | None:
    _ = recent_history
    if not _is_basic_consult_greeting(message):
        return None

    evidence = [message.strip()] if message.strip() else []
    consult_reply = (
        "Chào anh/chị, em sẵn sàng tư vấn bất động sản cho mình. "
        "Em có thể tư vấn theo toàn bộ dữ liệu dự án hiện có và bám sát nhu cầu ở thực hoặc đầu tư của anh/chị."
    )
    return TurnAnalysis(
        route="consult_discovery",
        decision_reason="deterministic_greeting_fastpath",
        need_update=NeedPainpointDelta(
            summary_delta="Khach mo dau bang loi chao, chua neu nhu cau cu the.",
            topics=[TopicWeight(label="loi_chao_mo_dau", weight=0.72)],
            evidence=evidence,
        ),
        painpoint_update=NeedPainpointDelta(summary_delta="", topics=[], evidence=evidence),
        routing_signal=RoutingSignal(
            should_route_project=False,
            project_query_hint=None,
            reason="deterministic_greeting_fastpath",
        ),
        consult_reply=consult_reply,
        query_type="clarification",
        retrieval_readiness="not_ready",
        route_source="deterministic_fastpath",
        engagement_state_after_hint=_promote_engagement_state(lead_state.engagement_state, "warm"),
        sales_state_after_hint="exploring",
        conversation_goal_hint="discover_need",
        extracted_name=lead_state.name,
        extracted_phone=lead_state.phone_contact,
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
    greeting_fastpath = _build_basic_consult_greeting_fastpath_analysis(
        message=message,
        lead_state=lead_state,
        recent_history=recent_history,
    )
    if greeting_fastpath is not None:
        return greeting_fastpath
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
    vision_identify=None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    app = FastAPI(title="Sales Orchestrator 2 Routes", version="1.0.0")
    settings = get_settings()

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    
    # Initialize optimization components
    init_optimization(
        enable_cache=settings.synthesis_cache_enabled,
        cache_max_entries=settings.synthesis_cache_max_entries,
        cache_ttl_seconds=settings.synthesis_cache_ttl_seconds,
        enable_timing=settings.enable_timing_instrumentation,
    )
    log.info(
        f"Optimization initialized: cache={settings.synthesis_cache_enabled}, "
        f"compression={settings.synthesis_compression_mode}, "
        f"streaming={settings.synthesis_enable_streaming}"
    )

    app.state.model_warmup_stop_event = None
    app.state.model_warmup_thread = None

    @app.on_event("startup")
    def _startup_model_warmup() -> None:
        if not settings.model_warmup_enabled:
            log.info("Periodic model warmup disabled")
            return
        stop_event = threading.Event()
        worker = threading.Thread(
            target=_run_model_warmup_loop,
            args=(settings, stop_event),
            name="model-warmup-worker",
            daemon=True,
        )
        app.state.model_warmup_stop_event = stop_event
        app.state.model_warmup_thread = worker
        worker.start()

    @app.on_event("shutdown")
    def _shutdown_model_warmup() -> None:
        stop_event = getattr(app.state, "model_warmup_stop_event", None)
        worker = getattr(app.state, "model_warmup_thread", None)
        if stop_event is not None:
            stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    client = RetrievalClient(
        base_url=settings.retrieval_service_url,
        timeout_sec=settings.retrieval_timeout_sec,
    )
    vision_client = VisionClient(
        base_url=settings.vision_service_url,
        timeout_sec=settings.vision_timeout_sec,
    )
    if project_grounded_fetcher is None:
        project_grounded_fetcher = client.retrieve_project_grounded
    if vision_identify is None:
        if settings.vision_enabled:
            vision_identify = vision_client.identify
    if session_store is None:
        session_store = SessionStore(settings.redis_url, settings.redis_namespace)

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

    def _identify_vision_context(
        *,
        image_base64: str,
        image_filename: str | None,
        image_content_type: str | None,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> VisionContext | None:
        if not settings.vision_enabled or vision_identify is None:
            return None
        if not str(image_base64 or "").strip():
            return None
        try:
            payload = vision_identify(
                image_base64=image_base64,
                image_filename=image_filename,
                image_content_type=image_content_type,
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
            )
            return VisionContext.model_validate(payload)
        except Exception as exc:
            log.warning("vision identify failed: %s", exc)
            return VisionContext(
                recognized=False,
                confidence=0.0,
                source="error",
                face_count=0,
                reason=str(exc)[:200],
            )

    def _build_session_record(
        *,
        session_id: str,
        face_session_key: str,
        customer_kind: str,
        ttl_sec: int,
        lead_state: LeadState,
        recent_history: list[HistoryTurn],
        vision_context: VisionContext | None,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "face_session_key": face_session_key,
            "customer_kind": customer_kind,
            "ttl_sec": ttl_sec,
            "lead_state": lead_state.model_dump(mode="json"),
            "recent_history": _serialize_history(recent_history),
            "vision_context": vision_context.model_dump(mode="json") if vision_context is not None else None,
            "updated_at": _now_iso(),
        }

    def _resolve_livetalking_session(
        *,
        image_base64: str,
        image_filename: str | None,
        image_content_type: str | None,
        requested_session_id: str | None,
        trace_id: str | None = None,
        user_id: str | None = None,
    ) -> tuple[LiveTalkingSessionState, LeadState, list[HistoryTurn], VisionContext | None]:
        vision_context = _identify_vision_context(
            image_base64=image_base64,
            image_filename=image_filename,
            image_content_type=image_content_type,
            trace_id=trace_id,
            session_id=requested_session_id,
            user_id=user_id,
        )
        customer_kind, face_identifier = _build_face_session_binding(
            vision_context,
            fallback_session_id=requested_session_id,
        )
        face_session_key = _build_face_session_key(customer_kind, face_identifier)
        ttl_sec = _build_session_ttl_sec(customer_kind, settings)
        record = session_store.load_by_face_key(face_session_key)
        if record is None and customer_kind == "anonymous" and requested_session_id:
            record = session_store.load_by_session_id(requested_session_id)
        if record is not None:
            lead_state, recent_history, stored_vision_context = _load_livetalking_session_record(record)
            effective_vision_context = vision_context or stored_vision_context
            lead_state = _merge_customer_profile(lead_state, effective_vision_context)
            session = LiveTalkingSessionState(
                session_id=str(record.get("session_id", requested_session_id or "")) or requested_session_id or uuid4().hex,
                face_session_key=str(record.get("face_session_key", face_session_key)) or face_session_key,
                customer_kind=str(record.get("customer_kind", customer_kind)) or customer_kind,
                ttl_sec=max(1, int(record.get("ttl_sec", ttl_sec) or ttl_sec)),
                resumed=True,
                should_greet=False,
                greeting=None,
            )
            return session, lead_state, recent_history, effective_vision_context

        base_state = _merge_customer_profile(LeadState(), vision_context)
        session = LiveTalkingSessionState(
            session_id=str(requested_session_id or uuid4().hex),
            face_session_key=face_session_key,
            customer_kind=customer_kind,
            ttl_sec=ttl_sec,
            resumed=False,
            should_greet=settings.vision_greeting_enabled,
            greeting=_build_livetalking_greeting(base_state) if settings.vision_greeting_enabled else None,
        )
        return session, base_state, [], vision_context

    @app.post("/sales/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        message = payload.message.strip()
        lead_state = _merge_customer_profile(payload.lead_state or LeadState(), payload.vision_context)
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
                                    if bool((final_state.name or final_state.customer_profile.name or "").strip())
                                    and final_state.phone_contact
                                    else "partial"
                                    if bool((final_state.name or final_state.customer_profile.name or "").strip())
                                    or final_state.phone_contact
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
                    fastpath_gate_reason = (
                        "deterministic_fastpath_with_llm_synthesis"
                        if analysis.route_source == "deterministic_fastpath"
                        else "forced_model_synthesis"
                    )

                    quick_intent_reply: str | None = None
                    if (
                        settings.quick_intent_fast_response_enabled
                        and final_route == "consult_discovery"
                        and _normalize_query_type(analysis.query_type) == "clarification"
                    ):
                        quick_intent_reply = _build_quick_intent_response(
                            message=message,
                            ask_policy=reply_plan.ask_policy,
                        )

                    if quick_intent_reply:
                        assistant_reply = _compact_text(
                            quick_intent_reply,
                            max_words=max(24, settings.quick_intent_response_max_words),
                        )
                        reply_source = "quick_intent_fast_response"
                        fastpath_gate_reason = "quick_intent_template"
                    else:
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
                                # Graceful degradation: keep the API responsive when model
                                # synthesis is slow/unavailable by returning fallback consult text.
                                if consult_result is None:
                                    consult_result = run_consult_discovery(
                                        message=message,
                                        lead_state=lead_state,
                                        analysis=analysis,
                                    )
                                fallback_reply = (
                                    str(consult_result.get("assistant_reply", "")).strip()
                                    if consult_result
                                    else ""
                                ) or analysis.consult_reply
                                assistant_reply = _compact_text(
                                    fallback_reply
                                    or (
                                        "Em xin lỗi, hệ thống trả lời đang bận. "
                                        "Anh/chị cho em 1 tiêu chí ưu tiên để em tư vấn nhanh hơn ạ."
                                    ),
                                    max_words=max(24, settings.quick_intent_response_max_words),
                                )
                                reply_source = "reply_synthesis_fallback_consult"
                                fastpath_gate_reason = "synthesis_error_fallback"
                                used_fastpath_consult_reply = True
                            reply_obs.update(
                                output={
                                    "assistant_reply_chars": len(assistant_reply),
                                    "assistant_reply_preview": assistant_reply[:240],
                                    "used_model_rewrite": False,
                                    "used_model_accent_normalize": False,
                                }
                            )

                    greeting_prefix = _build_customer_greeting_prefix(
                        final_state,
                        payload.recent_history,
                        settings,
                    )
                    assistant_reply = _prepend_customer_greeting(assistant_reply, greeting_prefix)
                    if greeting_prefix and _reply_has_greeting_prefix(assistant_reply):
                        final_state = _mark_customer_greeting_applied(final_state)

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

    @app.post("/sales/query-with-vision", response_model=QueryResponse)
    def query_with_vision(payload: VisionQueryRequest) -> QueryResponse:
        vision_context = _identify_vision_context(
            image_base64=payload.image_base64,
            image_filename=payload.image_filename,
            image_content_type=payload.image_content_type,
            session_id=payload.session_id,
        )
        query_payload = QueryRequest(
            message=payload.message,
            lead_state=payload.lead_state,
            vision_context=vision_context,
            recent_history=payload.recent_history,
            top_k=payload.top_k,
            session_id=payload.session_id,
            force_route=payload.force_route,
        )
        return query(query_payload)

    @app.post("/integrations/livetalking/start", response_model=LiveTalkingSessionStartResponse)
    def livetalking_start(payload: LiveTalkingSessionStartRequest) -> LiveTalkingSessionStartResponse:
        session, stored_state, recent_history, vision_context = _resolve_livetalking_session(
            image_base64=payload.image_base64,
            image_filename=payload.image_filename,
            image_content_type=payload.image_content_type,
            requested_session_id=payload.session_id,
            user_id=_derive_observability_user_id(payload.lead_state or LeadState()),
        )
        lead_state = payload.lead_state or stored_state
        if recent_history:
            lead_state = stored_state
        lead_state = _merge_customer_profile(lead_state, vision_context)

        history_to_store = list(recent_history)
        if session.should_greet and session.greeting:
            history_to_store.append(HistoryTurn(role="assistant", message=session.greeting))
            lead_state = _mark_customer_greeting_applied(lead_state)

        session_store.save(
            _build_session_record(
                session_id=session.session_id,
                face_session_key=session.face_session_key,
                customer_kind=session.customer_kind,
                ttl_sec=session.ttl_sec,
                lead_state=lead_state,
                recent_history=history_to_store,
                vision_context=vision_context,
            ),
            session.ttl_sec,
        )
        return LiveTalkingSessionStartResponse(
            session=session,
            lead_state=lead_state,
            vision_context=vision_context,
        )

    @app.post("/integrations/livetalking/query", response_model=LiveTalkingQueryResponse)
    def livetalking_query(payload: LiveTalkingQueryRequest) -> LiveTalkingQueryResponse:
        session, stored_state, recent_history, vision_context = _resolve_livetalking_session(
            image_base64=payload.image_base64,
            image_filename=payload.image_filename,
            image_content_type=payload.image_content_type,
            requested_session_id=payload.session_id,
            user_id=_derive_observability_user_id(payload.lead_state or LeadState()),
        )
        lead_state = payload.lead_state or stored_state
        if recent_history:
            lead_state = stored_state
        lead_state = _merge_customer_profile(lead_state, vision_context)

        query_payload = QueryRequest(
            message=payload.message,
            lead_state=lead_state,
            vision_context=vision_context,
            recent_history=recent_history,
            top_k=payload.top_k,
            session_id=session.session_id,
            force_route=payload.force_route,
        )
        query_response = query(query_payload)

        updated_history = [
            *recent_history,
            HistoryTurn(role="user", message=payload.message.strip()),
            HistoryTurn(role="assistant", message=query_response.assistant_reply),
        ]
        session_store.save(
            _build_session_record(
                session_id=session.session_id,
                face_session_key=session.face_session_key,
                customer_kind=session.customer_kind,
                ttl_sec=session.ttl_sec,
                lead_state=query_response.lead_state,
                recent_history=updated_history,
                vision_context=vision_context,
            ),
            session.ttl_sec,
        )
        return LiveTalkingQueryResponse(
            **query_response.model_dump(),
            session=session,
            vision_context=vision_context,
        )

    @app.post("/integrations/livetalking/stop", response_model=LiveTalkingStopResponse)
    def livetalking_stop(payload: LiveTalkingStopRequest) -> LiveTalkingStopResponse:
        deleted = session_store.delete(
            session_id=payload.session_id,
            face_session_key=payload.face_session_key,
        )
        return LiveTalkingStopResponse(
            stopped=deleted is not None,
            session_id=deleted.session_key if deleted is not None else payload.session_id,
            face_session_key=deleted.face_key if deleted is not None else payload.face_session_key,
        )

    return app


app = create_app()
