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

_QUERY_TYPES = {"advisory_strategy", "project_matching", "project_specific", "clarification"}
_READINESS_VALUES = {"not_ready", "soft_ready", "ready"}
_PROJECT_QUERY_TYPES = {"project_matching", "project_specific"}
_POI_TYPE_VALUES = {"hospital", "school", "park", "mall"}
_RESPONSE_MODES = {"inform_only", "recommendation", "clarify_light", "meeting_invite"}
_ASK_POLICIES = {"avoid_question", "allow_question", "must_clarify"}


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


@dataclass(frozen=True)
class ReplyPlan:
    response_mode: str
    ask_policy: str
    focus: str
    question_focus: str


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
        r"(?:mình|toi|tôi|em|anh|chi|chị)\s+là\s+([a-zA-ZÀ-ỹ][a-zA-ZÀ-ỹ\s]{1,30})",
        r"tên\s+(?:mình|toi|tôi|em|anh|chi|chị)\s+là\s+([a-zA-ZÀ-ỹ][a-zA-ZÀ-ỹ\s]{1,30})",
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
    )


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


def _call_project_grounded_fetcher(fetcher, message: str, intent: dict[str, Any], top_k: int) -> dict[str, Any]:
    try:
        return fetcher(message, intent, top_k)
    except TypeError:
        try:
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


def _normalize_response_mode(raw: Any, fallback: str = "inform_only") -> str:
    value = str(raw or "").strip().lower()
    if value in _RESPONSE_MODES:
        return value
    return fallback if fallback in _RESPONSE_MODES else "inform_only"


def _normalize_ask_policy(raw: Any, fallback: str = "avoid_question") -> str:
    value = str(raw or "").strip().lower()
    if value in _ASK_POLICIES:
        return value
    return fallback if fallback in _ASK_POLICIES else "avoid_question"


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


def _select_question_focus(query_type: str, final_route: str) -> str:
    if query_type == "project_matching":
        return "mức ưu tiên quan tâm"
    if query_type == "advisory_strategy":
        return "mục tiêu sử dụng"
    if query_type == "project_specific":
        return "thời điểm mua"
    if final_route == "project_grounded":
        return "mức chấp nhận rủi ro"
    return "điểm ưu tiên chính"


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
    final_route: str,
    grounded_result: dict[str, Any] | None,
) -> ReplyPlan:
    has_grounded_result = bool(grounded_result and (grounded_result.get("project_cards", []) or []))
    low_confidence = bool(grounded_result.get("low_confidence", False)) if grounded_result else False

    if has_grounded_result and not low_confidence:
        response_mode = "recommendation"
    elif has_grounded_result and low_confidence:
        response_mode = "meeting_invite"
    elif analysis.query_type == "advisory_strategy":
        response_mode = "inform_only"
    elif final_route == "consult_discovery":
        response_mode = "clarify_light"
    else:
        response_mode = "inform_only"

    has_state_context = bool(
        lead_state.need.summary.strip()
        or lead_state.painpoint.summary.strip()
        or lead_state.need.topics
        or lead_state.painpoint.topics
    )
    if response_mode in {"recommendation", "inform_only", "meeting_invite"}:
        ask_policy = "avoid_question"
    else:
        ask_policy = "allow_question"
    if response_mode == "clarify_light" and not has_state_context:
        ask_policy = "must_clarify"
    if has_grounded_result and not low_confidence:
        ask_policy = "avoid_question"

    if _recent_assistant_question_count(recent_history=recent_history, take_last_assistant_turns=2) >= 1:
        ask_policy = "avoid_question"
    if _is_short_user_reply_after_assistant_question(message=message, recent_history=recent_history):
        ask_policy = "avoid_question"

    return ReplyPlan(
        response_mode=_normalize_response_mode(response_mode),
        ask_policy=_normalize_ask_policy(ask_policy),
        focus=_build_reply_focus(message=message, lead_state=lead_state, grounded_result=grounded_result),
        question_focus=_select_question_focus(query_type=analysis.query_type, final_route=final_route),
    )


def _looks_unaccented_vietnamese(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    letters = [ch for ch in content if ch.isalpha()]
    if not letters:
        return False
    return all(ord(ch) < 128 for ch in letters)


def _build_grounded_snapshot(grounded_result: dict[str, Any] | None) -> dict[str, Any]:
    if not grounded_result:
        return {}
    project_cards_raw = grounded_result.get("project_cards", []) or []
    trait_tags_raw = grounded_result.get("trait_tags", []) or []
    proximity_facts_raw = grounded_result.get("proximity_facts", []) or []
    evidence_chunks_raw = grounded_result.get("evidence_chunks", []) or []
    snapshot_cards: list[dict[str, Any]] = []
    for card in project_cards_raw[:2]:
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
        for item in trait_tags_raw[:5]
    ]
    snapshot_proximity = [
        {
            "project_id": item.get("project_id"),
            "poi_type": item.get("poi_type"),
            "poi_name": item.get("poi_name"),
            "proximity_text": item.get("proximity_text"),
        }
        for item in proximity_facts_raw[:5]
    ]
    snapshot_evidence = [
        {"text": item.get("text"), "source": item.get("source"), "topic": item.get("topic")}
        for item in evidence_chunks_raw[:4]
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
    response_mode: str,
    ask_policy: str,
    focus: str,
    question_focus: str,
    grounded_result: dict[str, Any] | None,
) -> str:
    history = [{"role": turn.role, "message": turn.message} for turn in recent_history[-6:]]
    state_payload = {
        "need_summary": lead_state.need.summary,
        "need_topics": [{"label": t.label, "weight": t.weight} for t in lead_state.need.topics[:6]],
        "painpoint_summary": lead_state.painpoint.summary,
        "painpoint_topics": [{"label": t.label, "weight": t.weight} for t in lead_state.painpoint.topics[:6]],
        "name": lead_state.name,
    }
    grounded_snapshot = _build_grounded_snapshot(grounded_result=grounded_result)
    return (
        "Bạn là chuyên viên tư vấn bất động sản.\n"
        "Nhiệm vụ duy nhất: trò chuyện tư vấn và giới thiệu dự án cho khách hàng bằng ngôn ngữ đời thường.\n"
        "Collection hiện tại chỉ có 1 dự án chính: Noble Palace Tây Thăng Long.\n"
        "Hãy trả về DUY NHẤT 1 JSON object theo schema:\n"
        '{ "assistant_reply": "string" }\n'
        "Quy tắc bắt buộc:\n"
        "- Dùng tiếng Việt có dấu, rõ ràng, tự nhiên, không máy móc.\n"
        "- Không dùng cụm từ kỹ thuật như route/retrieval/metadata/payload/confidence/vector/schema.\n"
        "- Phản hồi 2-4 câu ngắn, tối đa 75 từ.\n"
        "- Luôn nêu ít nhất 1 nhận định cụ thể bám dữ liệu đã có.\n"
        "- Nếu người dùng cần phân tích quá sâu (tài chính chi tiết, pháp lý sâu, phương án căn cụ thể), đề xuất 1 buổi hẹn trực tiếp.\n"
        "- Không trả lời theo mẫu form/checklist; không hỏi dồn nhiều câu.\n"
        "- Không bịa thông tin ngoài dữ liệu cung cấp.\n"
        "- Nếu dữ liệu hiện tại chưa đủ chi tiết để kết luận sâu, nói ngắn gọn và đề xuất hẹn gặp trực tiếp.\n"
        "- Nếu grounded_context.single_project_mode=true: chỉ nói về Noble Palace Tây Thăng Long, không nói 'nhiều lựa chọn' hay 'nhiều dự án'.\n"
        "- Nếu grounded_context.single_project_mode=true: không hỏi khu vực/quận.\n"
        "- Không dùng câu hỏi nhị phân theo mẫu 'ở hay đầu tư'.\n"
        "Ràng buộc do orchestrator cung cấp:\n"
        "- Bạn KHÔNG tự quyết hỏi hay không, phải làm theo ask_policy.\n"
        "- response_mode=inform_only: tập trung cung cấp nhận định ngắn gọn, không kéo hội thoại vòng lặp.\n"
        "- response_mode=recommendation: nêu điểm phù hợp, nhận định grounded và gợi ý hành động ngắn.\n"
        "- response_mode=clarify_light: tư vấn trước 1-2 nhận định rồi mới làm rõ nhẹ nếu cần.\n"
        "- response_mode=meeting_invite: tư vấn ngắn gọn và đề xuất buổi hẹn trực tiếp.\n"
        "- ask_policy=avoid_question: kết thúc KHÔNG có dấu hỏi.\n"
        "- ask_policy=allow_question: có thể không hỏi, hoặc hỏi tối đa 1 câu mở.\n"
        "- ask_policy=must_clarify: hỏi đúng 1 câu ngắn về question_focus sau khi đã có nhận định.\n"
        "- Nếu có câu hỏi: chỉ 1 câu, không hỏi form, không lặp mẫu câu hỏi gần đây.\n\n"
        f"user_message={json.dumps(message, ensure_ascii=False)}\n"
        f"final_route={json.dumps(final_route, ensure_ascii=False)}\n"
        f"query_type={json.dumps(query_type, ensure_ascii=False)}\n"
        f"decision_reason={json.dumps(decision_reason, ensure_ascii=False)}\n"
        f"focus={json.dumps(focus, ensure_ascii=False)}\n"
        f"response_mode={json.dumps(_normalize_response_mode(response_mode), ensure_ascii=False)}\n"
        f"ask_policy={json.dumps(_normalize_ask_policy(ask_policy), ensure_ascii=False)}\n"
        f"question_focus={json.dumps(question_focus, ensure_ascii=False)}\n"
        f"lead_state={json.dumps(state_payload, ensure_ascii=False)}\n"
        f"recent_history={json.dumps(history, ensure_ascii=False)}\n"
        f"grounded_context={json.dumps(grounded_snapshot, ensure_ascii=False)}\n"
    )


def _call_model_generate(
    settings: Settings,
    prompt: str,
    temperature: float,
    response_format: str | None = "json",
) -> Any:
    payload: dict[str, Any] = {
        "model": settings.decider_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if response_format:
        payload["format"] = response_format
    if settings.decider_keep_alive:
        payload["keep_alive"] = settings.decider_keep_alive
    req = urllib.request.Request(
        settings.decider_api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=settings.decider_timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("model response is not a JSON object")
    return parsed.get("response", "")


def _normalize_reply_to_accented_vietnamese(reply: str, settings: Settings) -> str:
    normalized = (reply or "").strip()
    if not normalized or not _looks_unaccented_vietnamese(normalized):
        return normalized
    prompt = (
        "Chuyển đoạn sau sang tiếng Việt có dấu, giữ nguyên ý, giữ văn phong tư vấn ngắn gọn, "
        "không thêm thông tin mới. Trả về duy nhất JSON: {\"assistant_reply\":\"...\"}.\n"
        f"text={json.dumps(normalized, ensure_ascii=False)}"
    )
    response_payload = _call_model_generate(
        settings=settings,
        prompt=prompt,
        temperature=max(0.0, min(1.0, settings.decider_temperature)),
        response_format="json",
    )
    if isinstance(response_payload, dict):
        rewritten = str(response_payload.get("assistant_reply", "")).strip()
    else:
        rewritten_obj = _extract_json_object(str(response_payload))
        rewritten = str(rewritten_obj.get("assistant_reply", "")).strip()
    return rewritten or normalized


def _reply_needs_retry(reply: str, single_project_mode: bool, ask_policy: str) -> bool:
    cleaned = (reply or "").strip()
    if not cleaned:
        return True
    if len(cleaned.split()) > 90:
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
    if re.search(r"\bở\b.{0,25}\bđầu tư\b|\bđầu tư\b.{0,25}\bở\b", lowered):
        return True
    if single_project_mode and re.search(r"\bnhiều\s+(lựa chọn|dự án|căn hộ)\b", lowered):
        return True
    if single_project_mode and re.search(r"\bkhu\s*vực\b|\bquận\b", lowered):
        return True
    return False


def _rewrite_reply_by_policy(reply: str, single_project_mode: bool, ask_policy: str, settings: Settings) -> str:
    mode_text = "true" if single_project_mode else "false"
    normalized_ask_policy = _normalize_ask_policy(ask_policy)
    prompt = (
        "Viết lại phản hồi sau để đúng các quy tắc:\n"
        "- 2-4 câu ngắn, tối đa 75 từ.\n"
        "- Tiếng Việt tự nhiên, không kỹ thuật.\n"
        "- Nếu cần phân tích sâu thì gợi ý hẹn gặp trực tiếp.\n"
        f"- single_project_mode={mode_text}; nếu true thì không nói 'nhiều lựa chọn' hay 'nhiều dự án'.\n"
        "- Nếu single_project_mode=true thì không hỏi khu vực/quận.\n"
        "- Không dùng cùng một khuôn câu hỏi lặp lại máy móc; đổi góc hỏi theo phần thông tin còn thiếu.\n"
        "- Không dùng câu hỏi nhị phân theo mẫu 'ở hay đầu tư'.\n"
        f"- ask_policy={normalized_ask_policy}: avoid_question=không hỏi; allow_question=tối đa 1 câu hỏi; must_clarify=đúng 1 câu hỏi.\n"
        "Trả về duy nhất JSON: {\"assistant_reply\":\"...\"}\n"
        f"reply={json.dumps(reply, ensure_ascii=False)}"
    )
    response_payload = _call_model_generate(
        settings=settings,
        prompt=prompt,
        temperature=max(0.0, min(1.0, settings.decider_temperature)),
        response_format="json",
    )
    if isinstance(response_payload, dict):
        rewritten = str(response_payload.get("assistant_reply", "")).strip()
    else:
        rewritten_obj = _extract_json_object(str(response_payload))
        rewritten = str(rewritten_obj.get("assistant_reply", "")).strip()
    return rewritten


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


def _build_reply_fallback(
    message: str,
    analysis: TurnAnalysis,
    grounded_result: dict[str, Any] | None,
) -> str:
    if grounded_result:
        cards = grounded_result.get("project_cards", []) or []
        if cards:
            first = cards[0]
            project_name = _friendly_project_name(str(first.get("project_id", "Noble Palace Tây Thăng Long")))
            summary = _compact_text(str(first.get("summary", "")).strip(), max_words=24)
            if summary:
                return (
                    f"Với thông tin bạn vừa chia sẻ, {project_name} đang là phương án phù hợp để mình tư vấn trước cho bạn. "
                    f"{summary} Nếu bạn muốn đi sâu theo phương án căn cụ thể, mình đề xuất một buổi hẹn trực tiếp để trao đổi đầy đủ hơn."
                )
            return (
                f"Với nhu cầu bạn vừa nêu, {project_name} là dự án mình có thể tư vấn phù hợp nhất lúc này. "
                "Nếu bạn muốn phân tích sâu hơn theo phương án cụ thể, mình đề xuất một buổi hẹn trực tiếp."
            )
    if analysis.consult_reply.strip():
        return analysis.consult_reply.strip()
    focus = _compact_text(message, max_words=18)
    if focus:
        return (
            f"Mình đã ghi nhận nhu cầu chính của bạn là: {focus}. "
            "Mình sẽ tư vấn ngắn gọn theo dự án hiện có, và nếu cần phân tích sâu hơn mình đề xuất một buổi hẹn trực tiếp."
        )
    return (
        "Mình sẽ tư vấn ngắn gọn theo thông tin dự án hiện có. "
        "Nếu bạn muốn phân tích sâu theo phương án cụ thể, mình đề xuất một buổi hẹn trực tiếp."
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
    prompt = _build_reply_synthesis_prompt(
        message=message,
        lead_state=lead_state,
        recent_history=recent_history,
        final_route=final_route,
        query_type=analysis.query_type,
        decision_reason=decision_reason,
        response_mode=reply_plan.response_mode,
        ask_policy=reply_plan.ask_policy,
        focus=reply_plan.focus,
        question_focus=reply_plan.question_focus,
        grounded_result=grounded_result,
    )
    single_project_mode = False
    if grounded_result:
        single_project_mode = _is_single_project_mode(
            project_cards=grounded_result.get("project_cards", []) or [],
            proximity_facts=grounded_result.get("proximity_facts", []) or [],
        )
    try:
        response_payload = _call_model_generate(
            settings=settings,
            prompt=prompt,
            temperature=max(0.0, min(1.0, settings.decider_temperature + 0.18)),
            response_format="json",
        )
        if isinstance(response_payload, dict):
            reply = str(response_payload.get("assistant_reply", "")).strip()
        else:
            parsed_obj = _extract_json_object(str(response_payload))
            reply = str(parsed_obj.get("assistant_reply", "")).strip()
        reply = _normalize_reply_to_accented_vietnamese(reply=reply, settings=settings)
        reply = _sanitize_reply_for_policy(
            reply=reply,
            ask_policy=reply_plan.ask_policy,
            single_project_mode=single_project_mode,
            question_focus=reply_plan.question_focus,
        )
        if _reply_needs_retry(
            reply=reply,
            single_project_mode=single_project_mode,
            ask_policy=reply_plan.ask_policy,
        ):
            rewritten = _rewrite_reply_by_policy(
                reply=reply,
                single_project_mode=single_project_mode,
                ask_policy=reply_plan.ask_policy,
                settings=settings,
            )
            if rewritten:
                reply = _normalize_reply_to_accented_vietnamese(reply=rewritten, settings=settings)
                reply = _sanitize_reply_for_policy(
                    reply=reply,
                    ask_policy=reply_plan.ask_policy,
                    single_project_mode=single_project_mode,
                    question_focus=reply_plan.question_focus,
                )
        if _reply_needs_retry(
            reply=reply,
            single_project_mode=single_project_mode,
            ask_policy=reply_plan.ask_policy,
        ):
            fallback_reply = _build_reply_fallback(message=message, analysis=analysis, grounded_result=grounded_result)
            reply = _sanitize_reply_for_policy(
                reply=fallback_reply,
                ask_policy=reply_plan.ask_policy,
                single_project_mode=single_project_mode,
                question_focus=reply_plan.question_focus,
            )
        if reply:
            return _compact_text(reply, max_words=120)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:240]
        log.warning("reply synthesis HTTP %s: %s", exc.code, detail)
    except urllib.error.URLError as exc:
        log.warning("reply synthesis unreachable: %s", exc.reason)
    except socket.timeout:
        log.warning("reply synthesis timeout after %ss", settings.decider_timeout_sec)
    except Exception as exc:
        log.warning("reply synthesis failed: %s", exc)
    return _build_reply_fallback(message=message, analysis=analysis, grounded_result=grounded_result)


def _build_decider_prompt(message: str, lead_state: LeadState, recent_history: list[HistoryTurn]) -> str:
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
    }
    compact_history = [{"role": turn.role, "message": turn.message} for turn in recent_history[-6:]]
    return (
        "Ban la bo phan decider route cho tro ly tu van bat dong san.\n"
        "Nhiem vu duy nhat cua agent: tro chuyen tu van va gioi thieu du an cho khach hang.\n"
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
        "- Bat buoc co it nhat 1 nhan dinh huu ich truoc.\n"
        "- Chi dat cau hoi khi thieu 1 thong tin quan trong; khong mac dinh ket thuc bang cau hoi.\n"
        "- Toi da 3-4 cau ngan; neu hoi thi toi da 1 cau hoi.\n"
        "- Khong dat cau hoi dang thu thap form nhu 'ban o quan nao', 'ban muon khu vuc nao'.\n"
        "- Khong hoi lai thong tin user vua noi.\n"
        "- Neu can phan tich sau hon, uu tien de xuat buoi hen gap truc tiep thay vi co gang phan tich qua sau trong chat.\n"
        "- Neu query la project_matching, consult_reply chi la cau bridge ngan de chain sang project route, khong dong request o consult.\n\n"
        "Bat buoc tra ve 1 JSON object theo schema:\n"
        "{\n"
        '  "query_type": "advisory_strategy|project_matching|project_specific|clarification",\n'
        '  "retrieval_readiness": "not_ready|soft_ready|ready",\n'
        '  "start_route": "consult_discovery|project_grounded",\n'
        '  "route": "consult_discovery|project_grounded",\n'
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
        '  "consult_reply": "string"\n'
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
    payload: dict[str, Any] = {
        "model": settings.decider_model,
        "prompt": _build_decider_prompt(message=message, lead_state=lead_state, recent_history=recent_history),
        "stream": False,
        "format": "json",
        "options": {"temperature": settings.decider_temperature},
    }
    if settings.decider_keep_alive:
        payload["keep_alive"] = settings.decider_keep_alive

    req = urllib.request.Request(
        settings.decider_api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=settings.decider_timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("decider response is not a JSON object")

    response_payload = parsed.get("response", "")
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

    consult_reply = _compact_text(str(result_obj.get("consult_reply", "")).strip(), max_words=95)
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
    )


def analyze_turn(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    settings: Settings,
) -> TurnAnalysis:
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
) -> dict[str, Any]:
    retrieval_intent = _build_retrieval_intent(
        message=message,
        lead_state=lead_state,
        query_type=query_type,
        retrieval_readiness=retrieval_readiness,
    )
    retrieval_raw = _call_project_grounded_fetcher(retrieval_fetcher, message, retrieval_intent, top_k)
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
        # Phase 1 - Analyze
        message = payload.message.strip()
        lead_state = payload.lead_state or LeadState()
        top_k = payload.top_k or settings.default_top_k
        extracted_name = _extract_name(message)
        extracted_phone = _extract_phone_contact(message)

        analysis = turn_analyzer(message, lead_state, payload.recent_history)
        start_route = analysis.route
        reason = analysis.decision_reason
        route_source = analysis.route_source or "turn_analyzer"

        if payload.force_route is not None:
            start_route = payload.force_route
            reason = f"force_route={payload.force_route}"
            route_source = "force_route_debug"

        # Phase 2 - Execute
        consult_result: dict[str, Any] | None = None
        grounded_result: dict[str, Any] | None = None
        final_state: LeadState
        final_route = start_route
        chained_from_consult = False
        active_need_update = analysis.need_update
        active_painpoint_update = analysis.painpoint_update
        routing_signal: RoutingSignal | None = None

        if start_route == "consult_discovery":
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

            if should_chain_project:
                chained_from_consult = True
                final_route = "project_grounded"
                project_message = str(routing_signal.project_query_hint or message).strip() or message
                try:
                    grounded_result = run_project_grounded(
                        message=project_message,
                        lead_state=after_consult_state,
                        query_type=analysis.query_type,
                        retrieval_readiness=analysis.retrieval_readiness,
                        top_k=top_k,
                        retrieval_fetcher=project_grounded_fetcher,
                        need_update=active_need_update,
                        painpoint_update=active_painpoint_update,
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
                grounded_result = run_project_grounded(
                    message=message,
                    lead_state=lead_state,
                    query_type=analysis.query_type,
                    retrieval_readiness=analysis.retrieval_readiness,
                    top_k=top_k,
                    retrieval_fetcher=project_grounded_fetcher,
                    need_update=analysis.need_update,
                    painpoint_update=analysis.painpoint_update,
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

        # Phase 3 - Finalize
        trace = DecisionTrace(
            query_type=_normalize_query_type(analysis.query_type),
            retrieval_readiness=_normalize_retrieval_readiness(analysis.retrieval_readiness),
            start_route=start_route,
            final_route=final_route,
            chained_from_consult=chained_from_consult,
            route_source=route_source,
            decision_reason=reason,
        )
        reply_plan = build_reply_plan(
            message=message,
            recent_history=payload.recent_history,
            lead_state=final_state,
            analysis=analysis,
            final_route=final_route,
            grounded_result=grounded_result,
        )
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

        log.info(
            (
                "decision start_route=%s final_route=%s chained_from_consult=%s "
                "query_type=%s retrieval_readiness=%s route_source=%s response_mode=%s ask_policy=%s reason=%s"
            ),
            trace.start_route,
            trace.final_route,
            trace.chained_from_consult,
            trace.query_type,
            trace.retrieval_readiness,
            trace.route_source,
            reply_plan.response_mode,
            reply_plan.ask_policy,
            trace.decision_reason,
        )
        return response

    return app


app = create_app()
