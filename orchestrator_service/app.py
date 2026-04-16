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


@dataclass(frozen=True)
class TurnAnalysis:
    route: str
    decision_reason: str
    need_update: NeedPainpointDelta
    painpoint_update: NeedPainpointDelta
    routing_signal: RoutingSignal
    consult_reply: str


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


def _build_retrieval_intent(message: str, lead_state: LeadState) -> str:
    need_topics = ", ".join(topic.label for topic in lead_state.need.topics[:4])
    pain_topics = ", ".join(topic.label for topic in lead_state.painpoint.topics[:4])
    parts = [
        f"user_query: {message.strip()}",
        f"need_topics: {need_topics or 'none'}",
        f"painpoint_topics: {pain_topics or 'none'}",
    ]
    return " | ".join(parts)


def _call_project_grounded_fetcher(fetcher, message: str, intent: str, top_k: int) -> dict[str, Any]:
    try:
        return fetcher(message, intent, top_k)
    except TypeError:
        legacy = fetcher(message, top_k)
        return {
            "project_cards": [],
            "trait_tags": [],
            "evidence_chunks": legacy.get("results", []) if isinstance(legacy, dict) else [],
            "confidence": legacy.get("confidence", 0.0) if isinstance(legacy, dict) else 0.0,
            "low_confidence": bool(legacy.get("low_confidence", False)) if isinstance(legacy, dict) else True,
        }


def _build_project_reply(
    message: str,
    lead_state: LeadState,
    project_cards: list[dict],
    trait_tags: list[dict],
    low_confidence: bool,
) -> str:
    if low_confidence or not project_cards:
        context_hint = ""
        user_focus = (message or "").strip()
        if not user_focus:
            user_focus = lead_state.need.summary.strip() or lead_state.painpoint.summary.strip()
        if user_focus:
            context_hint = f"Minh da ghi nhan bo loc cua ban: {user_focus}. "
        return (
            context_hint
            + "Trong kho du lieu Noble hien tai, thong tin grounded de de xuat du an cu the chua du day. "
            + "De giu dung nhu cau that cua ban, ban muon minh di sau theo huong phu hop de o lau dai hay toi uu dong tien truoc?"
        )

    top_cards = project_cards[:2]
    segments: list[str] = []
    for card in top_cards:
        project_id = str(card.get("project_id", "unknown_project"))
        strengths = card.get("strengths", []) or []
        tradeoffs = card.get("tradeoffs", []) or []
        reason = ", ".join(str(x) for x in strengths[:2]) if strengths else "co du lieu phu hop voi bo tieu chi hien tai"
        tradeoff = str(tradeoffs[0]) if tradeoffs else "can doi chieu them voi lich di chuyen va ngan sach thuc te"
        segments.append(f"- {project_id}: hop vi {reason}. Can nhac: {tradeoff}.")

    top_traits = ", ".join(str(item.get("tag", "")) for item in trait_tags[:3] if item.get("tag"))
    trait_line = f"Trait noi bat tu du lieu: {top_traits}. " if top_traits else ""
    return (
        "Minh da grounding tren du lieu du an Noble va shortlist tam thoi:\n"
        + "\n".join(segments)
        + "\n"
        + trait_line
        + "Ban muon minh di sau phuong an de o on dinh hay phuong an toi uu dau tu truoc?"
    )


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


def _build_consult_reply_from_context(
    message: str,
    lead_state: LeadState,
    need_update: NeedPainpointDelta,
    painpoint_update: NeedPainpointDelta,
) -> str:
    cleaned_message = (message or "").strip()
    need_focus = need_update.summary_delta or lead_state.need.summary or "muc tieu mua va bo tieu chi phu hop"
    pain_focus = painpoint_update.summary_delta or lead_state.painpoint.summary or "khung ra quyet dinh chua du ro"
    if cleaned_message and len(cleaned_message.split()) <= 3:
        return (
            f"Minh ghi nhan '{cleaned_message}' la uu tien hien tai cua ban. "
            "Minh se dua tren bo loc nay de thu hep huong tu van thay vi hoi form. "
            "Ban muon minh mo nhanh goc de o on dinh hay goc toi uu tai chinh truoc?"
        )
    return (
        f"Minh da nam duoc huong uu tien cua ban: {need_focus}. "
        f"Diem dang lam ban can nhac la: {pain_focus}. "
        "Minh co the phac nhanh 2 huong de ban thay ngay diem khac nhau trong bo loc du an Noble. "
        "Ban muon xem huong de o on dinh truoc hay huong giu gia dau tu truoc?"
    )


def _is_form_like_consult_reply(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if cleaned.count("?") >= 2:
        return True
    if re.search(r"(?m)^\s*\d+\.", cleaned):
        return True
    lowered = cleaned.lower()
    form_signals = [
        "cho minh biet them",
        "de minh co the tim",
        "ve khu vuc",
        "ve ngan sach",
        "ve muc dich",
    ]
    return any(signal in lowered for signal in form_signals)


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
        "Ban la bo phan tich route cho tro ly tu van bat dong san.\n"
        "Nhiem vu: suy luan theo ngu canh hoi thoai, khong duoc dua vao bang keyword co dinh.\n"
        "Ban can xac dinh nhu cau (need), khuc mac/painpoint, va do san sang de grounding du an.\n"
        "Neu chua du clarity thi giu consult_discovery va tra loi de khai thac them.\n"
        "Neu da du clarity thi route project_grounded.\n\n"
        "Quy tac bat buoc cho consult_reply:\n"
        "- Giong consultant, khong giong form thu thap thong tin.\n"
        "- Khong dung checklist dang 1., 2., 3. hoac hoi don dap lien tuc.\n"
        "- Toi da 90 tu, toi da 1 cau hoi o cuoi cau tra loi.\n"
        "- Neu nguoi dung da dua ngan sach/khu vuc/muc tieu thi KHONG hoi lai y chang.\n"
        "- Neu user tra loi ngan 1-3 tu (vi du: 'ngan sach', 'long bien') thi xem do la thong tin bo sung va day tiep bang 1 huong goi mo.\n"
        "- Muc tieu la lam nguoi dung to mo de hoi sau hon ve huong du an Noble, khong phai thu thap keyword.\n\n"
        "Bat buoc tra ve dung 1 JSON object, khong markdown, theo schema:\n"
        "{\n"
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

    route_raw = str(result_obj.get("route", "consult_discovery")).strip().lower()
    route = "project_grounded" if route_raw == "project_grounded" else "consult_discovery"
    need_update = _coerce_delta(result_obj.get("need_update"), message)
    painpoint_update = _coerce_delta(result_obj.get("painpoint_update"), message)

    should_route_project = bool(result_obj.get("should_route_project", route == "project_grounded"))
    if route == "project_grounded":
        should_route_project = True
    project_query_hint_raw = str(result_obj.get("project_query_hint", "")).strip()
    if project_query_hint_raw.lower() in {"none", "null", "n/a"}:
        project_query_hint_raw = ""
    project_query_hint = project_query_hint_raw or (message.strip() if should_route_project else None)
    routing_reason = str(result_obj.get("decision_reason", "")).strip()
    if not routing_reason:
        routing_reason = "llm_decider"

    consult_reply_raw = str(result_obj.get("consult_reply", "")).strip()
    consult_reply = consult_reply_raw or _build_consult_reply_from_context(
        message=message,
        lead_state=lead_state,
        need_update=need_update,
        painpoint_update=painpoint_update,
    )
    if _is_form_like_consult_reply(consult_reply):
        consult_reply = _build_consult_reply_from_context(
            message=message,
            lead_state=lead_state,
            need_update=need_update,
            painpoint_update=painpoint_update,
        )
    return TurnAnalysis(
        route=route,
        decision_reason=routing_reason,
        need_update=need_update,
        painpoint_update=painpoint_update,
        routing_signal=RoutingSignal(
            should_route_project=should_route_project,
            project_query_hint=project_query_hint,
            reason=routing_reason,
        ),
        consult_reply=consult_reply,
    )


def _analyze_turn_fallback(message: str, lead_state: LeadState) -> TurnAnalysis:
    cleaned = message.strip()
    token_count = len(cleaned.split())
    has_question = "?" in cleaned
    existing_context = bool(
        lead_state.need.summary.strip()
        or lead_state.painpoint.summary.strip()
        or lead_state.need.topics
        or lead_state.painpoint.topics
    )
    top_need_weight = max((float(item.weight) for item in lead_state.need.topics), default=0.0)
    detail_score = min(
        1.0,
        (token_count / 24.0)
        + (0.2 if has_question else 0.0)
        + (0.2 if existing_context else 0.0),
    )
    should_route_project = bool(has_question and (top_need_weight >= 0.72 or detail_score >= 0.9))
    route = "project_grounded" if should_route_project else "consult_discovery"

    if route == "project_grounded":
        need_summary = "Khach da neu boi canh du de doi chieu phuong an cu the."
        pain_summary = "Khach can bang chung de giam do bat dinh truoc khi quyet dinh."
    else:
        need_summary = "Khach dang o giai doan lam ro muc tieu va tieu chi uu tien."
        pain_summary = "Khach chua co khung tieu chi du ro de shortlist phuong an."

    need_update = NeedPainpointDelta(
        summary_delta=need_summary,
        topics=lead_state.need.topics[:2],
        evidence=[cleaned] if cleaned else [],
    )
    painpoint_update = NeedPainpointDelta(
        summary_delta=pain_summary,
        topics=lead_state.painpoint.topics[:2],
        evidence=[cleaned] if cleaned else [],
    )
    consult_reply = _build_consult_reply_from_context(
        message=message,
        lead_state=lead_state,
        need_update=need_update,
        painpoint_update=painpoint_update,
    )
    reason = "fallback_context_decider"
    return TurnAnalysis(
        route=route,
        decision_reason=reason,
        need_update=need_update,
        painpoint_update=painpoint_update,
        routing_signal=RoutingSignal(
            should_route_project=should_route_project,
            project_query_hint=cleaned if should_route_project else None,
            reason=reason,
        ),
        consult_reply=consult_reply,
    )


def analyze_turn(
    message: str,
    lead_state: LeadState,
    recent_history: list[HistoryTurn],
    settings: Settings,
) -> TurnAnalysis:
    if not settings.decider_enabled:
        return _analyze_turn_fallback(message=message, lead_state=lead_state)
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
    return _analyze_turn_fallback(message=message, lead_state=lead_state)


def classify_route(message: str, lead_state: LeadState) -> tuple[str, str]:
    # Backward-compatible helper: now powered by context analyzer fallback.
    analysis = _analyze_turn_fallback(message=message, lead_state=lead_state)
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
    top_k: int,
    retrieval_fetcher,
    need_update: NeedPainpointDelta,
    painpoint_update: NeedPainpointDelta,
) -> dict[str, Any]:
    retrieval_intent = _build_retrieval_intent(message, lead_state)
    retrieval_raw = _call_project_grounded_fetcher(retrieval_fetcher, message, retrieval_intent, top_k)
    project_cards = retrieval_raw.get("project_cards", []) or []
    trait_tags = retrieval_raw.get("trait_tags", []) or []
    evidence_chunks = retrieval_raw.get("evidence_chunks", []) or []
    low_confidence = bool(retrieval_raw.get("low_confidence", False))

    assistant_reply = _build_project_reply(
        message=message,
        lead_state=lead_state,
        project_cards=project_cards,
        trait_tags=trait_tags,
        low_confidence=low_confidence,
    )
    used_projects = [str(card.get("project_id")) for card in project_cards if card.get("project_id")]
    return {
        "assistant_reply": assistant_reply,
        "used_projects": _dedupe_keep_order(used_projects, max_items=5),
        "project_cards": project_cards,
        "trait_tags": trait_tags,
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

        extracted_name = _extract_name(message)
        extracted_phone = _extract_phone_contact(message)

        analysis = turn_analyzer(message, lead_state, payload.recent_history)
        route = analysis.route
        reason = analysis.decision_reason
        if payload.force_route is not None:
            route = payload.force_route
            reason = f"force_route={payload.force_route}"

        if route == "consult_discovery":
            consult = run_consult_discovery(message=message, lead_state=lead_state, analysis=analysis)
            updated_state = merge_lead_state(
                lead_state=lead_state,
                need_update=consult["need_update"],
                painpoint_update=consult["painpoint_update"],
                extracted_name=extracted_name,
                extracted_phone=extracted_phone,
            )
            response = QueryResponse(
                route="consult_discovery",
                assistant_reply=consult["assistant_reply"],
                decision_reason=reason,
                lead_state=updated_state,
                need_update=consult["need_update"],
                painpoint_update=consult["painpoint_update"],
                routing_signal=consult["routing_signal"],
                project_grounded_payload=None,
            )
            log.info("decision route=%s reason=%s", response.route, response.decision_reason)
            return response

        try:
            grounded = run_project_grounded(
                message=message,
                lead_state=lead_state,
                top_k=top_k,
                retrieval_fetcher=project_grounded_fetcher,
                need_update=analysis.need_update,
                painpoint_update=analysis.painpoint_update,
            )
        except Exception as exc:
            log.exception("project_grounded retrieval failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"retrieval service error: {exc}") from exc

        final_state = merge_lead_state(
            lead_state=lead_state,
            need_update=grounded["need_update"],
            painpoint_update=grounded["painpoint_update"],
            extracted_name=extracted_name,
            extracted_phone=extracted_phone,
        )
        response = QueryResponse(
            route="project_grounded",
            assistant_reply=grounded["assistant_reply"],
            decision_reason=reason,
            lead_state=final_state,
            need_update=grounded["need_update"],
            painpoint_update=grounded["painpoint_update"],
            routing_signal=None,
            project_grounded_payload={
                "used_projects": grounded["used_projects"],
                "project_cards": grounded["project_cards"],
                "trait_tags": grounded["trait_tags"],
                "evidence_chunks": grounded["evidence_chunks"],
                "confidence": grounded["confidence"],
                "low_confidence": grounded["low_confidence"],
            },
        )
        log.info("decision route=%s reason=%s", response.route, response.decision_reason)
        return response

    return app


app = create_app()
