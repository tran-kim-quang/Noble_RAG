from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException

from orchestrator_service.config import get_settings
from orchestrator_service.retrieval_client import RetrievalClient
from orchestrator_service.schemas import (
    LeadState,
    NeedPainpointDelta,
    NeedPainpointState,
    QueryRequest,
    QueryResponse,
    RoutingSignal,
    TopicWeight,
)

log = logging.getLogger("sales-orchestrator")

_PROJECT_ROUTE_KEYWORDS = {
    "du an",
    "dự án",
    "gia",
    "giá",
    "dien tich",
    "diện tích",
    "phap ly",
    "pháp lý",
    "gan truong",
    "gần trường",
    "benh vien",
    "bệnh viện",
    "khu nao",
    "khu nào",
    "can nao",
    "căn nào",
    "co du an nao",
    "có dự án nào",
    "thanh khoan",
    "thanh khoản",
}

_CONSULT_ROUTE_KEYWORDS = {
    "nen bat dau",
    "nên bắt đầu",
    "tu van",
    "tư vấn",
    "nen chon",
    "nên chọn",
    "mua de o",
    "mua để ở",
    "mua de dau tu",
    "mua để đầu tư",
}

_NEED_TOPIC_RULES: list[tuple[str, float, tuple[str, ...]]] = [
    ("dau tu", 0.94, ("dau tu", "đầu tư", "thanh khoan", "thanh khoản", "giu gia", "giữ giá")),
    ("an toan", 0.82, ("an toan", "an toàn", "it rui ro", "ít rủi ro")),
    ("tang truong", 0.8, ("tang truong", "tăng trưởng", "bien do", "biên độ")),
    ("mua de o", 0.9, ("mua de o", "mua để ở", "o thuc", "ở thực")),
    ("gia dinh co con nho", 0.88, ("con nho", "con nhỏ", "gia dinh", "gia đình")),
    ("truong hoc", 0.84, ("truong hoc", "trường học")),
    ("benh vien", 0.79, ("benh vien", "bệnh viện")),
    ("phap ly ro rang", 0.86, ("phap ly", "pháp lý", "so hong", "sổ hồng")),
    ("ngan sach", 0.76, ("ngan sach", "ngân sách", "gia", "giá")),
    ("vi tri thuan tien", 0.8, ("vi tri", "vị trí", "di chuyen", "di chuyển", "khu dong", "khu đông")),
]

_PAINPOINT_RULES: list[tuple[str, float, tuple[str, ...]]] = [
    ("so chon sai", 0.82, ("so chon sai", "sợ", "chon sai", "chọn sai", "lo chon sai")),
    ("phan van nhieu lua chon", 0.77, ("phan van", "phân vân", "roi", "rối")),
    ("lo ap luc tai chinh", 0.79, ("ap luc tai chinh", "áp lực tài chính", "vay", "lãi suất", "lai suat")),
    ("thieu thong tin de quyet dinh", 0.74, ("khong biet", "không biết", "chua ro", "chưa rõ")),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


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


def classify_route(message: str, lead_state: LeadState) -> tuple[str, str]:
    text = _normalize_text(message)
    if _contains_any(text, _PROJECT_ROUTE_KEYWORDS):
        return "project_grounded", "query_has_project_specific_filters_or_project_intent"
    if _contains_any(text, _CONSULT_ROUTE_KEYWORDS):
        return "consult_discovery", "query_is_open_ended_strategy"

    specific_need_topics = {
        "truong hoc",
        "benh vien",
        "phap ly ro rang",
        "vi tri thuan tien",
        "dau tu",
        "mua de o",
    }
    state_has_specific_lens = any(
        topic.label in specific_need_topics and topic.weight >= 0.8 for topic in lead_state.need.topics
    )
    asks_for_options = any(token in text for token in ("co gi", "có gì", "goi y", "gợi ý", "lua chon", "lựa chọn"))
    if state_has_specific_lens and asks_for_options:
        return "project_grounded", "existing_state_has_clear_lens_and_user_asks_for_options"
    return "consult_discovery", "default_to_consult_for_discovery"


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


def _extract_delta(message: str, rules: list[tuple[str, float, tuple[str, ...]]], summary_prefix: str) -> NeedPainpointDelta:
    text = _normalize_text(message)
    topics: list[TopicWeight] = []
    labels: list[str] = []
    for label, weight, keywords in rules:
        if any(keyword in text for keyword in keywords):
            topics.append(TopicWeight(label=label, weight=weight))
            labels.append(label)

    if labels:
        summary_delta = f"{summary_prefix}: {', '.join(labels)}."
    else:
        summary_delta = ""
    evidence = [message.strip()] if message.strip() else []
    return NeedPainpointDelta(summary_delta=summary_delta, topics=topics, evidence=evidence)


def _build_routing_signal(message: str, need_update: NeedPainpointDelta) -> RoutingSignal:
    text = _normalize_text(message)
    user_has_clear_filters = _contains_any(text, _PROJECT_ROUTE_KEYWORDS)
    user_asks_real_options = any(token in text for token in ("du an nao", "dự án nào", "co can nao", "có căn nào", "khu nao", "khu nào"))
    extracted_need_is_specific = len(need_update.topics) >= 2

    should_route_project = user_has_clear_filters or user_asks_real_options or extracted_need_is_specific
    if should_route_project:
        reason = "Da co lens/criteria du ro de grounding du lieu du an."
        project_query_hint = message.strip()
    else:
        reason = "Chua du tieu chi cu the de retrieve du an."
        project_query_hint = None
    return RoutingSignal(
        should_route_project=should_route_project,
        project_query_hint=project_query_hint,
        reason=reason,
    )


def _build_consult_reply(message: str, need_update: NeedPainpointDelta) -> str:
    labels = {topic.label for topic in need_update.topics}
    if "dau tu" in labels:
        return (
            "Neu minh mua de dau tu, minh nen nhin theo 3 truc: do an toan khi xuong tien, "
            "kha nang giu gia/thanh khoan, va du dia tang truong theo khu vuc. "
            "Khi can bang 3 truc nay, quyet dinh se it cam tinh hon va de tranh chon sai. "
            "Ban dang uu tien phuong an an toan hay chap nhan bien do de ky vong tang truong cao hon?"
        )
    if "mua de o" in labels or "gia dinh co con nho" in labels:
        return (
            "Voi nhu cau o thuc, thuong nen uu tien 1 goc nhin chinh truoc: "
            "chat luong song hang ngay (di chuyen, truong hoc, y te, tien ich). "
            "Khi xac dinh duoc goc nhin nay, viec loc du an se sat va nhanh hon nhieu. "
            "Gia dinh minh hien uu tien nhat la di chuyen tien hay moi truong song cho con?"
        )
    return (
        "De tu van sat hon, minh thu chot 1 khung nho truoc: muc tieu mua (o hay dau tu), "
        "ngan sach linh hoat den dau, va khu vuc uu tien. "
        "Co du 3 diem nay, minh se de dang shortlist cac lua chon phu hop hon. "
        "Ban muon bat dau tu muc tieu mua hay ngan sach truoc?"
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


def _build_project_reply(project_cards: list[dict], trait_tags: list[dict], low_confidence: bool) -> str:
    if low_confidence or not project_cards:
        return (
            "Hien du lieu retrieve chua du vung de de xuat du an cu the. "
            "Ban cho minh them 2 tieu chi uu tien nhat (ngan sach, khu vuc, muc tieu o/de dau tu) de minh loc lai chinh xac hon?"
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
        "Minh da grounding tren du lieu du an va shortlist tam thoi:\n"
        + "\n".join(segments)
        + "\n"
        + trait_line
        + "Ban uu tien phuong an de o on dinh lau dai hay toi uu cho dau tu de minh sap xep thu tu uu tien?"
    )


def run_consult_discovery(message: str) -> dict[str, Any]:
    need_update = _extract_delta(message, _NEED_TOPIC_RULES, "Nhu cau duoc bo sung")
    painpoint_update = _extract_delta(message, _PAINPOINT_RULES, "Painpoint duoc bo sung")
    routing_signal = _build_routing_signal(message, need_update)
    assistant_reply = _build_consult_reply(message, need_update)
    return {
        "assistant_reply": assistant_reply,
        "need_update": need_update,
        "painpoint_update": painpoint_update,
        "routing_signal": routing_signal,
    }


def run_project_grounded(
    message: str,
    lead_state: LeadState,
    top_k: int,
    retrieval_fetcher,
) -> dict[str, Any]:
    retrieval_intent = _build_retrieval_intent(message, lead_state)
    retrieval_raw = _call_project_grounded_fetcher(retrieval_fetcher, message, retrieval_intent, top_k)
    project_cards = retrieval_raw.get("project_cards", []) or []
    trait_tags = retrieval_raw.get("trait_tags", []) or []
    evidence_chunks = retrieval_raw.get("evidence_chunks", []) or []
    low_confidence = bool(retrieval_raw.get("low_confidence", False))

    assistant_reply = _build_project_reply(project_cards=project_cards, trait_tags=trait_tags, low_confidence=low_confidence)
    used_projects = [str(card.get("project_id")) for card in project_cards if card.get("project_id")]
    need_update = _extract_delta(message, _NEED_TOPIC_RULES, "Nhu cau duoc bo sung")
    painpoint_update = _extract_delta(message, _PAINPOINT_RULES, "Painpoint duoc bo sung")
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


def create_app(project_grounded_fetcher=None) -> FastAPI:
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

        route, reason = classify_route(message, lead_state)
        if payload.force_route is not None:
            route = payload.force_route
            reason = f"force_route={payload.force_route}"

        if route == "consult_discovery":
            consult = run_consult_discovery(message)
            updated_state = merge_lead_state(
                lead_state=lead_state,
                need_update=consult["need_update"],
                painpoint_update=consult["painpoint_update"],
                extracted_name=extracted_name,
                extracted_phone=extracted_phone,
            )

            routing_signal: RoutingSignal = consult["routing_signal"]
            if routing_signal.should_route_project:
                try:
                    grounded = run_project_grounded(
                        message=message,
                        lead_state=updated_state,
                        top_k=top_k,
                        retrieval_fetcher=project_grounded_fetcher,
                    )
                except Exception as exc:
                    log.exception("project_grounded retrieval failed: %s", exc)
                    raise HTTPException(status_code=502, detail=f"retrieval service error: {exc}") from exc

                final_state = merge_lead_state(
                    lead_state=updated_state,
                    need_update=grounded["need_update"],
                    painpoint_update=grounded["painpoint_update"],
                    extracted_name=extracted_name,
                    extracted_phone=extracted_phone,
                )
                response = QueryResponse(
                    route="project_grounded",
                    assistant_reply=grounded["assistant_reply"],
                    decision_reason=f"{reason}; consult_escalated_to_project={routing_signal.reason}",
                    lead_state=final_state,
                    need_update=grounded["need_update"],
                    painpoint_update=grounded["painpoint_update"],
                    routing_signal=routing_signal,
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

            response = QueryResponse(
                route="consult_discovery",
                assistant_reply=consult["assistant_reply"],
                decision_reason=reason,
                lead_state=updated_state,
                need_update=consult["need_update"],
                painpoint_update=consult["painpoint_update"],
                routing_signal=routing_signal,
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
