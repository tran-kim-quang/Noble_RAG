from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field
from pydantic import field_validator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopicWeight(BaseModel):
    label: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def validate_label_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label must not be blank")
        return cleaned


class NeedPainpointState(BaseModel):
    summary: str = ""
    topics: list[TopicWeight] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    last_updated_at: str | None = None


class VisionContext(BaseModel):
    recognized: bool = False
    name: str | None = None
    age: Literal["trẻ", "trung niên"] | None = None
    gender: Literal["male", "female"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Literal["face_db", "local_face_analysis", "vlm", "none", "error"] = "none"
    face_count: int = Field(default=0, ge=0)
    bbox: list[int] | None = None
    reason: str | None = None
    face_id: str | None = None


class CustomerProfileState(BaseModel):
    recognized: bool = False
    name: str | None = None
    age: Literal["trẻ", "trung niên"] | None = None
    gender: Literal["male", "female"] | None = None
    source: Literal["face_db", "local_face_analysis", "vlm", "none", "error"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    last_seen_at: str | None = None
    greeted_by_name: bool = False
    greeted_generic: bool = False
    face_id: str | None = None


class LeadState(BaseModel):
    name: str | None = None
    phone_contact: str | None = None
    need: NeedPainpointState = Field(default_factory=NeedPainpointState)
    painpoint: NeedPainpointState = Field(default_factory=NeedPainpointState)
    customer_profile: CustomerProfileState = Field(default_factory=CustomerProfileState)
    engagement_state: Literal["cold", "warm", "interested", "ready"] = "cold"
    engagement_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sales_state: Literal[
        "unknown",
        "exploring",
        "need_identified",
        "qualified",
        "interested",
        "appointment_ready",
        "nurture",
        "handoff",
    ] = "unknown"
    lead_level: Literal["exploratory", "interested", "qualified", "hot"] | None = None
    last_conversation_goal: Literal[
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
    ] | None = None
    next_best_action: Literal[
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
    ] | None = None
    contact_capture_status: Literal["unknown", "requested", "partial", "complete"] = "unknown"


class NeedPainpointDelta(BaseModel):
    summary_delta: str = ""
    topics: list[TopicWeight] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class RoutingSignal(BaseModel):
    should_route_project: bool = False
    project_query_hint: str | None = None
    reason: str = ""


class ProjectCardPayload(BaseModel):
    project_id: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    area: str | None = None
    product_types: list[str] = Field(default_factory=list)
    fit_personas: list[str] = Field(default_factory=list)
    family_fit_score: float | None = None
    investor_fit_score: float | None = None
    proximity_tags: list[str] = Field(default_factory=list)
    key_pois: list[str] = Field(default_factory=list)
    score: float = 0.0


class TraitTagPayload(BaseModel):
    tag: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: str
    project_id: str | None = None


class EvidenceChunkPayload(BaseModel):
    text: str
    score: float
    source: str
    doc_id: str
    project_id: str | None = None
    topic: str | None = None


class ProximityFactPayload(BaseModel):
    project_id: str
    poi_type: Literal["hospital", "school", "park", "mall", "unknown"] = "unknown"
    poi_name: str | None = None
    proximity_text: str = ""
    distance_text: str | None = None
    travel_mode: Literal["walk", "drive", "unspecified"] = "unspecified"
    evidence_source: str = "unknown"
    semantic_tags: list[str] = Field(default_factory=list)


class ProjectGroundedPayload(BaseModel):
    used_projects: list[str] = Field(default_factory=list)
    project_cards: list[ProjectCardPayload] = Field(default_factory=list)
    trait_tags: list[TraitTagPayload] = Field(default_factory=list)
    proximity_facts: list[ProximityFactPayload] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunkPayload] = Field(default_factory=list)
    retrieval_intent: dict[str, Any] | str | None = None
    confidence: float | None = None
    low_confidence: bool = False


class DecisionTrace(BaseModel):
    query_type: Literal["advisory_strategy", "project_matching", "project_specific", "clarification"] = (
        "clarification"
    )
    retrieval_readiness: Literal["not_ready", "soft_ready", "ready"] = "not_ready"
    start_route: Literal["consult_discovery", "project_grounded"] = "consult_discovery"
    final_route: Literal["consult_discovery", "project_grounded"] = "consult_discovery"
    chained_from_consult: bool = False
    route_source: str = "unknown"
    decision_reason: str = ""
    engagement_state_before: Literal["cold", "warm", "interested", "ready"] | None = None
    engagement_state_after: Literal["cold", "warm", "interested", "ready"] | None = None
    sales_state_before: Literal[
        "unknown",
        "exploring",
        "need_identified",
        "qualified",
        "interested",
        "appointment_ready",
        "nurture",
        "handoff",
    ] | None = None
    sales_state_after: Literal[
        "unknown",
        "exploring",
        "need_identified",
        "qualified",
        "interested",
        "appointment_ready",
        "nurture",
        "handoff",
    ] | None = None
    conversation_goal: Literal[
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
    ] | None = None
    response_mode: Literal[
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
    ] | None = None
    ask_policy: Literal["avoid_question", "allow_question", "must_clarify"] | None = None


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned


class QueryRequest(BaseModel):
    message: str = Field(min_length=1)
    lead_state: LeadState | None = None
    vision_context: VisionContext | None = None
    recent_history: list[HistoryTurn] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    force_route: Literal["consult_discovery", "project_grounded"] | None = None

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned


class QueryResponse(BaseModel):
    route: Literal["consult_discovery", "project_grounded"]
    assistant_reply: str
    decision_reason: str
    lead_state: LeadState
    need_update: NeedPainpointDelta = Field(default_factory=NeedPainpointDelta)
    painpoint_update: NeedPainpointDelta = Field(default_factory=NeedPainpointDelta)
    routing_signal: RoutingSignal | None = None
    project_grounded_payload: ProjectGroundedPayload | None = None
    decision_trace: DecisionTrace | None = None
    timestamp: str = Field(default_factory=_now_iso)


class VisionQueryRequest(QueryRequest):
    image_base64: str = Field(min_length=1)
    image_filename: str | None = None
    image_content_type: str | None = None

    @field_validator("image_base64")
    @classmethod
    def validate_image_base64_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("image_base64 must not be blank")
        return cleaned


class LiveTalkingSessionStartRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    image_filename: str | None = None
    image_content_type: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    lead_state: LeadState | None = None

    @field_validator("image_base64")
    @classmethod
    def validate_start_image_base64_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("image_base64 must not be blank")
        return cleaned


class LiveTalkingQueryRequest(VisionQueryRequest):
    pass


class LiveTalkingStopRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    face_session_key: str | None = None


class LiveTalkingSessionState(BaseModel):
    session_id: str
    face_session_key: str
    customer_kind: Literal["known", "guest", "anonymous"]
    ttl_sec: int = Field(ge=1)
    resumed: bool = False
    should_greet: bool = False
    greeting: str | None = None


class LiveTalkingSessionStartResponse(BaseModel):
    session: LiveTalkingSessionState
    lead_state: LeadState
    vision_context: VisionContext | None = None


class LiveTalkingQueryResponse(QueryResponse):
    session: LiveTalkingSessionState
    vision_context: VisionContext | None = None


class LiveTalkingStopResponse(BaseModel):
    stopped: bool
    session_id: str | None = None
    face_session_key: str | None = None
