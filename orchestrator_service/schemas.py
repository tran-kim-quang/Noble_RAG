from __future__ import annotations

from datetime import datetime, timezone
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


class LeadState(BaseModel):
    name: str | None = None
    phone_contact: str | None = None
    need: NeedPainpointState = Field(default_factory=NeedPainpointState)
    painpoint: NeedPainpointState = Field(default_factory=NeedPainpointState)


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
    fit_personas: list[str] = Field(default_factory=list)
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


class ProjectGroundedPayload(BaseModel):
    used_projects: list[str] = Field(default_factory=list)
    project_cards: list[ProjectCardPayload] = Field(default_factory=list)
    trait_tags: list[TraitTagPayload] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunkPayload] = Field(default_factory=list)
    confidence: float | None = None
    low_confidence: bool = False


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
    recent_history: list[HistoryTurn] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
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
    timestamp: str = Field(default_factory=_now_iso)
