from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field
from pydantic import field_validator


class IngestDocument(BaseModel):
    text: str = Field(min_length=1)
    source: str | None = None
    doc_id: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("text")
    @classmethod
    def validate_text_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1)


class IngestResponse(BaseModel):
    ingested: int
    collection: str


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class RetrievedItem(BaseModel):
    text: str
    score: float
    source: str
    doc_id: str


class RetrieveResponse(BaseModel):
    results: list[RetrievedItem]
    confidence: float | None = None
    low_confidence: bool = False


class ProjectCard(BaseModel):
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


class TraitTag(BaseModel):
    tag: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: str
    project_id: str | None = None


class EvidenceChunk(BaseModel):
    text: str
    score: float
    source: str
    doc_id: str
    project_id: str | None = None
    topic: str | None = None


class ProximityFact(BaseModel):
    project_id: str
    poi_type: Literal["hospital", "school", "park", "mall", "unknown"] = "unknown"
    poi_name: str | None = None
    proximity_text: str = ""
    distance_text: str | None = None
    travel_mode: Literal["walk", "drive", "unspecified"] = "unspecified"
    evidence_source: str = "unknown"
    semantic_tags: list[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    goal: str | None = None
    semantic_focus: list[str] = Field(default_factory=list)
    persona_hint: list[str] = Field(default_factory=list)
    poi_types: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)


class ProjectGroundedRetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    retrieval_intent: RetrievalIntent | str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query_not_blank_for_grounded(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class ProjectGroundedRetrieveResponse(BaseModel):
    route: Literal["project_grounded"] = "project_grounded"
    project_cards: list[ProjectCard]
    trait_tags: list[TraitTag]
    proximity_facts: list[ProximityFact] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk]
    confidence: float | None = None
    low_confidence: bool = False


class HealthResponse(BaseModel):
    status: str
    collection: str
