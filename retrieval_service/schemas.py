from __future__ import annotations

from typing import Any

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


class HealthResponse(BaseModel):
    status: str
    collection: str
