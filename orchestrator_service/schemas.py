from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic import field_validator


class QueryRequest(BaseModel):
    message: str = Field(min_length=1)
    need_retrieval: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be blank")
        return cleaned


class ContextItem(BaseModel):
    text: str
    score: float
    source: str
    doc_id: str


class RetrievalPayload(BaseModel):
    results: list[ContextItem] = []
    confidence: float | None = None
    low_confidence: bool = False


class QueryResponse(BaseModel):
    route: Literal[
        "no_retrieval_needed",
        "retrieval_confident",
        "retrieval_low_confidence",
    ]
    action: Literal[
        "proceed_without_retrieval",
        "use_retrieved_context",
        "ask_clarifying_question",
    ]
    answer: str
    decision_reason: str
    retrieval: RetrievalPayload | None = None
