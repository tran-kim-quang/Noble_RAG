from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class StorageType(str, Enum):
    GRAPH = "graph"
    VECTOR = "vector"
    BOTH = "both"


# ── Document endpoints ─────────────────────────────────────────────────────
class UploadDocumentRequest(BaseModel):
    storage_type: StorageType = StorageType.VECTOR
    metadata: Optional[dict] = None


class UploadDocumentResponse(BaseModel):
    status: str
    document_id: str
    chunks_count: int
    storage_type: str
    message: str


class KnowledgeIngestResponse(BaseModel):
    document_id: str
    status: str
    chunks_created: int
    collection: str


class DocumentRecord(BaseModel):
    id: str
    status: Optional[str] = None
    file_path: Optional[str] = None
    content_length: Optional[int] = None
    chunks_count: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[dict] = None


# ── Query endpoint ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = ""
    message: Optional[str] = None
    top_k: int = 8
    return_structured_output: bool = False
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    context: List[str]
    tokens_used: Optional[dict] = None
    model: str


# ── Sales endpoints ────────────────────────────────────────────────────────
class SalesChatRequest(BaseModel):
    session_id: str
    message: str
    raw_transcript: Optional[str] = None


class SalesChatResponse(BaseModel):
    session_id: str
    response: str
    route_category: Optional[str] = None
    sales_state: Optional[str] = None
    lead_profile: Optional[Dict[str, Any]] = None
    missing_slots: Optional[List[str]] = None


class LeadUpdateRequest(BaseModel):
    updates: Dict[str, Any]


class VisionIdentifyAndContextResponse(BaseModel):
    matched: bool
    customer_id: str
    session_id: str
    vision_context_saved: bool
    vision_summary: Dict[str, Any]


class SalesChatV1Request(BaseModel):
    session_id: str
    customer_id: str
    message: str
    channel: Optional[str] = None
    stream: bool = False


class SalesChatV1Response(BaseModel):
    response: str
    intent: Optional[str] = None
    sales_stage: Optional[str] = None
    missing_slots: List[str] = Field(default_factory=list)
    context_used: bool = True


class SessionOpenRequest(BaseModel):
    customer_id: str
    session_id: Optional[str] = None
    channel: Optional[str] = None
    source: Optional[str] = "machine_b"
    allow_resume: bool = True
    customer_profile: Dict[str, Any] = Field(default_factory=dict)
    vision_summary: Optional[Dict[str, Any]] = None


class SessionOpenResponse(BaseModel):
    session_id: str
    customer_id: str
    reused_session: bool
    context_ready: bool
    context_used: bool
