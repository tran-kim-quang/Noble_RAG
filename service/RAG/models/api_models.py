from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class StorageType(str, Enum):
    GRAPH = "graph"
    VECTOR = "vector"
    BOTH = "both"


# ── Document endpoints ─────────────────────────────────────────────────────
class UploadDocumentRequest(BaseModel):
    storage_type: StorageType = StorageType.GRAPH
    metadata: Optional[dict] = None


class UploadDocumentResponse(BaseModel):
    status: str
    document_id: str
    chunks_count: int
    storage_type: str
    message: str


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
    sales_state: Optional[str] = None
    lead_profile: Optional[Dict[str, Any]] = None
    missing_slots: Optional[List[str]] = None


class LeadUpdateRequest(BaseModel):
    updates: Dict[str, Any]


class WatcherSessionPushRequest(BaseModel):
    session_id: str
    enabled: bool = True
    watcher_url: Optional[str] = None


class WatcherSessionPushResponse(BaseModel):
    ok: bool
    watcher_url: str
    session_id: str
    enabled: bool
    watcher_response: Dict[str, Any]
