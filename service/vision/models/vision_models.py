from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VisionCustomerContext(BaseModel):
    lead_profile: Dict[str, Any] = Field(default_factory=dict)
    recent_chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    recent_session_context: Dict[str, Any] = Field(default_factory=dict)
    purchase_history: List[Dict[str, Any]] = Field(default_factory=list)


class VisionIdentifyResponse(BaseModel):
    session_id: str
    customer_id: str
    is_existing_customer: bool
    matched: bool = False
    decision: str = "unknown"
    gender_estimate: str = "unknown"
    age_group_estimate: str = "unknown"
    match_score: Optional[float] = None
    vision_summary: Dict[str, Any] = Field(default_factory=dict)
    customer_context: VisionCustomerContext


class VisionSessionResponse(BaseModel):
    session_id: str
    customer_id: str
    source: str
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    customer_code: Optional[str] = None
    customer_metadata: Dict[str, Any] = Field(default_factory=dict)
    customer_context: VisionCustomerContext = Field(default_factory=VisionCustomerContext)
    known_session_count: int = 0


class VisionCustomerResponse(BaseModel):
    customer_id: str
    customer_code: str
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    customer_context: VisionCustomerContext


class VisionEnrollResponse(BaseModel):
    customer_id: str
    face_detected: bool
    face_count: int
    best_face_saved: bool
    embedding_saved: bool
    image_asset_id: str
    face_embedding_id: str


class VisionIdentifyAndContextResponse(BaseModel):
    matched: bool
    customer_id: str
    session_id: str
    vision_context_saved: bool
    vision_summary: Dict[str, Any] = Field(default_factory=dict)
    match_score: Optional[float] = None
