from pydantic import BaseModel, Field
from typing import Optional, List


class LeadProfile(BaseModel):
    lead_id: str
    name: Optional[str] = None
    purpose: Optional[str] = "khong_ro"
    property_type: Optional[str] = "khong_ro"
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    budget_text: Optional[str] = None
    location_preference: List[str] = Field(default_factory=list)
    region_flexibility: Optional[bool] = None
    financing_need: Optional[bool] = None
    financing_ratio: Optional[float] = None
    timeline: Optional[str] = "khong_ro"
    legal_sensitivity: Optional[str] = None
    risk_appetite: Optional[str] = None
    key_needs: List[str] = Field(default_factory=list)
    key_concerns: List[str] = Field(default_factory=list)
    objections: List[str] = Field(default_factory=list)
    recommended_projects: List[str] = Field(default_factory=list)
    shortlisted_projects: List[str] = Field(default_factory=list)
    rejected_projects: List[str] = Field(default_factory=list)
    current_state: str = "greeting"
    lead_temperature: Optional[str] = "cold"
    preferred_contact_channel: Optional[str] = "chat"
    contact_phone: Optional[str] = None
    last_user_intent: Optional[str] = None
    last_next_action: Optional[str] = None


class SessionContext(BaseModel):
    session_id: str
    current_state: str = "greeting"
    previous_state: Optional[str] = None
    last_agent_action: Optional[str] = None
    last_retrieved_context_ids: List[str] = Field(default_factory=list)
    conversation_turn_count: int = 0
