from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ProductProject(BaseModel):
    project_id: str
    project_name: str
    city: str
    district: str
    ward: Optional[str] = None
    property_types: List[str] = Field(default_factory=list)
    target_personas: List[str] = Field(default_factory=list)
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    area_min: Optional[float] = None
    area_max: Optional[float] = None
    legal_status: Optional[str] = None
    handover_time: Optional[str] = None
    payment_policy: Optional[str] = None
    bank_support: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    fit_rules: List[str] = Field(default_factory=list)
    disqualify_rules: List[str] = Field(default_factory=list)
    objection_answers: Dict = Field(default_factory=dict)
    cta_recommendation: Optional[str] = None
