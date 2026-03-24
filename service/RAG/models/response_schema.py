from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class SalesEventLog(BaseModel):
    session_id: str
    event_type: str
    event_data: Dict[str, Any]
    sales_state: str
    timestamp: Optional[str] = None


class RecommendationItem(BaseModel):
    project_id: str
    project_name: str
    reason: str
    cta: Optional[str] = None


class SalesRecommendationsResponse(BaseModel):
    session_id: str
    recommendations: List[RecommendationItem]
    sales_state: str
