from typing import TypedDict, List, Dict, Optional, Any


class SalesAgentState(TypedDict, total=False):
    # Input
    session_id: str
    user_text: str
    raw_transcript: Optional[str]

    # Memory
    chat_history: List[Dict[str, str]]
    lead_profile: Dict[str, Any]
    session_context: Dict[str, Any]

    # Intent classification
    detected_intent: Optional[str]
    objection_type: Optional[str]
    buy_signal: Optional[bool]

    # Intermediate slot extraction
    extracted_slots: Dict[str, Any]

    # State machine
    current_sales_state: Optional[str]
    next_sales_state: Optional[str]
    missing_slots: List[str]

    # Retrieval
    retrieved_context: List[Dict[str, Any]]

    # Response
    draft_response: Optional[str]
    final_response: Optional[str]

    # Control
    should_persist: bool
    errors: List[str]
