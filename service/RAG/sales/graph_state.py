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

    # Phase 5 knowledge payload context
    knowledge_payload: Optional[Dict[str, Any]]
    knowledge_decision_reason: Optional[str]
    kb_top_score: Optional[float]
    search_used: Optional[bool]

    # Intent classification
    detected_intent: Optional[str]
    objection_type: Optional[str]
    buy_signal: Optional[bool]
    turn_role: Optional[str]
    should_retrieve: Optional[bool]
    retrieval_goal: Optional[str]

    # Intermediate slot extraction
    extracted_slots: Dict[str, Any]
    fast_path_confidence: Optional[float]
    fast_lane: Optional[str]
    retrieval_mode: Optional[str]
    fast_ack_response: Optional[str]
    semantic_parse_done: Optional[bool]
    semantic_parse_confidence: Optional[float]

    # State machine
    current_sales_state: Optional[str]
    next_sales_state: Optional[str]
    current_script_step: Optional[str]
    next_script_step: Optional[str]
    response_action: Optional[str]
    missing_slots: List[str]

    # Retrieval
    retrieved_candidates: List[Dict[str, Any]]
    retrieved_context: List[Dict[str, Any]]
    catalog_projects: List[Dict[str, Any]]
    project_facts: List[Dict[str, Any]]
    has_retrieved_context: bool
    project_qa_blocked: bool
    resolved_project_name: Optional[str]
    project_resolution_source: Optional[str]
    project_resolution_confidence: Optional[float]
    candidate_count_before_llm: Optional[int]
    llm_disambiguation_called: Optional[bool]
    retrieval_skipped_due_to_unresolved_entity: Optional[bool]
    full_retrieval_used: Optional[bool]

    # Response
    draft_response: Optional[str]
    final_response: Optional[str]

    # Control
    should_persist: bool
    errors: List[str]
