"""Sales state machine.

Rules are implemented in code (not delegated to LLM) so that transitions
are deterministic and auditable.
"""

from typing import Any, Dict, List


def resolve_next_state(state: Dict[str, Any]) -> str:
    intent: str = state.get("detected_intent") or ""
    turn_role: str = state.get("turn_role") or ""
    retrieval_goal: str = state.get("retrieval_goal") or "none"
    missing: List[str] = state.get("missing_slots") or []
    extracted_slots: Dict[str, Any] = state.get("extracted_slots") or {}
    buy_signal: bool = bool(state.get("buy_signal", False))
    lead_profile: Dict[str, Any] = state.get("lead_profile") or {}
    current_state: str = state.get("current_sales_state") or "greeting"
    resolved_project_name = state.get("resolved_project_name")
    history = state.get("chat_history") or []
    is_first_turn = len(history) == 0

    # Hard exits
    if intent == "out_of_scope":
        return "out_of_scope"

    # Keep the opening greeting only on the actual first user turn.
    if current_state == "greeting" and is_first_turn and not lead_profile.get("purpose"):
        if not extracted_slots and (intent in ("greeting", "", "other") or not intent):
            return "greeting"

    if extracted_slots and missing and current_state in {"greeting", "need_discovery", "product_matching", "qualification"}:
        if intent in {"other", "follow_up", "ask_recommendation", "greeting"}:
            return "need_discovery"

    # If user is supplying discovery info while we are blocked in project QA
    # without a concrete project name, leave project_qa and continue qualification.
    if current_state == "project_qa" and not resolved_project_name and extracted_slots:
        if missing:
            return "need_discovery"
        return "product_matching"

    # Intent-based transitions
    if intent == "project_info":
        return "project_qa"
    if intent == "comparison":
        return "comparison"
    if intent == "objection":
        return "objection_handling"
    if buy_signal or intent == "buy_signal":
        return "closing_next_step"
    if current_state == "objection_handling" and intent == "ask_recommendation":
        if missing:
            return "need_discovery"
        return "product_matching"
    if retrieval_goal == "project_qa":
        return "project_qa"
    if retrieval_goal == "comparison":
        return "comparison"
    if retrieval_goal == "objection_support":
        return "objection_handling"
    if retrieval_goal == "closing_next_step":
        return "closing_next_step"
    if retrieval_goal == "shortlist" and not missing:
        return "product_matching"
    if intent == "follow_up":
        if current_state == "greeting":
            return "need_discovery"
        if current_state == "need_discovery" and not missing:
            return "product_matching"
        if current_state == "project_qa" and extracted_slots:
            if missing:
                return "need_discovery"
            return "product_matching"
        return current_state

    # Missing required slots only force discovery when the user is actually asking for recommendation/discovery.
    if missing and (
        intent in {"ask_recommendation", "greeting"}
        or (current_state == "need_discovery" and turn_role == "answer_previous_question")
    ):
        return "need_discovery"

    if intent == "ask_recommendation":
        if missing:
            return "need_discovery"
        return "product_matching"

    if current_state in ("greeting", "need_discovery", "qualification"):
        return "product_matching"

    # Safe default: keep current non-greeting flow, otherwise continue discovery.
    if current_state in ("product_matching", "comparison"):
        return current_state
    if current_state in ("objection_handling", "closing_next_step"):
        return current_state
    return "need_discovery"
