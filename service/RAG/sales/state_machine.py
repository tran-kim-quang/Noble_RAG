"""Sales state machine.

Rules are implemented in code (not delegated to LLM) so that transitions
are deterministic and auditable.
"""

from typing import Any, Dict, List


def resolve_next_state(state: Dict[str, Any]) -> str:
    intent: str = state.get("detected_intent") or ""
    missing: List[str] = state.get("missing_slots") or []
    buy_signal: bool = bool(state.get("buy_signal", False))
    lead_profile: Dict[str, Any] = state.get("lead_profile") or {}
    current_state: str = state.get("current_sales_state") or "greeting"

    # Hard exits
    if intent == "out_of_scope":
        return "out_of_scope"

    # First turn → greet
    if current_state == "greeting" and not lead_profile.get("purpose"):
        if intent in ("greeting", "", "other") or not intent:
            return "greeting"

    # Intent-based transitions
    if intent == "comparison":
        return "comparison"
    if intent == "objection":
        return "objection_handling"
    if buy_signal or intent == "buy_signal":
        return "closing_next_step"
    if intent == "follow_up":
        return current_state if current_state != "greeting" else "need_discovery"

    # Missing required slots → discover
    if missing:
        return "need_discovery"

    if intent == "ask_recommendation":
        return "product_matching"

    # Safe default: keep current non-greeting flow, otherwise continue discovery.
    if current_state in ("product_matching", "comparison"):
        return current_state
    if current_state in ("objection_handling", "closing_next_step"):
        return current_state
    return "need_discovery"
