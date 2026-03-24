"""Conditional edge functions for the LangGraph sales graph."""

from typing import Any, Dict

from models.sales_state import STATES_NEEDING_RETRIEVAL, SalesState


def route_after_state_resolution(state: Dict[str, Any]) -> str:
    """Return the next node name after resolve_sales_state."""
    next_state_str: str = state.get("next_sales_state") or "need_discovery"
    try:
        next_state = SalesState(next_state_str)
    except ValueError:
        next_state = SalesState.NEED_DISCOVERY

    if next_state in STATES_NEEDING_RETRIEVAL:
        return "retrieve_context"
    return "build_response"
