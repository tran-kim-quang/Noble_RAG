"""Conditional edge functions for the LangGraph sales graph."""

from typing import Any, Dict

from models.sales_state import ResponseAction, TEMPLATE_ACTIONS


def route_after_action_resolution(state: Dict[str, Any]) -> str:
    """Route to template rendering or grounded generation by response action."""
    action_str = state.get("response_action") or ResponseAction.ASK_OPENING.value
    try:
        action = ResponseAction(action_str)
    except ValueError:
        action = ResponseAction.ASK_OPENING

    if action in TEMPLATE_ACTIONS:
        return "render_response_from_template"
    return "retrieve_context"
