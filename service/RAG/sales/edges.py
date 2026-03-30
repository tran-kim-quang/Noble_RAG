"""Conditional edge functions for the LangGraph sales graph."""

from typing import Any, Dict

from models.sales_state import ResponseAction, TEMPLATE_ACTIONS


_DISCOVERY_RETRIEVAL_INTENTS = {"ask_recommendation", "project_info", "comparison", "objection"}


def route_after_fast_parse(state: Dict[str, Any]) -> str:
    """Route to fallback reasoning only when fast parse is uncertain."""
    lane = state.get("fast_lane") or "lane_c"
    confidence = float(state.get("fast_path_confidence") or 0.0)
    if lane != "lane_c":
        return "update_lead_profile"
    if confidence < 0.55:
        return "classify_and_extract"
    return "update_lead_profile"


def route_after_action_resolution(state: Dict[str, Any]) -> str:
    """Route to template rendering or grounded generation by response action."""
    action_str = state.get("response_action") or ResponseAction.ASK_OPENING.value
    try:
        action = ResponseAction(action_str)
    except ValueError:
        action = ResponseAction.ASK_OPENING

    if (
        state.get("next_sales_state") == "need_discovery"
        and action in {
            ResponseAction.ASK_FAMILY_SIZE,
            ResponseAction.ASK_CHILDREN,
            ResponseAction.ASK_PURPOSE,
            ResponseAction.ASK_LOCATION,
        }
        and (state.get("detected_intent") or "") in _DISCOVERY_RETRIEVAL_INTENTS
    ):
        return "retrieve_context"

    if action in TEMPLATE_ACTIONS:
        return "render_response_from_template"
    return "retrieve_context"


def route_after_retrieve_context(state: Dict[str, Any]) -> str:
    if (state.get("fast_lane") or "") == "lane_b":
        return "fast_ack_response"
    return "build_response"
