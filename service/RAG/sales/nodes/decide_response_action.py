"""Normalize response_action for the current scripted step."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState

log = logging.getLogger("rag-service")


def decide_response_action_node(state: SalesAgentState) -> Dict[str, Any]:
    action = state.get("response_action")
    step = state.get("next_script_step")
    log.info("decide_response_action: step=%s action=%s", step, action)
    return {"response_action": action}
