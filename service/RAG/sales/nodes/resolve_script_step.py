"""Resolve the next script step based on state and lead progress."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from sales.script_engine import resolve_script_step_and_action

log = logging.getLogger("rag-service")


def resolve_script_step_node(state: SalesAgentState) -> Dict[str, Any]:
    next_step, action = resolve_script_step_and_action(state)
    log.info(
        "resolve_script_step: state=%s step=%s action=%s",
        state.get("next_sales_state"),
        next_step,
        action,
    )
    return {
        "next_script_step": next_step,
        "response_action": action,
    }
