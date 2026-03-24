"""Node 5: Apply state machine to determine next_sales_state."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from sales.state_machine import resolve_next_state

log = logging.getLogger("rag-service")


def resolve_sales_state_node(state: SalesAgentState) -> Dict[str, Any]:
    next_state = resolve_next_state(state)
    log.info(
        "resolve_sales_state: %s → %s",
        state.get("current_sales_state"),
        next_state,
    )
    return {"next_sales_state": next_state}
