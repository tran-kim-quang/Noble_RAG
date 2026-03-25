"""Render deterministic responses for scripted ask/close steps."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from sales.response_templates import render_template_response

log = logging.getLogger("rag-service")


def render_response_from_template(state: SalesAgentState) -> Dict[str, Any]:
    action = state.get("response_action") or ""
    text = render_template_response(action, state).strip()
    log.info("render_response_from_template: action=%s response_len=%d", action, len(text))
    return {
        "draft_response": text,
        "final_response": text,
    }
