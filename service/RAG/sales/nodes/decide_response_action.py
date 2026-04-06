"""Normalize response_action for the current scripted step."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState

log = logging.getLogger("rag-service")


def _has_kb_evidence(state: SalesAgentState) -> bool:
    items = state.get("knowledge_payload", {}).get("kb_evidence") if isinstance(state.get("knowledge_payload"), dict) else None
    return bool(items)


def _has_search_evidence(state: SalesAgentState) -> bool:
    items = state.get("knowledge_payload", {}).get("search_evidence") if isinstance(state.get("knowledge_payload"), dict) else None
    return bool(items)


def _is_unresolved(state: SalesAgentState) -> bool:
    payload = state.get("knowledge_payload")
    if isinstance(payload, dict):
        return bool(payload.get("unresolved"))
    return False


def _derive_response_action(state: SalesAgentState) -> str:
    next_state = str(state.get("next_sales_state") or "need_discovery")
    has_kb = _has_kb_evidence(state)
    has_search = _has_search_evidence(state)
    unresolved = _is_unresolved(state)

    if unresolved:
        return "ask_follow_up"
    if next_state in {"greeting", "need_discovery"}:
        return "ask_follow_up"
    if next_state == "product_matching":
        return "recommend_products"
    if next_state == "comparison":
        if has_kb and has_search:
            return "answer_with_kb_and_search"
        return "compare_options"
    if next_state == "project_qa":
        if has_kb and has_search:
            return "answer_with_kb_and_search"
        if has_kb:
            return "answer_with_kb"
        if has_search:
            return "answer_with_search"
        return "ask_follow_up"
    if next_state == "objection_handling":
        return "handle_objection"
    if next_state == "closing_next_step":
        return "invite_next_step"
    return "ask_follow_up"


def decide_response_action_node(state: SalesAgentState) -> Dict[str, Any]:
    action = str(state.get("response_action") or "").strip() or _derive_response_action(state)
    step = state.get("next_script_step")
    log.info("decide_response_action: step=%s action=%s", step, action)
    return {"response_action": action}
