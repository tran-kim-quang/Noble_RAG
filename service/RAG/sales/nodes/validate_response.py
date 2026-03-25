"""Basic validator to keep responses aligned with script action."""

import logging
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from sales.response_templates import render_template_response
from utils.text import split_into_sentences

log = logging.getLogger("rag-service")

_GREETING_MARKERS = ("xin chào", "chào anh/chị", "chào anh chị", "em là tư vấn viên")


def _contains_greeting(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _GREETING_MARKERS)


def _validate_matching_response(state: SalesAgentState, text: str) -> List[str]:
    errors: List[str] = []
    lower = text.lower()
    if _contains_greeting(text):
        errors.append("matching_should_not_greet")
    if "ngân sách" in lower or "budget" in lower:
        errors.append("matching_should_not_ask_budget")
    lead = state.get("lead_profile") or {}
    property_type = lead.get("property_type")
    if lead.get("purpose") == "mua_o" and property_type in (None, "", "khong_ro"):
        if "shophouse" in lower:
            errors.append("matching_should_not_push_shophouse")
    stripped = text.strip()
    if stripped.endswith(("1.", "1)", "1:", "-", ":")):
        errors.append("matching_response_incomplete")
    return errors


def _fallback_from_retrieved_context(state: SalesAgentState) -> str:
    context = state.get("retrieved_context") or []
    for item in context:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        sentences = split_into_sentences(content)
        if sentences:
            return " ".join(sentences[:3]).strip()
    return (
        "Em đang rà lại dữ liệu dự án để gửi Anh/Chị phương án phù hợp nhất. "
        "Em sẽ gửi ngay bản tư vấn ngắn gọn, đúng nhu cầu của gia đình mình nhé."
    )


def validate_response_node(state: SalesAgentState) -> Dict[str, Any]:
    action = state.get("response_action") or ""
    final_response = (state.get("final_response") or state.get("draft_response") or "").strip()
    errors: List[str] = []

    if not final_response:
        errors.append("empty_response")

    if action.startswith("ask_") or action in {"soft_close", "followup_closeout", "redirect_out_of_scope", "check_interest"}:
        if _contains_greeting(final_response) and action != "ask_opening":
            errors.append("template_step_should_not_regreet")

    if action in {"match_options", "explain_option_detail"}:
        errors.extend(_validate_matching_response(state, final_response))

    if errors:
        log.warning("validate_response: action=%s errors=%s", action, errors)
        if action.startswith("ask_") or action in {"soft_close", "followup_closeout", "redirect_out_of_scope", "check_interest"}:
            fallback = render_template_response(action, state)
        elif "matching_should_not_push_shophouse" in errors:
            fallback = (
                "Em đang ưu tiên các phương án phù hợp để ở cho gia đình mình. "
                "Để tránh gợi ý lệch nhu cầu, em sẽ rà lại đúng các dự án phù hợp rồi gửi Anh/Chị ngay."
            )
        elif "matching_response_incomplete" in errors:
            fallback = _fallback_from_retrieved_context(state)
        else:
            fallback = (
                "Em đang rà lại thông tin để tư vấn đúng nhu cầu của Anh/Chị hơn. "
                "Em sẽ gửi lại phương án phù hợp nhất ngay nhé."
            )
        return {
            "draft_response": fallback,
            "final_response": fallback,
            "validation_errors": errors,
        }

    return {"validation_errors": []}
