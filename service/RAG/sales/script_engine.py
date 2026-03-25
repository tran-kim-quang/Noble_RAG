"""Deterministic script-step and response-action resolution."""

from typing import Any, Dict, List, Tuple

from models.sales_state import ResponseAction, ScriptStep

_DISCOVERY_SLOT_ORDER = [
    ("family_member_count", ScriptStep.S2_ASK_FAMILY_SIZE, ResponseAction.ASK_FAMILY_SIZE),
    ("children_count", ScriptStep.S3_ASK_CHILDREN, ResponseAction.ASK_CHILDREN),
    ("purpose", ScriptStep.S4_ASK_PURPOSE, ResponseAction.ASK_PURPOSE),
    ("location_preference", ScriptStep.S5_ASK_LOCATION, ResponseAction.ASK_LOCATION),
]


def resolve_script_step_and_action(state: Dict[str, Any]) -> Tuple[str, str]:
    next_state = state.get("next_sales_state") or "need_discovery"
    current_step = state.get("current_script_step") or ScriptStep.S1_OPENING.value
    missing_slots: List[str] = state.get("missing_slots") or []
    intent = state.get("detected_intent") or ""
    current_state = state.get("current_sales_state") or "greeting"

    if next_state == "out_of_scope":
        return ScriptStep.OUT_OF_SCOPE.value, ResponseAction.REDIRECT_OUT_OF_SCOPE.value

    if next_state == "greeting":
        return ScriptStep.S1_OPENING.value, ResponseAction.ASK_OPENING.value

    if next_state == "need_discovery":
        for slot_name, step, action in _DISCOVERY_SLOT_ORDER:
            if slot_name in missing_slots:
                return step.value, action.value
        return ScriptStep.S5_ASK_LOCATION.value, ResponseAction.ASK_LOCATION.value

    if next_state == "objection_handling":
        return ScriptStep.O1_HANDLE_OBJECTION.value, ResponseAction.HANDLE_OBJECTION.value

    if next_state == "closing_next_step":
        return ScriptStep.C1_SOFT_CLOSE.value, ResponseAction.SOFT_CLOSE.value

    if next_state == "follow_up":
        return ScriptStep.F1_FOLLOWUP_CLOSEOUT.value, ResponseAction.FOLLOWUP_CLOSEOUT.value

    if next_state == "comparison":
        return ScriptStep.M2_EXPLAIN_OPTION_DETAIL.value, ResponseAction.EXPLAIN_OPTION_DETAIL.value

    if next_state == "product_matching":
        if current_state == "product_matching" or current_step == ScriptStep.M1_MATCH_OPTIONS.value:
            if intent in ("follow_up", "buy_signal"):
                return ScriptStep.M3_INTEREST_CHECK.value, ResponseAction.CHECK_INTEREST.value
            return ScriptStep.M2_EXPLAIN_OPTION_DETAIL.value, ResponseAction.EXPLAIN_OPTION_DETAIL.value
        return ScriptStep.M1_MATCH_OPTIONS.value, ResponseAction.MATCH_OPTIONS.value

    return ScriptStep.S1_OPENING.value, ResponseAction.ASK_OPENING.value
