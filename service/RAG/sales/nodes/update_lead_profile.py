"""Node 4: Merge extracted slots into lead profile and compute missing_slots."""

import logging
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from models.sales_state import BUDGET_SLOT_ALIASES, REQUIRED_SLOTS_FOR_PITCH

log = logging.getLogger("rag-service")

_EMPTY_VALUES = {None, "khong_ro", ""}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", "khong_ro"):
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def update_lead_profile(state: SalesAgentState) -> Dict[str, Any]:
    profile: Dict[str, Any] = dict(state.get("lead_profile") or {})
    slots: Dict[str, Any] = state.get("extracted_slots") or {}

    for key, value in slots.items():
        if _is_empty(value):
            continue
        # Merge list fields
        if isinstance(value, list) and isinstance(profile.get(key), list):
            existing: list = profile[key]
            for item in value:
                if item not in existing:
                    existing.append(item)
            profile[key] = existing
        else:
            profile[key] = value

    # Lead temperature should reflect practical sales qualification first.
    purpose_known = not _is_empty(profile.get("purpose"))
    property_type_known = not _is_empty(profile.get("property_type"))
    location_known = not _is_empty(profile.get("location_preference"))
    budget_known = any(not _is_empty(profile.get(slot)) for slot in BUDGET_SLOT_ALIASES)
    family_known = not _is_empty(profile.get("family_member_count"))
    children_known = not _is_empty(profile.get("children_count"))

    core_filled_count = sum([purpose_known, property_type_known, location_known, budget_known])
    enrich_filled_count = sum([family_known, children_known])
    if core_filled_count >= 4:
        profile["lead_temperature"] = "hot"
    elif core_filled_count >= 3 or (core_filled_count >= 2 and enrich_filled_count >= 1):
        profile["lead_temperature"] = "warm"
    else:
        profile["lead_temperature"] = profile.get("lead_temperature", "cold")

    # Compute missing required slots
    missing: List[str] = []
    for slot in REQUIRED_SLOTS_FOR_PITCH:
        if _is_empty(profile.get(slot)):
            missing.append(slot)
    if all(_is_empty(profile.get(slot)) for slot in BUDGET_SLOT_ALIASES):
        missing.append("budget")

    log.info(
        "update_lead_profile: temp=%s missing=%s",
        profile.get("lead_temperature"),
        missing,
    )

    return {
        "lead_profile": profile,
        "missing_slots": missing,
    }
