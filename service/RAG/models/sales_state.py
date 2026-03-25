from enum import Enum


class SalesState(str, Enum):
    GREETING = "greeting"
    QUALIFICATION = "qualification"
    NEED_DISCOVERY = "need_discovery"
    BUDGET_ALIGNMENT = "budget_alignment"
    PRODUCT_MATCHING = "product_matching"
    COMPARISON = "comparison"
    OBJECTION_HANDLING = "objection_handling"
    BUY_SIGNAL = "buy_signal"
    CLOSING_NEXT_STEP = "closing_next_step"
    FOLLOW_UP = "follow_up"
    OUT_OF_SCOPE = "out_of_scope"


STATES_NEEDING_RETRIEVAL = {
    SalesState.PRODUCT_MATCHING,
    SalesState.COMPARISON,
    SalesState.OBJECTION_HANDLING,
    SalesState.CLOSING_NEXT_STEP,
}

REQUIRED_SLOTS_FOR_PITCH = [
    "purpose",
    "property_type",
    "location_preference",
]

BUDGET_SLOT_ALIASES = [
    "budget_text",
    "budget_min",
    "budget_max",
]
