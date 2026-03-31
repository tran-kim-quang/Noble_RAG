from enum import Enum


class SalesState(str, Enum):
    GREETING = "greeting"
    QUALIFICATION = "qualification"
    NEED_DISCOVERY = "need_discovery"
    BUDGET_ALIGNMENT = "budget_alignment"
    PROJECT_QA = "project_qa"
    PRODUCT_MATCHING = "product_matching"
    COMPARISON = "comparison"
    OBJECTION_HANDLING = "objection_handling"
    BUY_SIGNAL = "buy_signal"
    CLOSING_NEXT_STEP = "closing_next_step"
    FOLLOW_UP = "follow_up"
    OUT_OF_SCOPE = "out_of_scope"


class ScriptStep(str, Enum):
    S1_OPENING = "S1_opening"
    S2_ASK_FAMILY_SIZE = "S2_ask_family_size"
    S3_ASK_CHILDREN = "S3_ask_children"
    S4_ASK_PURPOSE = "S4_ask_purpose"
    S5_ASK_LOCATION = "S5_ask_location"
    Q1_PROJECT_QA = "Q1_project_qa"
    M1_MATCH_OPTIONS = "M1_match_options"
    M2_EXPLAIN_OPTION_DETAIL = "M2_explain_option_detail"
    M3_INTEREST_CHECK = "M3_interest_check"
    O1_HANDLE_OBJECTION = "O1_handle_objection"
    C1_SOFT_CLOSE = "C1_soft_close"
    F1_FOLLOWUP_CLOSEOUT = "F1_followup_closeout"
    OUT_OF_SCOPE = "out_of_scope"


class ResponseAction(str, Enum):
    ASK_OPENING = "ask_opening"
    ASK_FAMILY_SIZE = "ask_family_size"
    ASK_CHILDREN = "ask_children"
    ASK_PURPOSE = "ask_purpose"
    ASK_LOCATION = "ask_location"
    CATALOG_OVERVIEW = "catalog_overview"
    PROJECT_QA = "project_qa"
    MATCH_OPTIONS = "match_options"
    EXPLAIN_OPTION_DETAIL = "explain_option_detail"
    CHECK_INTEREST = "check_interest"
    HANDLE_OBJECTION = "handle_objection"
    SOFT_CLOSE = "soft_close"
    FOLLOWUP_CLOSEOUT = "followup_closeout"
    REDIRECT_OUT_OF_SCOPE = "redirect_out_of_scope"


STATES_NEEDING_RETRIEVAL = {
    SalesState.PROJECT_QA,
    SalesState.PRODUCT_MATCHING,
    SalesState.COMPARISON,
    SalesState.OBJECTION_HANDLING,
    SalesState.CLOSING_NEXT_STEP,
}

TEMPLATE_ACTIONS = {
    ResponseAction.ASK_OPENING,
    ResponseAction.ASK_FAMILY_SIZE,
    ResponseAction.ASK_CHILDREN,
    ResponseAction.ASK_PURPOSE,
    ResponseAction.ASK_LOCATION,
    ResponseAction.CHECK_INTEREST,
    ResponseAction.SOFT_CLOSE,
    ResponseAction.FOLLOWUP_CLOSEOUT,
    ResponseAction.REDIRECT_OUT_OF_SCOPE,
}

REQUIRED_SLOTS_FOR_PITCH = [
    "family_member_count",
    "children_count",
    "purpose",
    "location_preference",
]
