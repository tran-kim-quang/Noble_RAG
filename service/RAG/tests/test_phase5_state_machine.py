from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sales.nodes.decide_response_action import decide_response_action_node
from sales.state_machine import resolve_next_state


def test_state_machine_moves_to_project_qa_when_project_info_and_kb_exists():
    state = {
        "detected_intent": "project_info",
        "current_sales_state": "need_discovery",
        "chat_history": [{"role": "user", "content": "phap ly du an"}],
        "missing_slots": ["purpose"],
        "extracted_slots": {},
        "retrieval_goal": "none",
        "turn_role": "user",
        "lead_profile": {},
        "knowledge_payload": {
            "kb_evidence": [{"content": "co thong tin"}],
            "search_evidence": [],
            "unresolved": False,
        },
    }
    assert resolve_next_state(state) == "project_qa"


def test_state_machine_moves_to_discovery_when_project_info_but_no_evidence():
    state = {
        "detected_intent": "project_info",
        "current_sales_state": "need_discovery",
        "chat_history": [{"role": "user", "content": "phap ly du an"}],
        "missing_slots": ["purpose"],
        "extracted_slots": {},
        "retrieval_goal": "none",
        "turn_role": "user",
        "lead_profile": {},
        "knowledge_payload": {
            "kb_evidence": [],
            "search_evidence": [],
            "unresolved": True,
        },
    }
    assert resolve_next_state(state) == "need_discovery"


def test_decide_response_action_uses_kb_and_search_when_both_available():
    output = decide_response_action_node(
        {
            "next_sales_state": "project_qa",
            "knowledge_payload": {
                "kb_evidence": [{"content": "kb"}],
                "search_evidence": [{"content": "search"}],
                "unresolved": False,
            },
        }
    )
    assert output["response_action"] == "answer_with_kb_and_search"


def test_decide_response_action_asks_follow_up_on_unresolved():
    output = decide_response_action_node(
        {
            "next_sales_state": "comparison",
            "knowledge_payload": {
                "kb_evidence": [],
                "search_evidence": [],
                "unresolved": True,
            },
        }
    )
    assert output["response_action"] == "ask_follow_up"
