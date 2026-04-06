from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_base.evaluator import (
    evaluate_kb_candidates,
    is_project_specific_query,
    is_realtime_query,
    is_short_ambiguous_query,
)


def test_is_realtime_query_detects_live_intents():
    assert is_realtime_query("gia vang hom nay bao nhieu")
    assert is_realtime_query("thoi tiet ha noi hien tai")
    assert not is_realtime_query("phap ly du an noble place")


def test_is_project_specific_query_detects_internal_context():
    assert is_project_specific_query("bang gia du an Noble Place Tay Thang Long")
    assert not is_project_specific_query("ty gia usd hom nay")


def test_is_short_ambiguous_query_prefers_chat_first():
    assert is_short_ambiguous_query("gia?")
    assert is_short_ambiguous_query("which one is better?")
    assert not is_short_ambiguous_query("phap ly du an noble place")


def test_evaluate_kb_candidates_prefers_search_for_realtime():
    decision = evaluate_kb_candidates(
        "gia vang hom nay",
        [{"score": 0.91, "source": "doc", "document_id": "1", "metadata": {}}],
        threshold=0.5,
    )
    assert decision["should_search"] is True
    assert decision["decision_reason"] == "realtime_query"


def test_evaluate_kb_candidates_prefers_kb_for_project_specific_even_if_low_score():
    decision = evaluate_kb_candidates(
        "phap ly du an noble place",
        [{"score": 0.22, "source": "doc", "document_id": "1", "metadata": {}}],
        threshold=0.5,
    )
    assert decision["should_search"] is False
    assert decision["decision_reason"] == "project_specific_kb_first"


def test_evaluate_kb_candidates_triggers_low_confidence_search():
    decision = evaluate_kb_candidates(
        "so sanh thi truong biet thu ha noi",
        [{"score": 0.21, "source": "doc", "document_id": "1", "metadata": {}}],
        threshold=0.5,
    )
    assert decision["should_search"] is True
    assert decision["decision_reason"] == "low_kb_confidence"
