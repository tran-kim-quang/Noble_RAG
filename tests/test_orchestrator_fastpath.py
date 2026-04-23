from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator_service.app import _build_quick_intent_response
from orchestrator_service.app import analyze_turn
from orchestrator_service.config import Settings
from orchestrator_service.schemas import LeadState


def test_quick_intent_response_only_keeps_greeting():
    greeting_reply = _build_quick_intent_response(
        message="Xin chao",
        ask_policy="allow_question",
    )

    assert greeting_reply is not None
    assert _build_quick_intent_response(message="Toi mua de o lau dai", ask_policy="allow_question") is None
    assert _build_quick_intent_response(message="Cam on", ask_policy="allow_question") is None
    assert _build_quick_intent_response(message="ok", ask_policy="allow_question") is None


def test_analyze_turn_keeps_only_greeting_as_deterministic_fastpath(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_DECIDER_ENABLED", "false")
    settings = Settings()

    greeting_analysis = analyze_turn(
        message="Xin chao",
        lead_state=LeadState(),
        recent_history=[],
        settings=settings,
    )
    long_term_analysis = analyze_turn(
        message="Toi dang tim noi o lau dai cho gia dinh",
        lead_state=LeadState(),
        recent_history=[],
        settings=settings,
    )

    assert greeting_analysis.route_source == "deterministic_fastpath"
    assert greeting_analysis.decision_reason == "deterministic_greeting_fastpath"
    assert long_term_analysis.route_source == "fallback"
    assert long_term_analysis.decision_reason != "deterministic_long_term_living_fastpath"
