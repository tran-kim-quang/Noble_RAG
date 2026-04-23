from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator_service.app import _analyze_turn_with_model
from orchestrator_service.config import Settings
from orchestrator_service.schemas import LeadState


def test_minimal_decider_payload_still_builds_turn_analysis(monkeypatch):
    import orchestrator_service.app as app_module

    monkeypatch.setattr(
        app_module,
        "_call_model_generate",
        lambda **kwargs: {
            "query_type": "project_matching",
            "start_route": "consult_discovery",
            "should_route_project": True,
            "project_query_hint": "can nao gan truong hoc",
        },
    )

    settings = Settings(
        decider_api_format="openai",
        decider_api_url="https://api.moonshot.ai/v1/chat/completions",
        decider_api_key="kimi-key",
        decider_model="kimi-k2.5",
    )
    analysis = _analyze_turn_with_model(
        message="Co can nao gan truong hoc khong?",
        lead_state=LeadState(),
        recent_history=[],
        settings=settings,
    )

    assert analysis.route == "consult_discovery"
    assert analysis.query_type == "project_matching"
    assert analysis.routing_signal.should_route_project is True
    assert analysis.routing_signal.project_query_hint == "can nao gan truong hoc"
    assert analysis.retrieval_readiness == "soft_ready"
    assert analysis.need_update.summary_delta
    assert analysis.consult_reply == ""
