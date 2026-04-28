from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchestrator_service.app as app_module
from orchestrator_service.app import TurnAnalysis
from orchestrator_service.app import create_app
from orchestrator_service.schemas import NeedPainpointDelta
from orchestrator_service.schemas import RoutingSignal


def fake_turn_analyzer(message: str, lead_state, recent_history):
    _ = message
    _ = lead_state
    _ = recent_history
    return TurnAnalysis(
        route="consult_discovery",
        decision_reason="vision_test_consult",
        need_update=NeedPainpointDelta(),
        painpoint_update=NeedPainpointDelta(),
        routing_signal=RoutingSignal(should_route_project=False, project_query_hint=None, reason="vision_test_consult"),
        consult_reply="Em dang ho tro tu van.",
        query_type="advisory_strategy",
        retrieval_readiness="not_ready",
        sales_state_after_hint="exploring",
        conversation_goal_hint="build_trust",
    )


def fake_synthesize_assistant_reply(*args, **kwargs) -> str:
    _ = args
    _ = kwargs
    return "Em dang ho tro tu van."


PY313 = sys.version_info >= (3, 13)


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_query_with_vision_greets_recognized_customer(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MODEL_WARMUP_ENABLED", "false")
    monkeypatch.setattr(app_module, "synthesize_assistant_reply", fake_synthesize_assistant_reply)

    def fake_vision_identify(**kwargs):
        _ = kwargs
        return {
            "recognized": True,
            "name": "Linh",
            "age": 31,
            "gender": "female",
            "confidence": 0.93,
            "source": "face_db",
            "face_count": 1,
            "bbox": [10, 10, 100, 100],
            "reason": None,
        }

    client = TestClient(create_app(turn_analyzer=fake_turn_analyzer, vision_identify=fake_vision_identify))
    res = client.post(
        "/sales/query-with-vision",
        json={
            "message": "Cho toi thong tin tong quan",
            "image_base64": "ZmFrZQ==",
            "force_route": "consult_discovery",
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["lead_state"]["customer_profile"]["recognized"] is True
    assert payload["lead_state"]["customer_profile"]["name"] == "Linh"
    assert payload["lead_state"]["customer_profile"]["greeted_by_name"] is True
    assert "Linh" in payload["assistant_reply"]


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_query_with_vision_generic_greeting_for_unknown_customer(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MODEL_WARMUP_ENABLED", "false")
    monkeypatch.setattr(app_module, "synthesize_assistant_reply", fake_synthesize_assistant_reply)

    def fake_vision_identify(**kwargs):
        _ = kwargs
        return {
            "recognized": False,
            "name": None,
            "age": 28,
            "gender": "female",
            "confidence": 0.22,
            "source": "vlm",
            "face_count": 1,
            "bbox": [10, 10, 100, 100],
            "reason": "unknown_face_vlm_gender",
        }

    client = TestClient(create_app(turn_analyzer=fake_turn_analyzer, vision_identify=fake_vision_identify))
    res = client.post(
        "/sales/query-with-vision",
        json={
            "message": "Cho toi thong tin tong quan",
            "image_base64": "ZmFrZQ==",
            "force_route": "consult_discovery",
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["lead_state"]["customer_profile"]["recognized"] is False
    assert payload["lead_state"]["customer_profile"]["gender"] == "female"
    assert payload["lead_state"]["customer_profile"]["greeted_generic"] is True
    assert payload["assistant_reply"] != "Em dang ho tro tu van."


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_query_with_vision_falls_back_when_vision_errors(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MODEL_WARMUP_ENABLED", "false")
    monkeypatch.setattr(app_module, "synthesize_assistant_reply", fake_synthesize_assistant_reply)

    def failing_vision_identify(**kwargs):
        _ = kwargs
        raise RuntimeError("vision offline")

    client = TestClient(create_app(turn_analyzer=fake_turn_analyzer, vision_identify=failing_vision_identify))
    res = client.post(
        "/sales/query-with-vision",
        json={
            "message": "Cho toi thong tin tong quan",
            "image_base64": "ZmFrZQ==",
            "force_route": "consult_discovery",
        },
    )

    assert res.status_code == 200
    payload = res.json()
    assert payload["lead_state"]["customer_profile"]["recognized"] is False
    assert payload["lead_state"]["customer_profile"]["name"] is None
    assert payload["assistant_reply"] == "Em dang ho tro tu van."
