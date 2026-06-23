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
from orchestrator_service.session_store import SessionStore


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
            "face_id": "known-linh",
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
            "face_id": "guest-abc",
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


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_livetalking_session_start_resume_and_stop(monkeypatch):
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
            "face_id": "known-linh",
        }

    store = SessionStore("")
    client = TestClient(
        create_app(
            turn_analyzer=fake_turn_analyzer,
            vision_identify=fake_vision_identify,
            session_store=store,
        )
    )

    start_res = client.post("/integrations/livetalking/start", json={"image_base64": "ZmFrZQ=="})
    assert start_res.status_code == 200
    start_payload = start_res.json()
    assert start_payload["session"]["resumed"] is False
    assert start_payload["session"]["should_greet"] is True
    assert "Linh" in start_payload["session"]["greeting"]
    assert "khách hàng cũ" in start_payload["session"]["greeting"]
    assert "sản phẩm nào khác" in start_payload["session"]["greeting"]

    session_id = start_payload["session"]["session_id"]
    restart_res = client.post(
        "/integrations/livetalking/start",
        json={"image_base64": "ZmFrZQ==", "session_id": session_id},
    )
    assert restart_res.status_code == 200
    restart_payload = restart_res.json()
    assert restart_payload["session"]["resumed"] is True
    assert restart_payload["session"]["should_greet"] is False

    query_res = client.post(
        "/integrations/livetalking/query",
        json={
            "session_id": session_id,
            "message": "Cho toi thong tin tong quan",
            "force_route": "consult_discovery",
        },
    )
    assert query_res.status_code == 200
    query_payload = query_res.json()
    assert query_payload["session"]["session_id"] == session_id
    assert query_payload["assistant_reply"] == "Em dang ho tro tu van."

    stop_res = client.post("/integrations/livetalking/stop", json={"session_id": session_id})
    assert stop_res.status_code == 200
    stop_payload = stop_res.json()
    assert stop_payload["stopped"] is True
    assert stop_payload["session_id"] == session_id


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_livetalking_session_start_unknown_face_includes_catalog_overview(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MODEL_WARMUP_ENABLED", "false")
    monkeypatch.setattr(app_module, "synthesize_assistant_reply", fake_synthesize_assistant_reply)

    def fake_vision_identify(**kwargs):
        _ = kwargs
        return {
            "recognized": False,
            "name": None,
            "age": 28,
            "gender": "female",
            "confidence": 0.31,
            "source": "vlm",
            "face_count": 1,
            "bbox": [10, 10, 100, 100],
            "reason": "unknown_face",
            "face_id": "guest-xyz",
        }

    def fake_project_grounded_fetcher(*args, **kwargs):
        _ = args
        _ = kwargs
        return {
            "project_cards": [
                {
                    "project_id": "noble_palace_tay_thang_long",
                    "product_types": ["apartment", "shophouse", "villa"],
                }
            ],
            "trait_tags": [],
            "proximity_facts": [],
            "evidence_chunks": [],
            "confidence": 0.9,
            "low_confidence": False,
        }

    client = TestClient(
        create_app(
            turn_analyzer=fake_turn_analyzer,
            vision_identify=fake_vision_identify,
            project_grounded_fetcher=fake_project_grounded_fetcher,
            session_store=SessionStore(""),
        )
    )

    start_res = client.post("/integrations/livetalking/start", json={"image_base64": "ZmFrZQ=="})
    assert start_res.status_code == 200
    start_payload = start_res.json()
    greeting = str(start_payload["session"]["greeting"] or "")
    assert "Em chào anh/chị" in greeting
    assert "căn hộ" in greeting
    assert "shophouse" in greeting
