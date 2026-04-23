from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator_service.app import TurnAnalysis
from orchestrator_service.app import create_app
from orchestrator_service.schemas import NeedPainpointDelta
from orchestrator_service.schemas import RoutingSignal
from orchestrator_service.schemas import TopicWeight


PY313 = sys.version_info >= (3, 13)


def fake_project_grounded_fetcher(query: str, retrieval_intent: str, top_k: int):
    _ = retrieval_intent
    _ = top_k
    return {
        "project_cards": [
            {
                "project_id": "noble_palace_tay_thang_long",
                "summary": "Du an phu hop lens gia dinh o khu Tay Ho Tay.",
                "strengths": ["ket noi giao thong thuan tien", "tien ich noi khu da dang"],
                "tradeoffs": ["can doi chieu them ngan sach theo tung loai can"],
                "fit_personas": ["gia dinh co con nho"],
                "score": 0.91,
            }
        ],
        "trait_tags": [
            {
                "tag": "family_friendly",
                "weight": 0.89,
                "reason": "Tien ich va boi canh phu hop gia dinh.",
                "project_id": "noble_palace_tay_thang_long",
            }
        ],
        "evidence_chunks": [
            {
                "text": f"matched: {query}",
                "score": 0.91,
                "source": "amenities.md",
                "doc_id": "amen-001",
                "project_id": "noble_palace_tay_thang_long",
                "topic": "amenities",
            }
        ],
        "confidence": 0.91,
        "low_confidence": False,
    }


def fake_project_grounded_low_confidence(query: str, retrieval_intent: str, top_k: int):
    _ = query
    _ = retrieval_intent
    _ = top_k
    return {
        "project_cards": [],
        "trait_tags": [],
        "evidence_chunks": [],
        "confidence": 0.4,
        "low_confidence": True,
    }


def fake_project_grounded_error(query: str, retrieval_intent: str, top_k: int):
    _ = query
    _ = retrieval_intent
    _ = top_k
    raise RuntimeError("connection refused")


def fake_turn_analyzer(message: str, lead_state, recent_history):
    _ = lead_state
    _ = recent_history
    return TurnAnalysis(
        route="project_grounded",
        decision_reason="fake_context_decider_project",
        need_update=NeedPainpointDelta(
            summary_delta="Nhu cau duoc bo sung: lua chon du an phu hop.",
            topics=[TopicWeight(label="lua chon du an", weight=0.86)],
            evidence=[message],
        ),
        painpoint_update=NeedPainpointDelta(
            summary_delta="Khach can doi chieu thong tin de quyet dinh.",
            topics=[TopicWeight(label="can doi chieu thong tin", weight=0.79)],
            evidence=[message],
        ),
        routing_signal=RoutingSignal(
            should_route_project=True,
            project_query_hint=message,
            reason="fake_context_decider_project",
        ),
        consult_reply="",
        query_type="project_matching",
        retrieval_readiness="soft_ready",
        sales_state_after_hint="qualified",
        conversation_goal_hint="show_fit",
    )


def assert_sales_schema(payload: dict):
    lead_state = payload["lead_state"]
    trace = payload["decision_trace"]
    assert lead_state["sales_state"] in {
        "unknown",
        "exploring",
        "need_identified",
        "qualified",
        "interested",
        "appointment_ready",
        "nurture",
        "handoff",
    }
    assert lead_state["lead_level"] in {"exploratory", "interested", "qualified", "hot"}
    assert lead_state["last_conversation_goal"] is not None
    assert lead_state["next_best_action"] is not None
    assert trace["sales_state_before"] is not None
    assert trace["sales_state_after"] is not None
    assert trace["conversation_goal"] is not None
    assert trace["response_mode"] is not None


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_01_health():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher, turn_analyzer=fake_turn_analyzer))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_02_consult_discovery_route_for_strategy_query():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher, turn_analyzer=fake_turn_analyzer))
    res = client.post("/sales/query", json={"message": "Mua de dau tu thi nen bat dau tu dau?", "force_route": "consult_discovery"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] in {"consult_discovery", "project_grounded"}
    assert "assistant_reply" in payload
    assert payload["need_update"]["topics"]
    assert_sales_schema(payload)


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_03_project_grounded_route_for_filter_query():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher, turn_analyzer=fake_turn_analyzer))
    res = client.post("/sales/query", json={"message": "Co can nao gan truong hoc va benh vien khong?"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "project_grounded"
    assert payload["project_grounded_payload"]["used_projects"] == ["noble_palace_tay_thang_long"]
    assert_sales_schema(payload)


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_04_project_grounded_low_confidence_path():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_low_confidence, turn_analyzer=fake_turn_analyzer))
    res = client.post("/sales/query", json={"message": "Du an co san truot tuyet trong nha khong?", "force_route": "project_grounded"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "project_grounded"
    assert payload["project_grounded_payload"]["low_confidence"] is True
    assert_sales_schema(payload)


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_05_project_grounded_error_path():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_error, turn_analyzer=fake_turn_analyzer))
    res = client.post("/sales/query", json={"message": "Cho minh thong tin phap ly", "force_route": "project_grounded"})
    assert res.status_code == 502


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_06_state_merge_name_phone_and_need():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher, turn_analyzer=fake_turn_analyzer))
    res = client.post(
        "/sales/query",
        json={
            "message": "Toi la Nam, so cua toi la 0912345678, minh mua de o va uu tien truong hoc.",
            "force_route": "consult_discovery",
            "lead_state": {
                "name": None,
                "phone_contact": None,
                "need": {"summary": "", "topics": [], "evidence": [], "last_updated_at": None},
                "painpoint": {"summary": "", "topics": [], "evidence": [], "last_updated_at": None},
            },
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["lead_state"]["name"] is not None
    assert payload["lead_state"]["phone_contact"] == "0912345678"
    assert payload["lead_state"]["need"]["topics"]
    assert_sales_schema(payload)


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_07_validation_blank_message():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher, turn_analyzer=fake_turn_analyzer))
    res = client.post("/sales/query", json={"message": "   "})
    assert res.status_code == 422
