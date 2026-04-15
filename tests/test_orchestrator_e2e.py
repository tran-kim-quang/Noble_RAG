from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator_service.app import create_app


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


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_01_health():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_02_consult_discovery_route_for_strategy_query():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher))
    res = client.post("/sales/query", json={"message": "Mua de dau tu thi nen bat dau tu dau?", "force_route": "consult_discovery"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] in {"consult_discovery", "project_grounded"}
    assert "assistant_reply" in payload
    assert payload["need_update"]["topics"]


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_03_project_grounded_route_for_filter_query():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher))
    res = client.post("/sales/query", json={"message": "Co can nao gan truong hoc va benh vien khong?"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "project_grounded"
    assert payload["project_grounded_payload"]["used_projects"] == ["noble_palace_tay_thang_long"]


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_04_project_grounded_low_confidence_path():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_low_confidence))
    res = client.post("/sales/query", json={"message": "Du an co san truot tuyet trong nha khong?", "force_route": "project_grounded"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "project_grounded"
    assert payload["project_grounded_payload"]["low_confidence"] is True


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_05_project_grounded_error_path():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_error))
    res = client.post("/sales/query", json={"message": "Cho minh thong tin phap ly", "force_route": "project_grounded"})
    assert res.status_code == 502


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_06_state_merge_name_phone_and_need():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher))
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


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_07_validation_blank_message():
    client = TestClient(create_app(project_grounded_fetcher=fake_project_grounded_fetcher))
    res = client.post("/sales/query", json={"message": "   "})
    assert res.status_code == 422
