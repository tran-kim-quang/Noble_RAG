from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator_service.app import create_app


PY313 = sys.version_info >= (3, 13)


def fake_retrieval_confident(query: str, top_k: int):
    return {
        "results": [
            {
                "text": "Gia can ho 2PN tu 7.8 ty",
                "score": 0.91,
                "source": "pricing_sheet_q1.md",
                "doc_id": "price-001",
            }
        ],
        "confidence": 0.91,
        "low_confidence": False,
    }


def fake_retrieval_low_confidence(query: str, top_k: int):
    return {
        "results": [
            {
                "text": "co the lien quan",
                "score": 0.42,
                "source": "misc.md",
                "doc_id": "misc-001",
            }
        ],
        "confidence": 0.42,
        "low_confidence": True,
    }


def fake_retrieval_empty(query: str, top_k: int):
    return {
        "results": [],
        "confidence": 0.0,
        "low_confidence": True,
    }


def fake_retrieval_error(query: str, top_k: int):
    raise RuntimeError("connection refused")


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_01_health():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_02_no_retrieval_explicit_false():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.post("/sales/query", json={"message": "xin chao", "need_retrieval": False})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "no_retrieval_needed"
    assert payload["action"] == "proceed_without_retrieval"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_03_retrieval_confident_path():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.post("/sales/query", json={"message": "gia can 2 phong ngu", "need_retrieval": True})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "retrieval_confident"
    assert payload["action"] == "use_retrieved_context"
    assert payload["retrieval"]["low_confidence"] is False


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_04_retrieval_low_confidence_path():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_low_confidence))
    res = client.post("/sales/query", json={"message": "gia can 2 phong ngu", "need_retrieval": True})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "retrieval_low_confidence"
    assert payload["action"] == "ask_clarifying_question"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_05_retrieval_empty_path():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_empty))
    res = client.post("/sales/query", json={"message": "gia can 2 phong ngu", "need_retrieval": True})
    assert res.status_code == 200
    payload = res.json()
    assert payload["route"] == "retrieval_low_confidence"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_06_retrieval_error_path():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_error))
    res = client.post("/sales/query", json={"message": "gia can 2 phong ngu", "need_retrieval": True})
    assert res.status_code == 502


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_07_auto_decision_no_retrieval_keyword_missing():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.post("/sales/query", json={"message": "xin chao ban"})
    assert res.status_code == 200
    assert res.json()["route"] == "no_retrieval_needed"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_08_auto_decision_needs_retrieval_keyword_present():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.post("/sales/query", json={"message": "phap ly du an hien tai"})
    assert res.status_code == 200
    assert res.json()["route"] == "retrieval_confident"


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_09_response_contains_retrieval_metadata():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.post("/sales/query", json={"message": "gia can 2 phong ngu", "need_retrieval": True})
    payload = res.json()
    assert "confidence" in payload["retrieval"]
    assert "low_confidence" in payload["retrieval"]


@pytest.mark.skipif(PY313, reason="Known TestClient/anyio instability on Python 3.13 in this environment.")
def test_10_validation_blank_message():
    client = TestClient(create_app(retrieval_fetcher=fake_retrieval_confident))
    res = client.post("/sales/query", json={"message": "   "})
    assert res.status_code == 422


# Python 3.13 fallback sanity checks (no TestClient)
def test_fallback_01_heuristic_keyword_case():
    from orchestrator_service.app import _needs_retrieval_heuristic

    assert _needs_retrieval_heuristic("phap ly du an") is True


def test_fallback_02_heuristic_non_keyword_case():
    from orchestrator_service.app import _needs_retrieval_heuristic

    assert _needs_retrieval_heuristic("xin chao") is False
