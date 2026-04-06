from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import routes_query, routes_sales


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes_sales.router)
    app.include_router(routes_query.router)
    return app


def test_sales_chat_returns_phase6_metadata(monkeypatch):
    async def fake_ensure():
        return None

    async def fake_enrich(_session_id):
        return None

    async def fake_run_flow(*, session_id, user_text, raw_transcript=None):
        return {
            "route_category": "CHAT",
            "final_response": "Da nhan",
            "next_sales_state": "need_discovery",
            "lead_profile": {"lead_id": session_id},
            "missing_slots": ["purpose"],
            "knowledge_used": True,
            "search_used": False,
            "kb_top_score": 0.72,
        }

    monkeypatch.setattr(routes_sales, "ensure_sales_schema", fake_ensure)
    monkeypatch.setattr(routes_sales, "maybe_enrich_identity", fake_enrich)
    monkeypatch.setattr(routes_sales, "_run_sales_flow", fake_run_flow)

    client = TestClient(_build_test_app())
    response = client.post("/sales/chat", json={"session_id": "s1", "message": "xin tu van"})

    assert response.status_code == 200
    data = response.json()
    assert data["route_category"] == "CHAT"
    assert data["knowledge_used"] is True
    assert data["search_used"] is False
    assert data["kb_top_score"] == 0.72


def test_sales_chat_stream_keeps_ndjson_and_complete_metadata(monkeypatch):
    async def fake_ensure():
        return None

    async def fake_enrich(_session_id):
        return None

    async def fake_load_lead(_session_id):
        return {"lead_id": _session_id}

    async def fake_load_context(_session_id):
        return {}

    async def fake_run_flow(*, session_id, user_text, raw_transcript=None):
        return {
            "route_category": "CHAT",
            "final_response": "Noi dung tra loi",
            "next_sales_state": "project_qa",
            "lead_profile": {"lead_id": session_id},
            "missing_slots": [],
            "knowledge_used": True,
            "search_used": True,
            "kb_top_score": 0.44,
        }

    monkeypatch.setattr(routes_sales, "ensure_sales_schema", fake_ensure)
    monkeypatch.setattr(routes_sales, "maybe_enrich_identity", fake_enrich)
    monkeypatch.setattr(routes_sales, "load_lead_profile", fake_load_lead)
    monkeypatch.setattr(routes_sales, "load_session_context", fake_load_context)
    monkeypatch.setattr(routes_sales, "_run_sales_flow", fake_run_flow)

    client = TestClient(_build_test_app())
    response = client.post("/sales/chat/stream", json={"session_id": "s2", "message": "hoi gia"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.strip()]
    payloads = [json.loads(line) for line in lines]

    assert payloads[0]["phase"] == "thinking_ack"
    complete = payloads[-1]
    assert complete["phase"] == "complete"
    assert complete["session_id"] == "s2"
    assert complete["knowledge_used"] is True
    assert complete["search_used"] is True
    assert complete["kb_top_score"] == 0.44


def test_query_stream_alias_forwards_to_sales_stream(monkeypatch):
    calls = {}

    async def fake_sales_stream(request):
        calls["session_id"] = request.session_id
        calls["message"] = request.message
        return JSONResponse({"ok": True, "session_id": request.session_id, "message": request.message})

    monkeypatch.setattr(routes_query, "sales_chat_stream", fake_sales_stream)

    client = TestClient(_build_test_app())
    response = client.post("/query/stream", json={"query": "xin chao", "session_id": "q1"})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert calls["session_id"] == "q1"
    assert calls["message"] == "xin chao"


def test_lead_update_endpoint_persists_profile(monkeypatch):
    saved = {}

    async def fake_load(_session_id):
        return {"lead_id": _session_id, "current_state": "greeting"}

    async def fake_save(_session_id, profile):
        saved["session_id"] = _session_id
        saved["profile"] = profile

    monkeypatch.setattr(routes_sales, "load_lead_profile", fake_load)
    monkeypatch.setattr(routes_sales, "save_lead_profile", fake_save)

    client = TestClient(_build_test_app())
    response = client.patch("/sales/lead/s4", json={"updates": {"purpose": "mua_o"}})

    assert response.status_code == 200
    assert saved["session_id"] == "s4"
    assert saved["profile"]["purpose"] == "mua_o"


def test_history_endpoint_keeps_continuity(monkeypatch):
    async def fake_history(_session_id):
        return [
            {"role": "user", "content": "xin chao", "timestamp": "t1"},
            {"role": "assistant", "content": "em chao anh", "timestamp": "t2"},
        ]

    monkeypatch.setattr(routes_sales, "load_chat_history", fake_history)

    client = TestClient(_build_test_app())
    response = client.get("/sales/history/s5")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "s5"
    assert len(data["history"]) == 2
    assert data["history"][0]["content"] == "xin chao"


@pytest.mark.asyncio
async def test_run_sales_flow_applies_pronoun_before_persist(monkeypatch):
    persisted = {}

    async def fake_load_history(_session_id):
        return []

    async def fake_load_context(_session_id):
        return {"gender": "male"}

    async def fake_load_lead(_session_id):
        return {"lead_id": _session_id}

    async def fake_resolve_knowledge(**kwargs):
        return {"original_query": kwargs.get("query", "")}

    async def fake_run_chat_route(**kwargs):
        return {
            "route_category": "CHAT",
            "final_response": "Tra loi goc",
            "next_sales_state": "need_discovery",
            "lead_profile": kwargs.get("lead_profile") or {},
            "missing_slots": [],
            "knowledge_used": True,
            "search_used": False,
            "kb_top_score": None,
        }

    def fake_apply_pronoun(text, _state):
        return f"[patched] {text}"

    async def fake_persist(_session_id, _user_text, _assistant_text):
        persisted["assistant"] = _assistant_text

    monkeypatch.setattr(routes_sales, "load_chat_history", fake_load_history)
    monkeypatch.setattr(routes_sales, "load_session_context", fake_load_context)
    monkeypatch.setattr(routes_sales, "load_lead_profile", fake_load_lead)
    monkeypatch.setattr(routes_sales, "resolve_knowledge", fake_resolve_knowledge)
    monkeypatch.setattr(routes_sales, "run_chat_route", fake_run_chat_route)
    monkeypatch.setattr(routes_sales, "_apply_customer_pronoun", fake_apply_pronoun)
    monkeypatch.setattr(routes_sales, "_persist_chat_turn", fake_persist)

    result = await routes_sales._run_sales_flow(session_id="s6", user_text="hello")

    assert result["final_response"].startswith("[patched]")
    assert persisted["assistant"].startswith("[patched]")
