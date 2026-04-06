from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat import service


@pytest.mark.asyncio
async def test_run_chat_route_uses_payload_and_marks_search(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return "Noi dung tu van"

    monkeypatch.setattr(service, "llm_model_func", fake_llm)

    result = await service.run_chat_route(
        session_id="s1",
        message="Cho toi thong tin du an",
        history=[],
        lead_profile={},
        session_context={},
        knowledge_payload={
            "original_query": "Cho toi thong tin du an",
            "rewritten_query": "thong tin du an",
            "should_search": True,
            "decision_reason": "low_kb_confidence",
            "kb_evidence": [
                {
                    "source_type": "kb",
                    "content": "Du an co day du phap ly",
                    "source_name": "internal.md",
                }
            ],
            "search_evidence": [
                {
                    "source_type": "search",
                    "content": "Gia thi truong hien tai",
                    "source_name": "tavily",
                }
            ],
        },
        raw_transcript=None,
    )

    assert result["route_category"] == "CHAT"
    assert result["knowledge_used"] is True
    assert result["search_used"] is True
    assert result["final_response"] == "Noi dung tu van"
    assert result["next_sales_state"] in {"project_qa", "comparison", "need_discovery", "product_matching"}
    assert result["response_action"]
    assert isinstance(result["missing_slots"], list)
    assert result["knowledge_used"] is True
    assert isinstance(result["kb_top_score"], (float, type(None)))


@pytest.mark.asyncio
async def test_run_chat_route_fallback_when_llm_fails(monkeypatch):
    async def failing_llm(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(service, "llm_model_func", failing_llm)

    result = await service.run_chat_route(
        session_id="s2",
        message="Xin tu van",
        history=[],
        lead_profile={},
        session_context={},
        knowledge_payload=None,
        raw_transcript=None,
    )

    assert result["route_category"] == "CHAT"
    assert result["knowledge_used"] is False
    assert "hỗ trợ" in result["final_response"].lower()
    assert result["response_action"] == "ask_follow_up"
    assert result["search_used"] is False
    assert result["kb_top_score"] is None


@pytest.mark.asyncio
async def test_run_chat_route_handles_invalid_payload(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return "Tra loi"

    monkeypatch.setattr(service, "llm_model_func", fake_llm)

    result = await service.run_chat_route(
        session_id="s3",
        message="Gia du an",
        history=[],
        lead_profile={},
        session_context={},
        knowledge_payload={"foo": "bar"},
        raw_transcript=None,
    )

    assert result["knowledge_used"] is False
    assert result["search_used"] is False
