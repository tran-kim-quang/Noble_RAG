from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_base import service


@pytest.mark.asyncio
async def test_resolve_knowledge_builds_kb_only_payload(monkeypatch):
    async def fake_plan(query, history=None):
        return {
            "rewritten_query": "phap ly noble place",
            "multi_intent": False,
            "subqueries": [],
            "planner_notes": {"query_kind": "fact"},
        }

    async def fake_retrieve(query, top_k=6):
        return [
            {
                "score": 0.82,
                "content": "Phap ly du an da co day du giay to.",
                "source": "internal_doc.md",
                "document_id": "doc-1",
                "metadata": {"chunk": 1},
            }
        ]

    def fake_evaluate(query, candidates, history=None, threshold=0.5):
        return {
            "top_score": 0.82,
            "should_search": False,
            "decision_reason": "kb_confident",
        }

    async def fail_search(_query):
        raise AssertionError("search should not run")

    monkeypatch.setattr(service, "build_knowledge_plan", fake_plan)
    monkeypatch.setattr(service, "retrieve_kb_candidates", fake_retrieve)
    monkeypatch.setattr(service, "evaluate_kb_candidates", fake_evaluate)
    monkeypatch.setattr(service, "tavily_search", fail_search)
    monkeypatch.setattr(service.settings, "enable_cosine_search_fallback", True, raising=False)
    monkeypatch.setattr(service.settings, "kb_search_score_threshold", 0.5, raising=False)

    payload = await service.resolve_knowledge("phap ly du an")

    assert payload.should_search is False
    assert payload.decision_reason == "kb_confident"
    assert payload.unresolved is False
    assert len(payload.kb_evidence) == 1
    assert len(payload.search_evidence) == 0
    assert payload.kb_evidence[0].source_type == "kb"


@pytest.mark.asyncio
async def test_resolve_knowledge_builds_search_only_payload(monkeypatch):
    async def fake_plan(query, history=None):
        return {
            "rewritten_query": "thoi tiet ha noi hien tai",
            "multi_intent": False,
            "subqueries": [],
            "planner_notes": {},
        }

    async def fake_retrieve(query, top_k=6):
        return []

    def fake_evaluate(query, candidates, history=None, threshold=0.5):
        return {
            "top_score": 0.0,
            "should_search": True,
            "decision_reason": "realtime_query",
        }

    async def fake_search(query):
        return "Ha Noi hien tai 28C"

    monkeypatch.setattr(service, "build_knowledge_plan", fake_plan)
    monkeypatch.setattr(service, "retrieve_kb_candidates", fake_retrieve)
    monkeypatch.setattr(service, "evaluate_kb_candidates", fake_evaluate)
    monkeypatch.setattr(service, "tavily_search", fake_search)
    monkeypatch.setattr(service.settings, "enable_cosine_search_fallback", True, raising=False)

    payload = await service.resolve_knowledge("thoi tiet hom nay")

    assert payload.should_search is True
    assert payload.decision_reason == "realtime_query"
    assert len(payload.kb_evidence) == 0
    assert len(payload.search_evidence) == 1
    assert payload.search_evidence[0].source_type == "search"


@pytest.mark.asyncio
async def test_resolve_knowledge_builds_mixed_payload(monkeypatch):
    async def fake_plan(query, history=None):
        return {
            "rewritten_query": "so sanh du an voi thi truong",
            "multi_intent": False,
            "subqueries": [],
            "planner_notes": {},
        }

    async def fake_retrieve(query, top_k=6):
        return [
            {
                "score": 0.31,
                "content": "Bang gia noi bo du an.",
                "source": "bang_gia.md",
                "document_id": "doc-2",
                "metadata": {},
            }
        ]

    def fake_evaluate(query, candidates, history=None, threshold=0.5):
        return {
            "top_score": 0.31,
            "should_search": True,
            "decision_reason": "low_kb_confidence",
        }

    async def fake_search(query):
        return "Gia thi truong khu vuc Tay Ho"

    monkeypatch.setattr(service, "build_knowledge_plan", fake_plan)
    monkeypatch.setattr(service, "retrieve_kb_candidates", fake_retrieve)
    monkeypatch.setattr(service, "evaluate_kb_candidates", fake_evaluate)
    monkeypatch.setattr(service, "tavily_search", fake_search)
    monkeypatch.setattr(service.settings, "enable_cosine_search_fallback", True, raising=False)

    payload = await service.resolve_knowledge("so sanh gia")

    assert payload.should_search is True
    assert len(payload.kb_evidence) == 1
    assert len(payload.search_evidence) == 1


@pytest.mark.asyncio
async def test_resolve_knowledge_marks_unresolved_for_ambiguous_query(monkeypatch):
    async def fake_plan(query, history=None):
        return {
            "rewritten_query": "gia?",
            "multi_intent": False,
            "subqueries": [],
            "planner_notes": {},
        }

    async def fake_retrieve(query, top_k=6):
        return []

    def fake_evaluate(query, candidates, history=None, threshold=0.5):
        return {
            "top_score": 0.0,
            "should_search": False,
            "decision_reason": "short_ambiguous_query",
        }

    async def fake_search(query):
        return ""

    monkeypatch.setattr(service, "build_knowledge_plan", fake_plan)
    monkeypatch.setattr(service, "retrieve_kb_candidates", fake_retrieve)
    monkeypatch.setattr(service, "evaluate_kb_candidates", fake_evaluate)
    monkeypatch.setattr(service, "tavily_search", fake_search)
    monkeypatch.setattr(service.settings, "enable_cosine_search_fallback", True, raising=False)

    payload = await service.resolve_knowledge("gia?")

    assert payload.unresolved is True
    assert payload.decision_reason == "short_ambiguous_query"


@pytest.mark.asyncio
async def test_resolve_knowledge_sets_multi_intent_only_when_subqueries_present(monkeypatch):
    async def fake_plan(query, history=None):
        return {
            "rewritten_query": "so sanh phap ly va gia",
            "multi_intent": True,
            "subqueries": [
                {
                    "query": "phap ly du an noble place",
                    "intent_hint": "project_info",
                    "source_preference": "kb_first",
                },
                {
                    "query": "bang gia du an noble place",
                    "intent_hint": "comparison",
                    "source_preference": "kb_first",
                },
            ],
            "planner_notes": {},
        }

    async def fake_retrieve(query, top_k=6):
        return []

    def fake_evaluate(query, candidates, history=None, threshold=0.5):
        return {
            "top_score": 0.0,
            "should_search": False,
            "decision_reason": "project_specific_kb_first",
        }

    async def fake_search(query):
        return ""

    monkeypatch.setattr(service, "build_knowledge_plan", fake_plan)
    monkeypatch.setattr(service, "retrieve_kb_candidates", fake_retrieve)
    monkeypatch.setattr(service, "evaluate_kb_candidates", fake_evaluate)
    monkeypatch.setattr(service, "tavily_search", fake_search)
    monkeypatch.setattr(service.settings, "enable_cosine_search_fallback", True, raising=False)

    payload = await service.resolve_knowledge("so sanh phap ly va gia")

    assert payload.multi_intent is True
    assert len(payload.subqueries) == 2
