from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_base import planner


async def _unused_llm(*args, **kwargs):
    raise AssertionError("llm should not be called in this test")


def test_build_knowledge_plan_fallback_on_empty_query():
    plan = planner._fallback_plan("")
    assert plan["original_query"] == ""
    assert plan["rewritten_query"] == ""
    assert plan["multi_intent"] is False
    assert plan["subqueries"] == []


def test_normalize_subqueries_caps_and_sanitizes():
    raw = [
        {
            "query": "  so sanh gia va tien do ",
            "intent_hint": "comparison",
            "source_preference": "kb_first",
        },
        {
            "query": "  phap ly   ",
            "intent_hint": "unsupported",
            "source_preference": "search_first",
        },
        {"query": "extra", "intent_hint": "generic", "source_preference": "kb_first"},
    ]

    subqueries = planner._normalize_subqueries(raw)
    assert subqueries == [
        {
            "query": "so sanh gia va tien do",
            "intent_hint": "comparison",
            "source_preference": "kb_first",
        },
        {
            "query": "phap ly",
            "intent_hint": "unknown",
            "source_preference": "kb_first",
        },
    ]


@pytest.mark.asyncio
async def test_build_knowledge_plan_single_intent(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return """
        {
          "rewritten_query": "phap ly du an noble place tay thang long",
          "multi_intent": false,
          "subqueries": [],
          "planner_notes": {"query_kind": "fact"}
        }
        """

    monkeypatch.setattr(planner, "llm_model_func", fake_llm)
    plan = await planner.build_knowledge_plan("phap ly du an la gi?", history=[])

    assert plan["original_query"] == "phap ly du an la gi?"
    assert plan["normalized_query"] == "phap ly du an la gi?"
    assert plan["rewritten_query"] == "phap ly du an noble place tay thang long"
    assert plan["multi_intent"] is False
    assert plan["subqueries"] == []
    assert plan["planner_notes"]["query_kind"] == "fact"


@pytest.mark.asyncio
async def test_build_knowledge_plan_multi_intent(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return """
        {
          "rewritten_query": "so sanh du an va gia thi truong",
          "multi_intent": true,
          "subqueries": [
            {
              "query": "thong tin noi bo du an noble place tay thang long",
              "intent_hint": "comparison",
              "source_preference": "kb_first"
            },
            {
              "query": "mat bang gia thi truong biet thu ha noi",
              "intent_hint": "comparison",
              "source_preference": "kb_first"
            }
          ],
          "planner_notes": {"query_kind": "comparison"}
        }
        """

    monkeypatch.setattr(planner, "llm_model_func", fake_llm)
    plan = await planner.build_knowledge_plan("so sanh du an voi thi truong", history=[])

    assert plan["multi_intent"] is True
    assert len(plan["subqueries"]) == 2
    assert plan["subqueries"][0]["intent_hint"] == "comparison"
    assert plan["subqueries"][1]["source_preference"] == "kb_first"


@pytest.mark.asyncio
async def test_build_knowledge_plan_fallback_when_llm_fails(monkeypatch):
    async def failing_llm(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(planner, "llm_model_func", failing_llm)
    plan = await planner.build_knowledge_plan("xin tu van giup minh", history=[])

    assert plan["rewritten_query"] == "xin tu van giup minh"
    assert plan["multi_intent"] is False
    assert plan["subqueries"] == []
    assert plan["planner_notes"]["fallback_reason"] == "planner_failed"
