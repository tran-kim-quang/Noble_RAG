"""Reusable query planner for the knowledge-base route."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from core.dependencies import llm_model_func
from utils.json_extract import extract_first_json_object

log = logging.getLogger("rag-service")

_ALLOWED_INTENT_HINTS = {
    "project_info",
    "comparison",
    "objection",
    "generic",
    "unknown",
}
_ALLOWED_SOURCE_PREFERENCES = {"kb_first"}


class KnowledgeSubquery(TypedDict, total=False):
    query: str
    intent_hint: str
    source_preference: str


class KnowledgePlan(TypedDict, total=False):
    original_query: str
    normalized_query: str
    rewritten_query: str
    multi_intent: bool
    subqueries: List[KnowledgeSubquery]
    planner_notes: Dict[str, Any]


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _history_text(history: Optional[List[Dict[str, Any]]]) -> str:
    history_lines: List[str] = []
    for item in (history or [])[-6:]:
        role = str(item.get("role") or "user").strip()
        content = _normalize_query_text(str(item.get("content") or ""))
        if content:
            history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines) if history_lines else "(empty)"


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    payload = extract_first_json_object(str(raw or ""))
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _fallback_plan(text: str) -> KnowledgePlan:
    return {
        "original_query": text,
        "normalized_query": text,
        "rewritten_query": text,
        "multi_intent": False,
        "subqueries": [],
        "planner_notes": {"fallback_reason": "planner_failed"},
    }


def _normalize_subqueries(raw_subqueries: Any) -> List[KnowledgeSubquery]:
    normalized: List[KnowledgeSubquery] = []
    if not isinstance(raw_subqueries, list):
        return normalized

    for item in raw_subqueries[:2]:
        if not isinstance(item, dict):
            continue
        query = _normalize_query_text(str(item.get("query") or ""))
        if not query:
            continue

        intent_hint = _normalize_query_text(str(item.get("intent_hint") or "")).lower() or "unknown"
        if intent_hint not in _ALLOWED_INTENT_HINTS:
            intent_hint = "unknown"

        source_preference = _normalize_query_text(str(item.get("source_preference") or "")).lower() or "kb_first"
        if source_preference not in _ALLOWED_SOURCE_PREFERENCES:
            source_preference = "kb_first"

        normalized.append(
            {
                "query": query,
                "intent_hint": intent_hint,
                "source_preference": source_preference,
            }
        )

    return normalized


def _normalize_planner_notes(raw_notes: Any) -> Dict[str, Any]:
    if not isinstance(raw_notes, dict):
        return {}
    notes: Dict[str, Any] = {}
    for key, value in raw_notes.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            notes[key] = value
    return notes


async def build_knowledge_plan(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> KnowledgePlan:
    """Rewrite and decompose a user query without deciding the final chat route."""
    text = _normalize_query_text(query)
    if not text:
        return _fallback_plan("")

    prompt = f"""
You are a query planner for a real-estate assistant.

Return JSON only:
{{
  "rewritten_query": "short rewritten query with same meaning",
  "multi_intent": true|false,
  "subqueries": [
    {{
      "query": "...",
      "intent_hint": "project_info|comparison|objection|generic|unknown",
      "source_preference": "kb_first"
    }}
  ],
  "planner_notes": {{
    "query_kind": "fact|consultative|comparison|greeting|generic|unknown"
  }}
}}

Rules:
- Preserve user meaning. Do not invent facts.
- If no rewrite is needed, keep the same meaning in a cleaner form.
- Only decompose when the user truly has 2 independent asks.
- Max 2 subqueries.
- If decomposition is not needed, set multi_intent=false and subqueries=[].
- Greetings or vague social turns should usually stay single-intent.
- For comparison requests, use intent_hint="comparison".
- For concrete project fact questions, use intent_hint="project_info".
- For concerns or pushback, use intent_hint="objection".
- For anything unclear, use intent_hint="unknown".
- Every subquery must use source_preference="kb_first".

History:
{_history_text(history)}

User query:
{text}
""".strip()

    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        data = _parse_json_dict(str(raw))
    except Exception as exc:
        log.warning("build_knowledge_plan failed, using fallback: %s", exc)
        return _fallback_plan(text)

    rewritten_query = _normalize_query_text(str(data.get("rewritten_query") or text)) or text
    subqueries = _normalize_subqueries(data.get("subqueries"))
    multi_intent = bool(data.get("multi_intent")) and len(subqueries) > 1
    if not multi_intent:
        subqueries = []

    planner_notes = _normalize_planner_notes(data.get("planner_notes"))
    return {
        "original_query": text,
        "normalized_query": text,
        "rewritten_query": rewritten_query,
        "multi_intent": multi_intent,
        "subqueries": subqueries,
        "planner_notes": planner_notes,
    }


async def decompose_subqueries(query: str) -> List[str]:
    """Compatibility helper for legacy callers that still want raw subquery strings."""
    plan = await build_knowledge_plan(query)
    subqueries = plan.get("subqueries") or []
    if plan.get("multi_intent") and subqueries:
        return [item["query"] for item in subqueries if item.get("query")]
    rewritten = str(plan.get("rewritten_query") or "").strip()
    return [rewritten] if rewritten else []
