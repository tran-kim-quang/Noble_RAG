"""Build structured knowledge payloads from KB retrieval and optional search fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.config import get_settings
from knowledge_base.evaluator import evaluate_kb_candidates
from knowledge_base.planner import KnowledgePlan, build_knowledge_plan
from rag.retriever import retrieve_kb_candidates
from tools.tavily_tool import tavily_search

log = logging.getLogger("rag-service")
settings = get_settings()


async def build_knowledge_payload(
    *,
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    plan: Optional[KnowledgePlan] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    knowledge_plan = plan or await build_knowledge_plan(query, history)
    rewritten_query = str(knowledge_plan.get("rewritten_query") or query).strip() or query

    kb_evidence = await retrieve_kb_candidates(rewritten_query, top_k=top_k)
    decision = evaluate_kb_candidates(
        rewritten_query,
        kb_evidence,
        history=history,
        threshold=settings.kb_search_score_threshold,
    )

    search_evidence: List[Dict[str, Any]] = []
    if settings.enable_cosine_search_fallback and decision.get("should_search"):
        search_text = await tavily_search(rewritten_query)
        cleaned = str(search_text or "").strip()
        if cleaned:
            search_evidence.append(
                {
                    "content": cleaned,
                    "source": "tavily",
                    "document_id": None,
                    "metadata": {"query": rewritten_query},
                }
            )

    log.info(
        "knowledge_payload: query=%r rewritten=%r top_score=%.4f threshold=%.4f should_search=%s reason=%s kb_hits=%s search_hits=%s",
        query,
        rewritten_query,
        float(decision.get("top_score") or 0.0),
        float(settings.kb_search_score_threshold),
        bool(decision.get("should_search")),
        str(decision.get("decision_reason") or ""),
        len(kb_evidence),
        len(search_evidence),
    )

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "multi_intent": bool(knowledge_plan.get("multi_intent")),
        "subqueries": list(knowledge_plan.get("subqueries") or []),
        "planner_notes": dict(knowledge_plan.get("planner_notes") or {}),
        "kb_evidence": kb_evidence,
        "search_evidence": search_evidence,
        "decision": decision,
    }
