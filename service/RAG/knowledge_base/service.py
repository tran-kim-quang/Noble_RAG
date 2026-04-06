"""Build structured knowledge payloads from KB retrieval and optional search fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.config import get_settings
from knowledge_base.evaluator import evaluate_kb_candidates
from knowledge_base.planner import KnowledgePlan, build_knowledge_plan
from models.api_models import KnowledgeDecisionPayload, KnowledgeEvidenceItem
from rag.retriever import retrieve_kb_candidates
from tools.tavily_tool import tavily_search

log = logging.getLogger("rag-service")
settings = get_settings()

_MAX_EVIDENCE_ITEMS = 6
_MAX_EVIDENCE_CHARS = 2400
_MAX_METADATA_DEPTH = 3
_MAX_METADATA_LIST_ITEMS = 20


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_metadata(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_METADATA_DEPTH:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            clean_item = _sanitize_metadata(item, depth + 1)
            if clean_item is not None:
                sanitized[key] = clean_item
        return sanitized
    if isinstance(value, list):
        sanitized_list: List[Any] = []
        for item in value[:_MAX_METADATA_LIST_ITEMS]:
            clean_item = _sanitize_metadata(item, depth + 1)
            if clean_item is not None:
                sanitized_list.append(clean_item)
        return sanitized_list
    return str(value)


def _sanitize_evidence(
    raw_items: List[Dict[str, Any]],
    *,
    source_type: str,
) -> List[KnowledgeEvidenceItem]:
    seen: set[tuple[str, str, str]] = set()
    sanitized: List[KnowledgeEvidenceItem] = []

    for item in raw_items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        content = content[:_MAX_EVIDENCE_CHARS]

        source_name = str(item.get("source_name") or item.get("source") or "").strip() or None
        document_id = str(item.get("document_id") or "").strip() or None
        dedupe_key = (
            source_type,
            source_name or "",
            " ".join(content.lower().split()),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        metadata = _sanitize_metadata(item.get("metadata") or {})
        if not isinstance(metadata, dict):
            metadata = {}

        evidence_item = KnowledgeEvidenceItem(
            source_type=source_type,
            content=content,
            score=_safe_float(item.get("score")),
            source_name=source_name,
            document_id=document_id,
            metadata=metadata,
        )
        sanitized.append(evidence_item)
        if len(sanitized) >= _MAX_EVIDENCE_ITEMS:
            break

    return sanitized


def _is_unresolved(
    *,
    decision_reason: str,
    kb_evidence: List[KnowledgeEvidenceItem],
    search_evidence: List[KnowledgeEvidenceItem],
) -> bool:
    if decision_reason == "short_ambiguous_query":
        return True
    if not kb_evidence and not search_evidence:
        return True
    return False


def _payload_to_dict(payload: KnowledgeDecisionPayload) -> Dict[str, Any]:
    data = payload.model_dump(mode="python")
    for key in ("kb_evidence", "search_evidence"):
        evidence = data.get(key) or []
        for item in evidence:
            item["source"] = item.get("source_name")

    data["decision"] = {
        "top_score": payload.top_score,
        "avg_top_k_score": None,
        "should_search": payload.should_search,
        "decision_reason": payload.decision_reason,
        "matched_documents": [
            {
                "score": item.get("score"),
                "source": item.get("source_name"),
                "document_id": item.get("document_id"),
                "metadata": item.get("metadata") or {},
            }
            for item in (data.get("kb_evidence") or [])[:3]
        ],
        "low_confidence": (payload.top_score or 0.0) < float(payload.threshold or 0.0),
        "special_case_triggered": payload.decision_reason in {"realtime_query"},
    }
    return data


async def resolve_knowledge(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    session_context: Optional[Dict[str, Any]] = None,
    *,
    plan: Optional[KnowledgePlan] = None,
    top_k: int = 6,
) -> KnowledgeDecisionPayload:
    del session_context
    knowledge_plan = plan or await build_knowledge_plan(query, history)
    rewritten_query = str(knowledge_plan.get("rewritten_query") or query).strip() or query

    kb_candidates = await retrieve_kb_candidates(rewritten_query, top_k=top_k)
    decision = evaluate_kb_candidates(
        rewritten_query,
        kb_candidates,
        history=history,
        threshold=settings.kb_search_score_threshold,
    )

    search_candidates: List[Dict[str, Any]] = []
    if settings.enable_cosine_search_fallback and decision.get("should_search"):
        search_text = await tavily_search(rewritten_query)
        cleaned = str(search_text or "").strip()
        if cleaned:
            search_candidates.append(
                {
                    "content": cleaned,
                    "source": "tavily",
                    "document_id": None,
                    "metadata": {"query": rewritten_query},
                }
            )

    kb_evidence = _sanitize_evidence(kb_candidates, source_type="kb")
    search_evidence = _sanitize_evidence(search_candidates, source_type="search")
    subqueries = list(knowledge_plan.get("subqueries") or [])
    decision_reason = str(decision.get("decision_reason") or "")

    payload = KnowledgeDecisionPayload(
        original_query=query,
        rewritten_query=rewritten_query,
        multi_intent=bool(knowledge_plan.get("multi_intent")) and bool(subqueries),
        subqueries=subqueries,
        top_score=_safe_float(decision.get("top_score")),
        threshold=float(settings.kb_search_score_threshold),
        should_search=bool(decision.get("should_search")),
        decision_reason=decision_reason or None,
        kb_evidence=kb_evidence,
        search_evidence=search_evidence,
        unresolved=_is_unresolved(
            decision_reason=decision_reason,
            kb_evidence=kb_evidence,
            search_evidence=search_evidence,
        ),
        planner_notes=dict(knowledge_plan.get("planner_notes") or {}),
    )

    log.info(
        "knowledge_payload: query=%r rewritten=%r top_score=%.4f threshold=%.4f should_search=%s reason=%s kb_hits=%s search_hits=%s unresolved=%s",
        query,
        rewritten_query,
        float(payload.top_score or 0.0),
        float(payload.threshold or 0.0),
        payload.should_search,
        payload.decision_reason or "",
        len(payload.kb_evidence),
        len(payload.search_evidence),
        payload.unresolved,
    )
    return payload


async def build_knowledge_payload(
    *,
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    plan: Optional[KnowledgePlan] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    payload = await resolve_knowledge(
        query=query,
        history=history,
        session_context=None,
        plan=plan,
        top_k=top_k,
    )
    return _payload_to_dict(payload)
