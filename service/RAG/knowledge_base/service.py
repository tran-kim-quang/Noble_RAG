"""Build structured knowledge payloads from KB retrieval and optional search fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.config import get_settings
from knowledge_base.evaluator import evaluate_kb_candidates, is_project_specific_query, is_realtime_query
from knowledge_base.planner import KnowledgePlan, build_knowledge_plan
from models.api_models import KnowledgeDecisionPayload, KnowledgeEvidenceItem, KnowledgeSubqueryResult
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


def _merge_evidence_lists(
    items: List[KnowledgeEvidenceItem],
    *,
    max_items: int = 6,
) -> List[KnowledgeEvidenceItem]:
    """Merge and deduplicate evidence items, maintaining order and score ranking."""
    seen: set[tuple[str, str, str]] = set()
    merged: List[KnowledgeEvidenceItem] = []

    # Sort by source type (kb first) and score descending
    sorted_items = sorted(
        items,
        key=lambda x: (
            x.source_type != "kb",  # False (kb) sorts before True (search)
            -(x.score or 0.0),  # Descending score
        ),
    )

    for item in sorted_items:
        content = str(item.content or "").strip()
        if not content:
            continue

        # Dedupe key: (source_type, source_name, normalized content)
        source_name = item.source_name or ""
        normalized_content = " ".join(content.lower().split())
        dedupe_key = (item.source_type, source_name, normalized_content[:200])

        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        merged.append(item)
        if len(merged) >= max_items:
            break

    return merged


async def _resolve_single_query(
    *,
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 6,
    rerank_profile: Optional[str] = None,
    force_external_search: bool = False,
    human_loop_threshold: float = 0.4,
) -> Dict[str, Any]:
    """Resolve a single query through KB retrieval and optional search fallback."""
    rewritten_query = str(query).strip() or query

    kb_candidates = await retrieve_kb_candidates(
        rewritten_query,
        top_k=top_k,
        rerank_profile=rerank_profile,
    )
    decision = evaluate_kb_candidates(
        rewritten_query,
        kb_candidates,
        history=history,
        threshold=settings.kb_search_score_threshold,
    )

    top_score = _safe_float(decision.get("top_score"))
    should_search_by_decision = bool(decision.get("should_search"))
    query_is_project_scope = bool(is_project_specific_query(rewritten_query, history))
    query_is_realtime = bool(is_realtime_query(rewritten_query))
    project_search_eligible = bool(query_is_project_scope)
    out_of_scope_non_project = bool((not project_search_eligible) and should_search_by_decision)

    if out_of_scope_non_project:
        should_search_by_decision = False

    approval_needed = (
        (not force_external_search)
        and project_search_eligible
        and (not out_of_scope_non_project)
        and top_score is not None
        and top_score < float(human_loop_threshold)
    )
    should_search_now = (project_search_eligible and (not out_of_scope_non_project)) and (bool(force_external_search) or (
        settings.enable_cosine_search_fallback
        and should_search_by_decision
        and (not approval_needed)
    ))

    search_candidates: List[Dict[str, Any]] = []
    if should_search_now:
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

    decision_reason = str(decision.get("decision_reason") or "")
    if out_of_scope_non_project:
        decision_reason = "out_of_scope_non_project"
    elif force_external_search:
        decision_reason = "human_approved_external_search"
    elif approval_needed and decision_reason:
        decision_reason = f"{decision_reason}:awaiting_user_search_approval"

    return {
        "query": rewritten_query,
        "top_score": top_score,
        "threshold": float(settings.kb_search_score_threshold),
        "should_search": bool(should_search_now),
        "decision_reason": decision_reason or None,
        "kb_evidence": kb_evidence,
        "search_evidence": search_evidence,
        "project_search_eligible": bool(project_search_eligible),
        "query_is_project_scope": bool(query_is_project_scope),
        "query_is_realtime": bool(query_is_realtime),
        "out_of_scope_non_project": bool(out_of_scope_non_project),
        "human_loop_search_approval_needed": bool(approval_needed),
        "human_loop_search_forced": bool(force_external_search),
        "unresolved": _is_unresolved(
            decision_reason=decision_reason,
            kb_evidence=kb_evidence,
            search_evidence=search_evidence,
        ),
    }


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
    force_external_search: bool = False,
    human_loop_threshold: float = 0.4,
) -> KnowledgeDecisionPayload:
    """Resolve knowledge with multi-intent support.
    
    When multi_intent=true, processes each subquery separately.
    When multi_intent=false, processes rewritten_query only.
    Maintains backward compatibility with chat.service by keeping merged payload fields.
    """
    del session_context
    knowledge_plan = plan or await build_knowledge_plan(query, history)
    rewritten_query = str(knowledge_plan.get("rewritten_query") or query).strip() or query
    subqueries = list(knowledge_plan.get("subqueries") or [])
    planner_notes = dict(knowledge_plan.get("planner_notes") or {})

    # Multi-intent path: process each subquery
    if knowledge_plan.get("multi_intent") and subqueries:
        log.info(
            "resolve_knowledge multi-intent: query=%r multi_intent=true subquery_count=%d",
            query,
            len(subqueries),
        )

        subquery_results_list: List[KnowledgeSubqueryResult] = []
        merged_kb: List[KnowledgeEvidenceItem] = []
        merged_search: List[KnowledgeEvidenceItem] = []
        all_should_search = False
        any_unresolved = False
        any_approval_needed = False
        any_out_of_scope = False
        max_score = None

        for idx, subquery_item in enumerate(subqueries):
            sq_query = str(subquery_item.get("query") or "").strip()
            if not sq_query:
                continue

            single = await _resolve_single_query(
                query=sq_query,
                history=history,
                top_k=top_k,
                rerank_profile=str(subquery_item.get("intent_hint") or ""),
                force_external_search=force_external_search,
                human_loop_threshold=human_loop_threshold,
            )

            log.info(
                "resolve_knowledge subquery[%d]: query=%r top_score=%.4f should_search=%s reason=%s kb_hits=%d search_hits=%d",
                idx,
                sq_query,
                float(single.get("top_score") or 0),
                single.get("should_search"),
                single.get("decision_reason"),
                len(single.get("kb_evidence", [])),
                len(single.get("search_evidence", [])),
            )

            subquery_result = KnowledgeSubqueryResult(
                query=sq_query,
                intent_hint=subquery_item.get("intent_hint"),
                source_preference=subquery_item.get("source_preference"),
                top_score=single["top_score"],
                threshold=single["threshold"],
                should_search=single["should_search"],
                decision_reason=single["decision_reason"],
                kb_evidence=single["kb_evidence"],
                search_evidence=single["search_evidence"],
                unresolved=single["unresolved"],
            )
            subquery_results_list.append(subquery_result)

            merged_kb.extend(single["kb_evidence"])
            merged_search.extend(single["search_evidence"])
            all_should_search = all_should_search or single["should_search"]
            any_unresolved = any_unresolved or single["unresolved"]
            any_approval_needed = any_approval_needed or bool(single.get("human_loop_search_approval_needed"))
            any_out_of_scope = any_out_of_scope or bool(single.get("out_of_scope_non_project"))

            score = single.get("top_score")
            if score is not None:
                max_score = score if max_score is None else max(max_score, score)

        # Merge and deduplicate evidence
        deduped_kb = _merge_evidence_lists(merged_kb, max_items=6)
        deduped_search = _merge_evidence_lists(merged_search, max_items=3)

        payload = KnowledgeDecisionPayload(
            original_query=query,
            rewritten_query=rewritten_query,
            multi_intent=True,
            subqueries=subqueries,
            subquery_results=subquery_results_list,
            top_score=max_score,
            threshold=float(settings.kb_search_score_threshold),
            should_search=all_should_search,
            decision_reason="multi_intent_mixed",
            kb_evidence=deduped_kb,
            search_evidence=deduped_search,
            unresolved=any_unresolved or (not deduped_kb and not deduped_search),
            planner_notes={
                **planner_notes,
                "out_of_scope_non_project": bool(any_out_of_scope),
                "human_loop_search_approval_needed": bool(any_approval_needed),
                "human_loop_search_forced": bool(force_external_search),
                "human_loop_cosine_threshold": float(human_loop_threshold),
            },
        )

        log.info(
            "knowledge_payload multi-intent merged: query=%r top_score=%s should_search=%s total_kb=%d total_search=%d unresolved=%s",
            query,
            payload.top_score,
            payload.should_search,
            len(payload.kb_evidence),
            len(payload.search_evidence),
            payload.unresolved,
        )
        return payload

    # Single-intent path: process rewritten_query only
    single = await _resolve_single_query(
        query=rewritten_query,
        history=history,
        top_k=top_k,
        rerank_profile=str(planner_notes.get("query_kind") or ""),
        force_external_search=force_external_search,
        human_loop_threshold=human_loop_threshold,
    )

    kb_evidence = single["kb_evidence"]
    search_evidence = single["search_evidence"]
    decision_reason = single["decision_reason"]

    payload = KnowledgeDecisionPayload(
        original_query=query,
        rewritten_query=rewritten_query,
        multi_intent=False,
        subqueries=subqueries,
        subquery_results=[],
        top_score=_safe_float(single["top_score"]),
        threshold=float(settings.kb_search_score_threshold),
        should_search=bool(single["should_search"]),
        decision_reason=decision_reason or None,
        kb_evidence=kb_evidence,
        search_evidence=search_evidence,
        unresolved=single["unresolved"],
        planner_notes={
            **planner_notes,
            "query_is_project_scope": bool(single.get("query_is_project_scope")),
            "query_is_realtime": bool(single.get("query_is_realtime")),
            "out_of_scope_non_project": bool(single.get("out_of_scope_non_project")),
            "human_loop_search_approval_needed": bool(single.get("human_loop_search_approval_needed")),
            "human_loop_search_forced": bool(force_external_search),
            "human_loop_cosine_threshold": float(human_loop_threshold),
        },
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
