"""RAG retrieval logic - routing, decomposition, and querying."""

import asyncio
import json
import logging
import re
import unicodedata
from typing import Any, AsyncIterator, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue

from knowledge_base.planner import (
    KnowledgePlan,
    decompose_subqueries as planner_decompose_subqueries,
    build_knowledge_plan,
)
from core.config import get_settings
from core.dependencies import llm_model_func, rag
from tools.tavily_tool import tavily_search
from utils.json_extract import extract_first_json_object
from utils.time import now_vietnam_str

settings = get_settings()
log = logging.getLogger("rag-service")


def _fold_vn(text: str) -> str:
    raw = (text or "").strip().lower()
    folded = unicodedata.normalize("NFD", raw)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def _normalize_search_query(original: str, refined: Optional[str]) -> str:
    candidate = (refined or original or "").strip()
    if not candidate:
        return "Việt Nam"
    lowered = _fold_vn(candidate)
    if "viet nam" in lowered or "vietnam" in lowered:
        return candidate
    return f"{candidate} tại Việt Nam"


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    payload = extract_first_json_object(str(raw or ""))
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


async def kb_evidence_probe(query: str, history: List[Dict[str, Any]]) -> bool:
    """Probe whether KB can answer; returns True/False."""
    try:
        probe_query = (
            "Bạn là bộ kiểm tra bằng chứng nội bộ. "
            "Dựa trên ngữ cảnh truy xuất, chỉ trả về 1 token: KB_HIT hoặc KB_MISS.\n"
            f"Câu hỏi: {query}"
        )
        probe_response = await asyncio.wait_for(
            rag.aquery(
                probe_query,
                top_k=2,
                conversation_history=history[-2:],
            ),
            timeout=min(settings.query_timeout_sec, 20),
        )
        text = (probe_response if isinstance(probe_response, str) else str(probe_response)).upper()
        kb_hit = "KB_HIT" in text and "KB_MISS" not in text
        log.info("KB probe: %s", "KB_HIT" if kb_hit else "KB_MISS")
        return kb_hit
    except Exception as e:
        log.warning("KB probe failed: %s", e)
        return False


async def decompose_subqueries(query: str) -> List[str]:
    """Compatibility wrapper around the extracted knowledge-base planner."""
    return await planner_decompose_subqueries(query)


def _legacy_route_from_plan(plan: KnowledgePlan) -> str:
    notes = plan.get("planner_notes") or {}
    query_kind = str(notes.get("query_kind") or "").strip().lower()
    if query_kind == "greeting":
        return "OTHER"
    if query_kind in {"fact", "consultative", "comparison", "objection", "generic", "unknown"}:
        return "RAG"
    return "RAG"


def _legacy_subqueries_from_plan(plan: KnowledgePlan, route: str) -> List[Dict[str, str]]:
    planned_subqueries = plan.get("subqueries") or []
    if not (plan.get("multi_intent") and planned_subqueries):
        return []

    normalized: List[Dict[str, str]] = []
    for item in planned_subqueries[:2]:
        query = re.sub(r"\s+", " ", str(item.get("query") or "")).strip()
        if not query:
            continue
        intent_hint = str(item.get("intent_hint") or "").strip().lower()
        sub_route = "RAG"
        if intent_hint == "unknown" and route == "OTHER":
            sub_route = "OTHER"
        normalized.append({"query": query, "route": sub_route})
    return normalized


async def route_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Optional[str]]:
    """LLM-based router only (SALES/RAG/SEARCH/OTHER)."""
    if history is None:
        history = []

    text = re.sub(r"\s+", " ", (query or "")).strip()
    if not text:
        return "OTHER", text

    history_lines = []
    for item in (history or [])[-6:]:
        role = str(item.get("role") or "user").strip()
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "(empty)"

    prompt = f"""
Bạn là bộ định tuyến truy vấn cho trợ lý bất động sản.
Chọn đúng 1 route:
- SALES: câu hội thoại tư vấn/mở nhu cầu/khám phá nhu cầu, cần vào sales flow theo trạng thái hội thoại.
- RAG: câu hỏi cần tri thức nội bộ dự án/sản phẩm/chính sách trong kho dữ liệu.
- SEARCH: câu hỏi cần thông tin realtime ngoài hệ thống (thời tiết, tin tức, tỷ giá, giá vàng, giờ theo địa điểm...).
- OTHER: chào hỏi, xã giao, hoặc câu không cần RAG/SEARCH.

Trả về JSON duy nhất:
{{
  "route": "SALES|RAG|SEARCH|OTHER",
  "refined_query": "<chuẩn hóa query ngắn gọn, giữ nguyên ý>"
}}

History:
{history_text}

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
        route = str(data.get("route") or "").upper().strip()
        refined = re.sub(r"\s+", " ", str(data.get("refined_query") or text)).strip()

        if route not in {"SALES", "RAG", "SEARCH", "OTHER"}:
            route = "SALES"

        if route == "SEARCH":
            refined = _normalize_search_query(text, refined)
        if not refined:
            refined = text

        log.info("Router (LLM): %s", route)
        return route, refined
    except Exception as e:
        log.warning("route_query failed, fallback SALES: %s", e)
        return "SALES", text


async def plan_query_adaptive(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper that adapts the extracted planner to the legacy schema."""
    if history is None:
        history = []

    text = re.sub(r"\s+", " ", (query or "")).strip()
    if not text:
        return {
            "route": "OTHER",
            "refined_query": "",
            "multi_intent": False,
            "subqueries": [],
        }
    try:
        plan = await build_knowledge_plan(text, history)
    except Exception as e:
        log.warning("plan_query_adaptive failed, fallback route_query: %s", e)
        route, refined = await route_query(text, history)
        return {
            "route": route,
            "refined_query": refined or text,
            "multi_intent": False,
            "subqueries": [],
        }

    refined = re.sub(r"\s+", " ", str(plan.get("rewritten_query") or text)).strip() or text
    route = _legacy_route_from_plan(plan)
    planned_subqueries = _legacy_subqueries_from_plan(plan, route)
    multi_intent = bool(plan.get("multi_intent")) and len(planned_subqueries) > 1
    if not multi_intent:
        planned_subqueries = []

    log.info(
        "Adaptive planner: route=%s multi_intent=%s subqueries=%s",
        route,
        multi_intent,
        len(planned_subqueries),
    )
    return {
        "route": route,
        "refined_query": refined,
        "multi_intent": multi_intent,
        "subqueries": planned_subqueries,
    }


async def summarize_search_answer(
    user_query: str,
    search_query: str,
    history: List[Dict[str, Any]],
    system_persona: str,
    max_sentences: int = 3,
) -> str:
    normalized = _normalize_search_query(user_query, search_query)
    search_results = await tavily_search(normalized)
    return await summarize_search_results(
        user_query=user_query,
        search_results=search_results,
        history=history,
        system_persona=system_persona,
        max_sentences=max_sentences,
    )


async def summarize_search_results(
    user_query: str,
    search_results: str,
    history: List[Dict[str, Any]],
    system_persona: str,
    max_sentences: int = 3,
) -> str:
    summary_prompt = f"""{system_persona}

Hãy trả lời dựa trên kết quả tìm kiếm.
Trả lời ngắn gọn (dưới {max_sentences} câu). Không dùng ký hiệu toán học.
Câu hỏi: {user_query}
Kết quả tìm kiếm:
{search_results}
Trả lời:"""
    raw = await llm_model_func(summary_prompt, history_messages=history)
    return str(raw)


async def retrieve_kb_candidates(
    query: str,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """Low-level KB retrieval with raw similarity scores for confidence evaluation."""
    search_query = re.sub(r"\s+", " ", (query or "")).strip()
    if not search_query:
        return []

    try:
        vector = (await rag._embed_texts([search_query]))[0]  # type: ignore[attr-defined]
        query_response = rag.qdrant.query_points(
            collection_name=rag.settings.collection_name,
            query=vector,
            limit=max(1, top_k),
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="workspace",
                        match=MatchValue(value=rag.settings.workspace),
                    )
                ]
            ),
        )
    except Exception as e:
        log.warning("retrieve_kb_candidates failed: %s", e)
        return []

    candidates: List[Dict[str, Any]] = []
    for point in list(getattr(query_response, "points", []) or []):
        payload = dict(getattr(point, "payload", None) or {})
        content = str(payload.get("content") or "").strip()
        if not content:
            continue
        source = str(payload.get("file_path") or payload.get("document_id") or "unknown")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"content", "file_path", "document_id"}
        }
        try:
            score = float(getattr(point, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        candidates.append(
            {
                "score": score,
                "content": content,
                "source": source,
                "document_id": str(payload.get("document_id") or ""),
                "metadata": metadata,
            }
        )
    return candidates


async def query_rag(
    text: str,
    top_k: int = 5,
    history: Optional[List[Dict[str, Any]]] = None,
    system_context: str = "",
) -> str:
    """Simple RAG query returning a string answer."""
    if history is None:
        history = []
    full_query = f"{system_context}\n\n{text}".strip() if system_context else text
    try:
        response = await asyncio.wait_for(
            rag.aquery(
                full_query,
                top_k=top_k,
                conversation_history=history,
            ),
            timeout=settings.query_timeout_sec,
        )
        if response is None:
            return ""
        return response if isinstance(response, str) else str(response)
    except asyncio.TimeoutError:
        log.warning("RAG query timed out: %s", text[:80])
        return ""
    except Exception as e:
        log.error("RAG query error: %s", e)
        return ""


async def query_rag_stream(
    text: str,
    top_k: int = 5,
    history: Optional[List[Dict[str, Any]]] = None,
    system_context: str = "",
) -> AsyncIterator[str]:
    """Stream RAG answer deltas as they are generated."""
    if history is None:
        history = []
    full_query = f"{system_context}\n\n{text}".strip() if system_context else text
    try:
        async for delta in rag.aquery_stream(
            full_query,
            top_k=top_k,
            conversation_history=history,
        ):
            if delta:
                yield delta if isinstance(delta, str) else str(delta)
    except Exception as e:
        log.error("RAG stream query error: %s", e)


def _extract_generic_evidence_lines(query: str, corpus: str, max_lines: int = 8) -> List[str]:
    query_folded = _fold_vn(query)
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", query_folded)
        if len(token) >= 3
    }
    lines = re.split(r"[\r\n]+", corpus or "")
    results: List[str] = []
    seen: set[str] = set()

    for raw_line in lines:
        line = re.sub(r"\s+", " ", (raw_line or "").strip())
        if not line:
            continue
        folded = _fold_vn(line)
        has_number = bool(re.search(r"\d", line))
        has_query_overlap = any(token in folded for token in query_tokens) if query_tokens else False
        if not (has_number or has_query_overlap):
            continue
        if folded in seen:
            continue
        seen.add(folded)
        results.append(line)
        if len(results) >= max_lines:
            break

    return results


async def build_rag_fact_constraints(user_text: str, top_k: int = 8) -> str:
    """Retrieve evidence from Qdrant and build generic constraints."""
    try:
        corpus_parts: List[str] = []
        candidates = await retrieve_kb_candidates(user_text, top_k=max(4, top_k))
        for item in candidates:
            content = str(item.get("content") or "").strip()
            if content:
                corpus_parts.append(content)
        if not corpus_parts:
            return ""

        evidence_lines = _extract_generic_evidence_lines(user_text, "\n".join(corpus_parts))
        if not evidence_lines:
            return ""

        rules = [
            "Evidence constraints (must follow):",
            "- When answering factual details, only use statements in 'Evidence lines' below.",
            "- Do not invent, swap, or rewrite numeric mappings from those lines.",
            "Evidence lines:",
        ]
        rules.extend(f"- {line}" for line in evidence_lines)
        return "\n".join(rules)
    except Exception as e:
        log.warning("build_rag_fact_constraints failed: %s", e)
        return ""
