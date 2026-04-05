"""RAG retrieval logic - routing, decomposition, and querying."""

import asyncio
import json
import logging
import re
import unicodedata
from typing import Any, AsyncIterator, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue

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
    """LLM-based decomposition only (no keyword hardcode)."""
    text = re.sub(r"\s+", " ", (query or "")).strip()
    if not text:
        return []

    prompt = f"""
Phân rã câu hỏi người dùng thành các ý độc lập để xử lý.
Trả về JSON duy nhất:
{{
  "subqueries": ["..."]
}}

Quy tắc:
- Nếu câu hỏi chỉ có 1 ý thì trả đúng 1 phần tử.
- Không thêm thông tin mới, không suy diễn ngoài câu gốc.
- Tối đa 3 subqueries.

User query: "{text}"
""".strip()

    try:
        raw = await llm_model_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        data = _parse_json_dict(str(raw))
        subqueries = data.get("subqueries")
        if isinstance(subqueries, list):
            cleaned = []
            for item in subqueries[:3]:
                value = re.sub(r"\s+", " ", str(item or "")).strip()
                if value:
                    cleaned.append(value)
            if cleaned:
                return cleaned
    except Exception as e:
        log.warning("decompose_subqueries failed, fallback single query: %s", e)

    return [text]


async def route_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Optional[str]]:
    """LLM-based router only (RAG/SEARCH/OTHER)."""
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
- RAG: câu hỏi cần tri thức nội bộ dự án/sản phẩm/chính sách trong kho dữ liệu.
- SEARCH: câu hỏi cần thông tin realtime ngoài hệ thống (thời tiết, tin tức, tỷ giá, giá vàng, giờ theo địa điểm...).
- OTHER: chào hỏi, xã giao, hoặc câu không cần RAG/SEARCH.

Trả về JSON duy nhất:
{{
  "route": "RAG|SEARCH|OTHER",
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

        if route not in {"RAG", "SEARCH", "OTHER"}:
            route = "RAG"

        if route == "SEARCH":
            refined = _normalize_search_query(text, refined)
        if not refined:
            refined = text

        log.info("Router (LLM): %s", route)
        return route, refined
    except Exception as e:
        log.warning("route_query failed, fallback RAG: %s", e)
        return "RAG", text


async def plan_query_adaptive(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Single LLM planner for route + optional decomposition."""
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

    history_lines = []
    for item in (history or [])[-6:]:
        role = str(item.get("role") or "user").strip()
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "(empty)"

    prompt = f"""
You are a query planner for a real-estate assistant.

Allowed routes:
- RAG: answer from internal project knowledge base.
- SEARCH: answer requires external realtime/public information.
- OTHER: social/small talk or generic conversation.

Return JSON only:
{{
  "route": "RAG|SEARCH|OTHER",
  "refined_query": "short rewritten query with same meaning",
  "multi_intent": true|false,
  "subqueries": [
    {{"query": "...", "route": "RAG|SEARCH|OTHER"}}
  ]
}}

Rules:
- Use decomposition only when the user has 2 independent intents.
- If not needed, set multi_intent=false and subqueries=[].
- Max 2 subqueries.
- Keep intent and facts faithful to original user query.
- If user asks to compare external market/projects with "dự án nhà mình",
  treat "dự án nhà mình" as internal project knowledge (RAG) and external market part as SEARCH.
- For comparison queries, prefer:
  multi_intent=true with exactly 2 subqueries:
  one SEARCH subquery (external comparison baseline),
  one RAG subquery (internal project side to compare).

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
    except Exception as e:
        log.warning("plan_query_adaptive failed, fallback route_query: %s", e)
        route, refined = await route_query(text, history)
        return {
            "route": route,
            "refined_query": refined or text,
            "multi_intent": False,
            "subqueries": [],
        }

    route = str(data.get("route") or "").upper().strip()
    if route not in {"RAG", "SEARCH", "OTHER"}:
        route = "RAG"

    refined = re.sub(r"\s+", " ", str(data.get("refined_query") or text)).strip() or text
    if route == "SEARCH":
        refined = _normalize_search_query(text, refined)

    raw_subqueries = data.get("subqueries")
    planned_subqueries: List[Dict[str, str]] = []
    if isinstance(raw_subqueries, list):
        for item in raw_subqueries[:2]:
            if not isinstance(item, dict):
                continue
            sq_query = re.sub(r"\s+", " ", str(item.get("query") or "")).strip()
            sq_route = str(item.get("route") or "").upper().strip()
            if not sq_query:
                continue
            if sq_route not in {"RAG", "SEARCH", "OTHER"}:
                sq_route = route
            if sq_route == "SEARCH":
                sq_query = _normalize_search_query(sq_query, sq_query)
            planned_subqueries.append({"query": sq_query, "route": sq_route})

    multi_intent = bool(data.get("multi_intent")) and len(planned_subqueries) > 1
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
    summary_prompt = f"""{system_persona}

Hãy trả lời dựa trên kết quả tìm kiếm.
Trả lời ngắn gọn (dưới {max_sentences} câu). Không dùng ký hiệu toán học.
Câu hỏi: {user_query}
Kết quả tìm kiếm:
{search_results}
Trả lời:"""
    raw = await llm_model_func(summary_prompt, history_messages=history)
    return str(raw)


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
        vector = (await rag._embed_texts([user_text]))[0]  # type: ignore[attr-defined]
        query_response = rag.qdrant.query_points(
            collection_name=rag.settings.collection_name,
            query=vector,
            limit=max(4, top_k),
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
        points = list(getattr(query_response, "points", []) or [])
        corpus_parts: List[str] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            content = str(payload.get("content") or "").strip()
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
