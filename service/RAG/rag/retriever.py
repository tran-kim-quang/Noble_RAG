"""RAG retrieval logic - routing, decomposition, and querying."""

import asyncio
import json
import logging
import os
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
    return folded.replace("Ä‘", "d")


def _normalize_search_query(original: str, refined: Optional[str]) -> str:
    candidate = (refined or original or "").strip()
    if not candidate:
        return "Viá»‡t Nam"
    lowered = _fold_vn(candidate)
    if "viet nam" in lowered or "vietnam" in lowered:
        return candidate
    return f"{candidate} táº¡i Viá»‡t Nam"


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    payload = extract_first_json_object(str(raw or ""))
    if not payload:
        return {}
    try:
        data = json.loads(payload)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _token_overlap_ratio(query: str, content: str) -> float:
    q_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", _fold_vn(query))
        if len(tok) >= 2
    }
    if not q_tokens:
        return 0.0
    c_tokens = set(re.findall(r"[a-z0-9]+", _fold_vn(content)))
    if not c_tokens:
        return 0.0
    return len(q_tokens.intersection(c_tokens)) / float(len(q_tokens))


def _looks_like_boundary_reply(text: str) -> bool:
    folded = _fold_vn(text)
    boundary_hints = (
        "chi ho tro tu van du an bat dong san noble palace tay thang long",
        "vui long hoi ve san pham",
        "ngoai pham vi",
        "khong nam trong pham vi",
        "khong du du lieu noi bo",
    )
    return any(hint in folded for hint in boundary_hints)


def _extract_tavily_items(search_results: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for raw_line in (search_results or "").splitlines():
        line = (raw_line or "").strip()
        if not line.startswith("- "):
            continue

        match = re.match(r"^- (.*?): (.*) \((https?://[^)]+)\)\s*$", line)
        if match:
            items.append(
                {
                    "title": (match.group(1) or "").strip(),
                    "content": (match.group(2) or "").strip(),
                    "url": (match.group(3) or "").strip(),
                }
            )
            continue

        body = line[2:].strip()
        url_match = re.search(r"(https?://\S+)$", body)
        url = url_match.group(1).rstrip(")") if url_match else ""
        body_no_url = body[: url_match.start()].strip() if url_match else body
        if ":" in body_no_url:
            title, content = body_no_url.split(":", 1)
        else:
            title, content = body_no_url, ""
        items.append(
            {
                "title": title.strip(),
                "content": content.strip(),
                "url": url.strip(),
            }
        )
    return items


def _build_local_search_fallback_answer(user_query: str, search_results: str) -> str:
    text = (search_results or "").strip()
    if not text:
        return ""

    folded = _fold_vn(text)
    if "search tool not available" in folded:
        return "Sunny chưa thể dùng công cụ tìm kiếm bên ngoài ở thời điểm này."
    if "loi khi tim kiem" in folded:
        return "Sunny đã thử tìm nguồn bên ngoài nhưng gặp lỗi kết nối tạm thời."
    if "khong tim thay ket qua" in folded:
        return "Sunny chưa tìm thấy nguồn bên ngoài đủ rõ cho câu hỏi này."

    items = _extract_tavily_items(text)
    if not items:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:900]

    snippets: List[str] = []
    seen: set[str] = set()
    for item in items[:5]:
        snippet = re.sub(r"\s+", " ", item.get("content") or "").strip()
        if not snippet:
            snippet = re.sub(r"\s+", " ", item.get("title") or "").strip()
        if not snippet:
            continue
        key = _fold_vn(snippet)
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet.rstrip(".") + ".")
        if len(snippets) >= 2:
            break

    if not snippets:
        return "Sunny đã tìm được nguồn bên ngoài nhưng chưa trích được nội dung rõ ràng để tóm tắt."

    answer = f'Sunny tổng hợp nhanh từ nguồn bên ngoài cho câu hỏi "{user_query}": ' + " ".join(snippets)
    return answer

def _rerank_router_points(query: str, points: List[Any]) -> List[Any]:
    """Apply adapter reranker if available; keep behavior stable if unavailable."""
    try:
        rerank_fn = getattr(rag, "_rerank_points", None)
        if callable(rerank_fn):
            return list(rerank_fn(query, points) or points)
    except Exception as e:
        log.warning("router rerank failed, fallback raw points: %s", e)
    return points


def _router_rerank_weights() -> tuple[float, float]:
    vector_w = float((os.getenv("SIMPLE_RERANK_VECTOR_WEIGHT") or "0.7").strip() or "0.7")
    lexical_w = float((os.getenv("SIMPLE_RERANK_LEXICAL_WEIGHT") or "0.3").strip() or "0.3")
    return vector_w, lexical_w


def _point_final_rerank_score(query: str, point: Any) -> tuple[float, float, float]:
    payload = dict(getattr(point, "payload", None) or {})
    content = str(payload.get("content") or "")
    vector_score = float(getattr(point, "score", 0.0) or 0.0)
    lexical_score = _token_overlap_ratio(query, content)
    vector_w, lexical_w = _router_rerank_weights()
    final_score = vector_w * vector_score + lexical_w * lexical_score
    return vector_score, lexical_score, final_score


def _probe_point_source(point: Any) -> str:
    payload = dict(getattr(point, "payload", None) or {})
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    source = str(
        payload.get("file_path")
        or payload.get("source")
        or metadata.get("file_path")
        or metadata.get("source")
        or ""
    ).strip()
    if source:
        return source
    point_id = getattr(point, "id", None)
    return f"point_id={point_id}" if point_id is not None else "unknown_source"


def _probe_point_snippet(point: Any, max_len: int = 140) -> str:
    payload = dict(getattr(point, "payload", None) or {})
    content = re.sub(r"\s+", " ", str(payload.get("content") or "")).strip()
    if not content:
        return ""
    if len(content) <= max_len:
        return content
    return content[: max_len - 3].rstrip() + "..."


async def kb_evidence_probe(query: str, history: List[Dict[str, Any]]) -> bool:
    """Probe whether KB can answer using direct vector evidence (LLM-independent)."""
    try:
        q = (query or "").strip()
        if not q:
            return False

        # Use adapter-normalized vectors so probe dimension matches indexed collection.
        vector = (await rag._embed_texts([q]))[0]

        query_response = rag.qdrant.query_points(
            collection_name=rag.settings.collection_name,
            query=vector,
            # Retrieve wider candidates, then rerank and keep top window for probe.
            limit=12,
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
        points = _rerank_router_points(q, points)[:3]
        max_score = 0.0
        max_final_score = 0.0
        max_overlap = 0.0
        scored_points: List[tuple[Any, float, float, float]] = []
        for point in points:
            vector_score, lexical_score, final_score = _point_final_rerank_score(q, point)
            max_score = max(max_score, vector_score)
            max_overlap = max(max_overlap, lexical_score)
            max_final_score = max(max_final_score, final_score)
            scored_points.append((point, vector_score, lexical_score, final_score))

        # Rule: if final rerank score < 0.2, treat as KB miss and route to SEARCH tool.
        min_score = float((os.getenv("ROUTER_MIN_FINAL_SCORE") or "0.20").strip() or "0.20")
        kb_hit = max_final_score >= min_score

        log.info(
            "KB probe(rerank): %s final=%.3f vector=%.3f overlap=%.3f min_score=%.3f",
            "KB_HIT" if kb_hit else "KB_MISS",
            max_final_score,
            max_score,
            max_overlap,
            min_score,
        )
        for idx, (point, vector_score, lexical_score, final_score) in enumerate(
            sorted(scored_points, key=lambda item: item[3], reverse=True),
            start=1,
        ):
            log.info(
                "KB probe top_chunk[%d]: final=%.3f vector=%.3f overlap=%.3f source=%s snippet=%s",
                idx,
                final_score,
                vector_score,
                lexical_score,
                _probe_point_source(point),
                _probe_point_snippet(point),
            )
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
PhÃ¢n rÃ£ cÃ¢u há»i ngÆ°á»i dÃ¹ng thÃ nh cÃ¡c Ã½ Ä‘á»™c láº­p Ä‘á»ƒ xá»­ lÃ½.
Tráº£ vá» JSON duy nháº¥t:
{{
  "subqueries": ["..."]
}}

Quy táº¯c:
- Náº¿u cÃ¢u há»i chá»‰ cÃ³ 1 Ã½ thÃ¬ tráº£ Ä‘Ãºng 1 pháº§n tá»­.
- KhÃ´ng thÃªm thÃ´ng tin má»›i, khÃ´ng suy diá»…n ngoÃ i cÃ¢u gá»‘c.
- Tá»‘i Ä‘a 3 subqueries.

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
Báº¡n lÃ  bá»™ Ä‘á»‹nh tuyáº¿n truy váº¥n cho trá»£ lÃ½ báº¥t Ä‘á»™ng sáº£n.
Chá»n Ä‘Ãºng 1 route:
- RAG: cÃ¢u há»i cáº§n tri thá»©c ná»™i bá»™ dá»± Ã¡n/sáº£n pháº©m/chÃ­nh sÃ¡ch trong kho dá»¯ liá»‡u.
- SEARCH: cÃ¢u há»i cáº§n thÃ´ng tin realtime ngoÃ i há»‡ thá»‘ng (thá»i tiáº¿t, tin tá»©c, tá»· giÃ¡, giÃ¡ vÃ ng, giá» theo Ä‘á»‹a Ä‘iá»ƒm...).
- OTHER: chÃ o há»i, xÃ£ giao, hoáº·c cÃ¢u khÃ´ng cáº§n RAG/SEARCH.

Tráº£ vá» JSON duy nháº¥t:
{{
  "route": "RAG|SEARCH|OTHER",
  "refined_query": "<chuáº©n hÃ³a query ngáº¯n gá»n, giá»¯ nguyÃªn Ã½>"
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
- If user asks to compare external market/projects with "dá»± Ã¡n nhÃ  mÃ¬nh",
  treat "dá»± Ã¡n nhÃ  mÃ¬nh" as internal project knowledge (RAG) and external market part as SEARCH.
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
    fallback_answer = _build_local_search_fallback_answer(user_query, search_results)
    log.info(
        "search summarize mode=local_only query_len=%d result_len=%d",
        len(str(user_query or "")),
        len(str(search_results or "")),
    )
    return fallback_answer

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
        stream_iter = rag.aquery_stream(
            full_query,
            top_k=top_k,
            conversation_history=history,
        ).__aiter__()

        first_token_timeout = max(2.0, float(settings.query_stream_first_token_timeout_sec))
        chunk_timeout = max(2.0, float(settings.query_stream_chunk_timeout_sec))
        max_total = max(first_token_timeout, float(settings.query_timeout_sec))

        started = asyncio.get_event_loop().time()
        emitted_any = False

        while True:
            elapsed = asyncio.get_event_loop().time() - started
            if elapsed >= max_total:
                log.warning("RAG stream total timeout: %s", text[:80])
                break

            timeout = first_token_timeout if not emitted_any else chunk_timeout
            try:
                delta = await asyncio.wait_for(stream_iter.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                if not emitted_any:
                    log.warning("RAG stream first-token timeout: %s", text[:80])
                else:
                    log.warning("RAG stream chunk timeout: %s", text[:80])
                break

            if not delta:
                continue
            emitted_any = True
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
            # Wider fetch so reranker can lift lexically-relevant chunks.
            limit=max(12, top_k * 2),
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
        points = _rerank_router_points(user_text, points)[: max(4, top_k)]
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


async def get_rag_citation_sources(user_text: str, top_k: int = 8) -> List[str]:
    """Retrieve candidate source file paths for citation validation."""
    try:
        vector = (await rag._embed_texts([user_text]))[0]  # type: ignore[attr-defined]
        query_response = rag.qdrant.query_points(
            collection_name=rag.settings.collection_name,
            query=vector,
            # Wider fetch so reranker can prioritize citation-relevant chunks.
            limit=max(12, top_k * 2),
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
        points = _rerank_router_points(user_text, points)[: max(4, top_k)]
        sources: List[str] = []
        seen: set[str] = set()
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            file_path = str(
                payload.get("file_path")
                or metadata.get("file_path")
                or ""
            ).strip()
            if not file_path:
                continue
            key = file_path.lower()
            if key in seen:
                continue
            seen.add(key)
            sources.append(file_path)
        return sources
    except Exception as e:
        log.warning("get_rag_citation_sources failed: %s", e)
        return []


