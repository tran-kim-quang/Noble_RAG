"""RAG retrieval logic — routing, KB probe, decomposition, querying.

All functions here use the singletons from core.dependencies.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from lightrag import QueryParam

from core.dependencies import rag, llm_model_func
from core.config import get_settings
from utils.json_extract import extract_first_json_object
from utils.time import now_vietnam_str, timed_await
from tools.tavily_tool import tavily_search

settings = get_settings()
log = logging.getLogger("rag-service")


# ── Helpers ──────────────────────────────────────────────────────────────

def _normalize_search_query(original: str, refined: Optional[str]) -> str:
    candidate = (refined or original or "").strip()
    if not candidate:
        return "Việt Nam"
    lowered = candidate.lower()
    if "việt nam" in lowered or "vietnam" in lowered:
        return candidate
    return f"{candidate} tại Việt Nam"


# ── KB evidence probe ─────────────────────────────────────────────────────

async def kb_evidence_probe(
    query: str, history: List[Dict[str, Any]]
) -> bool:
    try:
        probe_query = (
            "Bạn là bộ kiểm tra bằng chứng nội bộ. "
            "Dựa trên ngữ cảnh truy xuất, chỉ trả về 1 token: KB_HIT hoặc KB_MISS.\n"
            f"Câu hỏi: {query}"
        )
        probe_response = await asyncio.wait_for(
            rag.aquery(
                probe_query,
                param=QueryParam(
                    top_k=2, mode="naive", conversation_history=history[-2:]
                ),
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


# ── Subquery decomposition ────────────────────────────────────────────────

async def decompose_subqueries(query: str) -> List[str]:
    prompt = f"""Bạn là bộ tách ý câu hỏi.
Trả về DUY NHẤT JSON:
{{
  "subqueries": ["...", "..."]
}}
Quy tắc:
- Nếu 1 ý → mảng 1 phần tử.
- Nếu nhiều ý → tách thành các câu độc lập, ngắn gọn nhưng đủ nghĩa.
- Giữ đúng THỨ TỰ ý như câu gốc.
- Không bỏ sót ý, không thêm ý mới.
Câu hỏi gốc: "{query}"""
    try:
        response = await llm_model_func(
            prompt, enable_cot=False, response_format={"type": "json_object"}
        )
        payload = extract_first_json_object(str(response))
        if not payload:
            return [query]
        data = json.loads(payload)
        items = data.get("subqueries", [])
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned if cleaned else [query]
    except Exception as e:
        log.error("Subquery decomposition failed: %s", e)
        return [query]


# ── Intent router ─────────────────────────────────────────────────────────

async def route_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Optional[str]]:
    """Route query to: RAG | SEARCH | OTHER.  Returns (category, refined_query)."""
    if history is None:
        history = []

    t0 = time.perf_counter()

    def _finish(cat: str, rq: Optional[str]) -> tuple[str, Optional[str]]:
        log.info("Router: %s (%.2fs)", cat, time.perf_counter() - t0)
        return cat, rq

    history_str = ""
    if history:
        recent = history[-2:]
        history_str = "LỊCH SỬ GẦN ĐÂY:\n" + "\n".join(
            [f"{m['role']}: {m['content']}" for m in recent]
        )

    prompt = f"""{history_str}
Thời gian hiện tại: {now_vietnam_str()}

BẠN LÀ BỘ ĐỊNH TUYẾN cho hệ thống RAG bất động sản.
Phân tích câu hỏi và trả về JSON:
{{
  "category": "SEARCH" | "RAG" | "OTHER",
  "search_query": "câu truy vấn tối ưu hoặc null",
  "confidence": 0.0-1.0,
  "has_project_intent": true|false,
  "needs_external_realtime": true|false
}}
- RAG: câu hỏi về dự án Noble, sản phẩm, chính sách, pháp lý.
- SEARCH: thông tin phụ thuộc thời gian hoặc nguồn bên ngoài như:
  giờ hiện tại, ngày tháng hiện tại, thời tiết hiện tại, lãi suất mới nhất, thị trường bên ngoài.
- OTHER: chào hỏi xã giao hoặc ngoài phạm vi.
Câu hỏi: "{query}"
CHỈ TRẢ VỀ JSON."""

    try:
        response = await timed_await(
            "router.classify",
            llm_model_func(prompt, enable_cot=False, response_format={"type": "json_object"}),
        )
        payload = extract_first_json_object(str(response))
        if not payload:
            raise ValueError("No JSON in router response")
        data = json.loads(payload)
        category = str(data.get("category", "RAG")).upper()
        if category not in {"SEARCH", "RAG", "OTHER"}:
            category = "RAG"
        search_query = str(data.get("search_query") or query).strip() or query
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.5
        has_project_intent = bool(data.get("has_project_intent", category == "RAG"))
        needs_external = bool(data.get("needs_external_realtime", category == "SEARCH"))

        # KB probe (skip if very confident)
        if confidence >= settings.router_skip_kb_probe_confidence:
            kb_hit = False
        else:
            kb_hit = await timed_await("router.kb_probe", kb_evidence_probe(query, history))

        if category == "SEARCH":
            if has_project_intent or kb_hit:
                return _finish("RAG", query)
            return _finish("SEARCH", _normalize_search_query(query, search_query))

        if category == "RAG":
            low_conf = confidence < settings.router_low_confidence_threshold
            if (not kb_hit) and (needs_external or (low_conf and not has_project_intent)):
                return _finish("SEARCH", _normalize_search_query(query, search_query))
            return _finish("RAG", query)

        if has_project_intent or kb_hit:
            return _finish("RAG", query)
        return _finish("OTHER", query)

    except Exception as e:
        log.error("Route query failed: %s", e)
        return _finish("RAG", query)


# ── Summarize search answer ───────────────────────────────────────────────

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
Thời gian hiện tại (UTC+7): {now_vietnam_str()}
Trả lời ngắn gọn (dưới {max_sentences} câu). Không dùng ký hiệu toán học.
Câu hỏi: {user_query}
Kết quả tìm kiếm:
{search_results}
Trả lời:"""
    raw = await llm_model_func(summary_prompt, history_messages=history)
    return str(raw)


# ── Direct RAG query (used by sales agent) ───────────────────────────────

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
                param=QueryParam(
                    top_k=top_k, mode="naive", conversation_history=history
                ),
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
