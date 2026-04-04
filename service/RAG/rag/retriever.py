"""RAG retrieval logic — routing, KB probe, decomposition, querying.

All functions here use the singletons from core.dependencies.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from core.dependencies import rag, llm_model_func
from core.config import get_settings
from utils.time import now_vietnam_str
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


# ── Subquery decomposition ────────────────────────────────────────────────

async def decompose_subqueries(query: str) -> List[str]:
    text = re.sub(r"\s+", " ", (query or "")).strip()
    if not text:
        return []
    # Vector-only fast path: avoid extra LLM call for decomposition.
    return [text]


# ── Intent router ─────────────────────────────────────────────────────────

async def route_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Optional[str]]:
    """Fast rule-based router for vector-only pipeline."""
    if history is None:
        history = []
    text = re.sub(r"\s+", " ", (query or "")).strip()
    lowered = text.lower()
    if not text:
        return "OTHER", text

    project_keywords = {
        "noble",
        "tây thăng long",
        "tay thang long",
        "dự án",
        "du an",
        "pháp lý",
        "phap ly",
        "shophouse",
        "liền kề",
        "lien ke",
    }
    search_keywords = {
        "thời tiết",
        "thoi tiet",
        "tin tức",
        "tin tuc",
        "lãi suất",
        "lai suat",
        "tỷ giá",
        "ty gia",
        "giá vàng",
        "gia vang",
        "giờ hiện tại",
        "mấy giờ",
        "mấy giờ rồi",
        "hôm nay là ngày",
        "newest",
        "latest news",
    }
    greeting_patterns = (
        "xin chào",
        "chào",
        "hello",
        "hi",
    )

    if any(key in lowered for key in project_keywords):
        log.info("Router: RAG (rule project)")
        return "RAG", text

    if any(key in lowered for key in search_keywords):
        log.info("Router: SEARCH (rule realtime)")
        return "SEARCH", _normalize_search_query(text, text)

    if any(lowered == greet or lowered.startswith(greet + " ") for greet in greeting_patterns):
        log.info("Router: OTHER (rule greeting)")
        return "OTHER", text

    log.info("Router: RAG (rule default)")
    return "RAG", text


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
