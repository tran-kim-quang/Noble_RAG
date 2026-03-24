"""
LightRAG HTTP API Service
------------------------
Endpoints:
  POST /upload-document   - Upload document, save to graph/vector storage
  POST /query             - Query with document context, receive LLM response
  GET  /health            - Health check
  GET  /models            - List available LLM models
"""

import io
import os
import time
import json
import re
import asyncio
import logging
import tempfile
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from enum import Enum
from functools import partial

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncpg

# LightRAG imports
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed,
    wrap_embedding_func_with_attrs,
)

# Tools imports
try:
    from scripts.tools.search import tavily_search
except ImportError:
    log.warning("scripts.tools.search.tavily_search not found. Search functionality will be limited.")
    async def tavily_search(query, **kwargs):
        return "Search tool not available."

# ── Config via env vars ────────────────────────────────────────────────
LLM_PROVIDER     = os.getenv("LLM_PROVIDER",       "openai")     # openai, azure, ollama, gemini, claude
LLM_MODEL        = os.getenv("LLM_MODEL",          "gpt-4o-mini")
LLM_API_KEY      = os.getenv("LLM_API_KEY",        "")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER)
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",    "text-embedding-3-large")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM",  "3072"))

POSTGRES_URL     = os.getenv("POSTGRES_URL",       "postgresql://rag:rag@localhost:5432/rag_db")
QDRANT_URL       = os.getenv("QDRANT_URL",         "http://localhost:6333")
REDIS_URL        = os.getenv("REDIS_URL",          "redis://localhost:6379/0")

KV_STORAGE       = os.getenv("KV_STORAGE",         "PGKVStorage")
VECTOR_STORAGE   = os.getenv("VECTOR_STORAGE",     "QdrantVectorDBStorage")
GRAPH_STORAGE    = os.getenv("GRAPH_STORAGE",      "PGGraphStorage")
DOC_STATUS_STORAGE = os.getenv("DOC_STATUS_STORAGE", "PGDocStatusStorage")
RAG_WORKSPACE    = os.getenv("RAG_WORKSPACE",      "default")
RAG_WORKING_DIR  = os.getenv("RAG_WORKING_DIR",    "./rag_db")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",     "1024"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",  "20"))
LOG_LEVEL        = os.getenv("LOG_LEVEL",          "INFO")
QUERY_TIMEOUT_SEC = float(os.getenv("QUERY_TIMEOUT_SEC", "45"))
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
ROUTER_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_LOW_CONFIDENCE_THRESHOLD", "0.60"))
SEARCH_FALLBACK_ON_EMPTY_RAG = os.getenv("SEARCH_FALLBACK_ON_EMPTY_RAG", "true").lower() in {"1", "true", "yes", "on"}
SUBQUERY_MAX_CONCURRENCY = max(1, int(os.getenv("SUBQUERY_MAX_CONCURRENCY", "4")))
ROUTER_SKIP_KB_PROBE_CONFIDENCE = float(os.getenv("ROUTER_SKIP_KB_PROBE_CONFIDENCE", "0.90"))
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rag-service")


def configure_storage_env() -> None:
    """Map compose-style URLs to LightRAG backend environment variables."""
    parsed_postgres = urlparse(POSTGRES_URL)

    if parsed_postgres.scheme.startswith("postgres"):
        if parsed_postgres.hostname and not os.getenv("POSTGRES_HOST"):
            os.environ["POSTGRES_HOST"] = parsed_postgres.hostname
        if parsed_postgres.port and not os.getenv("POSTGRES_PORT"):
            os.environ["POSTGRES_PORT"] = str(parsed_postgres.port)
        if parsed_postgres.username and not os.getenv("POSTGRES_USER"):
            os.environ["POSTGRES_USER"] = parsed_postgres.username
        if parsed_postgres.password and not os.getenv("POSTGRES_PASSWORD"):
            os.environ["POSTGRES_PASSWORD"] = parsed_postgres.password
        db_name = parsed_postgres.path.lstrip("/")
        if db_name and not os.getenv("POSTGRES_DATABASE"):
            os.environ["POSTGRES_DATABASE"] = db_name

    if QDRANT_URL and not os.getenv("QDRANT_URL"):
        os.environ["QDRANT_URL"] = QDRANT_URL

    if REDIS_URL and not os.getenv("REDIS_URI"):
        os.environ["REDIS_URI"] = REDIS_URL


# ── Enums ──────────────────────────────────────────────────────────────
class StorageType(str, Enum):
    """Storage type for documents"""
    GRAPH = "graph"
    VECTOR = "vector"
    BOTH = "both"


class LLMProviderType(str, Enum):
    """LLM Provider types"""
    OPENAI = "openai"
    CLAUDE = "claude"
    OLLAMA = "ollama"


# ── Pydantic Models ────────────────────────────────────────────────────
class UploadDocumentRequest(BaseModel):
    """Request model for document upload"""
    storage_type: StorageType = StorageType.GRAPH
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    """Request model for query"""
    query: str
    top_k: int = 8
    return_structured_output: bool = False
    session_id: Optional[str] = "default_session"


class UploadDocumentResponse(BaseModel):
    """Response model for document upload"""
    status: str
    document_id: str
    chunks_count: int
    storage_type: str
    message: str


class QueryResponse(BaseModel):
    """Response model for query"""
    response: str
    context: List[str]
    tokens_used: Optional[dict] = None
    model: str


class DocumentRecord(BaseModel):
    id: str
    status: Optional[str] = None
    file_path: Optional[str] = None
    content_length: Optional[int] = None
    chunks_count: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[dict] = None


# ── External Tools ─────────────────────────────────────────────────────
# (Imported from scripts.tools.search)


# ── Conversation History (Redis) ──────────────────────────────────────
async def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """Retrieve chat history from Redis"""
    import redis.asyncio as redis
    client = None
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
        history_json = await client.get(f"chat_history:{session_id}")
        if history_json:
            data = json.loads(history_json)
            log.info(f"Redis: Found {len(data)} messages for session {session_id}")
            return data
    except Exception as e:
        log.error(f"Redis get error (session={session_id}): {e}")
    finally:
        if client:
            await client.aclose()
    return []


async def save_chat_history(session_id: str, history: List[Dict[str, Any]], max_len: int = 10):
    """Save chat history to Redis with a limit"""
    import redis.asyncio as redis
    client = None
    try:
        # Keep only the last N messages (N * 2 for user+assistant)
        history_trimmed = history[-(max_len * 2):]  # type: ignore
        data = json.dumps(history_trimmed)
        
        client = redis.Redis.from_url(REDIS_URL, socket_timeout=5, socket_connect_timeout=5)
        await client.setex(f"chat_history:{session_id}", 86400, data)
        log.info(f"Redis: Saved {len(history_trimmed)} messages for {session_id}")
    except Exception as e:
        log.error(f"Redis save error (session={session_id}): {e}")
    finally:
        if client:
            await client.aclose()


# ── Real Estate Assistant Persona ─────────────────────────────────────
SYSTEM_PERSONA = """Bạn là trợ lý tư vấn chuyên sâu về các dự án bất động sản của Noble.

PHẠM VI TRẢ LỜI CỐ ĐỊNH:
1. Thông tin dự án, sản phẩm, chính sách bán hàng, pháp lý bất động sản của Noble.
2. Thông tin về ngày giờ hiện tại và thời tiết tại một địa điểm (thông qua tra cứu).

LUẬT TỪ CHỐI (QUAN TRỌNG):
- TUYỆT ĐỐI KHÔNG trả lời các chủ đề: Thể thao, Giải trí, Chính trị, Tôn giáo, Kiến thức chung không liên quan (như công nghệ, nấu ăn, y tế, v.v.), hoặc đời tư.
- Nếu khách hỏi ngoài phạm vi trên, hãy lịch sự từ chối: "Dạ, em là trợ lý chuyên biệt về dự án Noble Palace, em chỉ có thể hỗ trợ Anh/Chị thông tin về dự án, bất động sản, thời tiết và ngày giờ ạ. Anh/Chị có thắc mắc nào về [tên một tiện ích/dự án trong Noble] không ạ?"

QUY TẮC GIAO TIẾP:
- Xưng "em", gọi khách là "Anh/Chị".
- Trả lời ngắn gọn, đi thẳng vào vấn đề.
- KHÔNG gửi lời chào "Dạ em chào Anh/Chị" một cách máy móc trong mỗi câu trả lời nếu cuộc hội thoại đang diễn ra.
- KHÔNG bịa đặt thông tin. Nếu không biết, mời để lại số điện thoại."""


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from LLM output text."""
    if not text:
        return None

    def _extract_balanced_from(source: str) -> Optional[str]:
        start = source.find("{")
        if start < 0:
            return None

        depth = 0
        for idx in range(start, len(source)):
            char = source[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[start : idx + 1]
        return None

    clean_text = text.strip()

    if "```" in clean_text:
        blocks = clean_text.split("```")
        for block in blocks:
            candidate = block.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            extracted = _extract_balanced_from(candidate)
            if extracted:
                return extracted

    return _extract_balanced_from(clean_text)


def _now_vietnam_str() -> str:
    """Current time in Vietnam timezone (Asia/Ho_Chi_Minh)."""
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _now_vietnam_human() -> str:
    """Human-friendly current Vietnam time in Vietnamese."""
    now_vn = datetime.now(VN_TZ)
    return (
        f"{now_vn.hour} giờ {now_vn.minute:02d} phút "
        f"ngày {now_vn.day:02d} tháng {now_vn.month:02d} năm {now_vn.year}"
    )


async def _analyze_time_intent(query: str, llm_func) -> tuple[bool, bool]:
    """Use LLM to detect if query asks current time and whether a specific location is provided."""
    prompt = f"""Bạn là bộ phân tích intent.
Trả về DUY NHẤT JSON:
{{
  "asks_current_time": true|false,
  "has_specific_location": true|false
}}

Định nghĩa:
- asks_current_time=true nếu câu hỏi yêu cầu thời điểm hiện tại (mấy giờ, hiện tại bao nhiêu giờ, current time, now time...).
- has_specific_location=true nếu người dùng đã nêu rõ địa điểm/khu vực cần lấy giờ.

Câu hỏi: "{query}"""  # noqa: E501

    try:
        response = await llm_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        payload = _extract_first_json_object(str(response))
        if not payload:
            raise ValueError("No JSON object found in time intent response")

        data = json.loads(payload)
        asks_current_time = bool(data.get("asks_current_time", False))
        has_specific_location = bool(data.get("has_specific_location", False))
        log.info(
            "Time intent: asks_current_time=%s has_specific_location=%s",
            asks_current_time,
            has_specific_location,
        )
        return asks_current_time, has_specific_location
    except Exception as e:
        log.error(f"Time intent parse failed (pass 1): {e}")
        try:
            retry_prompt = prompt + "\n\nBẮT BUỘC: Trả về DUY NHẤT JSON object hợp lệ, không markdown, không text thừa."
            retry_response = await llm_func(
                retry_prompt,
                enable_cot=False,
                response_format={"type": "json_object"},
            )
            retry_payload = _extract_first_json_object(str(retry_response))
            if not retry_payload:
                raise ValueError("No JSON object found in time intent retry response")

            retry_data = json.loads(retry_payload)
            asks_current_time = bool(retry_data.get("asks_current_time", False))
            has_specific_location = bool(retry_data.get("has_specific_location", False))
            log.info(
                "Time intent (retry): asks_current_time=%s has_specific_location=%s",
                asks_current_time,
                has_specific_location,
            )
            return asks_current_time, has_specific_location
        except Exception as retry_error:
            log.error(f"Time intent parse failed (pass 2): {retry_error}")
            return False, False


async def _kb_evidence_probe(query: str, history: List[Dict[str, Any]]) -> bool:
    """Quick KB evidence probe using a constrained RAG call to reduce mis-routing."""
    try:
        probe_query = (
            "Bạn là bộ kiểm tra bằng chứng nội bộ. "
            "Dựa trên ngữ cảnh truy xuất từ kho tài liệu, chỉ trả về đúng 1 token: KB_HIT hoặc KB_MISS.\n"
            f"Câu hỏi: {query}"
        )
        probe_response = await asyncio.wait_for(
            rag.aquery(
                probe_query,
                param=QueryParam(
                    top_k=2,
                    mode="naive",
                    conversation_history=history[-2:],
                ),
            ),
            timeout=min(QUERY_TIMEOUT_SEC, 20),
        )
        probe_text = (probe_response if isinstance(probe_response, str) else str(probe_response)).strip().upper()
        kb_hit = "KB_HIT" in probe_text and "KB_MISS" not in probe_text
        log.info("KB evidence probe: %s", "KB_HIT" if kb_hit else "KB_MISS")
        return kb_hit
    except Exception as e:
        log.warning(f"KB evidence probe failed, defaulting to no evidence: {e}")
        return False


async def _decompose_subqueries(query: str, llm_func) -> List[str]:
    """Decompose a mixed-intent query into standalone subqueries using LLM."""
    prompt = f"""Bạn là bộ tách ý câu hỏi.
Trả về DUY NHẤT JSON:
{{
  "subqueries": ["...", "..."]
}}

Quy tắc:
- Nếu câu hỏi chỉ có 1 ý, trả mảng gồm đúng 1 phần tử là câu gốc.
- Nếu câu có nhiều ý (ví dụ vừa hỏi thời gian vừa hỏi dự án), tách thành các câu độc lập, ngắn gọn, đủ nghĩa.
- Không thêm thông tin mới.
- KHÔNG được bỏ sót ý nào trong câu gốc, kể cả mệnh đề cuối sau từ nối.
- Giữ thứ tự ý như câu gốc.

Câu hỏi gốc: "{query}"""  # noqa: E501

    try:
        response = await llm_func(
            prompt,
            enable_cot=False,
            response_format={"type": "json_object"},
        )
        payload = _extract_first_json_object(str(response))
        if not payload:
            raise ValueError("No JSON object found in decomposition response")

        data = json.loads(payload)
        raw_items = data.get("subqueries", [])
        if not isinstance(raw_items, list):
            return [query]

        cleaned: List[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)

        if not cleaned:
            return [query]

        log.info("Subquery decomposition count=%s items=%s", len(cleaned), cleaned)
        return cleaned
    except Exception as e:
        log.error(f"Subquery decomposition failed: {e}")
        return [query]


async def _timed_await(label: str, awaitable):
    """Await a coroutine and emit elapsed time to logs."""
    started = time.perf_counter()
    try:
        return await awaitable
    finally:
        elapsed = time.perf_counter() - started
        log.info("Timing[%s]: %.2fs", label, elapsed)


def _normalize_search_query_for_vietnam(original_query: str, refined_query: Optional[str]) -> str:
    """Prioritize Vietnam context for SEARCH queries without keyword hardcoding."""
    candidate = (refined_query or original_query or "").strip()
    if not candidate:
        return "Việt Nam"

    lowered = candidate.lower()
    if "việt nam" in lowered or "vietnam" in lowered:
        return candidate

    return f"{candidate} tại Việt Nam"


async def _summarize_search_answer(
    user_query: str,
    search_query: str,
    history: List[Dict[str, Any]],
    max_sentences: int = 3,
) -> str:
    """Run Tavily search then summarize answer in assistant persona."""
    normalized_query = _normalize_search_query_for_vietnam(user_query, search_query)
    search_results = await tavily_search(normalized_query)
    summary_prompt = f"""{SYSTEM_PERSONA}

Anh/Chị vừa hỏi về thông tin bên ngoài dự án. Em sẽ tra cứu và cung cấp thông tin chính xác.
Thời gian hiện tại (Việt Nam - UTC+7): {_now_vietnam_str()}

QUY TẮC TRẢ LỜI:
- Trả lời ngắn gọn, dưới {max_sentences} câu.
- Chỉ trích dẫn thông tin từ kết quả tìm kiếm.
- Không dùng ký hiệu toán học hay LaTeX.
- Ưu tiên thông tin theo ngữ cảnh Việt Nam; nếu có nhiều khu vực, ưu tiên dữ liệu tại Việt Nam.
- Với câu hỏi về thời gian hiện tại, trả theo múi giờ Việt Nam (UTC+7) nếu người dùng không chỉ định nơi khác.
- Nếu kết quả chứa múi giờ nước ngoài, quy đổi hoặc diễn giải lại theo giờ Việt Nam trước khi trả lời.
- Nếu câu hỏi không nêu rõ dự án, đừng tự gắn vào một dự án cụ thể; chỉ nói theo dữ liệu đã retrieve được.

Câu hỏi của Anh/Chị: {user_query}
Thông tin tra cứu được:
{search_results}

Câu trả lời:"""
    raw_answer = await llm_model_func(summary_prompt, history_messages=history)
    return str(raw_answer)


async def route_query(query: str, llm_func, history: Optional[List[Dict[str, Any]]] = None) -> tuple[str, Optional[str]]:
    t_route_total = time.perf_counter()

    def _finish(category: str, routed_query: Optional[str]) -> tuple[str, Optional[str]]:
        log.info(
            "Timing[router.total]: %.2fs category=%s",
            time.perf_counter() - t_route_total,
            category,
        )
        return category, routed_query

    if history is None:
        history = []
    """
    Classify user query into categories: RAG, SEARCH, OTHER.
    SEARCH is for real-time info like weather, news, etc.
    Returns: (category, refined_query)
    """
    history_str = ""
    recent_history = history[-2:]
    if history:
        history_str = "LỊCH SỬ TRÒ CHUYỆN GẦN ĐÂY:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])  # type: ignore

    date_time = _now_vietnam_str()

    prompt = f"""{history_str}
Thời gian hiện tại: {date_time}

BẠN LÀ BỘ ĐỊNH TUYẾN CÂU HỎI cho hệ thống RAG doanh nghiệp.
Phân tích câu hỏi và trả về JSON:
{{
  "category": "SEARCH" | "RAG" | "OTHER",
  "search_query": "câu truy vấn tối ưu hoặc null",
  "confidence": 0.0-1.0,
  "has_project_intent": true|false,
  "needs_external_realtime": true|false,
  "reason": "lý do ngắn"
}}

- 'RAG': Câu hỏi về tri thức NỘI BỘ của hệ thống (dự án Noble Palace, sản phẩm, chính sách, pháp lý dự án...).
- 'SEARCH': CHỈ chọn cho câu hỏi về THỜI TIẾT hiện tại hoặc các thông tin THỊ TRƯỜNG BẤT ĐỘNG SẢN bên ngoài (lãi suất ngân hàng mới nhất, quy hoạch khu vực dự án).
- 'OTHER': Các câu hỏi chào hỏi xã giao HOẶC các câu hỏi NGOÀI PHẠM VI (thể thao, bóng đá, nấu ăn, kiến thức chung...). Các câu này sẽ bị AI từ chối ở bước sau.

LUẬT ƯU TIÊN:
- Nếu câu hỏi có thể trả lời từ dữ liệu nội bộ thì ưu tiên 'RAG'.
- Chỉ chọn 'SEARCH' khi bản chất câu hỏi là dữ liệu ngoài hệ thống, cần nguồn cập nhật bên ngoài.

QUY TẮC SEARCH:
- Ưu tiên ngữ cảnh tại Việt Nam.
- Nếu câu hỏi thời gian thực chưa nêu địa điểm, hãy viết search_query có hậu tố "tại Việt Nam".
- Nếu người dùng đã nêu địa điểm cụ thể ngoài Việt Nam, giữ đúng địa điểm đó.

Câu hỏi: "{query}"

YÊU CẦU: CHỈ TRẢ VỀ JSON. KHÔNG GIẢI THÍCH.
"""
    try:
        response = await _timed_await(
            "router.classify_llm",
            llm_func(
                prompt,
                enable_cot=False,
                response_format={"type": "json_object"},
            ),
        )

        json_payload = _extract_first_json_object(str(response))
        if not json_payload:
            raise ValueError("No JSON object found in router response")

        data = json.loads(json_payload)
        category = str(data.get("category", "RAG")).upper()
        if category not in {"SEARCH", "RAG", "OTHER"}:
            category = "RAG"

        search_query = str(data.get("search_query") or query).strip() or query
        confidence_raw = data.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        has_project_intent = bool(data.get("has_project_intent", category == "RAG"))
        needs_external_realtime = bool(data.get("needs_external_realtime", category == "SEARCH"))

        log.info(
            "Router intent flags: project_intent=%s external_realtime=%s confidence=%.2f",
            has_project_intent,
            needs_external_realtime,
            confidence,
        )

        if confidence >= ROUTER_SKIP_KB_PROBE_CONFIDENCE:
            kb_hit = False
            log.info(
                "KB evidence probe skipped: confidence=%.2f threshold=%.2f",
                confidence,
                ROUTER_SKIP_KB_PROBE_CONFIDENCE,
            )
        else:
            kb_hit = await _timed_await("router.kb_probe", _kb_evidence_probe(query, history))

        if category == "SEARCH":
            if has_project_intent or kb_hit:
                log.info(
                    "Router override SEARCH->RAG (confidence=%.2f, project_intent=%s, kb_hit=%s)",
                    confidence,
                    has_project_intent,
                    kb_hit,
                )
                return _finish("RAG", query)
            normalized_search_query = _normalize_search_query_for_vietnam(query, search_query)
            log.info("Router final: SEARCH (confidence=%.2f)", confidence)
            return _finish("SEARCH", normalized_search_query)

        if category == "RAG":
            low_confidence = confidence < ROUTER_LOW_CONFIDENCE_THRESHOLD
            if (not kb_hit) and (
                needs_external_realtime
                or (low_confidence and not has_project_intent)
            ):
                normalized_search_query = _normalize_search_query_for_vietnam(query, search_query)
                log.info(
                    "Router override RAG->SEARCH (confidence=%.2f, threshold=%.2f, kb_hit=%s, external=%s, project_intent=%s)",
                    confidence,
                    ROUTER_LOW_CONFIDENCE_THRESHOLD,
                    kb_hit,
                    needs_external_realtime,
                    has_project_intent,
                )
                return _finish("SEARCH", normalized_search_query)
            log.info("Router final: RAG (confidence=%.2f)", confidence)
            return _finish("RAG", query)

        if has_project_intent or kb_hit:
            log.info("Router override OTHER->RAG (project_intent=%s, kb_hit=%s)", has_project_intent, kb_hit)
            return _finish("RAG", query)
        # Removed OTHER->SEARCH override to prevent off-topic questions from triggerring search.
        log.info("Router final: OTHER (confidence=%.2f)", confidence)
        return _finish("OTHER", query)
    except Exception as e:
        log.error(f"Routing logic failed (pass 1): {e}")
        try:
            retry_prompt = prompt + "\n\nBẮT BUỘC: Trả về DUY NHẤT một JSON object hợp lệ, không có markdown, không có text thừa."
            retry_response = await _timed_await(
                "router.classify_llm_retry",
                llm_func(
                    retry_prompt,
                    enable_cot=False,
                    response_format={"type": "json_object"},
                ),
            )
            retry_json_payload = _extract_first_json_object(str(retry_response))
            if not retry_json_payload:
                raise ValueError("No JSON object found in router retry response")

            retry_data = json.loads(retry_json_payload)
            retry_category = str(retry_data.get("category", "RAG")).upper()
            if retry_category not in {"SEARCH", "RAG", "OTHER"}:
                retry_category = "RAG"

            retry_has_project_intent = bool(retry_data.get("has_project_intent", retry_category == "RAG"))
            retry_needs_external_realtime = bool(retry_data.get("needs_external_realtime", retry_category == "SEARCH"))
            log.info(
                "Router intent flags (retry): project_intent=%s external_realtime=%s",
                retry_has_project_intent,
                retry_needs_external_realtime,
            )

            retry_search_query = str(retry_data.get("search_query") or query).strip() or query
            if retry_category == "SEARCH":
                retry_search_query = _normalize_search_query_for_vietnam(query, retry_search_query)
            log.info(f"Router classification (retry): {retry_category}")
            return _finish(retry_category, retry_search_query)
        except Exception as retry_error:
            log.error(f"Routing logic failed (pass 2): {retry_error}")
            return _finish("RAG", query)


# ── Initialize LLM ─────────────────────────────────────────────────────
def init_llm():
    """Initialize LLM based on provider"""
    log.info(f"Initializing LLM provider='{LLM_PROVIDER}' model='{LLM_MODEL}'...")

    provider_lower = LLM_PROVIDER.lower()

    if provider_lower == "openai":
        async def llm_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            **kwargs,
        ):
            # Patch for DeepSeek: It does not support Pydantic Structured Outputs.
            # Convert any pydantic response_format to {"type": "json_object"} and append instruction.
            # Also intercept 'keyword_extraction' because LightRAG's openai_complete_if_cache
            # automatically adds Pydantic models if this flag is True!
            force_json = False
            if kwargs.get("keyword_extraction"):
                kwargs["keyword_extraction"] = False
                force_json = True
            
            if "response_format" in kwargs:
                if type(kwargs["response_format"]) is not dict or kwargs["response_format"].get("type") != "json_object":
                    force_json = True
            
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}
                # Force JSON adherence via prompt string
                prompt += "\n\nIMPORTANT: Return strictly a valid JSON object matching the requested schema. No additional text."

            # enable_cot=True để hỗ trợ cả deepseek-chat và deepseek-reasoner
            # deepseek-reasoner trả về reasoning_content thay vì content
            cot = kwargs.pop("enable_cot", True)
            
            result = await openai_complete_if_cache(
                LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=LLM_API_KEY,
                base_url=os.getenv("LLM_API_URL") or None,
                enable_cot=cot,
                **kwargs,
            )
            if result is None:
                log.error(f"LLM returned None for model={LLM_MODEL}, prompt_len={len(prompt)}")
                return ""
            return result

        return llm_func
    elif provider_lower == "ollama":
        from lightrag.llm.ollama import ollama_model_complete

        return partial(
            ollama_model_complete,
            model=LLM_MODEL,
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider for this service build: {LLM_PROVIDER}. "
            "Use 'openai' or 'ollama'."
        )


def init_embedding():
    """Initialize embedding model"""
    log.info(
        f"Initializing embedding provider='{EMBEDDING_PROVIDER}' model='{EMBEDDING_MODEL}'..."
    )

    provider_lower = EMBEDDING_PROVIDER.lower()

    if provider_lower == "ollama":
        from lightrag.llm.ollama import ollama_embed

        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=int(os.getenv("MAX_TOKEN_SIZE", "8192")),
            model_name=EMBEDDING_MODEL,
        )
        async def embedding_func(texts: list[str]):
            if not texts:
                return await ollama_embed.func(
                    texts,
                    embed_model=EMBEDDING_MODEL,
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                    api_key=os.getenv("OLLAMA_API_KEY") or None,
                )

            batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "4")))
            if len(texts) <= batch_size:
                return await ollama_embed.func(
                    texts,
                    embed_model=EMBEDDING_MODEL,
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                    api_key=os.getenv("OLLAMA_API_KEY") or None,
                )

            import numpy as np

            vectors = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                batch_vectors = await ollama_embed.func(
                    batch,
                    embed_model=EMBEDDING_MODEL,
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                    api_key=os.getenv("OLLAMA_API_KEY") or None,
                )
                vectors.append(batch_vectors)

            return np.concatenate(vectors, axis=0)

        return embedding_func

    if provider_lower == "openai":
        @wrap_embedding_func_with_attrs(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=int(os.getenv("MAX_TOKEN_SIZE", "8192")),
            model_name=EMBEDDING_MODEL,
        )
        async def embedding_func(texts: list[str]):
            return await openai_embed.func(
                texts,
                model=EMBEDDING_MODEL,
                api_key=LLM_API_KEY,
                base_url=os.getenv("LLM_API_URL") or None,
            )

        return embedding_func
    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}. "
        "Use 'openai' or 'ollama'."
    )


# ── Initialize LightRAG ────────────────────────────────────────────────
log.info("Loading LightRAG components...")
_load_start = time.perf_counter()

try:
    configure_storage_env()
    llm_model_func = init_llm()
    embedding_func = init_embedding()
    
    # Initialize LightRAG instance
    rag = LightRAG(
        working_dir=RAG_WORKING_DIR,
        kv_storage=KV_STORAGE,
        vector_storage=VECTOR_STORAGE,
        graph_storage=GRAPH_STORAGE,
        doc_status_storage=DOC_STATUS_STORAGE,
        workspace=RAG_WORKSPACE,
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=embedding_func,
        embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "2")),
        default_embedding_timeout=int(os.getenv("EMBEDDING_TIMEOUT", "120")),
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
    )
    
    _load_time = time.perf_counter() - _load_start
    log.info(f"LightRAG initialized in {_load_time:.2f}s")
    
except Exception as e:
    log.error(f"Failed to initialize LightRAG: {str(e)}")
    raise


# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="LightRAG API",
    description="Retrieval-Augmented Generation service powered by LightRAG",
    version="1.0.0",
)

QUERY_LOCK = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    """Initialize LightRAG storages required by latest LightRAG pipeline."""
    await rag.initialize_storages()
    log.info("LightRAG storages initialized")


# ── Health Check ───────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "service": "rag-service",
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        }
    )


# ── Models Info ────────────────────────────────────────────────────────
@app.get("/models")
async def get_models():
    """Get available LLM models and embedding info"""
    return JSONResponse(
        status_code=200,
        content={
            "llm": {
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "max_tokens": 2048,
            },
            "embedding": {
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIM,
            },
            "storage": {
                "kv_storage": KV_STORAGE,
                "vector_storage": VECTOR_STORAGE,
                "graph_storage": GRAPH_STORAGE,
                "doc_status_storage": DOC_STATUS_STORAGE,
                "postgres": POSTGRES_URL.replace(
                    POSTGRES_URL.split("@")[0].split(":")[-1], 
                    "****"
                ) if "@" in POSTGRES_URL else POSTGRES_URL,
                "qdrant": QDRANT_URL,
                "redis": REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL,
            }
        }
    )


# ── Document Upload ────────────────────────────────────────────────────
@app.post("/upload-document", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    storage_type: StorageType = Form(StorageType.GRAPH),
    metadata: Optional[str] = Form(None),
):
    """
    Upload a document and process it into graph/vector storage
    
    Args:
        file: Document file (txt, pdf, doc, etc.)
        storage_type: "graph" (default), "vector", or "both"
        metadata: JSON metadata associated with document
    
    Returns:
        UploadDocumentResponse with document_id and chunk count
    """
    
    start_time = time.perf_counter()
    
    try:
        # Read file content
        content = await file.read()
        
        # Decode content
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be UTF-8 encoded text"
            )
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        doc_metadata: Dict[str, Any] = {}
        if metadata:
            try:
                parsed = json.loads(metadata)
                if isinstance(parsed, dict):
                    doc_metadata.update(parsed)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON metadata"
                )
        
        # Add filename and upload timestamp to metadata
        doc_metadata['filename'] = file.filename
        doc_metadata['upload_timestamp'] = time.time()
        
        log.info(
            f"Processing document '{file.filename}' "
            f"storage_type={storage_type.value} size={len(text_content)} chars"
        )
        
        # Process with LightRAG
        # Note: ainsert() always inserts both vector and graph for optimization.
        # Storage type parameter is retained for future extensibility.
        document_id = await rag.ainsert(
            text_content,
            file_paths=file.filename,
        )
        
        # Estimate chunks (rough calculation)
        chunks_count = max(1, len(text_content) // CHUNK_SIZE)
        
        elapsed_time = time.perf_counter() - start_time
        
        log.info(
            f"Document '{file.filename}' processed successfully "
            f"document_id={document_id} chunks={chunks_count} time={elapsed_time:.2f}s"
        )
        
        return UploadDocumentResponse(
            status="success",
            document_id=document_id,
            chunks_count=chunks_count,
            storage_type=storage_type.value,
            message=f"Document processed and stored ({chunks_count} chunks in {elapsed_time:.2f}s)"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing document '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# ── Query Endpoint ─────────────────────────────────────────────────────


# ── Status Endpoint ────────────────────────────────────────────────────
@app.get("/status")
async def get_status():
    """Get RAG system status"""
    return JSONResponse(
        status_code=200,
        content={
            "service": "rag-service",
            "status": "running",
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "storage": {
                "type": "hybrid",
                "graph": GRAPH_STORAGE,
                "vector": VECTOR_STORAGE,
                "session": "Redis",
                "kv": KV_STORAGE,
                "doc_status": DOC_STATUS_STORAGE,
            }
        }
    )


@app.get("/documents")
async def list_documents(limit: int = 50):
    """List uploaded documents from PostgreSQL LightRAG doc status table."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    parsed = urlparse(POSTGRES_URL)
    if not parsed.scheme.startswith("postgres"):
        raise HTTPException(status_code=500, detail="Invalid POSTGRES_URL")

    db_name = parsed.path.lstrip("/") or "noble_rag"

    try:
        conn = await asyncpg.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=db_name,
        )
        rows = await conn.fetch(
            """
            SELECT id, status, file_path, content_length, chunks_count, created_at, updated_at, metadata
            FROM lightrag_doc_status
            WHERE workspace = $1
            ORDER BY updated_at DESC NULLS LAST
            LIMIT $2
            """,
            RAG_WORKSPACE,
            limit,
        )
        await conn.close()
    except Exception as e:
        log.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

    documents = []
    for row in rows:
        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = None

        documents.append(
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "file_path": row.get("file_path"),
                "content_length": row.get("content_length"),
                "chunks_count": row.get("chunks_count"),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
                "metadata": metadata,
            }
        )

    return JSONResponse(
        {
            "workspace": RAG_WORKSPACE,
            "count": len(documents),
            "documents": documents,
        }
    )


@app.get("/documents/track/{track_id}")
async def get_track_status(track_id: str):
    """Get processing status summary for a LightRAG upload track id."""
    parsed = urlparse(POSTGRES_URL)
    if not parsed.scheme.startswith("postgres"):
        raise HTTPException(status_code=500, detail="Invalid POSTGRES_URL")

    db_name = parsed.path.lstrip("/") or "noble_rag"

    try:
        conn = await asyncpg.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=db_name,
        )
        rows = await conn.fetch(
            """
            SELECT status, COUNT(*) AS count
            FROM lightrag_doc_status
            WHERE workspace = $1 AND track_id = $2
            GROUP BY status
            """,
            RAG_WORKSPACE,
            track_id,
        )
        await conn.close()
    except Exception as e:
        log.error(f"Failed to get track status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get track status: {e}")

    status_counts: dict[str, int] = {}
    total = 0
    for row in rows:
        status_value = row.get("status") or "unknown"
        count_value = int(row.get("count") or 0)
        status_counts[status_value] = count_value
        total += count_value

    processing = status_counts.get("processing", 0)
    pending = status_counts.get("pending", 0)
    ready = total > 0 and processing == 0 and pending == 0

    return JSONResponse(
        {
            "workspace": RAG_WORKSPACE,
            "track_id": track_id,
            "total": total,
            "ready": ready,
            "status_counts": status_counts,
        }
    )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document from LightRAG storage by its ID."""
    try:
        log.info(f"Attempting to delete document_id={doc_id}...")
        # LightRAG ados_delete is the method for deleting documents
        await rag.adelete_by_doc_id(doc_id)
        log.info(f"Document {doc_id} deletion triggered.")
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": f"Deletion triggered for document {doc_id}"}
        )
    except Exception as e:
        log.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


# ── Streaming Query Endpoint ──────────────────────────────────────────
@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """
    Stream query response as NDJSON lines.
    Each line is a JSON object: {"chunk": "...", "done": false}
    Final line: {"chunk": "", "done": true, "model": "..."}
    
    Sentences are emitted as soon as they are complete (split on .!?),
    so the client receives the first sentence within ~Router+first-token time.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    log.info(f"Stream query: {request.query[:100]}...")

    async def generate():
        import re
        t_stream_total = time.perf_counter()
        session_id = request.session_id or "default_session"
        history = await _timed_await(f"stream.history_load[{session_id}]", get_chat_history(session_id))



        # Heuristic optimization: skip decomposition for very short queries to save ~1.5s
        if len(request.query.strip().split()) < 5:
            subqueries = [request.query]
            log.info("Skipping decomposition for short query: '%s'", request.query)
        else:
            subqueries = await _timed_await(
                "stream.decompose_subqueries",
                _decompose_subqueries(request.query, llm_model_func),
            )

        if len(subqueries) > 1:
            completed_answers: Dict[int, str] = {}

            semaphore = asyncio.Semaphore(SUBQUERY_MAX_CONCURRENCY)
            log.info("Stream multi-intent parallel execution: subqueries=%s concurrency=%s", len(subqueries), SUBQUERY_MAX_CONCURRENCY)

            async def _run_stream_subquery(idx: int, sub_query: str) -> tuple[int, str]:
                async with semaphore:
                    t_sub_total = time.perf_counter()
                    sub_category, sub_refined_query = await _timed_await(
                        f"stream.subquery[{idx}].route",
                        route_query(sub_query, llm_model_func, history),
                    )

                    if sub_category == "SEARCH":
                        if sub_refined_query:
                            search_query = sub_refined_query.strip()
                        else:
                            search_query = sub_query.strip()
                        sub_answer = await _timed_await(
                            f"stream.subquery[{idx}].search_summarize",
                            _summarize_search_answer(
                                user_query=sub_query,
                                search_query=search_query,
                                history=history,
                                max_sentences=2,
                            ),
                        )
                        log.info("Timing[stream.subquery[%s].total]: %.2fs", idx, time.perf_counter() - t_sub_total)
                        return idx, sub_answer

                    if sub_category == "OTHER":
                        other_prompt = f"""{SYSTEM_PERSONA}
Tiểu câu hỏi/lời nhắn: {sub_query}
Hãy phản hồi ngắn gọn, lịch sự, đúng vai trò."""
                        raw = await _timed_await(
                            f"stream.subquery[{idx}].other_llm",
                            llm_model_func(other_prompt, history_messages=history),
                        )
                        log.info("Timing[stream.subquery[%s].total]: %.2fs", idx, time.perf_counter() - t_sub_total)
                        return idx, str(raw)

                    rag_sub_query = (
                        f"[NGỮ CẢNH: {SYSTEM_PERSONA}]\n\n"
                        f"Tiểu câu hỏi của khách hàng: {sub_query}\n\n"
                        f"Hãy trả lời dựa trên thông tin retrieve từ toàn bộ kho tài liệu đã index. "
                        f"Không mặc định dự án cụ thể nếu câu hỏi mơ hồ. Viết bằng tiếng Việt, "
                        f"ngắn gọn, không dùng ký hiệu đặc biệt."
                    )
                    response = await _timed_await(
                        f"stream.subquery[{idx}].rag_aquery",
                        asyncio.wait_for(
                            rag.aquery(
                                rag_sub_query,
                                param=QueryParam(
                                    top_k=request.top_k,
                                    mode="naive",
                                    conversation_history=history,
                                ),
                            ),
                            timeout=QUERY_TIMEOUT_SEC,
                        ),
                    )
                    sub_answer = response if isinstance(response, str) else str(response)
                    log.info("Timing[stream.subquery[%s].total]: %.2fs", idx, time.perf_counter() - t_sub_total)
                    return idx, sub_answer

            tasks = [
                asyncio.create_task(_run_stream_subquery(idx, sub_query))
                for idx, sub_query in enumerate(subqueries, start=1)
            ]
            for task in asyncio.as_completed(tasks):
                try:
                    idx, sub_answer = await task
                except Exception as e:
                    log.error("Stream subquery task failed: %s", e)
                    continue

                numbered = f"{idx}) {sub_answer}"
                completed_answers[idx] = numbered
                yield json.dumps({"chunk": numbered, "done": False}, ensure_ascii=False) + "\n"

            # Fill missing answers (if any task failed unexpectedly) to keep final history consistent.
            for idx in range(1, len(subqueries) + 1):
                if idx not in completed_answers:
                    fallback = "Xin lỗi, em gặp lỗi khi xử lý ý này. Anh/Chị vui lòng thử lại giúp em nhé."
                    completed_answers[idx] = f"{idx}) {fallback}"

            full_answer = "\n\n".join(completed_answers[idx] for idx in sorted(completed_answers.keys()))
            history.append({"role": "user", "content": request.query})
            history.append({"role": "assistant", "content": full_answer})
            await _timed_await(f"stream.history_save[{session_id}]", save_chat_history(session_id, history))

            yield json.dumps({"chunk": "", "done": True, "model": "multi-intent-composed"}, ensure_ascii=False) + "\n"
            log.info("Timing[stream.total]: %.2fs", time.perf_counter() - t_stream_total)
            return

        # -- Router step --
        category, refined_query = await _timed_await(
            "stream.route_query",
            route_query(request.query, llm_model_func, history),
        )
        log.info("Stream router category: %s", category)

        full_answer = ""

        if category == "SEARCH":
            # UX: Let the user know search is happening
            yield json.dumps({"chunk": "  \n*(Em đang tìm kiếm thông tin mới nhất trên mạng...)*", "done": False}, ensure_ascii=False) + "\n"
            
            if refined_query:
                search_query = refined_query.strip()
            else:
                search_query = request.query.strip()
            search_query = _normalize_search_query_for_vietnam(request.query, search_query)
            log.info(f"Stream SEARCH query normalized: {search_query}")
            search_results = await _timed_await("stream.search.tavily", tavily_search(search_query))
            summary_prompt = f"""{SYSTEM_PERSONA}

Hãy trả lời câu hỏi dựa trên kết quả tìm kiếm sau.
Thời gian hiện tại (Việt Nam - UTC+7): {_now_vietnam_str()}
YÊU CẦU:
- TUÂN THỦ PHẠM VI TRẢ LỜI TRONG SYSTEM_PERSONA. Nếu thông tin tìm kiếm không thuộc các chủ đề cho phép (bất động sản, thời tiết, ngày giờ), hãy từ chối lịch sự.
- Nếu thuộc chủ đề cho phép, trả lời ngắn gọn (dưới 3 câu), không dùng ký hiệu toán học.
- Ưu tiên thông tin theo ngữ cảnh Việt Nam.
Câu hỏi: {request.query}
Kết quả tìm kiếm:
{search_results}
Trả lời:"""
            raw = await _timed_await(
                "stream.search.summarize_llm",
                llm_model_func(summary_prompt, history_messages=history),
            )
            full_answer = str(raw)
            line = json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False)
            yield line + "\n"

        elif category == "OTHER":
            other_prompt = f"""{SYSTEM_PERSONA}

Thời gian hiện tại (Việt Nam): {_now_vietnam_str()}
Lịch sử cuộc trò chuyện gần đây:
{chr(10).join([f"{m['role']}: {m['content']}" for m in history[-4:]]) if history else "(chưa có)"}  # type: ignore

Câu hỏi/lời nhắn mới nhất của Anh/Chị: {request.query}

Hãy phản hồi theo đúng SYSTEM_PERSONA. Nếu khách chào hỏi, hãy chào lại một cách tự nhiên (đừng quá máy móc). Nếu là câu hỏi ngoài lề, hãy từ chối lịch sự.
Trả lời (ngắn gọn, không quá 3 câu):"""
            raw = await _timed_await(
                "stream.other.llm",
                llm_model_func(other_prompt, history_messages=history),
            )
            full_answer = str(raw)
            line = json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False)
            yield line + "\n"

        else:
            # RAG path — stream token by token, emit complete sentences immediately
            try:
                # UX: Let user know RAG search is happening
                yield json.dumps({"chunk": "  \n*(Em đang truy xuất thông tin từ kho tài liệu dự án...)*", "done": False}, ensure_ascii=False) + "\n"
                
                buffer = ""
                t_rag_total = time.perf_counter()
                first_token_sec: Optional[float] = None

                # LightRAG aquery_stream
                rag_query = (
                    f"{SYSTEM_PERSONA}\n\n"
                    f"Câu hỏi của khách hàng: {request.query}\n\n"
                    f"Hãy trả lời dựa trên thông tin retrieve từ toàn bộ kho tài liệu đã index. "
                    f"LUÔN TUÂN THỦ PHẠM VI TRẢ LỜI. "
                    f"Viết bằng tiếng Việt, ngắn gọn."
                )
                
                # Use aquery with stream=True to get an AsyncIterator
                response = await _timed_await(
                    "stream.rag.aquery_wait",
                    asyncio.wait_for(
                        rag.aquery(
                            rag_query,
                            param=QueryParam(
                                top_k=request.top_k,
                                mode="naive",
                                conversation_history=history,
                                stream=True,
                            ),
                        ),
                        timeout=QUERY_TIMEOUT_SEC,
                    ),
                )

                if response is None:
                    full_answer = "Xin lỗi, em chưa tìm thấy thông tin phù hợp trong kho tài liệu hiện tại."
                    yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"
                elif isinstance(response, str):
                    # In case streaming is not supported, it returns the full string
                    if first_token_sec is None:
                        first_token_sec = time.perf_counter() - t_rag_total
                    full_answer = response
                    if full_answer.strip().lower() == "none":
                        full_answer = "Xin lỗi, em chưa tìm thấy thông tin phù hợp trong kho tài liệu hiện tại."
                    sentences = re.split(r'(?<=[.!?。！？\n])\s*', full_answer)
                    for sent in sentences:
                        sent = sent.strip()
                        if sent:
                            yield json.dumps({"chunk": sent, "done": False}, ensure_ascii=False) + "\n"
                            await asyncio.sleep(0)
                else:
                    # Async generation
                    async for token in response:
                        if not token:
                            continue
                        if first_token_sec is None:
                            first_token_sec = time.perf_counter() - t_rag_total
                        buffer += token
                        # Flush complete sentences immediately
                        parts = re.split(r'(?<=[.!?。！？\n])\s*', buffer)
                        if len(parts) > 1:
                            for sent in parts[:-1]:
                                sent = sent.strip()
                                if sent:
                                    full_answer = str(full_answer) + str(sent) + " "
                                    yield json.dumps({"chunk": sent, "done": False}, ensure_ascii=False) + "\n"
                            buffer = parts[-1]

                if first_token_sec is not None:
                    log.info("Timing[stream.rag.first_token]: %.2fs", first_token_sec)
                log.info("Timing[stream.rag.total]: %.2fs", time.perf_counter() - t_rag_total)

                # Flush remaining buffer
                if buffer.strip():
                    full_answer = str(full_answer) + buffer.strip()
                    yield json.dumps({"chunk": buffer.strip(), "done": False}, ensure_ascii=False) + "\n"

            except asyncio.TimeoutError:
                yield json.dumps({"chunk": "Xin lỗi, hệ thống đang bận. Bạn thử lại nhé.", "done": False}, ensure_ascii=False) + "\n"
                full_answer = ""
            except Exception as e:
                log.error(f"Stream RAG error: {e}")
                full_answer = "Xin lỗi, em gặp lỗi khi truy xuất dữ liệu. Anh/Chị thử lại giúp em nhé."
                yield json.dumps({"chunk": full_answer, "done": False}, ensure_ascii=False) + "\n"

        # Save history
        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": full_answer.strip()})
        await _timed_await(f"stream.history_save[{session_id}]", save_chat_history(session_id, history))

        # Done sentinel
        yield json.dumps({"chunk": "", "done": True, "model": LLM_MODEL}, ensure_ascii=False) + "\n"
        log.info("Timing[stream.total]: %.2fs", time.perf_counter() - t_stream_total)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


# ── Run Server ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("RAG_SERVICE_PORT", "8001"))
    host = os.getenv("RAG_SERVICE_HOST", "0.0.0.0")
    
    log.info(f"Starting RAG service on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=LOG_LEVEL.lower(),
    )
