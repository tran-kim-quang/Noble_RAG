"""Sales agent API endpoints."""

import asyncio
import json
import logging
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from models.api_models import LeadUpdateRequest, SalesChatRequest, SalesChatResponse
from core.dependencies import llm_model_func
from core.config import get_settings
from memory.chat_history_store import (
    append_turn,
    delete_chat_history,
    load_chat_history,
    purge_all_chat_history,
)
from memory.lead_profile_store import (
    delete_lead_profile_cache,
    ensure_sales_schema,
    load_lead_profile,
    save_lead_profile,
)
from memory.local_snapshot_store import (
    get_local_snapshot_path,
    load_local_session_snapshot,
    read_local_session_snapshot_text,
)
from memory.session_store import delete_session_context, load_session_context
from rag.retriever import (
    build_rag_fact_constraints,
    get_rag_citation_sources,
    kb_evidence_probe,
    query_rag,
    query_rag_stream,
    summarize_search_answer,
)
from integrations.machine_b_face_client import notify_machine_b_customer_removed
from sales.session_export import export_session_to_txt
from tools.tavily_tool import tavily_search
from tools.sales_advisor_tool import is_sales_advisor_query, run_sales_advisor_tool
from utils.text import iter_stream_chunks
from utils.time import now_vietnam_str

router = APIRouter(prefix="/sales", tags=["sales"])
log = logging.getLogger("rag-service")
settings = get_settings()

# Đồng bộ timeout toàn bộ nhánh sales theo cấu hình LLM, tránh timeout cứng theo từng flow.
_SALES_LLM_TIMEOUT_SEC = max(
    5.0,
    float((os.getenv("SALES_LLM_TIMEOUT_SEC") or "").strip() or settings.llm_request_timeout_max_sec),
)
_SALES_LLM_CLASSIFIER_TIMEOUT_SEC = max(
    2.0,
    float((os.getenv("SALES_LLM_CLASSIFIER_TIMEOUT_SEC") or "").strip() or settings.llm_request_timeout_sec),
)


def _require_chat_purge_token(x_chat_history_purge_token: Optional[str]) -> None:
    expected = (get_settings().chat_history_purge_token or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="CHAT_HISTORY_PURGE_TOKEN is not configured on server",
        )
    if (x_chat_history_purge_token or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid X-Chat-History-Purge-Token")
_THINKING_ACK_MESSAGES = [
    "Sunny đã nhận thông tin của bạn rồi, cho Sunny vài giây để kiểm tra nhanh nhé.",
    "Sunny đang xử lý yêu cầu của bạn và phản hồi ngay cho bạn đây.",
    "Sunny đã ghi nhận câu hỏi, Sunny rà nhanh dữ liệu rồi trả lời bạn ngay nhé.",
    "Sunny nhận được rồi, Sunny đối chiếu thông tin một chút để trả lời bạn thật rõ.",
    "Sunny đang kiểm tra nhanh và sẽ gửi cho bạn câu trả lời ngắn gọn, dễ hiểu.",
]
_SEARCH_WAITING_ACK_MESSAGES = [
    "Sunny đang tổng hợp thông tin từ nguồn bên ngoài. Bạn đợi Sunny một chút nhé.",
]
_SEARCH_PERSONA = (
    "Bạn là Sunny, trợ lý tư vấn của Noble Palace Tây Thăng Long. "
    "Luôn xưng Sunny và gọi khách là bạn. "
    "Với câu hỏi ngoài kho tri thức dự án, hãy trả lời trực tiếp, tự nhiên, ngắn gọn nhưng hữu ích. "
    "Nếu thông tin phụ thuộc thời gian, hãy ưu tiên dữ liệu vừa tìm được."
)

_LOW_KB_EXTERNAL_SEARCH_NOTICE = (
    "Lưu ý: Thông tin dưới đây đến từ nguồn bên ngoài vì dữ liệu nội bộ hiện chưa đủ để xác nhận đầy đủ câu hỏi này."
)
_PROFILE_TXT_DIR = Path("service/RAG/live_snapshots/customer_profiles")
_PHONE_PATTERN = re.compile(r"(?:\+84|0)(?:[\s\.\-]?\d){8,10}")
_PREFERENCE_HINTS = (
    "gu nhà",
    "gu sống",
    "phong cách",
    "thích",
    "ưu tiên",
    "muốn",
    "cần",
    "để ở",
    "đầu tư",
    "kinh doanh",
    "shophouse",
    "căn hộ",
    "nhà phố",
    "biệt thự",
    "diện tích",
)

_NOBLE_PROJECT_KEYS = (
    "noble palace tay thang long",
    "noble tay thang long",
    "noble palace",
)

_REAL_ESTATE_KEYS = (
    "du an",
    "bat dong san",
    "nha pho",
    "biet thu",
    "shophouse",
    "can ho",
    "thi truong",
    "gia nha",
    "quy hoach",
    "phap ly",
)

_PROJECT_REAL_ESTATE_HINTS = (
    "noble",
    "tay thang long",
    "du an",
    "bat dong san",
    "shophouse",
    "nha pho",
    "biet thu",
    "can ho",
    "san pham",
    "uu dai",
    "chinh sach",
    "gia",
    "thanh toan",
    "phap ly",
    "quy hoach",
    "mat bang",
    "tien ich",
    "vi tri",
    "ban giao",
)

_SEARCH_TOOL_HINTS = (
    "so sanh",
    "so sánh",
    "du an khac",
    "dự án khác",
    "thi truong",
    "thị trường",
    "lan can",
    "lân cận",
    "cap nhat",
    "cập nhật",
    "moi nhat",
    "mới nhất",
    "hom nay",
    "hôm nay",
    "xu huong",
    "xu hướng",
    "tim web",
    "tìm web",
    "nguon ngoai",
    "nguồn ngoài",
)


_COMPARISON_INTENT_HINTS = (
    "so sanh",
    "doi chieu",
    "khac gi",
    "vs",
    "versus",
)

_OUT_OF_SCOPE_HINTS = (
    "ty gia",
    "usd",
    "eur",
    "jpy",
    "gia vang",
    "vang sjc",
    "chung khoan",
    "co phieu",
    "bitcoin",
    "crypto",
    "thoi tiet",
    "du bao thoi tiet",
    "bong da",
    "lich thi dau",
    "ket qua bong da",
    "gia xang",
    "tu vi",
)

_IN_SCOPE_FINANCE_HINTS = (
    "lai suat vay",
    "goi vay",
    "ho tro lai suat",
    "chinh sach vay",
    "ngan hang",
    "an han",
    "thanh toan",
)

_NON_DOMAIN_BOUNDARY_REPLY = (
    "Sunny chỉ hỗ trợ tư vấn dự án bất động sản Noble Palace Tây Thăng Long. "
    "Bạn vui lòng hỏi về sản phẩm, giá, chính sách, pháp lý hoặc tiến độ của dự án."
)

_SHORT_STYLE_POLICY = (
    "Phong cách trả lời bắt buộc:\n"
    "- Trả lời ngắn gọn, rõ ràng, tối đa 4 câu.\n"
    "- Không dùng từ viết tắt.\n"
    "- Không lan man, không suy diễn ngoài dữ liệu có sẵn.\n"
)

_GENERAL_CHAT_SYSTEM_POLICY = (
    "Bạn là Sunny, trợ lý tư vấn dự án Noble Palace Tây Thăng Long.\n"
    "Chỉ được trao đổi trong phạm vi dự án bất động sản này.\n"
    "Nếu người dùng hỏi ngoài phạm vi dự án, từ chối lịch sự và hướng người dùng quay lại câu hỏi về dự án.\n"
    + _SHORT_STYLE_POLICY
)

_RAG_STYLE_POLICY = (
    "Yêu cầu bắt buộc khi trả lời:\n"
    "- Chỉ dùng dữ liệu nội bộ đã truy xuất.\n"
    "- Trả lời ngắn gọn, rõ ràng, tối đa 5 câu.\n"
    "- Không dùng từ viết tắt.\n"
    "- Nếu thiếu dữ liệu nội bộ, nói rõ là chưa đủ dữ liệu nội bộ để xác nhận.\n"
)



_SEARCH_APPROVE_PATTERNS = (
    r"\b(co|có)\b",
    r"\b(cho\s*phep|cho\s*phép)\b",
    r"\b(ok|oke|okay|dong\s*y|đồng\s*ý|duoc|được)\b",
    r"\b(search|tim\s*ngoai|tìm\s*ngoài|tim\s*web|tìm\s*web)\b",
)
_SEARCH_DENY_PATTERNS = (
    r"\b(khong|không)\b",
    r"\b(khong\s*can|không\s*cần|khong\s*tim|không\s*tìm)\b",
    r"\b(khong\s*cho\s*phep|không\s*cho\s*phép)\b",
    r"\b(no|cancel|dung|dừng)\b",
)
_SEARCH_APPROVAL_CANDIDATE_HINTS = (
    "tìm",
    "tim",
    "search",
    "tra cứu",
    "tra cuu",
    "nguồn ngoài",
    "nguon ngoai",
    "bên ngoài",
    "ben ngoai",
)


_FAST_REPLY_SOCIAL_RULES: List[tuple[List[str], str]] = [
    (["xin chao", "chao", "hello", "hi", "alo"], "Sunny chào bạn. Sunny sẵn sàng tư vấn Noble Palace Tây Thăng Long ngay bây giờ."),
    (
        ["ban la ai", "sunny la ai", "gioi thieu sunny", "gioi thieu ban"],
        "Sunny là trợ lý tư vấn dự án Noble Palace Tây Thăng Long, chuyên hỗ trợ thông tin sản phẩm, chính sách và quy trình mua.",
    ),
    (
        ["ban lam duoc gi", "sunny lam duoc gi", "ho tro gi", "co the giup gi"],
        "Sunny có thể hỗ trợ bạn về tổng quan dự án, loại hình sản phẩm, giá, chính sách bán hàng và ưu đãi theo tài liệu nội bộ.",
    ),
    (
        ["toi can tu van", "tu van giup toi", "ho tro toi"],
        "Sunny sẵn sàng hỗ trợ. Bạn cho Sunny biết mục tiêu chính là ở thực, đầu tư hay kinh doanh để tư vấn sát nhất.",
    ),
    (
        ["toi muon xem nhanh", "tom tat nhanh", "noi ngan gon"],
        "Sunny sẽ trả lời ngắn gọn, đúng trọng tâm và ưu tiên thông tin quan trọng nhất trước cho bạn.",
    ),
    (
        ["bat dau di", "bat dau tu dau", "minh nen lam gi dau tien"],
        "Mình bắt đầu từ nhu cầu chính của bạn nhé: ngân sách dự kiến, mục tiêu mua và loại sản phẩm bạn đang ưu tiên.",
    ),
    (["cam on", "ok", "duoc roi", "tot roi"], "Sunny luôn sẵn sàng. Khi bạn cần, cứ hỏi tiếp để Sunny hỗ trợ ngay."),
]

_FAST_REPLY_PROJECT_RULES: List[tuple[List[str], str]] = [
    (
        ["gioi thieu du an", "tong quan du an", "du an nay la gi"],
        "Noble Palace Tây Thăng Long là dự án thấp tầng với định hướng sống hiện đại, phù hợp cả nhu cầu ở và khai thác kinh doanh.",
    ),
    (
        ["du an o dau", "vi tri du an", "dia chi du an"],
        "Sunny có thể gửi bạn thông tin vị trí dự án theo từng mốc kết nối giao thông và tiện ích lân cận.",
    ),
    (
        ["co nhung loai hinh nao", "san pham nao", "co can ho khong", "co shophouse khong"],
        "Dự án có các dòng sản phẩm thấp tầng, Sunny sẽ lọc nhanh theo nhu cầu ở thực hay đầu tư để tư vấn đúng loại phù hợp.",
    ),
    (
        ["gia bao nhieu", "tam gia", "muc gia", "gia nhu the nao"],
        "Sunny sẽ tư vấn khung giá theo từng loại sản phẩm và diện tích để bạn dễ so sánh trước khi chốt lựa chọn.",
    ),
    (
        ["co uu dai gi", "uu dai mua hang", "khuyen mai", "chinh sach uu dai"],
        "Sunny sẽ kiểm tra ngay chính sách ưu đãi mua hàng hiện hành và gửi bạn các mốc thanh toán quan trọng.",
    ),
]


_FAST_REPLY_SOCIAL_REGEX_RULES: List[tuple[str, str]] = [
    (r"\b(xin\s*chao|chao|hello|hi|a\s*lo\w*)\b", "Sunny chào bạn. Sunny sẵn sàng tư vấn Noble Palace Tây Thăng Long ngay bây giờ."),
    (
        r"\b(gioi\s*thieu)\b.*\b(sunny|ban)\b|\b(sunny|ban)\s+la\s+ai\b",
        "Sunny là trợ lý tư vấn dự án Noble Palace Tây Thăng Long, chuyên hỗ trợ thông tin sản phẩm, chính sách và quy trình mua.",
    ),
]

_FAST_REPLY_MAX_TOKENS = 10
_FAST_REPLY_MAX_CHARS = 80
_FAST_REPLY_BLOCK_HINTS = (
    "phan tich",
    "phan tich",
    "trade off",
    "trade-off",
    "so sanh",
    "khuyen nghi",
    "de xuat",
    "phuong an",
    "chi tiet",
    "cu the",
    "theo ngan sach",
    "ngan sach",
    "nen mua",
    "dau tu",
    "de o",
)


def _tokenize_folded(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text or "")


def _phrase_tokens_match(folded_text: str, phrase: str, max_gap: int = 3) -> bool:
    phrase_tokens = _tokenize_folded(_fold_text(phrase))
    text_tokens = _tokenize_folded(folded_text)
    if not phrase_tokens or not text_tokens:
        return False

    last_idx = -1
    for token in phrase_tokens:
        found_idx = -1
        for idx in range(last_idx + 1, len(text_tokens)):
            if text_tokens[idx] != token:
                continue
            if last_idx >= 0 and (idx - last_idx - 1) > max_gap:
                continue
            found_idx = idx
            break
        if found_idx < 0:
            return False
        last_idx = found_idx
    return True





def _fold_text(text: str) -> str:
    raw = (text or "").strip().lower()
    folded = unicodedata.normalize("NFD", raw)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def _allow_fast_reply(user_text: str, folded_text: str) -> bool:
    text = (user_text or "").strip()
    tokens = _tokenize_folded(folded_text)
    if not tokens:
        return False
    if len(tokens) > _FAST_REPLY_MAX_TOKENS:
        return False
    if len(text) > _FAST_REPLY_MAX_CHARS:
        return False
    if any(hint in folded_text for hint in _FAST_REPLY_BLOCK_HINTS):
        return False
    return True


def _fast_reply_from_sys_prompt(user_text: str) -> Optional[str]:
    folded = _fold_text(user_text)
    if not folded:
        return None
    if not _allow_fast_reply(user_text, folded):
        return None
    for pattern, answer in _FAST_REPLY_SOCIAL_REGEX_RULES:
        if re.search(pattern, folded):
            return answer
    for phrases, answer in _FAST_REPLY_SOCIAL_RULES:
        for phrase in phrases:
            if _phrase_tokens_match(folded, phrase):
                return answer

    project_deictic_hints = (
        "du an nay",
        "du an ben minh",
        "du an nha minh",
        "ben minh",
        "nha minh",
    )
    allow_project_fast_reply = _is_noble_related_text(user_text) or any(h in folded for h in project_deictic_hints)
    if not allow_project_fast_reply:
        return None

    for phrases, answer in _FAST_REPLY_PROJECT_RULES:
        for phrase in phrases:
            if _phrase_tokens_match(folded, phrase):
                return answer
    return None


def _is_noble_related_text(text: str) -> bool:
    folded = _fold_text(text)
    return any(key in folded for key in _NOBLE_PROJECT_KEYS)


def _is_real_estate_text(text: str) -> bool:
    folded = _fold_text(text)
    return any(key in folded for key in _REAL_ESTATE_KEYS)


def _is_project_real_estate_query(text: str) -> bool:
    folded = _fold_text(text)
    if not folded:
        return False
    return any(hint in folded for hint in _PROJECT_REAL_ESTATE_HINTS)


def _has_explicit_comparison_intent(text: str) -> bool:
    folded = _fold_text(text)
    if not folded:
        return False
    return any(h in folded for h in _COMPARISON_INTENT_HINTS)


def _format_recent_history(history: List[Dict[str, Any]], max_turns: int = 4) -> str:
    rows: List[str] = []
    tail = history[-max_turns:] if history else []
    for item in tail:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role not in {"user", "assistant"}:
            continue
        rows.append(f"{role}: {content}")
    return "\n".join(rows)


def _build_general_chat_prompt(user_text: str, history: List[Dict[str, Any]]) -> str:
    history_text = _format_recent_history(history)
    return (
        f"{_GENERAL_CHAT_SYSTEM_POLICY}\n"
        "Ngữ cảnh hội thoại gần nhất:\n"
        f"{history_text or '(không có)'}\n\n"
        f"Câu người dùng: {user_text}\n\n"
        "Hãy trả lời theo đúng chính sách."
    )


def _fallback_domain_scope(user_text: str) -> str:
    if _is_out_of_scope_query(user_text):
        return "OUT_OF_SCOPE"
    if _is_project_real_estate_query(user_text):
        return "PROJECT"
    return "OUT_OF_SCOPE"


async def _classify_domain_scope(user_text: str, history: List[Dict[str, Any]]) -> str:
    """LLM decides whether query is project-domain or out-of-scope.
    Returns: PROJECT | OUT_OF_SCOPE
    """
    text = (user_text or "").strip()
    if not text:
        return "OUT_OF_SCOPE"

    history_text = _format_recent_history(history, max_turns=4)
    prompt = f"""
Bạn là bộ phân loại domain cho trợ lý tư vấn dự án bất động sản Noble Palace Tây Thăng Long.
Nhiệm vụ: phân loại câu hỏi người dùng thuộc phạm vi dự án hay ngoài phạm vi.

Trả JSON duy nhất:
{{
  "scope": "PROJECT|OUT_OF_SCOPE",
  "reason": "<ngắn gọn>"
}}

Quy tắc:
- PROJECT: Câu hỏi liên quan tư vấn dự án bất động sản (thông tin dự án, sản phẩm, giá, chính sách, pháp lý, tiến độ, vị trí, so sánh dự án, tài chính mua dự án).
- OUT_OF_SCOPE: Câu hỏi kiến thức chung không phục vụ tư vấn dự án (thời tiết, thể thao, tử vi, kỹ thuật lập trình, tin tức không liên quan dự án...).
- Nếu câu hỏi mơ hồ nhưng có ngữ cảnh hội thoại gần nhất đang hỏi về dự án thì ưu tiên PROJECT.

Ngữ cảnh gần nhất:
{history_text or "(không có)"}

Câu người dùng:
{text}
""".strip()

    try:
        raw = await asyncio.wait_for(
            llm_model_func(
                prompt,
                enable_cot=False,
                response_format={"type": "json_object"},
            ),
            timeout=_SALES_LLM_CLASSIFIER_TIMEOUT_SEC,
        )
        payload = json.loads(str(raw or "{}"))
        scope = str(payload.get("scope") or "").strip().upper()
        if scope in {"PROJECT", "OUT_OF_SCOPE"}:
            if scope == "OUT_OF_SCOPE" and _is_project_real_estate_query(text):
                return "PROJECT"
            return scope
    except Exception as e:
        log.debug("domain classifier failed, fallback rule-based: %s", e)

    return _fallback_domain_scope(text)


def _fallback_project_intent(user_text: str) -> str:
    folded = _fold_text(user_text)
    if _is_noble_related_text(user_text):
        return "INTERNAL_PROJECT"
    if any(
        hint in folded
        for hint in (
            "du an nay",
            "du an ben minh",
            "du an nha minh",
            "du an cua minh",
            "ben minh",
            "nha minh",
        )
    ):
        return "INTERNAL_PROJECT"
    if _has_explicit_comparison_intent(user_text) or _has_search_tool_hint(user_text):
        return "EXTERNAL_PROJECT"
    return "UNKNOWN"

async def _classify_project_intent(user_text: str, history: List[Dict[str, Any]]) -> str:
    """Classify project question intent: INTERNAL_PROJECT | EXTERNAL_PROJECT | UNKNOWN."""
    text = (user_text or "").strip()
    if not text:
        return "UNKNOWN"
    if not _is_project_real_estate_query(text):
        return "UNKNOWN"
    # Hard guard: explicit Noble mention must stay in internal-project lane.
    if _is_noble_related_text(text):
        return "INTERNAL_PROJECT"
    if any(
        hint in _fold_text(text)
        for hint in ("du an nay", "du an ben minh", "du an nha minh", "du an cua minh")
    ):
        return "INTERNAL_PROJECT"

    history_text = _format_recent_history(history, max_turns=4)
    prompt = f"""
Bạn là bộ phân tích ý định cho trợ lý bất động sản Noble Palace Tây Thăng Long.
Nhiệm vụ: xác định câu hỏi dự án hiện tại là hỏi về dự án nội bộ Noble hay dự án/thị trường bên ngoài.

Trả JSON duy nhất:
{{
  "intent": "INTERNAL_PROJECT|EXTERNAL_PROJECT|UNKNOWN",
  "reason": "<ngắn gọn>"
}}

Quy tắc:
- INTERNAL_PROJECT: hỏi trực tiếp về Noble Palace Tây Thăng Long hoặc ngữ cảnh hội thoại cho thấy "dự án này/bên mình".
- EXTERNAL_PROJECT: hỏi về dự án khác, so sánh thị trường, giá ngoài hệ thống nội bộ.
- UNKNOWN: mơ hồ, không đủ tín hiệu rõ.

Ngữ cảnh gần nhất:
{history_text or "(không có)"}

Câu người dùng:
{text}
""".strip()

    try:
        raw = await asyncio.wait_for(
            llm_model_func(
                prompt,
                enable_cot=False,
                response_format={"type": "json_object"},
            ),
            timeout=_SALES_LLM_CLASSIFIER_TIMEOUT_SEC,
        )
        payload = json.loads(str(raw or "{}"))
        intent = str(payload.get("intent") or "").strip().upper()
        if intent in {"INTERNAL_PROJECT", "EXTERNAL_PROJECT", "UNKNOWN"}:
            return intent
    except Exception as e:
        log.debug("project intent classifier failed, fallback rule-based: %s", e)

    return _fallback_project_intent(text)


def _extract_kb_probe_lines(fact_context: str, max_items: int = 8) -> List[str]:
    lines: List[str] = []
    for raw in str(fact_context or "").splitlines():
        line = str(raw or "").strip()
        if not line.startswith("- "):
            continue
        value = re.sub(r"\s+", " ", line[2:].strip())
        if not value:
            continue
        lowered = _fold_text(value)
        if "do not invent" in lowered or "when answering factual details" in lowered:
            continue
        if "evidence lines" in lowered:
            continue
        if "hien chua truy xuat du lieu noi bo" in lowered:
            continue
        lines.append(value)
        if len(lines) >= max_items:
            break
    return lines


def _compress_fact_context(user_text: str, fact_context: str, max_lines: int = 4, max_line_chars: int = 260) -> str:
    lines = _extract_kb_probe_lines(fact_context, max_items=12)
    if not lines:
        return ""

    query_tokens = set(re.findall(r"[a-z0-9]+", _fold_text(user_text)))
    scored: List[tuple[float, str]] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", str(line or "")).strip()
        if not normalized:
            continue
        folded = _fold_text(normalized)
        if folded.startswith("#"):
            continue
        line_tokens = set(re.findall(r"[a-z0-9]+", folded))
        overlap = 0.0
        if query_tokens and line_tokens:
            overlap = len(query_tokens.intersection(line_tokens)) / float(len(query_tokens))
        scored.append((overlap, normalized))

    if not scored:
        return ""

    compact_lines: List[str] = []
    for _, line in sorted(scored, key=lambda item: item[0], reverse=True):
        clipped = line
        if len(clipped) > max_line_chars:
            clipped = clipped[: max_line_chars - 3].rstrip(" ,;:-.") + "..."
        compact_lines.append(clipped)
        if len(compact_lines) >= max_lines:
            break

    if not compact_lines:
        return ""

    rules = [
        "Evidence constraints (must follow):",
        "- When answering factual details, only use statements in 'Evidence lines' below.",
        "- Do not invent, swap, or rewrite numeric mappings from those lines.",
        "Evidence lines:",
    ]
    rules.extend(f"- {line}" for line in compact_lines)
    return "\n".join(rules)


def _compact_citation_sources(sources: List[str], max_items: int = 3) -> List[str]:
    compact: List[str] = []
    seen: set[str] = set()
    for src in sources or []:
        value = str(src or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        compact.append(value)
        if len(compact) >= max_items:
            break
    return compact


def _is_location_query(user_text: str) -> bool:
    folded = _fold_text(user_text)
    hints = (
        "o dau",
        "toa lac",
        "vi tri",
        "dia chi",
        "thuoc khu nao",
        "nam o dau",
    )
    return any(h in folded for h in hints)


def _is_price_query(user_text: str) -> bool:
    folded = _fold_text(user_text)
    if "gia" not in folded:
        return False
    hints = ("bao nhieu", "don gia", "muc gia", "gia tu", "gia ban", "so tien")
    return any(h in folded for h in hints)


def _is_product_type_query(user_text: str) -> bool:
    folded = _fold_text(user_text)
    hints = (
        "loai hinh",
        "san pham gi",
        "co may loai",
        "mau san pham",
        "dong san pham",
    )
    return any(h in folded for h in hints) and ("san pham" in folded or "du an" in folded or "noble" in folded)


def _extract_location_from_evidence_line(line: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return None

    patterns = [
        r"(?i)\bvị trí\s*:\s*(.+?)(?=\s+-\s+(?:định vị|tagline|slogan|quy mô)\b|$)",
        r"(?i)\btọa lạc(?: tại)?\s*(.+?)(?=[\.;]|$)",
        r"(?i)\bthuộc\s+(.+?)(?=[\.;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = re.sub(r"\s+", " ", (match.group(1) or "")).strip(" -,:;.")
        if value:
            return value
    return None


def _extract_product_type_from_evidence_line(line: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    if not text:
        return None
    patterns = [
        r"(?i)\bloại hình\s*:\s*(.+?)(?=\s+[+\-]\s+số căn\b|-\s+tổng mức đầu tư\b|$)",
        r"(?i)\bsản phẩm áp dụng\s*:\s*(.+?)(?=\s+-\s+điều kiện áp dụng\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = re.sub(r"\s+", " ", (match.group(1) or "")).strip(" -,:;.")
        if value:
            return value
    return None


def _extract_price_line_from_evidence(lines: List[str]) -> Optional[str]:
    strong_hints = ("gia ban", "don gia", "gia tu", "muc gia", "bang gia")
    ignore_hints = ("qua tang", "the debit", "han muc the", "uu dai", "chiet khau")
    money_pattern = re.compile(r"\d[\d\.,]*\s*(?:ty|trieu|vnd|dong|/m2|m2)")

    for line in lines:
        text = re.sub(r"\s+", " ", str(line or "")).strip()
        folded = _fold_text(text)
        if not text:
            continue
        if any(h in folded for h in ignore_hints):
            continue
        if any(h in folded for h in strong_hints) and money_pattern.search(folded):
            return text
    return None


def _append_internal_citation(answer: str, citation_sources: List[str]) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""
    src = str((citation_sources or [""])[0] or "").strip()
    if not src:
        return text
    if re.search(r"(?is)\bnguồn\s*:", text):
        return text
    return f"{text} [Nguồn: {src}]"


def _fast_fact_answer_from_evidence(
    *,
    user_text: str,
    fact_context: str,
    citation_sources: List[str],
) -> Optional[str]:
    lines = _extract_kb_probe_lines(fact_context, max_items=12)
    if not lines:
        return None

    if _is_location_query(user_text):
        for line in lines:
            value = _extract_location_from_evidence_line(line)
            if not value:
                continue
            answer = f"Noble Palace Tây Thăng Long có vị trí tại {value}."
            return _append_internal_citation(answer, citation_sources)

    if _is_product_type_query(user_text):
        for line in lines:
            product_type = _extract_product_type_from_evidence_line(line)
            if not product_type:
                continue
            answer = f"Các loại hình sản phẩm của dự án gồm: {product_type}."
            return _append_internal_citation(answer, citation_sources)

    if _is_price_query(user_text):
        price_line = _extract_price_line_from_evidence(lines)
        if price_line:
            answer = f"Theo dữ liệu nội bộ hiện có: {price_line}"
            return _append_internal_citation(answer, citation_sources)
        return None

    return None


def _fallback_evidence_relevance(user_text: str, evidence_lines: List[str]) -> bool:
    if not evidence_lines:
        return False
    q_tokens = {
        tok for tok in re.findall(r"[a-z0-9]+", _fold_text(user_text))
        if len(tok) >= 3
    }
    if not q_tokens:
        return True
    e_tokens: set[str] = set()
    for line in evidence_lines:
        e_tokens.update(tok for tok in re.findall(r"[a-z0-9]+", _fold_text(line)) if len(tok) >= 3)
    if not e_tokens:
        return False
    overlap_ratio = len(q_tokens.intersection(e_tokens)) / float(len(q_tokens))
    return overlap_ratio >= 0.18


async def _is_internal_kb_relevant(user_text: str, history: List[Dict[str, Any]]) -> bool:
    fact_context = await build_rag_fact_constraints(user_text, top_k=8)
    evidence_lines = _extract_kb_probe_lines(fact_context, max_items=8)
    if not evidence_lines:
        return False

    evidence_block = "\n".join(f"- {line}" for line in evidence_lines)
    history_text = _format_recent_history(history, max_turns=4)
    prompt = f"""
Bạn là bộ đánh giá mức liên quan của bằng chứng nội bộ cho câu hỏi người dùng.
Nhiệm vụ: quyết định liệu dữ liệu nội bộ hiện có có đủ liên quan để trả lời trực tiếp không.

Trả JSON duy nhất:
{{
  "decision": "RELEVANT|IRRELEVANT",
  "reason": "<ngắn gọn>"
}}

Quy tắc:
- RELEVANT: bằng chứng trả lời trực tiếp trọng tâm câu hỏi.
- IRRELEVANT: bằng chứng chỉ liên quan chung chung, lệch trọng tâm, thiếu dữ kiện để trả lời.
- Nếu không chắc chắn, ưu tiên IRRELEVANT để hệ thống chuyển sang tìm nguồn ngoài.

Ngữ cảnh gần nhất:
{history_text or "(không có)"}

Câu hỏi người dùng:
{user_text}

Bằng chứng nội bộ:
{evidence_block}
""".strip()

    try:
        raw = await asyncio.wait_for(
            llm_model_func(
                prompt,
                enable_cot=False,
                response_format={"type": "json_object"},
            ),
            timeout=_SALES_LLM_CLASSIFIER_TIMEOUT_SEC,
        )
        payload = json.loads(str(raw or "{}"))
        decision = str(payload.get("decision") or "").strip().upper()
        if decision == "RELEVANT":
            return True
        if decision == "IRRELEVANT":
            return False
    except Exception as e:
        log.debug("kb relevance gate failed, fallback lexical check: %s", e)

    return _fallback_evidence_relevance(user_text, evidence_lines)




def _normalize_sunny_style(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    # Ưu tiên văn bản rõ ràng cho TTS: hạn chế viết tắt thông dụng.
    text = re.sub(r"\bTP\.\s*", "Thành phố ", text)
    text = re.sub(r"\bTP\b", "Thành phố", text)
    text = re.sub(r"\bKĐT\b", "Khu đô thị", text)
    text = re.sub(r"\bDT\b", "Diện tích", text)

    text = re.sub(r"\bchào\s+anh\s*/\s*chị\b", "Chào bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchào\s+anh\s+chị\b", "Chào bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchào\s+anh\b", "Chào bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchào\s+chị\b", "Chào bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\banh\s*/\s*chị\b", "bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\banh\s+chị\b", "bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\banh\b", "bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchị\b", "bạn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bquý\s*khách\b", "bạn", text, flags=re.IGNORECASE)
    # Tránh thay đại từ ngôi thứ nhất toàn cục (dễ làm sai nghĩa câu của người dùng,
    # ví dụ "gia đình tôi" bị biến thành "gia đình Sunny").
    # Chỉ chuẩn hóa khi đại từ đứng ở đầu câu như một cách xưng hô của trợ lý.
    text = re.sub(r"(^|[.!?]\s+)(em|tôi|mình)\b", r"\1Sunny", text, flags=re.IGNORECASE)
    return text


def _is_search_approval_yes(user_text: str) -> bool:
    lowered = (user_text or "").strip().lower()
    if not lowered:
        return False
    if _is_search_approval_no(user_text):
        return False
    return any(re.search(p, lowered) for p in _SEARCH_APPROVE_PATTERNS)


def _is_search_approval_no(user_text: str) -> bool:
    lowered = (user_text or "").strip().lower()
    if not lowered:
        return False
    return any(re.search(p, lowered) for p in _SEARCH_DENY_PATTERNS)


def _looks_like_search_approval_candidate(user_text: str) -> bool:
    lowered = (user_text or "").strip().lower()
    if not lowered:
        return False
    return any(hint in lowered for hint in _SEARCH_APPROVAL_CANDIDATE_HINTS)


async def _interpret_search_approval_semantic(user_text: str) -> Optional[bool]:
    """Return True/False/None for approve/deny/unclear using LLM semantic parsing."""
    text = (user_text or "").strip()
    if not text:
        return None
    if not _looks_like_search_approval_candidate(text):
        return None

    prompt = f"""
Bạn là bộ phân loại đồng ý tìm kiếm nguồn bên ngoài.
Nhiệm vụ: đọc câu người dùng và trả JSON duy nhất:
{{"decision":"approve|deny|unclear"}}

Quy tắc:
- approve: người dùng cho phép tìm nguồn ngoài/web ngay (ví dụ: "tìm đi em", "ok tìm giúp", "cứ search đi").
- deny: người dùng từ chối tìm nguồn ngoài (ví dụ: "không cần", "đừng tìm ngoài").
- unclear: không rõ ý hoặc không liên quan việc cho phép tìm nguồn ngoài.

Câu người dùng: "{text}"
""".strip()

    try:
        raw = await asyncio.wait_for(
            llm_model_func(
                prompt,
                enable_cot=False,
                response_format={"type": "json_object"},
            ),
            timeout=_SALES_LLM_CLASSIFIER_TIMEOUT_SEC,
        )
        payload = json.loads(str(raw or "{}"))
        decision = str(payload.get("decision") or "").strip().lower()
        if decision == "approve":
            return True
        if decision == "deny":
            return False
        return None
    except Exception as e:
        log.debug("semantic approval parse failed: %s", e)
        return None


def _search_approval_prompt() -> str:
    return (
        "Sunny thấy dữ liệu nội bộ hiện chưa đủ để trả lời chắc chắn. "
        "Bạn có cho phép Sunny tìm thêm nguồn bên ngoài để lấy thông tin mới nhất không? "
        "Bạn chỉ cần trả lời: có hoặc không."
    )


def _search_denied_reply() -> str:
    return (
        "Sunny hiểu rồi, mình sẽ không tìm nguồn bên ngoài. "
        "Nếu bạn muốn, Sunny sẽ tiếp tục tư vấn trong phạm vi tài liệu nội bộ hiện có."
    )


async def _guard_external_search_with_hil(
    *,
    session_id: str,
    user_text: str,
) -> Optional[Dict[str, Any]]:
    profile = await load_lead_profile(session_id) or {"lead_id": session_id, "current_state": "greeting"}
    pending_query = str(profile.get("external_search_request_query") or "").strip()
    pending = bool(profile.get("external_search_pending")) and bool(pending_query)
    semantic_decision = await _interpret_search_approval_semantic(user_text)

    if _is_search_approval_no(user_text):
        profile["external_search_pending"] = False
        profile["external_search_approved"] = False
        profile["external_search_request_query"] = ""
        await save_lead_profile(session_id, profile)
        answer = _normalize_sunny_style(_search_denied_reply())
        await _persist_chat_turn(session_id, user_text, answer)
        return {
            "route_category": "HIL",
            "final_response": answer,
            "next_sales_state": None,
            "lead_profile": profile,
            "missing_slots": None,
        }

    approved_now = _is_search_approval_yes(user_text) or semantic_decision is True
    denied_now = semantic_decision is False

    if denied_now:
        profile["external_search_pending"] = False
        profile["external_search_approved"] = False
        profile["external_search_request_query"] = ""
        await save_lead_profile(session_id, profile)
        answer = _normalize_sunny_style(_search_denied_reply())
        await _persist_chat_turn(session_id, user_text, answer)
        return {
            "route_category": "HIL",
            "final_response": answer,
            "next_sales_state": None,
            "lead_profile": profile,
            "missing_slots": None,
        }

    if pending and approved_now:
        profile["external_search_pending"] = False
        profile["external_search_approved"] = True
        await save_lead_profile(session_id, profile)
        return None

    if not pending:
        profile["external_search_pending"] = True
        profile["external_search_approved"] = False
        profile["external_search_request_query"] = (user_text or "").strip()
    await save_lead_profile(session_id, profile)
    answer = _normalize_sunny_style(_search_approval_prompt())
    await _persist_chat_turn(session_id, user_text, answer)
    return {
        "route_category": "HIL",
        "final_response": answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


async def _consume_approved_search_query(session_id: str, fallback_query: str) -> str:
    profile = await load_lead_profile(session_id) or {}
    approved = bool(profile.get("external_search_approved"))
    pending_query = str(profile.get("external_search_request_query") or "").strip()

    if approved and pending_query:
        profile["external_search_approved"] = False
        profile["external_search_request_query"] = ""
        await save_lead_profile(session_id, profile)
        return pending_query

    return fallback_query


def _has_pending_external_search_request(profile: Optional[Dict[str, Any]]) -> bool:
    data = profile or {}
    return bool(data.get("external_search_pending")) and bool(str(data.get("external_search_request_query") or "").strip())


def _strip_citation_markers(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""

    cleaned = re.sub(r"(?is)\[\s*(?:nguồn|nguon|source)\s*:[^\]]*\]", " ", raw)
    cleaned = re.sub(r"(?is)\(\s*(?:nguồn|nguon|source)\s*:[^\)]*\)", " ", cleaned)
    cleaned = re.sub(
        r"(?im)(?:^|\s)(?:nguồn|nguon|source)(?:\s*tham\s*(?:khảo|khao))?\s*:\s*[^\n\r]+",
        " ",
        cleaned,
    )
    # TTS output must not read raw links.
    cleaned = re.sub(r"(?i)\bhttps?://\S+\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_citation_policy_context(sources: List[str]) -> str:
    strict_required = (os.getenv("STRICT_CITATION_REQUIRED") or "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not strict_required:
        return (
            "Citation policy (relaxed):\n"
            "- Chỉ dùng dữ liệu trong tài liệu nội bộ đã truy xuất.\n"
            "- Không bắt buộc in nguồn trong câu trả lời cuối cùng cho người dùng."
        )

    cleaned = [str(item).strip() for item in (sources or []) if str(item).strip()]
    if not cleaned:
        return (
            "Citation policy (strict):\n"
            "- Không có nguồn tài liệu nội bộ hợp lệ trong ngữ cảnh truy xuất.\n"
            "- Chỉ trả đúng câu sau và không thêm nội dung khác: \"Sunny chưa tìm thấy nguồn trong tài liệu nội bộ để xác nhận thông tin này.\""
        )

    rules = [
        "Citation policy (strict):",
        "- Chỉ được trả lời bằng thông tin có trong tài liệu nội bộ đã truy xuất.",
        "- Bắt buộc có trích nguồn cuối câu trả lời theo định dạng: [Nguồn: <file_path>]",
        "- Ít nhất 1 nguồn phải thuộc danh sách cho phép dưới đây.",
        "- Nếu không chắc nguồn, chỉ trả đúng: \"Sunny chưa tìm thấy nguồn trong tài liệu nội bộ để xác nhận thông tin này.\"",
        "Allowed sources:",
    ]
    rules.extend(f"- {item}" for item in cleaned)
    return "\n".join(rules)


def _answer_has_allowed_citation(answer: str, sources: List[str]) -> bool:
    strict_required = (os.getenv("STRICT_CITATION_REQUIRED") or "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not strict_required:
        return True

    text = str(answer or "")
    if not text.strip() or not sources:
        return False

    # Accept flexible forms, e.g. [Nguồn: file.md] or Nguồn: file.md
    citation_tokens = re.findall(r"(?is)nguồn\s*:\s*([^\]\n\r]+)", text)
    if not citation_tokens:
        return False

    allowed = {s.lower() for s in sources}
    allowed_basenames = {Path(s).name.lower() for s in sources}

    for token in citation_tokens:
        norm = re.sub(r"\s+", " ", str(token or "").strip()).lower()
        if not norm:
            continue
        if norm in allowed:
            return True
        if Path(norm).name in allowed_basenames:
            return True
        for src in allowed:
            if src in norm:
                return True
    return False


_FACT_COVERAGE_STOPWORDS = {
    "la",
    "co",
    "cua",
    "cho",
    "voi",
    "ve",
    "duoc",
    "nhung",
    "nhu",
    "nao",
    "ra",
    "sao",
    "bao",
    "nhieu",
    "gioi",
    "thieu",
    "toi",
    "mua",
    "ban",
    "du",
    "an",
}


def _coverage_tokens(text: str, *, min_len: int = 3) -> set[str]:
    folded = _fold_text(text)
    tokens = set(re.findall(r"[a-z0-9]+", folded))
    return {tok for tok in tokens if len(tok) >= min_len and tok not in _FACT_COVERAGE_STOPWORDS}


def _coverage_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", str(text or "")))


def _compute_fact_coverage(user_text: str, rag_answer: str, fact_context: str) -> Dict[str, Any]:
    evidence_lines = _extract_kb_probe_lines(fact_context, max_items=12)
    query_tokens = _coverage_tokens(user_text)
    answer_tokens = _coverage_tokens(rag_answer)
    answer_numbers = _coverage_numbers(rag_answer)
    line_overlap_min = float((os.getenv("RAG_FACT_LINE_OVERLAP_MIN") or "0.30").strip() or "0.30")

    query_overlap = 1.0
    if query_tokens:
        query_overlap = len(query_tokens.intersection(answer_tokens)) / float(len(query_tokens))

    hit_lines = 0
    best_line_overlap = 0.0
    for line in evidence_lines:
        line_tokens = _coverage_tokens(line)
        if not line_tokens:
            continue
        token_overlap = len(line_tokens.intersection(answer_tokens)) / float(len(line_tokens))
        line_numbers = _coverage_numbers(line)
        number_hit = bool(line_numbers.intersection(answer_numbers)) if line_numbers and answer_numbers else False
        best_line_overlap = max(best_line_overlap, token_overlap)
        if token_overlap >= line_overlap_min or number_hit:
            hit_lines += 1

    evidence_coverage = 0.0
    if evidence_lines:
        evidence_coverage = hit_lines / float(len(evidence_lines))

    return {
        "query_overlap": query_overlap,
        "hit_lines": hit_lines,
        "total_lines": len(evidence_lines),
        "evidence_coverage": evidence_coverage,
        "best_line_overlap": best_line_overlap,
        "has_evidence": bool(evidence_lines),
    }


def _is_fact_coverage_sufficient(user_text: str, rag_answer: str, fact_context: str) -> bool:
    metrics = _compute_fact_coverage(user_text=user_text, rag_answer=rag_answer, fact_context=fact_context)
    min_query_overlap = float((os.getenv("RAG_FACT_MIN_QUERY_OVERLAP") or "0.12").strip() or "0.12")
    min_evidence_coverage = float((os.getenv("RAG_FACT_MIN_EVIDENCE_COVERAGE") or "0.12").strip() or "0.12")
    min_hit_lines = int((os.getenv("RAG_FACT_MIN_HIT_LINES") or "1").strip() or "1")

    covered = (
        metrics["has_evidence"]
        and metrics["query_overlap"] >= min_query_overlap
        and metrics["evidence_coverage"] >= min_evidence_coverage
        and metrics["hit_lines"] >= min_hit_lines
    )

    log.info(
        "RAG fact coverage: %s query_overlap=%.3f evidence_coverage=%.3f hit_lines=%d/%d best_line_overlap=%.3f thresholds=(q>=%.2f ev>=%.2f hit>=%d)",
        "COVERED" if covered else "UNCOVERED",
        float(metrics["query_overlap"]),
        float(metrics["evidence_coverage"]),
        int(metrics["hit_lines"]),
        int(metrics["total_lines"]),
        float(metrics["best_line_overlap"]),
        min_query_overlap,
        min_evidence_coverage,
        min_hit_lines,
    )
    return covered


def _extract_phone_number(user_text: str) -> Optional[str]:
    raw = (user_text or "").strip()
    if not raw:
        return None
    match = _PHONE_PATTERN.search(raw)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(0))
    if digits.startswith("84") and len(digits) >= 11:
        digits = "0" + digits[2:]
    if len(digits) < 9 or len(digits) > 11:
        return None
    return digits


def _extract_name(user_text: str) -> Optional[str]:
    text = (user_text or "").strip()
    if not text:
        return None
    patterns = [
        r"(?:tôi là|toi la|mình là|minh la|em là|tên là|ten la|gọi tôi là|goi toi la|gọi mình là|goi minh la)\s+([A-Za-zÀ-ỹ\s]{2,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        name = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;")
        if len(name) < 2:
            continue
        return name.title()
    return None


def _extract_housing_preference(user_text: str) -> Optional[str]:
    text = (user_text or "").strip()
    lowered = text.lower()
    if _extract_phone_number(text):
        return None
    if re.search(r"\b(tên là|toi la|tôi là|mình là|minh la|gọi tôi là|goi toi la)\b", lowered):
        return None
    if not text or not any(token in lowered for token in _PREFERENCE_HINTS):
        return None
    return text[:320]


def _is_consulting_intent(user_text: str) -> bool:
    lowered = (user_text or "").strip().lower()
    if not lowered:
        return False
    intent_hints = (
        "tư vấn",
        "tu van",
        "báo giá",
        "bao gia",
        "giá",
        "gia",
        "chính sách",
        "chinh sach",
        "đặt lịch",
        "dat lich",
        "xem nhà",
        "xem du an",
        "liên hệ",
        "lien he",
        "mua",
        "đầu tư",
        "dau tu",
        "shophouse",
    )
    return any(hint in lowered for hint in intent_hints)


async def _update_customer_profile(session_id: str, user_text: str) -> Dict[str, Any]:
    profile = await load_lead_profile(session_id) or {"lead_id": session_id, "current_state": "greeting"}
    updated = False

    housing_preference = _extract_housing_preference(user_text)
    if housing_preference and not str(profile.get("housing_preference") or "").strip():
        profile["housing_preference"] = housing_preference
        updated = True

    name = _extract_name(user_text)
    if name and not profile.get("name"):
        profile["name"] = name
        updated = True

    phone = _extract_phone_number(user_text)
    if phone and not profile.get("contact_phone"):
        profile["contact_phone"] = phone
        updated = True

    info_complete = bool(
        str(profile.get("housing_preference") or "").strip()
        and str(profile.get("name") or "").strip()
        and str(profile.get("contact_phone") or "").strip()
    )
    if bool(profile.get("information_customer")) != info_complete:
        profile["information_customer"] = info_complete
        updated = True

    if updated:
        await save_lead_profile(session_id, profile)
    return profile


def _build_follow_up_question(profile: Dict[str, Any], user_text: str) -> str:
    info_complete = bool(profile.get("information_customer"))
    has_preference = bool(str(profile.get("housing_preference") or "").strip())
    has_name = bool(str(profile.get("name") or "").strip())
    has_phone = bool(str(profile.get("contact_phone") or "").strip())
    consulting_intent = _is_consulting_intent(user_text)

    if info_complete:
        preference = str(profile.get("housing_preference") or "").strip()
        if preference:
            return (
                "Bạn muốn Sunny đào sâu thêm điểm nào của sản phẩm bạn đang quan tâm, "
                f"dựa trên gu nhà ở này: {preference}?"
            )
        return "Bạn muốn Sunny đào sâu thêm điểm nào của sản phẩm bạn đang quan tâm tại Noble Palace Tây Thăng Long?"

    if not has_preference:
        return (
            "Bạn đang tò mò nhất điều gì về Noble Palace Tây Thăng Long? "
            "Nếu tiện, bạn chia sẻ thêm gu nhà ở để Sunny tư vấn sát hơn nhé."
        )
    if not consulting_intent:
        return "Bạn muốn Sunny đào sâu thêm điểm nào của Noble Palace Tây Thăng Long để trả lời sát điều bạn vừa hỏi?"
    if not has_name and not has_phone:
        return (
            "Bạn còn tò mò điểm nào của Noble Palace Tây Thăng Long? "
            "Sunny cũng xin tên và số điện thoại để hỗ trợ bạn sát gu nhà ở nhé."
        )
    if not has_name:
        return (
            "Bạn còn tò mò thêm điều gì về Noble Palace Tây Thăng Long? "
            "Sunny xin tên của bạn để tiện xưng hô và theo sát nhu cầu nhé."
        )
    if not has_phone:
        return (
            "Bạn muốn Sunny phân tích thêm điểm nào của Noble Palace Tây Thăng Long? "
            "Bạn cho Sunny xin số điện thoại để tiện hỗ trợ nhanh khi cần nhé."
        )
    return "Bạn muốn Sunny phân tích sâu thêm điểm nào của Noble Palace Tây Thăng Long để mình đi tiếp?"


def _save_customer_profile_txt(session_id: str, profile: Dict[str, Any]) -> Optional[str]:
    name = str(profile.get("name") or "").strip()
    phone = str(profile.get("contact_phone") or "").strip()
    housing_preference = str(profile.get("housing_preference") or "").strip()
    if not (name and phone and housing_preference):
        return None

    _PROFILE_TXT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _PROFILE_TXT_DIR / f"{session_id}.txt"
    content = (
        f"updated_at: {now_vietnam_str()}\n"
        f"session_id: {session_id}\n"
        f"name: {name}\n"
        f"contact_phone: {phone}\n"
        f"housing_preference: {housing_preference}\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


async def _finalize_assistant_answer(
    session_id: str,
    user_text: str,
    answer: str,
    *,
    append_follow_up: bool = False,
) -> Dict[str, Any]:
    profile = await _update_customer_profile(session_id, user_text)
    text = _normalize_sunny_style(_strip_citation_markers(answer))
    if not text:
        text = "Sunny đang ở đây để hỗ trợ bạn."

    if append_follow_up:
        follow_up = _build_follow_up_question(profile, user_text)
        if follow_up not in text:
            text = f"{text}\n\n{follow_up}"

    _save_customer_profile_txt(session_id, profile)
    return {"answer": text, "lead_profile": profile}


async def _persist_chat_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    user = (user_text or "").strip()
    assistant = (assistant_text or "").strip()
    if not user or not assistant:
        return
    try:
        await append_turn(session_id, user, assistant)
    except Exception as e:
        log.warning("append_turn failed: session=%s error=%s", session_id, e)


async def _run_search_flow(
    *,
    session_id: str,
    user_text: str,
    search_query: str,
    history: List[Dict[str, Any]],
    external_notice: Optional[str] = None,
) -> Dict[str, Any]:
    answer = await summarize_search_answer(
        user_query=user_text,
        search_query=search_query or user_text,
        history=history,
        system_persona=_SEARCH_PERSONA,
        max_sentences=4,
    )
    merged_answer = str(answer or "").strip()
    notice = str(external_notice or "").strip()
    if notice:
        if merged_answer:
            merged_answer = f"{notice}\n\n{merged_answer}"
        else:
            merged_answer = notice

    finalized = await _finalize_assistant_answer(session_id, user_text, merged_answer)
    patched_answer = finalized["answer"]
    profile = finalized["lead_profile"]

    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "SEARCH",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


async def _run_comparison_flow(
    *,
    session_id: str,
    user_text: str,
    search_query: str,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fact_context = await build_rag_fact_constraints(user_text)
    citation_sources = await get_rag_citation_sources(user_text, top_k=8)
    citation_policy = _build_citation_policy_context(citation_sources)
    system_context = "\n\n".join(part for part in [fact_context, citation_policy] if part)
    rag_answer = (
        await query_rag(
            text=user_text,
            top_k=6,
            history=history,
            system_context=system_context,
        )
    ).strip()

    if rag_answer and not _answer_has_allowed_citation(rag_answer, citation_sources):
        rag_answer = ""

    search_answer = await summarize_search_answer(
        user_query=user_text,
        search_query=search_query or user_text,
        history=history,
        system_persona=_SEARCH_PERSONA,
        max_sentences=4,
    )
    search_answer = (search_answer or "").strip()

    internal_part = rag_answer or "Sunny chưa thấy đủ bằng chứng nội bộ mạnh cho phần so sánh này."
    external_part = search_answer or "Sunny chưa lấy được nguồn bên ngoài phù hợp ở lần thử này."
    merged = (
        "So sánh nhanh theo 2 nguồn:\n"
        f"- Dữ liệu nội bộ: {internal_part}\n"
        f"- Nguồn bên ngoài: {external_part}\n"
        "Sunny có thể đi sâu từng tiêu chí bạn muốn ưu tiên (giá, vị trí, pháp lý, tiến độ, thanh khoản)."
    )

    finalized = await _finalize_assistant_answer(session_id, user_text, merged)
    patched_answer = finalized["answer"]
    profile = finalized["lead_profile"]
    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "COMPARISON",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


async def _run_fast_reply_flow(
    *,
    session_id: str,
    user_text: str,
    reply: str,
) -> Dict[str, Any]:
    finalized = await _finalize_assistant_answer(
        session_id,
        user_text,
        reply,
        append_follow_up=False,
    )
    patched_answer = finalized["answer"]
    profile = finalized["lead_profile"]
    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "FAST_REPLY",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


async def _run_sales_advisor_flow(
    *,
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
    lead_profile: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Sales advisor tool: consult product options from internal KB and ask next qualifying question."""
    answer = await run_sales_advisor_tool(
        user_query=user_text,
        history=history,
        lead_profile=lead_profile or {},
    )
    if not str(answer or "").strip():
        return None

    finalized = await _finalize_assistant_answer(
        session_id,
        user_text,
        answer,
        append_follow_up=False,
    )
    patched_answer = finalized["answer"]
    profile = finalized["lead_profile"]
    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "SALES_TOOL",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


async def _run_general_chat_flow(
    *,
    session_id: str,
    user_text: str,
    history: List[Dict[str, Any]],
    force_boundary: bool = False,
) -> Dict[str, Any]:
    if force_boundary or (not _is_project_real_estate_query(user_text)):
        answer = _NON_DOMAIN_BOUNDARY_REPLY
    else:
        prompt = _build_general_chat_prompt(user_text, history)
        try:
            raw = await asyncio.wait_for(
                llm_model_func(prompt, enable_cot=False),
                timeout=_SALES_LLM_TIMEOUT_SEC,
            )
            answer = str(raw or "").strip()
        except Exception as e:
            log.warning("general chat fallback failed: session=%s error=%s", session_id, e)
            answer = ""
        if not answer:
            answer = _NON_DOMAIN_BOUNDARY_REPLY

    finalized = await _finalize_assistant_answer(
        session_id,
        user_text,
        answer,
        append_follow_up=False,
    )
    patched_answer = finalized["answer"]
    profile = finalized["lead_profile"]
    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "CHAT",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


def _stream_meta_from_flow_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Các khóa final_meta cho /chat/stream sau khi chạy _run_*_flow."""
    return {
        "route_category": result.get("route_category"),
        "sales_state": result.get("next_sales_state"),
        "missing_slots": result.get("missing_slots"),
        "lead_profile": result.get("lead_profile"),
    }


def _ndjson_response_line(session_id: str, chunk: str) -> str:
    """Một dòng NDJSON phase=response (dùng cho stream token hoặc chunk)."""
    return json.dumps(
        {
            "chunk": chunk,
            "done": False,
            "phase": "response",
            "session_id": session_id,
        },
        ensure_ascii=False,
    ) + "\n"


async def _yield_ndjson_response_chunks(session_id: str, full_text: str):
    """Sinh các dòng NDJSON (phase=response) từ nội dung đã có sẵn."""
    for chunk in iter_stream_chunks((full_text or "").strip()):
        yield _ndjson_response_line(session_id, chunk)
        await asyncio.sleep(0)


async def _should_auto_comparison(user_text: str, kb_hit: bool) -> bool:
    if not kb_hit:
        return False
    if not _has_explicit_comparison_intent(user_text):
        return False

    try:
        raw = await tavily_search(user_text, max_results=3)
    except Exception as e:
        log.warning("comparison search probe failed: %s", e)
        return False

    text = str(raw or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if "search tool not available" in lowered:
        return False
    if "kh?ng t?m th?y k?t qu?" in lowered or "khong tim thay ket qua" in lowered:
        return False
    if lowered.startswith("error"):
        return False

    return _is_real_estate_text(text) and (not _is_noble_related_text(text))


def _should_try_rag_first(user_text: str, kb_hit: bool) -> bool:
    """Prefer internal KB attempt for project-domain questions, even on probe miss."""
    if kb_hit:
        return True
    folded = _fold_text(user_text)
    if _is_noble_related_text(user_text):
        return True
    internal_hints = (
        "uu dai",
        "chinh sach",
        "gia",
        "thanh toan",
        "san pham",
        "shophouse",
        "biet thu",
        "nha pho",
        "du an",
        "tay thang long",
        "noble palace",
    )
    return any(h in folded for h in internal_hints)
def _has_search_tool_hint(user_text: str) -> bool:
    folded = _fold_text(user_text)
    return any(h in folded for h in _SEARCH_TOOL_HINTS)


def _should_try_search_tool(user_text: str, kb_hit: bool) -> bool:
    return _has_search_tool_hint(user_text) or (not kb_hit)


def _is_out_of_scope_query(user_text: str) -> bool:
    folded = _fold_text(user_text)
    if any(h in folded for h in _IN_SCOPE_FINANCE_HINTS):
        return False
    return any(h in folded for h in _OUT_OF_SCOPE_HINTS)




async def _resolve_evidence_route(user_text: str, history: List[Dict[str, Any]]) -> str:
    """Evidence-first routing with search-evidence override for external comparison asks."""
    kb_hit = await kb_evidence_probe(user_text, history)
    route = "RAG" if kb_hit else "SEARCH"

    log.info("Evidence router: route=%s kb_hit=%s", route, kb_hit)
    return route


async def _run_rag_flow(
    *,
    session_id: str,
    user_text: str,
    rag_query: str,
    history: List[Dict[str, Any]],
    raw_transcript: Optional[str] = None,
    allow_search_fallback: bool = False,
) -> Dict[str, Any]:
    fact_context = await build_rag_fact_constraints(rag_query or user_text)
    citation_sources = await get_rag_citation_sources(rag_query or user_text, top_k=8)
    fast_answer = _fast_fact_answer_from_evidence(
        user_text=rag_query or user_text,
        fact_context=fact_context,
        citation_sources=citation_sources,
    )
    if fast_answer:
        rag_answer = fast_answer
        log.info("RAG fast-path hit: session=%s query=%s", session_id, (rag_query or user_text)[:120])
    else:
        compact_fact_context = _compress_fact_context(rag_query or user_text, fact_context, max_lines=4, max_line_chars=260)
        compact_sources = _compact_citation_sources(citation_sources, max_items=3)
        citation_policy = _build_citation_policy_context(compact_sources)
        system_context = "\n\n".join(part for part in [compact_fact_context, citation_policy, _RAG_STYLE_POLICY] if part)
        log.info(
            "RAG context compacted: session=%s fact_chars=%d->%d sources=%d->%d",
            session_id,
            len(fact_context or ""),
            len(compact_fact_context or ""),
            len(citation_sources or []),
            len(compact_sources or []),
        )
        rag_answer = await query_rag(
            text=rag_query or user_text,
            top_k=6,
            history=history,
            system_context=system_context,
        )
    if rag_answer.strip() and not _answer_has_allowed_citation(rag_answer, citation_sources):
        log.warning("Blocked uncited RAG answer for session=%s", session_id)
        rag_answer = ""

    if rag_answer.strip():
        fact_covered = _is_fact_coverage_sufficient(
            user_text=rag_query or user_text,
            rag_answer=rag_answer,
            fact_context=fact_context,
        )
        if allow_search_fallback and (not fact_covered):
            return await _run_search_flow(
                session_id=session_id,
                user_text=user_text,
                search_query=user_text,
                history=history,
                external_notice=(
                    "Lưu ý: Sunny thấy độ bao phủ dữ kiện nội bộ hiện chưa đủ chắc chắn cho câu hỏi này, "
                    "nên Sunny đang tổng hợp thêm từ nguồn bên ngoài."
                ),
            )

        finalized = await _finalize_assistant_answer(
            session_id,
            user_text,
            rag_answer,
            append_follow_up=False,
        )
        patched_answer = finalized["answer"]
        profile = finalized["lead_profile"]

        await _persist_chat_turn(session_id, user_text, patched_answer)
        return {
            "route_category": "RAG",
            "final_response": patched_answer,
            "next_sales_state": None,
            "lead_profile": profile,
            "missing_slots": None,
        }

    if allow_search_fallback:
        return await _run_search_flow(
            session_id=session_id,
            user_text=user_text,
            search_query=user_text,
            history=history,
            external_notice=_LOW_KB_EXTERNAL_SEARCH_NOTICE,
        )

    fallback_answer = (
        "Sunny chưa có đủ dữ liệu nội bộ để xác nhận câu hỏi này. "
        "Bạn vui lòng hỏi chi tiết hơn về sản phẩm, giá, chính sách, pháp lý hoặc tiến độ của dự án."
    )
    finalized = await _finalize_assistant_answer(
        session_id,
        user_text,
        fallback_answer,
        append_follow_up=False,
    )
    patched_answer = finalized["answer"]
    profile = finalized["lead_profile"]
    await _persist_chat_turn(session_id, user_text, patched_answer)
    return {
        "route_category": "RAG",
        "final_response": patched_answer,
        "next_sales_state": None,
        "lead_profile": profile,
        "missing_slots": None,
    }


async def _emit_route_hint(
    on_route_hint: Optional[Callable[[str], Optional[Awaitable[None]]]],
    route: str,
) -> None:
    if not callable(on_route_hint):
        return
    try:
        maybe_awaitable = on_route_hint(str(route or "").upper())
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable
    except Exception as e:
        log.debug("route hint callback failed: %s", e)


async def _run_sales_or_search(
    *,
    session_id: str,
    user_text: str,
    raw_transcript: Optional[str] = None,
    on_route_hint: Optional[Callable[[str], Optional[Awaitable[None]]]] = None,
) -> Dict[str, Any]:
    history = await load_chat_history(session_id)
    allow_search_fallback_for_internal = True

    async def _run_search_with_hint(**kwargs: Any) -> Dict[str, Any]:
        await _emit_route_hint(on_route_hint, "SEARCH")
        return await _run_search_flow(**kwargs)

    fast_reply = _fast_reply_from_sys_prompt(user_text)
    if fast_reply:
        return await _run_fast_reply_flow(
            session_id=session_id,
            user_text=user_text,
            reply=fast_reply,
        )

    profile = await load_lead_profile(session_id)
    if _has_pending_external_search_request(profile):
        profile["external_search_pending"] = False
        profile["external_search_approved"] = False
        profile["external_search_request_query"] = ""
        await save_lead_profile(session_id, profile)

    if is_sales_advisor_query(user_text) and (not _has_explicit_comparison_intent(user_text)) and (not _has_search_tool_hint(user_text)):
        advisor_result = await _run_sales_advisor_flow(
            session_id=session_id,
            user_text=user_text,
            history=history,
            lead_profile=profile,
        )
        if advisor_result:
            return advisor_result

    scope = await _classify_domain_scope(user_text, history)
    log.info(
        "sales route gate: scope=%s real_estate=%s project_query=%s search_hint=%s comparison=%s text=%s",
        scope,
        _is_real_estate_text(user_text),
        _is_project_real_estate_query(user_text),
        _has_search_tool_hint(user_text),
        _has_explicit_comparison_intent(user_text),
        user_text[:160],
    )
    if scope == "OUT_OF_SCOPE":
        # For real-estate/project-like questions outside internal KB scope,
        # prefer external web search over hard boundary replies.
        if _is_real_estate_text(user_text) or _has_search_tool_hint(user_text) or _has_explicit_comparison_intent(user_text):
            log.info("sales route decision: OUT_OF_SCOPE -> SEARCH")
            return await _run_search_with_hint(
                session_id=session_id,
                user_text=user_text,
                search_query=user_text,
                history=history,
                external_notice=(
                    "Lưu ý: Câu hỏi này vượt ngoài phạm vi dữ liệu nội bộ dự án, "
                    "Sunny đang tổng hợp thêm từ nguồn bên ngoài để trả lời bạn."
                ),
            )
        log.info("sales route decision: OUT_OF_SCOPE -> CHAT_BOUNDARY")
        return await _run_general_chat_flow(
            session_id=session_id,
            user_text=user_text,
            history=history,
            force_boundary=True,
        )

    if scope == "PROJECT":
        project_intent = await _classify_project_intent(user_text, history)
        if project_intent == "EXTERNAL_PROJECT":
            return await _run_search_with_hint(
                session_id=session_id,
                user_text=user_text,
                search_query=user_text,
                history=history,
                external_notice=(
                    "Lưu ý: Câu hỏi này đang cần thông tin bên ngoài phạm vi dữ liệu nội bộ, "
                    "Sunny sẽ tổng hợp từ nguồn ngoài trước."
                ),
            )

        kb_hit = await kb_evidence_probe(user_text, history)
        if kb_hit:
            kb_relevant = await _is_internal_kb_relevant(user_text, history)
            if not kb_relevant:
                log.info("sales route decision: KB_IRRELEVANT -> SEARCH (skip RAG generate)")
                return await _run_search_with_hint(
                    session_id=session_id,
                    user_text=user_text,
                    search_query=user_text,
                    history=history,
                    external_notice=(
                        "Lưu ý: Sunny thấy bằng chứng nội bộ hiện chưa đủ liên quan để trả lời chắc chắn, "
                        "nên Sunny đang tổng hợp thêm từ nguồn bên ngoài."
                    ),
                )

        if await _should_auto_comparison(user_text, kb_hit):
            await _emit_route_hint(on_route_hint, "SEARCH")
            return await _run_comparison_flow(
                session_id=session_id,
                user_text=user_text,
                search_query=user_text,
                history=history,
            )

        if not kb_hit:
            log.info("sales route decision: KB_MISS -> SEARCH")
            return await _run_search_with_hint(
                session_id=session_id,
                user_text=user_text,
                search_query=user_text,
                history=history,
                external_notice=_LOW_KB_EXTERNAL_SEARCH_NOTICE,
            )

        if _has_search_tool_hint(user_text):
            return await _run_search_with_hint(
                session_id=session_id,
                user_text=user_text,
                search_query=user_text,
                history=history,
            )

        return await _run_rag_flow(
            session_id=session_id,
            user_text=user_text,
            rag_query=user_text,
            history=history,
            raw_transcript=raw_transcript,
            allow_search_fallback=allow_search_fallback_for_internal,
        )

    return await _run_general_chat_flow(
        session_id=session_id,
        user_text=user_text,
        history=history,
    )


@router.post("/chat", response_model=SalesChatResponse)
async def sales_chat(request: SalesChatRequest):
    """
    Main sales chat endpoint.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await ensure_sales_schema()
    try:
        result = await _run_sales_or_search(
            session_id=request.session_id,
            user_text=request.message.strip(),
            raw_transcript=request.raw_transcript,
        )
    except Exception as e:
        log.error("sales_chat error: %s", e)
        raise HTTPException(status_code=500, detail=f"Sales agent error: {e}")

    return SalesChatResponse(
        session_id=request.session_id,
        response=result.get("final_response") or "",
        route_category=result.get("route_category"),
        sales_state=result.get("next_sales_state"),
        lead_profile=result.get("lead_profile"),
        missing_slots=result.get("missing_slots"),
    )


@router.post("/chat/stream")
async def sales_chat_stream(request: SalesChatRequest):
    """Streaming sales chat endpoint."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    await ensure_sales_schema()
    async def generate():
        t_total = time.perf_counter()
        user_text = request.message.strip()
        search_hint_queue: "asyncio.Queue[bool]" = asyncio.Queue()
        search_ack_sent = False

        async def _on_route_hint(route: str) -> None:
            nonlocal search_ack_sent
            if str(route or "").upper() != "SEARCH":
                return
            if search_ack_sent:
                return
            if search_hint_queue.empty():
                search_hint_queue.put_nowait(True)

        fast_reply = _fast_reply_from_sys_prompt(user_text)
        if not fast_reply:
            yield json.dumps(
                {
                    "chunk": random.choice(_THINKING_ACK_MESSAGES),
                    "done": False,
                    "phase": "thinking_ack",
                    "session_id": request.session_id,
                },
                ensure_ascii=False,
            ) + "\n"

        try:
            run_task = asyncio.create_task(
                _run_sales_or_search(
                    session_id=request.session_id,
                    user_text=user_text,
                    raw_transcript=request.raw_transcript,
                    on_route_hint=_on_route_hint,
                )
            )

            while not run_task.done():
                if not search_ack_sent:
                    try:
                        await asyncio.wait_for(search_hint_queue.get(), timeout=0.08)
                        yield json.dumps(
                            {
                                "chunk": random.choice(_SEARCH_WAITING_ACK_MESSAGES),
                                "done": False,
                                "phase": "thinking_ack",
                                "session_id": request.session_id,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                        search_ack_sent = True
                        continue
                    except asyncio.TimeoutError:
                        pass
                await asyncio.sleep(0.02)

            result = await run_task
            route_category = str(result.get("route_category") or "").upper()
            if (not search_ack_sent) and route_category in {"SEARCH", "COMPARISON"}:
                yield json.dumps(
                    {
                        "chunk": random.choice(_SEARCH_WAITING_ACK_MESSAGES),
                        "done": False,
                        "phase": "thinking_ack",
                        "session_id": request.session_id,
                    },
                    ensure_ascii=False,
                ) + "\n"
                search_ack_sent = True
            async for line in _yield_ndjson_response_chunks(
                request.session_id,
                str(result.get("final_response") or ""),
            ):
                yield line
            final_meta = _stream_meta_from_flow_result(result)
            if final_meta.get("lead_profile") is None:
                final_meta["lead_profile"] = await load_lead_profile(request.session_id)

        except Exception as e:
            log.error("sales_chat_stream error: %s", e)
            fallback = "Sunny xin lỗi, Sunny gặp lỗi khi xử lý yêu cầu. Bạn thử lại giúp Sunny nhé."
            yield json.dumps(
                {
                    "chunk": fallback,
                    "done": False,
                    "phase": "error",
                    "session_id": request.session_id,
                },
                ensure_ascii=False,
            ) + "\n"
            yield json.dumps(
                {"chunk": "", "done": True, "phase": "complete", "session_id": request.session_id},
                ensure_ascii=False,
            ) + "\n"
            return

        yield json.dumps(
            {
                "chunk": "",
                "done": True,
                "phase": "complete",
                "session_id": request.session_id,
                "route_category": final_meta.get("route_category"),
                "sales_state": final_meta.get("sales_state"),
                "missing_slots": final_meta.get("missing_slots"),
                "lead_profile": final_meta.get("lead_profile"),
                "latency_sec": round(time.perf_counter() - t_total, 3),
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/lead/{session_id}")
async def get_lead(session_id: str):
    """Get lead profile for a session."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")
    return JSONResponse({"session_id": session_id, "lead_profile": profile})


@router.patch("/lead/{session_id}")
async def update_lead(session_id: str, body: LeadUpdateRequest):
    """Manually update lead profile fields."""
    profile = await load_lead_profile(session_id) or {"lead_id": session_id, "current_state": "greeting"}
    profile.update(body.updates)
    await save_lead_profile(session_id, profile)
    return JSONResponse({"session_id": session_id, "lead_profile": profile})


@router.get("/state/{session_id}")
async def get_sales_state(session_id: str):
    """Get current sales state for a session."""
    ctx = await load_session_context(session_id)
    profile = await load_lead_profile(session_id) or {}
    return JSONResponse(
        {
            "session_id": session_id,
            "current_state": ctx.get("current_state", "greeting"),
            "previous_state": ctx.get("previous_state"),
            "conversation_turn_count": ctx.get("conversation_turn_count", 0),
            "lead_temperature": profile.get("lead_temperature", "cold"),
        }
    )


@router.get("/session/{session_id}/snapshot")
async def get_session_snapshot(session_id: str):
    """Read the local txt snapshot for a session."""
    snapshot = load_local_session_snapshot(session_id)
    raw_text = read_local_session_snapshot_text(session_id)
    if not snapshot and not raw_text:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return JSONResponse(
        {
            "session_id": session_id,
            "snapshot_path": get_local_snapshot_path(session_id),
            "updated_at": snapshot.get("updated_at"),
            "lead_profile": snapshot.get("lead_profile"),
            "session_context": snapshot.get("session_context"),
            "chat_history": snapshot.get("chat_history"),
            "raw_text": raw_text,
        }
    )


@router.post("/recommendations/refresh")
async def refresh_recommendations(session_id: str):
    """Generate a natural refresh recommendation message for the session."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")

    try:
        history = await load_chat_history(session_id)
        result = await _run_search_flow(
            session_id=session_id,
            user_text="Bạn giúp Sunny tổng hợp lại các sản phẩm phù hợp nhất với nhu cầu hiện tại nhé.",
            search_query="Bạn giúp Sunny tổng hợp lại các sản phẩm phù hợp nhất với nhu cầu hiện tại nhé.",
            history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh error: {e}")

    return JSONResponse(
        {
            "session_id": session_id,
            "response": result.get("final_response", ""),
            "sales_state": result.get("next_sales_state"),
        }
    )


@router.post("/followup/generate")
async def generate_followup(session_id: str):
    """Generate a natural follow-up message for a lead."""
    profile = await load_lead_profile(session_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Lead not found")

    try:
        history = await load_chat_history(session_id)
        result = await _run_search_flow(
            session_id=session_id,
            user_text="Sunny muốn follow-up nhẹ nhàng và gợi mở đúng điều bạn đang tò mò về sản phẩm.",
            search_query="Sunny muốn follow-up nhẹ nhàng và gợi mở đúng điều bạn đang tò mò về sản phẩm.",
            history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up error: {e}")

    return JSONResponse(
        {
            "session_id": session_id,
            "response": result.get("final_response", ""),
        }
    )


@router.post("/session/{session_id}/close")
async def close_session(session_id: str):
    """Export session data to txt and clear ephemeral session caches."""
    await ensure_sales_schema()
    profile = await load_lead_profile(session_id) or {}
    ctx = await load_session_context(session_id)
    history = await load_chat_history(session_id)

    if not profile and not history:
        raise HTTPException(status_code=404, detail="Session not found")

    export_path = export_session_to_txt(
        session_id=session_id,
        lead_profile=profile,
        session_context=ctx,
        chat_history=history,
    )

    await delete_chat_history(session_id)
    await delete_session_context(session_id)
    await delete_lead_profile_cache(session_id)

    # Vision/camera sync is disabled; skip face embedding removal.
    customer_id = str(profile.get("customer_id") or "").strip()
    if not customer_id and isinstance(ctx, dict):
        customer_id = str(ctx.get("customer_id") or "").strip()
    machine_b_deleted: bool = False

    return JSONResponse(
        {
            "session_id": session_id,
            "status": "closed",
            "export_path": export_path,
            "messages_exported": len(history),
            "machine_b_face_deleted": machine_b_deleted,
        }
    )


@router.post("/customer/{customer_id}/machine-b-face-removal")
async def notify_machine_b_after_customer_record_removed(
    customer_id: str,
    x_chat_history_purge_token: Optional[str] = Header(
        None, alias="X-Chat-History-Purge-Token"
    ),
):
    """Sau khi đã xóa / vô hiệu hóa khách trong DB (khóa `customer_id`), báo Máy B xóa embedding tương ứng.

    Gọi từ job/worker phía A ngay sau khi commit xóa bản ghi Postgres (hoặc tương đương).
    Cùng header bảo vệ với `purge-all`: `X-Chat-History-Purge-Token`.
    """
    _require_chat_purge_token(x_chat_history_purge_token)
    cid = (customer_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="customer_id is required")
    result = await notify_machine_b_customer_removed(cid)
    if result is None:
        raise HTTPException(
            status_code=502,
            detail="Could not notify Machine B (check MACHINE_B_BASE_URL, network, VISION_FACE_DELETE_TOKEN)",
        )
    return JSONResponse({"ok": True, "customer_id": cid, "machine_b": result})


@router.post("/chat-history/purge-all")
async def purge_all_chat_history_endpoint(
    x_chat_history_purge_token: Optional[str] = Header(
        None, alias="X-Chat-History-Purge-Token"
    ),
    notify_machine_b: bool = Query(
        True,
        description=(
            "True: trước khi xóa Redis, gom customer_id từ session_context + lead_profile_cache; "
            "sau purge gọi Máy B DELETE /v1/face/{id} cho từng id (cần MACHINE_B_BASE_URL + VISION_FACE_DELETE_TOKEN). "
            "False: chỉ xóa cache/snapshot, không gọi B."
        ),
    ),
):
    """Xóa sạch phiên sales: Redis, Postgres (`lead_profiles` + `customer_sessions` và CASCADE),
    toàn bộ file trong `live_snapshots/*.txt` (không chỉ xóa chat trong file).

    Gộp với nhu cầu \"quên phiên\": không chỉ xóa tin nhắn mà xóa luôn context/lead cache Redis
    để lần sau không đọc nhầm `customer_id` cũ.

    Response gồm `postgres_*_deleted`, `redis_session_context_keys_deleted`, `live_snapshot_files_deleted`,
    `customer_ids_collected_before_purge`, `session_ids_seen_in_redis_before_purge`, và `machine_b`.

    Tắt gọi B toàn cục: `.env` `PURGE_ALL_NOTIFY_MACHINE_B=false` (ghi đè ý định gọi API).

    Cần CHAT_HISTORY_PURGE_TOKEN; gửi header X-Chat-History-Purge-Token.
    """
    _require_chat_purge_token(x_chat_history_purge_token)
    stats = await purge_all_chat_history(notify_machine_b=notify_machine_b)
    return JSONResponse({"ok": True, **stats})


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get all chat messages (human and AI) for a specific session.
    Useful for frontend to poll or display session conversation history.
    """
    history = await load_chat_history(session_id)
    if not history:
        # Don't 404, just return empty list as session might be new
        return JSONResponse({"session_id": session_id, "history": []})

    formatted_history = []
    for msg in history:
        # history returned from store usually contains langchain objects or dicts
        # format according to your storage implementation pattern
        formatted_history.append({
            "role": msg.type if hasattr(msg, "type") else msg.get("role", "unknown"),
            "content": msg.content if hasattr(msg, "content") else msg.get("content", ""),
            "timestamp": msg.additional_kwargs.get("timestamp") if hasattr(msg, "additional_kwargs") else msg.get("timestamp")
        })

    return JSONResponse(
        {
            "session_id": session_id,
            "history": formatted_history
        }
    )


