"""Sales advisor tool for project product recommendation from internal KB."""

import asyncio
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from core.config import get_settings
from core.dependencies import llm_model_func
from rag.retriever import build_rag_fact_constraints


log = logging.getLogger("rag-service")
_SETTINGS = get_settings()
_SALES_ADVISOR_LLM_TIMEOUT_SEC = max(
    5.0,
    float((os.getenv("SALES_ADVISOR_LLM_TIMEOUT_SEC") or "").strip() or _SETTINGS.llm_request_timeout_max_sec),
)


_SALES_ADVISOR_HINTS = (
    "ngan sach",
    "tai chinh",
    "du toan",
    "toi co",
    "nen mua",
    "mua loai nao",
    "goi y san pham",
    "goi y",
    "phu hop",
    "de o",
    "dau tu",
    "dong tien",
    "cho thue",
    "tang gia",
)

_INVESTMENT_HINTS = ("dau tu", "kinh doanh", "sinh loi", "dong tien", "cho thue", "tang gia")
_INVESTMENT_GOAL_HINTS = ("cho thue", "dong tien", "tang gia", "giu tai san", "an toan")
_LIVING_HINTS = (
    "de o",
    "o thuc",
    "gia dinh",
    "tien ich",
    "moi truong",
    "yen tinh",
    "truong hoc",
    "di chuyen",
    "an ninh",
    "khong gian song",
)


def _fold_vn(text: str) -> str:
    raw = (text or "").strip().lower()
    folded = unicodedata.normalize("NFD", raw)
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return folded.replace("đ", "d")


def _to_float(num_text: str) -> Optional[float]:
    text = (num_text or "").strip().replace(",", ".")
    try:
        return float(text)
    except Exception:
        return None


def _extract_budget_billion(user_text: str) -> Optional[Tuple[float, float]]:
    folded = _fold_vn(user_text)
    if not folded:
        return None

    range_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:-|den|toi)\s*(\d+(?:[.,]\d+)?)\s*(ty|trieu)",
        folded,
    )
    if range_match:
        left = _to_float(range_match.group(1))
        right = _to_float(range_match.group(2))
        unit = range_match.group(3)
        if left is None or right is None:
            return None
        if unit == "trieu":
            left, right = left / 1000.0, right / 1000.0
        return (min(left, right), max(left, right))

    single_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(ty|trieu)", folded)
    if single_match:
        value = _to_float(single_match.group(1))
        unit = single_match.group(2)
        if value is None:
            return None
        if unit == "trieu":
            value = value / 1000.0
        return (value, value)

    return None


def is_sales_advisor_query(user_text: str) -> bool:
    folded = _fold_vn(user_text)
    if not folded:
        return False
    if _extract_budget_billion(user_text) is not None:
        return True
    return any(hint in folded for hint in _SALES_ADVISOR_HINTS)


def _profile_summary(profile: Optional[Dict[str, Any]]) -> str:
    data = profile or {}
    rows: List[str] = []
    for key in ("name", "housing_preference", "budget_text", "budget_min", "budget_max"):
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            rows.append(f"- {key}: {text}")
    return "\n".join(rows) if rows else "- none"


def _strip_question_sentences(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", raw)
    kept = [p.strip() for p in parts if p.strip() and "?" not in p]
    return " ".join(kept).strip()


def _clean_evidence_line(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("**", "")
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    text = re.sub(r"^#+\s*", "", text).strip()
    text = re.sub(r"^\d+\.\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if "Luu y:" in _fold_vn(text):
        text = text.split("Luu y:", 1)[0].strip()
    if len(text) > 160 and ". " in text:
        text = text.split(". ", 1)[0].strip() + "."
    return text


def _extract_evidence_lines(fact_context: str, max_items: int = 8) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for raw in str(fact_context or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        value = _clean_evidence_line(line[2:].strip())
        if not value:
            continue
        lowered = value.lower()
        if "hien chua truy xuat" in lowered:
            continue
        if "when answering factual details" in lowered:
            continue
        if "do not invent" in lowered:
            continue
        if lowered.startswith("evidence lines"):
            continue
        if len(value) < 18:
            continue
        if value in seen:
            continue
        seen.add(value)
        items.append(value)
        if len(items) >= max_items:
            break
    return items


def _select_relevant_evidence_lines(
    evidence_lines: List[str],
    *,
    folded_query: str,
    living_intent: bool,
    investment_intent: bool,
    max_items: int = 3,
) -> List[str]:
    query_tokens = set(re.findall(r"[a-z0-9]+", folded_query))
    living_keys = {"tien", "ich", "moi", "truong", "yen", "tinh", "ninh", "truong", "hoc", "cong", "vien"}
    invest_keys = {"gia", "thanh", "toan", "chiet", "khau", "uu", "dai", "dau", "tu", "kinh", "doanh", "cho", "thue"}
    banned_meta = {
        "ten thuong mai",
        "ten phap ly",
        "tagline",
        "slogan",
        "quy mo",
        "dinh vi",
        "tong dien tich",
    }

    scored: List[tuple[int, str]] = []
    for line in evidence_lines:
        folded_line = _fold_vn(line)
        if any(meta in folded_line for meta in banned_meta):
            continue
        line_tokens = set(re.findall(r"[a-z0-9]+", folded_line))
        score = len(query_tokens.intersection(line_tokens))
        living_match = len(living_keys.intersection(line_tokens))
        invest_match = len(invest_keys.intersection(line_tokens))
        if living_intent and living_match == 0:
            continue
        if investment_intent and invest_match == 0:
            continue
        if living_intent:
            score += living_match
        if investment_intent:
            score += invest_match
        scored.append((score, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [line for score, line in scored if score > 0][:max_items]
    if selected and scored and scored[0][0] >= 2:
        return selected
    return []


def _project_grounding_sentence(evidence_lines: List[str]) -> str:
    def _shorten(value: str, max_len: int = 120) -> str:
        text = str(value or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    if not evidence_lines:
        return "Theo dữ liệu nội bộ của Noble Palace Tây Thăng Long, hiện mới truy xuất được một phần thông tin sản phẩm và chính sách."
    compact = [_shorten(line, 120) for line in evidence_lines[:2]]
    return "Theo dữ liệu nội bộ của Noble Palace Tây Thăng Long: " + "; ".join(compact) + "."


async def _llm_sales_advice_from_retrieval(
    *,
    query: str,
    history: List[Dict[str, Any]],
    profile_summary: str,
    budget_note: str,
    investment_intent: bool,
    has_investment_goal: bool,
    living_intent: bool,
    evidence_lines: List[str],
    project_grounding: str,
) -> str:
    evidence_block = "\n".join(f"- {line}" for line in evidence_lines) if evidence_lines else f"- {project_grounding}"
    intent_mode = "living" if living_intent else ("investment" if investment_intent else "generic")

    prompt = f"""
Bạn là Sunny, sales advisor của Noble Palace Tây Thăng Long.
NHIỆM VỤ: đưa phương án cụ thể dựa trên evidence retrieval nội bộ.

Quy tắc bắt buộc:
- Chỉ dùng thông tin trong Retrieval Evidence.
- Trả lời bằng tiếng Việt có dấu, gọn rõ, tối đa 6 câu.
- Không hỏi ngược người dùng.
- Bắt buộc nêu rõ "Noble Palace Tây Thăng Long" ít nhất 1 lần.
- Bắt buộc đưa các phương án hành động cụ thể dạng "Phương án 1/2/3".
- Không nói chung chung.

Input:
- User query: {query}
- Intent mode: {intent_mode}
- Budget signal: {budget_note}
- Investment goal known: {"yes" if has_investment_goal else "no"}
- Lead profile:
{profile_summary}

Retrieval Evidence:
{evidence_block}
""".strip()

    try:
        raw = await asyncio.wait_for(
            llm_model_func(prompt, history_messages=history, enable_cot=False),
            timeout=_SALES_ADVISOR_LLM_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        log.warning("sales_advisor llm timeout after %.1fs", _SALES_ADVISOR_LLM_TIMEOUT_SEC)
        return ""
    except Exception:
        return ""

    text = _strip_question_sentences(str(raw or "").strip())
    if not text:
        return ""
    folded = _fold_vn(text)
    if "noble palace tay thang long" not in folded:
        text = f"{project_grounding} {text}".strip()
    return text


async def run_sales_advisor_tool(
    *,
    user_query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    lead_profile: Optional[Dict[str, Any]] = None,
) -> str:
    history = history or []
    query = (user_query or "").strip()
    if not query:
        return ""

    folded = _fold_vn(query)
    budget_range = _extract_budget_billion(query)
    investment_intent = any(h in folded for h in _INVESTMENT_HINTS)
    has_investment_goal = any(h in folded for h in _INVESTMENT_GOAL_HINTS)
    living_intent = any(h in folded for h in _LIVING_HINTS)

    budget_note = "khong ro"
    if budget_range is not None:
        budget_note = f"{budget_range[0]:.2f}-{budget_range[1]:.2f} ty VND"

    fact_context = await build_rag_fact_constraints(query, top_k=12)
    if not fact_context.strip():
        fact_context = "Evidence constraints (must follow):\n- Hien chua truy xuat du bang chung noi bo manh cho cau hoi nay."
    evidence_all = _extract_evidence_lines(fact_context, max_items=10)
    evidence_lines = _select_relevant_evidence_lines(
        evidence_all,
        folded_query=folded,
        living_intent=living_intent,
        investment_intent=investment_intent,
        max_items=3,
    )
    if living_intent:
        project_grounding = (
            "Theo dữ liệu nội bộ của Noble Palace Tây Thăng Long, dự án có các dòng sản phẩm thấp tầng như nhà liền kề và shophouse, "
            "đồng thời có chính sách thanh toán và ưu đãi theo từng giai đoạn mở bán."
        )
    elif investment_intent:
        project_grounding = (
            "Theo dữ liệu nội bộ của Noble Palace Tây Thăng Long, dự án có sản phẩm thấp tầng có khả năng khai thác kinh doanh "
            "và được hỗ trợ bằng các chính sách ưu đãi, thanh toán, tài chính nội bộ."
        )
    else:
        project_grounding = _project_grounding_sentence(evidence_lines)

    llm_text = await _llm_sales_advice_from_retrieval(
        query=query,
        history=history,
        profile_summary=_profile_summary(lead_profile),
        budget_note=budget_note,
        investment_intent=investment_intent,
        has_investment_goal=has_investment_goal,
        living_intent=living_intent,
        evidence_lines=evidence_lines,
        project_grounding=project_grounding,
    )
    if llm_text:
        return llm_text

    if investment_intent and not has_investment_goal:
        return (
            f"{project_grounding} "
            "Sunny đề xuất 3 phương án đầu tư để triển khai ngay. "
            "Phương án 1 ưu tiên dòng tiền: tập trung sản phẩm có khả năng khai thác kinh doanh/cho thuê sớm trong khu đô thị. "
            "Phương án 2 ưu tiên tăng giá: ưu tiên vị trí kết nối hạ tầng và cụm tiện ích để tăng biên độ giá trị dài hạn. "
            "Phương án 3 ưu tiên an toàn: ưu tiên sản phẩm diện tích vừa, thanh khoản tốt, bám sát chính sách hỗ trợ tài chính nội bộ."
        )
    if living_intent:
        return (
            f"{project_grounding} "
            "Sunny đề xuất 3 phương án ở thực theo nhu cầu sống gia đình. "
            "Phương án 1 cân bằng tài chính: ưu tiên nhà liền kề diện tích vừa để tối ưu tổng chi phí sở hữu. "
            "Phương án 2 ưu tiên trải nghiệm sống: ưu tiên sản phẩm gần cụm tiện ích nội khu và trục di chuyển chính. "
            "Phương án 3 ưu tiên yên tĩnh: ưu tiên vị trí giảm lưu lượng, tách khỏi trục thương mại để bảo toàn chất lượng sống lâu dài."
        )
    if budget_range is not None:
        return (
            f"{project_grounding} "
            f"Với ngân sách {budget_range[0]:.2f}-{budget_range[1]:.2f} tỷ, Sunny đưa 3 phương án hợp lý. "
            "Hướng 1: sản phẩm diện tích vừa để tối ưu tổng tiền vào và thanh khoản. "
            "Hướng 2: sản phẩm gần trục tiện ích để cân bằng ở thực và tiềm năng tăng giá. "
            "Hướng 3: sản phẩm có khả năng khai thác kinh doanh/cho thuê nếu ưu tiên dòng tiền."
        )
    return (
        f"{project_grounding} "
        "Sunny đề xuất ngay 2 hướng: hướng ở thực ưu tiên chất lượng sống và hướng đầu tư ưu tiên thanh khoản, "
        "sau đó lọc tiếp theo khung giá và chính sách nội bộ để chốt phương án tối ưu."
    )
