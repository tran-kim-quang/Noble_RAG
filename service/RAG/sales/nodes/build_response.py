"""Node 7: Build and generate the response using LLM."""

import logging
import re
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from sales.prompt_builder import build_prompt
from sales.response_templates import render_match_options_from_candidates
from core.dependencies import llm_model_func
from utils.text import normalize_whitespace, split_into_sentences

log = logging.getLogger("rag-service")

_GROUNDED_STATES = {"project_qa", "product_matching", "comparison", "objection_handling", "closing_next_step"}
_DISCOVERY_STATES = {"greeting", "need_discovery"}


def _has_retrieved_context(state: SalesAgentState) -> bool:
    context = state.get("retrieved_context") or []
    return any(
        (item.get("content") or item.get("text") or "").strip()
        for item in context
        if isinstance(item, dict)
    )


def _question_count(text: str) -> int:
    return text.count("?")


def _enforce_short_form(text: str, max_sentences: int) -> str:
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


def _discovery_max_sentences(state: SalesAgentState) -> int:
    return 3 if _has_retrieved_context(state) else 2


def _sanitize_project_qa_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    clean = re.sub(r"(?<=\d)\.\s+(?=\d{3}\b)", ".", clean)
    clean = clean.replace("**", "")
    clean = clean.replace("__", "")
    clean = re.sub(r"\*\s+", "", clean)
    clean = re.sub(r"\s+\)", ")", clean)
    clean = re.sub(r"\(\s+", "(", clean)
    return normalize_whitespace(clean)


def _fallback_without_context(state: SalesAgentState) -> str:
    missing_slots: List[str] = state.get("missing_slots") or []
    next_state = state.get("next_sales_state") or "need_discovery"
    if next_state == "closing_next_step":
        return (
            "Em chưa có đủ dữ liệu từ kho dự án để chốt bước tiếp theo thật chính xác. "
            "Nếu Anh/Chị muốn, em sẽ kiểm tra lại thông tin phù hợp rồi quay lại ngay."
        )
    if next_state == "objection_handling":
        return (
            "Em chưa có đủ dữ liệu từ kho dự án để phản hồi chắc chắn cho băn khoăn này. "
            "Anh/Chị cho em kiểm tra lại thông tin chính xác rồi em phản hồi ngay nhé."
        )
    if next_state == "comparison":
        return (
            "Em chưa có đủ dữ liệu đã retrieve để so sánh chính xác lúc này. "
            "Anh/Chị cho em kiểm tra lại thông tin dự án rồi em so sánh ngắn gọn, đúng trọng tâm giúp mình nhé."
        )
    if missing_slots:
        return (
            "Em chưa có đủ dữ liệu từ kho dự án để đề xuất chính xác. "
            f"Anh/Chị chia sẻ thêm giúp em các mục còn thiếu: {', '.join(missing_slots)} nhé?"
        )
    return (
        "Em chưa có đủ dữ liệu từ kho dự án để đề xuất chính xác lúc này. "
        "Anh/Chị cho em kiểm tra thêm rồi em gửi lại phương án phù hợp nhất nhé."
    )


async def _repair_response(state: SalesAgentState, invalid_response: str) -> str:
    next_state = state.get("next_sales_state") or "need_discovery"
    missing_slots = ", ".join(state.get("missing_slots") or []) or "không có"
    has_context = _has_retrieved_context(state)
    repair_prompt = (
        "Hãy viết lại câu trả lời sales dưới đây để tuân thủ đúng policy.\n\n"
        f"STATE: {next_state}\n"
        f"MISSING_SLOTS: {missing_slots}\n"
        f"HAS_CONTEXT: {'yes' if has_context else 'no'}\n"
        "RULES:\n"
        "- Nếu state là greeting: tối đa 2 câu, chỉ 1 câu hỏi.\n"
        "- Nếu state là need_discovery và HAS_CONTEXT=no: tối đa 2 câu, chỉ 1 câu hỏi.\n"
        "- Nếu state là need_discovery và HAS_CONTEXT=yes: tối đa 3 câu; 1-2 câu đầu tóm tắt định hướng dựa trên context, câu cuối cùng là tối đa 1 câu hỏi.\n"
        "- Chỉ hỏi về các slot còn thiếu.\n"
        "- Không nêu tên dự án hoặc địa danh ví dụ nếu chưa có context.\n"
        "- Nếu khách chưa rõ nhu cầu ở, có thể gợi ý tiêu chí sống trung lập dựa trên gia đình, con cái, mục đích mua.\n"
        "- Không ép khách sang một loại hình sản phẩm nếu khách chưa tự nêu.\n"
        "- Giữ giọng điệu tự nhiên, xưng em, gọi Anh/Chị.\n\n"
        f"Câu trả lời cần viết lại:\n{invalid_response}"
    )
    repaired = await llm_model_func(repair_prompt, history_messages=[])
    return _enforce_short_form(str(repaired).strip(), max_sentences=3 if has_context else 2)


async def build_response(state: SalesAgentState) -> Dict[str, Any]:
    next_state = state.get("next_sales_state") or "need_discovery"
    response_action = state.get("response_action") or ""
    if response_action == "project_qa" and state.get("project_qa_blocked"):
        project_prompt = (
            "Để em trả lời chính xác về số lượng căn, phân khu, pháp lý hay tiện ích, "
            "Anh/Chị cho em xin đúng tên dự án Noble mình đang quan tâm nhé?"
        )
        return {"draft_response": project_prompt, "final_response": project_prompt}
    if next_state in _GROUNDED_STATES and not _has_retrieved_context(state):
        fallback = _fallback_without_context(state)
        return {"draft_response": fallback, "final_response": fallback}
    if response_action == "match_options":
        deterministic = render_match_options_from_candidates(
            state,
            state.get("retrieved_candidates") or [],
        )
        return {"draft_response": deterministic, "final_response": deterministic}

    prompt = build_prompt(state)
    history = state.get("chat_history") or []

    try:
        raw = await llm_model_func(prompt, history_messages=history)
        response_text = str(raw).strip()
        if next_state in _DISCOVERY_STATES:
            response_text = _enforce_short_form(response_text, max_sentences=_discovery_max_sentences(state))
            if _question_count(response_text) > 1:
                response_text = await _repair_response(state, response_text)
        if response_action == "project_qa":
            response_text = _sanitize_project_qa_text(response_text)
        if response_action in {"match_options", "explain_option_detail"}:
            if _question_count(response_text) > 0:
                response_text = _enforce_short_form(response_text, max_sentences=4)
        log.info(
            "build_response: state=%s action=%s response_len=%d",
            next_state,
            response_action,
            len(response_text),
        )
        return {"draft_response": response_text, "final_response": response_text}
    except Exception as e:
        log.error("build_response error: %s", e)
        fallback = "Xin lỗi, em gặp lỗi khi xử lý. Anh/Chị thử lại giúp em nhé."
        return {"draft_response": fallback, "final_response": fallback, "errors": [str(e)]}
