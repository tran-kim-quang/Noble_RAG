"""Node 6: Retrieve knowledge from LightRAG based on current sales state."""

import logging
import re
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from rag.retriever import query_rag
from utils.text import split_into_sentences

log = logging.getLogger("rag-service")

_PROJECT_NAME_PATTERN = re.compile(r"(Noble[^\n,.;:!?()]*)", re.IGNORECASE)

_STATE_QUERY_TEMPLATES: Dict[str, str] = {
    "need_discovery": (
        "Dựa trên kho tri thức dự án Noble, trích ra các dữ kiện và định hướng tư vấn liên quan trực tiếp đến nhu cầu hiện có của khách. "
        "Không chốt dự án cụ thể nếu còn thiếu tiêu chí cốt lõi. "
        "Ưu tiên thông tin thực tế về tiện ích, phong cách sống, nhóm sản phẩm, vị trí, kết nối, hoặc điểm phù hợp với nhu cầu sau: "
        "mục đích {purpose}, loại hình {property_type}, khu vực {location}, gia đình {family_size} người, "
        "có {children_count} con nhỏ. "
        "Nếu dữ liệu chưa đủ để kết luận, hãy nêu 2-4 dữ kiện/định hướng hữu ích nhất để sales dùng trả lời ngắn gọn rồi hỏi tiếp. "
        "Câu khách: {user_text}"
    ),
    "project_qa": (
        "Trả lời trực tiếp câu hỏi sau về dự án/sản phẩm/chính sách của Noble bằng dữ kiện có trong kho tri thức. "
        "Nếu dữ liệu thiếu thì nêu rõ phần chưa đủ. Câu hỏi: {user_text}"
    ),
    "product_matching": (
        "Tư vấn dự án bất động sản Noble phù hợp với mục đích {purpose}, "
        "loại hình ưu tiên {property_type}, ngân sách tham khảo {budget}, "
        "ưu tiên gần khu vực {location}, gia đình {family_size} người, "
        "có {children_count} con nhỏ. {matching_guidance} "
        "Chỉ nêu phương án phù hợp với các tiêu chí này "
        "và các dữ kiện có trong kho tri thức."
    ),
    "comparison": (
        "So sánh các dự án bất động sản Noble về: giá, vị trí, pháp lý, tiến độ, "
        "chính sách thanh toán, tiềm năng cho thuê. Khách quan tâm: {user_text}"
    ),
    "objection_handling": (
        "Playbook xử lý phản đối bất động sản: {objection_type}. "
        "FAQ và chính sách liên quan. Khách nói: {user_text}"
    ),
    "closing_next_step": (
        "CTA playbook và bước tiếp theo phù hợp cho khách quan tâm: {user_text}. "
        "Shortlist, bảng giá, lịch xem dự án."
    ),
}


def _build_retrieval_query(state: Dict[str, Any]) -> str:
    next_state: str = state.get("next_sales_state") or "product_matching"
    template = _STATE_QUERY_TEMPLATES.get(next_state)
    if not template:
        return ""

    lead = state.get("lead_profile") or {}
    query = template.format(
        family_size=lead.get("family_member_count") or "không rõ",
        children_count=lead.get("children_count") or "không rõ",
        purpose=lead.get("purpose") or "không xác định",
        budget=lead.get("budget_text") or "không xác định",
        location=", ".join(lead.get("location_preference") or []) or "linh hoạt",
        property_type=lead.get("property_type") or "không nêu rõ",
        matching_guidance=_build_matching_guidance(lead),
        objection_type=state.get("objection_type") or "chung",
        user_text=state.get("user_text") or "",
    )
    return query


def _build_retrieval_query_lite(state: Dict[str, Any]) -> str:
    user_text = (state.get("user_text") or "").strip()
    next_state = state.get("next_sales_state") or "project_qa"
    project_name = state.get("resolved_project_name") or _resolve_project_name_from_user_context(state)

    if next_state == "project_qa" and project_name:
        return f"Thông tin chính xác và ngắn gọn về {project_name}: {user_text}"
    if next_state == "comparison":
        return f"So sánh ngắn gọn theo câu hỏi: {user_text}"
    if next_state == "objection_handling":
        return f"Xử lý băn khoăn phổ biến bất động sản cho câu: {user_text}"
    if next_state == "closing_next_step":
        return f"Gợi ý bước tiếp theo phù hợp cho khách với yêu cầu: {user_text}"
    return user_text


def _extract_project_name(text: str) -> str:
    match = _PROJECT_NAME_PATTERN.search(text or "")
    return match.group(1).strip() if match else ""


def _resolve_project_name_from_user_context(state: Dict[str, Any]) -> str:
    current = _extract_project_name(state.get("user_text") or "")
    if current:
        return current

    history = state.get("chat_history") or []
    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "user":
            continue
        found = _extract_project_name(str(message.get("content") or ""))
        if found:
            return found
    return ""


def _build_matching_guidance(lead: Dict[str, Any]) -> str:
    purpose = lead.get("purpose")
    property_type = lead.get("property_type")
    family_size = lead.get("family_member_count")
    children_count = lead.get("children_count")

    if property_type:
        return f"Ưu tiên đúng loại hình khách đã nêu: {property_type}."

    if purpose == "mua_o" and (family_size or children_count):
        return (
            "Khách đang mua để ở cho gia đình nhưng chưa chốt loại hình. "
            "Ưu tiên mô tả phương án theo tiêu chí sống phù hợp cho gia đình, "
            "không ép sang một loại hình cụ thể nếu dữ liệu chưa đủ."
        )

    return "Nếu khách chưa chốt loại hình, ưu tiên phương án phù hợp với nhu cầu thực tế thay vì áp sẵn một loại hình."


async def retrieve_context(state: SalesAgentState) -> Dict[str, Any]:
    if (state.get("response_action") or "") == "project_qa":
        resolved_project_name = _resolve_project_name_from_user_context(state)
        if not resolved_project_name:
            log.info("retrieve_context: project_qa blocked because project name is unresolved")
            return {
                "retrieved_candidates": [],
                "retrieved_context": [],
                "has_retrieved_context": False,
                "project_qa_blocked": True,
                "resolved_project_name": None,
            }
        state = dict(state)
        state["resolved_project_name"] = resolved_project_name

    retrieval_mode = (state.get("retrieval_mode") or "full").lower()
    if retrieval_mode == "lite":
        query = _build_retrieval_query_lite(state)
        top_k = 3
        history = (state.get("chat_history") or [])[-2:]
    else:
        query = _build_retrieval_query(state)
        top_k = 5
        history = state.get("chat_history") or []

    if not query:
        log.info("retrieve_context: no query needed for state=%s", state.get("next_sales_state"))
        return {"retrieved_context": []}

    log.info("retrieve_context: mode=%s query='%s...'", retrieval_mode, query[:80])
    raw_answer = await query_rag(
        query,
        top_k=top_k,
        history=history,
    )

    context_items: List[Dict[str, Any]] = []
    if raw_answer and raw_answer.strip():
        context_items.append({"content": raw_answer, "source": "lightrag"})
    candidates = _extract_candidates(raw_answer)

    log.info("retrieve_context: got %d items %d candidates", len(context_items), len(candidates))
    return {
        "retrieved_candidates": candidates,
        "retrieved_context": context_items,
        "has_retrieved_context": bool(context_items),
        "project_qa_blocked": False,
        "resolved_project_name": state.get("resolved_project_name"),
        "retrieval_mode": retrieval_mode,
    }


def _extract_candidates(raw_answer: str) -> List[Dict[str, Any]]:
    text = (raw_answer or "").strip()
    if not text:
        return []

    candidates = _extract_candidates_from_numbered_blocks(text)
    if candidates:
        return candidates[:2]
    return _extract_candidates_from_project_mentions(text)[:2]


def _extract_candidates_from_numbered_blocks(text: str) -> List[Dict[str, Any]]:
    matches = list(re.finditer(r"(?m)^\s*(\d+)[\.\)]\s+(.+)$", text))
    if not matches:
        return []

    candidates: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        first_line = match.group(2).strip()
        project_name = re.split(r"\s*[-:]\s*", first_line, maxsplit=1)[0].strip()
        fit_reasons: List[str] = []
        risk_notes: List[str] = []
        for line in block.splitlines()[1:]:
            clean = line.strip(" -\t")
            if not clean:
                continue
            lower = clean.lower()
            if lower.startswith(("lưu ý", "rủi ro", "hạn chế", "cân nhắc")):
                risk_notes.append(clean)
            elif not lower.startswith("lý do phù hợp"):
                fit_reasons.append(clean.rstrip("."))
        candidates.append(
            {
                "project_name": project_name or f"Phương án {idx + 1}",
                "fit_reasons": fit_reasons[:2],
                "risk_notes": risk_notes[:1],
                "content": block,
            }
        )
    return candidates


def _extract_candidates_from_project_mentions(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"(Noble[^\n,.;:]+)", re.IGNORECASE)
    names: List[str] = []
    for match in pattern.findall(text):
        clean = match.strip()
        if clean and clean.lower() not in {item.lower() for item in names}:
            names.append(clean)

    sentences = split_into_sentences(text)
    candidates: List[Dict[str, Any]] = []
    for name in names[:2]:
        related = [sentence for sentence in sentences if name.lower() in sentence.lower()]
        candidates.append(
            {
                "project_name": name,
                "fit_reasons": [sentence.rstrip(".") for sentence in related[:2]],
                "risk_notes": [],
                "content": " ".join(related[:3]).strip(),
            }
        )
    return candidates
