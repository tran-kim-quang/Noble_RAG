"""Node 6: Retrieve knowledge from LightRAG based on current sales state."""

import logging
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from rag.retriever import query_rag

log = logging.getLogger("rag-service")

_STATE_QUERY_TEMPLATES: Dict[str, str] = {
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
    query = _build_retrieval_query(state)
    if not query:
        log.info("retrieve_context: no query needed for state=%s", state.get("next_sales_state"))
        return {"retrieved_context": []}

    log.info("retrieve_context: query='%s...'", query[:80])
    raw_answer = await query_rag(
        query,
        top_k=5,
        history=state.get("chat_history") or [],
    )

    context_items: List[Dict[str, Any]] = []
    if raw_answer and raw_answer.strip():
        context_items.append({"content": raw_answer, "source": "lightrag"})

    log.info("retrieve_context: got %d items", len(context_items))
    return {
        "retrieved_context": context_items,
        "has_retrieved_context": bool(context_items),
    }
