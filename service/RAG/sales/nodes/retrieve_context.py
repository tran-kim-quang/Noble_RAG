"""Node 6: Retrieve knowledge from LightRAG based on current sales state."""

import logging
from typing import Any, Dict, List

from sales.graph_state import SalesAgentState
from rag.retriever import query_rag

log = logging.getLogger("rag-service")

_STATE_QUERY_TEMPLATES: Dict[str, str] = {
    "product_matching": (
        "Dự án bất động sản phù hợp với: "
        "mục đích {purpose}, ngân sách {budget}, khu vực {location}, loại hình {property_type}. "
        "Cung cấp thông tin về dự án, mức giá, pháp lý, chính sách thanh toán."
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
        purpose=lead.get("purpose") or "không xác định",
        budget=lead.get("budget_text") or "không xác định",
        location=", ".join(lead.get("location_preference") or []) or "linh hoạt",
        property_type=lead.get("property_type") or "không xác định",
        objection_type=state.get("objection_type") or "chung",
        user_text=state.get("user_text") or "",
    )
    return query


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
    return {"retrieved_context": context_items}
