"""Fast deterministic acknowledgement for grounded lane."""

from typing import Dict

from sales.graph_state import SalesAgentState


_ACK_BY_STATE = {
    "project_qa": "Để em kiểm tra đúng dữ liệu dự án này và trả lời ngắn gọn, chính xác cho Anh/Chị nhé.",
    "comparison": "Em lọc nhanh 2 phương án rồi so sánh đúng trọng tâm cho Anh/Chị ngay nhé.",
    "objection_handling": "Em hiểu băn khoăn của Anh/Chị, để em đối chiếu lại thông tin phù hợp nhất nhé.",
    "closing_next_step": "Em kiểm tra nhanh phương án và bước tiếp theo phù hợp nhất cho Anh/Chị ngay nhé.",
}


def fast_ack_response(state: SalesAgentState) -> Dict[str, str]:
    next_state = state.get("next_sales_state") or "project_qa"
    ack = _ACK_BY_STATE.get(
        next_state,
        "Để em kiểm tra đúng dữ liệu và phản hồi ngắn gọn cho Anh/Chị ngay nhé.",
    )
    return {"fast_ack_response": ack}
