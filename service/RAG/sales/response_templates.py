"""Deterministic response templates for scripted sales steps."""

from typing import Any, Dict, List


_SLOT_LABELS = {
    "family_member_count": "số người trong gia đình",
    "children_count": "số con nhỏ",
    "purpose": "mục đích mua để ở hay kinh doanh",
    "location_preference": "khu vực ưu tiên",
}


def _missing_slots_text(missing_slots: List[str]) -> str:
    labels = [_SLOT_LABELS.get(slot, slot) for slot in missing_slots]
    return ", ".join(labels)


def render_template_response(action: str, state: Dict[str, Any]) -> str:
    lead = state.get("lead_profile") or {}
    missing_slots = state.get("missing_slots") or []

    if action == "ask_opening":
        return (
            "Chào Anh/Chị, em là tư vấn viên bất động sản của Noble. "
            "Để em hỗ trợ đúng nhu cầu, gia đình mình hiện có bao nhiêu người ạ?"
        )

    if action == "ask_family_size":
        return "Gia đình mình hiện có khoảng bao nhiêu người sẽ ở thường xuyên ạ?"

    if action == "ask_children":
        return "Gia đình mình hiện có bao nhiêu con nhỏ để em ưu tiên tiện ích phù hợp ạ?"

    if action == "ask_purpose":
        return "Hiện tại Anh/Chị đang tìm sản phẩm để ở hay để kinh doanh/đầu tư ạ?"

    if action == "ask_location":
        return "Anh/Chị đang ưu tiên khu vực nào để em lọc đúng dự án phù hợp ạ?"

    if action == "check_interest":
        return "Trong phương án em vừa tư vấn, Anh/Chị thấy hướng nào phù hợp hơn để em đi sâu tiếp ạ?"

    if action == "soft_close":
        return "Nếu phù hợp, em có thể gửi thêm shortlist ngắn gọn hoặc thông tin chi tiết để Anh/Chị tham khảo tiếp ạ?"

    if action == "followup_closeout":
        return (
            "Em cảm ơn Anh/Chị đã chia sẻ thông tin. "
            "Em sẽ tổng hợp phương án phù hợp nhất để mình tiện theo dõi, khi cần thêm gì Anh/Chị cứ nhắn em nhé."
        )

    if action == "redirect_out_of_scope":
        return (
            "Em hiện hỗ trợ tư vấn các dự án bất động sản của Noble. "
            "Nếu Anh/Chị muốn, mình quay lại nhu cầu mua để ở hoặc kinh doanh để em hỗ trợ đúng hơn nhé."
        )

    if missing_slots:
        return (
            "Để em tư vấn sát hơn, Anh/Chị chia sẻ thêm giúp em "
            f"{_missing_slots_text(missing_slots)} nhé?"
        )

    if lead.get("purpose") == "mua_o":
        return "Em đã nắm được nhu cầu cơ bản của gia đình mình rồi. Em sẽ tư vấn ngay dự án phù hợp nhất ạ."

    return "Em đã nắm được nhu cầu cơ bản rồi. Em sẽ tư vấn phương án phù hợp nhất cho Anh/Chị ngay ạ."
