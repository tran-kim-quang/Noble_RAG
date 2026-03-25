"""Deterministic response templates for scripted sales steps."""

import re
from typing import Any, Dict, List

from utils.text import split_into_sentences


_SLOT_LABELS = {
    "family_member_count": "số người trong gia đình",
    "children_count": "số con nhỏ",
    "purpose": "mục đích mua để ở hay kinh doanh",
    "location_preference": "khu vực ưu tiên",
}


def _missing_slots_text(missing_slots: List[str]) -> str:
    labels = [_SLOT_LABELS.get(slot, slot) for slot in missing_slots]
    return ", ".join(labels)


def _extract_reason_candidates(content: str) -> List[str]:
    reasons: List[str] = []
    for line in content.splitlines():
        clean = line.strip(" -\t")
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith(("lý do", "lưu ý", "cần lưu ý")):
            continue
        reasons.append(clean.rstrip("."))
    if reasons:
        return reasons[:2]

    sentences = split_into_sentences(content)
    return [sentence.rstrip(".") for sentence in sentences[:2]]


def render_match_options_from_candidates(state: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    top2 = [candidate for candidate in candidates if isinstance(candidate, dict)][:2]
    if len(top2) < 2:
        return (
            "Dựa trên thông tin hiện tại, em chưa lọc được 2 phương án thật sự rõ ràng để gửi Anh/Chị. "
            "Anh/Chị cho em thêm 1 tiêu chí ưu tiên nhất để em lọc sát hơn nhé."
        )

    lines = ["Dựa trên nhu cầu hiện tại, em thấy có 2 phương án phù hợp để Anh/Chị tham khảo:"]
    for idx, candidate in enumerate(top2, start=1):
        name = str(candidate.get("project_name") or f"Phương án {idx}").strip()
        reasons = [str(reason).strip().rstrip(".") for reason in candidate.get("fit_reasons") or [] if str(reason).strip()]
        if not reasons:
            reasons = _extract_reason_candidates(str(candidate.get("content") or ""))
        reasons = reasons[:2]
        risk_notes = [str(note).strip() for note in candidate.get("risk_notes") or [] if str(note).strip()]

        lines.append(f"{idx}. {name}")
        if reasons:
            lines.append("Lý do phù hợp:")
            for reason in reasons:
                lines.append(f"- {reason}")
        if risk_notes:
            lines.append(f"Lưu ý: {risk_notes[0]}")
        lines.append("")

    lines.append("Nếu Anh/Chị muốn, em sẽ đi sâu hơn vào phương án phù hợp hơn với gia đình mình ạ.")
    return "\n".join(lines).strip()


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
