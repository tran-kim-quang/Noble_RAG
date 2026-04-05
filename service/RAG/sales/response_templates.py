"""Deterministic response templates for scripted sales steps."""

import re
import unicodedata
from typing import Any, Dict, List

from utils.text import split_into_sentences


_SLOT_LABELS = {
    "family_member_count": "số người trong gia đình",
    "children_count": "số con nhỏ",
    "purpose": "mục đích mua để ở hay kinh doanh",
    "location_preference": "khu vực ưu tiên",
}


def _fold_vn(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    decomp = unicodedata.normalize("NFD", text)
    no_mark = "".join(ch for ch in decomp if unicodedata.category(ch) != "Mn")
    return no_mark.replace("đ", "d")


def _normalize_gender(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "unknown"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value)
        if n == 1:
            return "male"
        if n == 2:
            return "female"
        return "unknown"
    folded = _fold_vn(value)
    if folded in {"1", "01"}:
        return "male"
    if folded in {"2", "02"}:
        return "female"
    if folded in {"male", "nam", "man", "m", "anh", "ong"}:
        return "male"
    if folded in {"female", "nu", "woman", "f", "chi", "co"}:
        return "female"
    return "unknown"


def _resolve_gender_from_state(state: Dict[str, Any]) -> str:
    lead = state.get("lead_profile") or {}
    session_context = state.get("session_context") or {}
    context_json = session_context.get("context_json") if isinstance(session_context, dict) else {}
    if not isinstance(context_json, dict):
        context_json = {}
    vision_context = context_json.get("vision_context") if isinstance(context_json, dict) else {}
    if not isinstance(vision_context, dict):
        vision_context = {}

    candidates = [
        vision_context.get("gender_guess"),
        vision_context.get("gender_estimate"),
        vision_context.get("gioi_tinh"),
        vision_context.get("gender"),
        vision_context.get("sex"),
        session_context.get("gender_estimate") if isinstance(session_context, dict) else None,
        session_context.get("gender_guess") if isinstance(session_context, dict) else None,
        session_context.get("gioi_tinh") if isinstance(session_context, dict) else None,
        lead.get("gender_estimate"),
        lead.get("gender_guess"),
        lead.get("gioi_tinh"),
    ]
    for item in candidates:
        gender = _normalize_gender(item)
        if gender != "unknown":
            return gender
    return "unknown"


def _customer_ref(state: Dict[str, Any], *, title_case: bool = True) -> str:
    gender = _resolve_gender_from_state(state)
    if gender == "male":
        return "Anh" if title_case else "anh"
    if gender == "female":
        return "Chị" if title_case else "chị"
    return "Anh/Chị" if title_case else "anh/chị"


def _apply_customer_pronoun(text: str, state: Dict[str, Any]) -> str:
    if not text:
        return text
    cap = _customer_ref(state, title_case=True)
    low = _customer_ref(state, title_case=False)
    patched = text
    replacements = {
        "Anh/Chị": cap,
        "Anh chị": cap,
        "Anh Chi": cap,
        "Anh/Chi": cap,
        "anh/chị": low,
        "anh chị": low,
        "anh chi": low,
        "anh/chi": low,
    }
    for old, new in replacements.items():
        patched = patched.replace(old, new)
    return patched


def _normalize_candidate_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip(" -\t#")).strip()


def _missing_slots_text(missing_slots: List[str]) -> str:
    labels = [_SLOT_LABELS.get(slot, slot) for slot in missing_slots]
    return ", ".join(labels)


def _extract_reason_candidates(content: str, project_name: str = "") -> List[str]:
    reasons: List[str] = []
    project_name_lower = project_name.strip().lower()
    for line in content.splitlines():
        clean = _normalize_candidate_line(line)
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith(("lý do", "lưu ý", "cần lưu ý", "dự án ")):
            continue
        if project_name_lower and clean.lower() == project_name_lower:
            continue
        reasons.append(clean.rstrip("."))
    if reasons:
        return reasons[:2]

    sentences = split_into_sentences(content)
    return [sentence.rstrip(".") for sentence in sentences[:2]]


def render_match_options_from_candidates(state: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    valid = [candidate for candidate in candidates if isinstance(candidate, dict)]
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in valid:
        name = str(candidate.get("project_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    if not deduped:
        return _apply_customer_pronoun(
            (
            "Dựa trên thông tin hiện tại, em chưa lọc được phương án thật sự rõ ràng để gửi Anh/Chị. "
            "Anh/Chị cho em thêm 1 tiêu chí ưu tiên nhất để em lọc sát hơn nhé."
            ),
            state,
        )

    if len(deduped) == 1:
        candidate = deduped[0]
        name = str(candidate.get("project_name") or "dự án phù hợp").strip()
        reasons = [str(reason).strip().rstrip(".") for reason in candidate.get("fit_reasons") or [] if str(reason).strip()]
        if not reasons:
            reasons = _extract_reason_candidates(str(candidate.get("content") or ""), project_name=name)
        reasons = reasons[:2]
        lines = [f"Với thông tin hiện tại, phương án em thấy phù hợp nhất là {name}."]
        if reasons:
            lines.append("Lý do em thấy phù hợp là:")
            for reason in reasons:
                lines.append(f"- {reason}")
        lines.append("Nếu Anh/Chị muốn, em sẽ đi sâu tiếp loại căn phù hợp nhất với nhu cầu của gia đình mình ạ.")
        return _apply_customer_pronoun("\n".join(lines).strip(), state)

    top_k = min(3, len(deduped))
    top_items = deduped[:top_k]
    lines = [f"Dựa trên nhu cầu hiện tại, em thấy có {top_k} phương án đáng cân nhắc để Anh/Chị tham khảo:"]
    for idx, candidate in enumerate(top_items, start=1):
        name = str(candidate.get("project_name") or f"Phương án {idx}").strip()
        reasons = [str(reason).strip().rstrip(".") for reason in candidate.get("fit_reasons") or [] if str(reason).strip()]
        if not reasons:
            reasons = _extract_reason_candidates(str(candidate.get("content") or ""), project_name=name)
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

    lines.append("Nếu Anh/Chị muốn, em sẽ bóc tách tiếp phương án nổi bật nhất hoặc so nhanh giữa các lựa chọn này để mình chốt hướng dễ hơn ạ.")
    return _apply_customer_pronoun("\n".join(lines).strip(), state)


def render_template_response(action: str, state: Dict[str, Any]) -> str:
    lead = state.get("lead_profile") or {}
    missing_slots = state.get("missing_slots") or []
    objection_type = state.get("objection_type") or "khac"

    if action == "ask_opening":
        return _apply_customer_pronoun(
            (
            "Chào Anh/Chị, em là tư vấn viên bất động sản của Noble. "
            "Để em hỗ trợ đúng nhu cầu, gia đình mình hiện có bao nhiêu người ạ?"
            ),
            state,
        )

    if action == "ask_family_size":
        return "Gia đình mình hiện có khoảng bao nhiêu người sẽ ở thường xuyên ạ?"

    if action == "ask_children":
        return "Gia đình mình hiện có bao nhiêu con nhỏ để em ưu tiên tiện ích phù hợp ạ?"

    if action == "ask_purpose":
        return _apply_customer_pronoun("Hiện tại Anh/Chị đang tìm sản phẩm để ở hay để kinh doanh/đầu tư ạ?", state)

    if action == "ask_location":
        return _apply_customer_pronoun("Anh/Chị đang ưu tiên khu vực nào để em lọc đúng dự án phù hợp ạ?", state)

    if action == "check_interest":
        return _apply_customer_pronoun(
            "Trong phương án em vừa tư vấn, Anh/Chị thấy hướng nào phù hợp hơn để em đi sâu tiếp ạ?",
            state,
        )

    if action == "handle_objection":
        if objection_type == "gia_cao":
            return _apply_customer_pronoun(
                (
                "Em hiểu băn khoăn của Anh/Chị về mức giá, vì với bất động sản cao cấp thì mình cần nhìn rất kỹ vào giá trị thực nhận và phương án thanh toán. "
                "Nếu Anh/Chị muốn, em sẽ gửi bảng giá kèm chính sách thanh toán để mình đối chiếu rõ xem mức giá có thực sự phù hợp hay không ạ."
                ),
                state,
            )
        if objection_type == "phap_ly":
            return _apply_customer_pronoun(
                (
                "Em hiểu pháp lý là phần mình cần chắc chắn trước khi đi tiếp. "
                "Em sẽ ưu tiên rà lại hồ sơ pháp lý và gửi Anh/Chị phần thông tin xác thực, rõ ràng nhất để mình yên tâm đánh giá ạ."
                ),
                state,
            )
        if objection_type == "vi_tri":
            return _apply_customer_pronoun(
                (
                "Em hiểu vị trí là yếu tố ảnh hưởng trực tiếp tới trải nghiệm sống và tính thanh khoản nên mình cần cân nhắc kỹ. "
                "Nếu Anh/Chị muốn, em sẽ đối chiếu lại đúng tiêu chí di chuyển và khu vực ưu tiên để xem dự án này có thật sự phù hợp không ạ."
                ),
                state,
            )
        if objection_type == "chua_du_tien":
            return _apply_customer_pronoun(
                (
                "Em hiểu mình cần cân đối dòng tiền thật an toàn trước khi quyết định. "
                "Em có thể giúp Anh/Chị rà lại phương án thanh toán và mức tài chính phù hợp hơn để mình xem có cửa nào dễ vào hơn không ạ."
                ),
                state,
            )
        return _apply_customer_pronoun(
            (
            "Em hiểu băn khoăn của Anh/Chị và đây là điều mình nên làm rõ trước khi đi tiếp. "
            "Nếu Anh/Chị đồng ý, em sẽ kiểm tra lại đúng phần thông tin liên quan và gửi mình câu trả lời ngắn gọn, rõ ràng nhất ạ."
            ),
            state,
        )

    if action == "soft_close":
        return _apply_customer_pronoun(
            "Nếu phù hợp, em có thể gửi thêm shortlist ngắn gọn hoặc thông tin chi tiết để Anh/Chị tham khảo tiếp ạ?",
            state,
        )

    if action == "followup_closeout":
        return _apply_customer_pronoun(
            (
            "Em cảm ơn Anh/Chị đã chia sẻ thông tin. "
            "Em sẽ tổng hợp phương án phù hợp nhất để mình tiện theo dõi, khi cần thêm gì Anh/Chị cứ nhắn em nhé."
            ),
            state,
        )

    if action == "redirect_out_of_scope":
        return _apply_customer_pronoun(
            (
            "Em hiện hỗ trợ tư vấn các dự án bất động sản của Noble. "
            "Nếu Anh/Chị muốn, mình quay lại nhu cầu mua để ở hoặc kinh doanh để em hỗ trợ đúng hơn nhé."
            ),
            state,
        )

    if missing_slots:
        return _apply_customer_pronoun(
            (
            "Để em tư vấn sát hơn, Anh/Chị chia sẻ thêm giúp em "
            f"{_missing_slots_text(missing_slots)} nhé?"
            ),
            state,
        )

    if lead.get("purpose") == "mua_o":
        return "Em đã nắm được nhu cầu cơ bản của gia đình mình rồi. Em sẽ tư vấn ngay dự án phù hợp nhất ạ."

    return _apply_customer_pronoun(
        "Em đã nắm được nhu cầu cơ bản rồi. Em sẽ tư vấn phương án phù hợp nhất cho Anh/Chị ngay ạ.",
        state,
    )
