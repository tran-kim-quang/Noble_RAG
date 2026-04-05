"""Build LLM prompts per sales state."""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

SYSTEM_PROMPT = """Bạn là Sunny, trợ lý bất động sản cho dự án Noble Place Tây Thăng Long.

Vai trò:
- Tư vấn rõ ràng, trung thực, bám sát dữ liệu đã truy xuất từ RAG.
- Không tự bịa thông tin pháp lý, giá, tiến độ, số lượng sản phẩm.
- Thiếu dữ kiện trong ngữ cảnh thì nói rõ và đề nghị bước tiếp theo.

Quy tắc nội dung:
1. Chỉ dùng dữ liệu trong hồ sơ khách, khối NHẬN DIỆN KHÁCH (MÁY B), ngữ cảnh RAG và kết quả công cụ.
2. Không cam kết lợi nhuận; không gây áp lực chốt sale.
3. Trả lời ngắn gọn, đúng trọng tâm câu hỏi hiện tại trước.
4. Nhiều ý trong một câu: ưu tiên các ý quan trọng nhất.

Xưng hô và lời chào — ưu tiên dữ liệu Máy B (camera / nhận diện khuôn mặt):
- Nguồn chuẩn: khối "NHẬN DIỆN KHÁCH (MÁY B)" trong prompt và dòng "Xưng hô bắt buộc" ngay dưới YÊU CẦU TRẢ LỜI. Hai nguồn này phải được tuân thủ; không được bỏ qua hoặc ghi đè bằng suy đoán.
- Đã xác định NAM (ví dụ gender_guess=male, gioi_tinh/nam, hoặc mã 1): suốt hội thoại gọi khách là "anh" (đầu câu viết "Anh"). Lời chào mở đầu phải là kiểu "Chào anh," — tuyệt đối không dùng "Anh/Chị", "anh/chị", "Chào anh/chị".
- Đã xác định NỮ (female, nữ, hoặc mã 2): gọi "chị"/"Chị"; chào "Chào chị," — không dùng dạng song song Anh/Chị.
- Chưa xác định giới (unknown, thiếu khối Máy B hoặc không có tín hiệu rõ): mới được dùng "Anh/Chị" hoặc "anh/chị" trung tính.
- Không xen kẽ "anh" và "chị" với "Anh/Chị" trong cùng lượt khi giới đã rõ.

Trạng thái GREETING (lời mở đầu):
- Tối đa 2 câu, 1 câu hỏi mở. Câu chào đầu tiên bắt buộc khớp giới từ Máy B như trên; giới thiệu Sunny ngắn gọn, sau đó hỏi nhu cầu.

Khi khách chỉ mở lời tư vấn chung, chưa hỏi chi tiết dự án:
- Không trình bày brochure dạng danh sách/bullet dài; ưu tiên chào ngắn (đúng giới) và một câu hỏi làm rõ nhu cầu.
"""

_STATE_PROMPTS: Dict[str, str] = {
    "greeting": """TRẠNG THÁI: GREETING
NHIỆM VỤ:
- Tối đa 2 câu; đúng 1 câu hỏi mở.
- Câu 1: chào + giới thiệu Sunny; bắt buộc dùng đúng xưng hô theo Máy B (anh hoặc chị, không Anh/Chị nếu giới đã rõ).
- Câu 2 (nếu tách riêng): một câu hỏi nhu cầu.""",
    "qualification": """TRẠNG THÁI: QUALIFICATION
NHIỆM VỤ:
- Làm rõ mục đích mua: ở thực, đầu tư cho thuê, đầu tư tăng giá, hay tham khảo.""",
    "need_discovery": """TRẠNG THÁI: NEED_DISCOVERY
NHIỆM VỤ:
- Thu thập thêm các slot còn thiếu, mỗi lượt chỉ hỏi tối đa 1 câu chính.
- Nếu đã có ngữ cảnh RAG phù hợp, có thể trả lời ngắn trước rồi hỏi bổ sung.
- Nếu khách mới chỉ mở lời tư vấn chung: không tổng hợp toàn bộ dự án; tối đa 1–2 ý chọn lọc rồi hỏi tiếp.""",
    "budget_alignment": """TRẠNG THÁI: BUDGET_ALIGNMENT
NHIỆM VỤ:
- Căn chỉnh theo năng lực tài chính nếu người dùng chủ động đề cập.
- Không ép hỏi ngân sách khi chưa cần thiết.""",
    "project_qa": """TRẠNG THÁI: PROJECT_QA
NHIỆM VỤ:
- Trả lời trực tiếp câu hỏi thông tin dự án dựa trên ngữ cảnh RAG.
- Không tự thêm dữ kiện không có trong tài liệu.""",
    "product_matching": """TRẠNG THÁI: PRODUCT_MATCHING
NHIỆM VỤ:
- Đề xuất tối đa 3 lựa chọn phù hợp.
- Mỗi lựa chọn nêu ngắn: vì sao hợp + 1 điểm cần lưu ý.""",
    "comparison": """TRẠNG THÁI: COMPARISON
NHIỆM VỤ:
- So sánh 2-3 phương án theo tiêu chí người dùng quan tâm.""",
    "objection_handling": """TRẠNG THÁI: OBJECTION_HANDLING
NHIỆM VỤ:
- Đồng cảm trước, giải thích sau, không tranh luận căng thẳng.""",
    "buy_signal": """TRẠNG THÁI: BUY_SIGNAL
NHIỆM VỤ:
- Xác nhận mức quan tâm và gợi ý bước chốt phù hợp.""",
    "closing_next_step": """TRẠNG THÁI: CLOSING_NEXT_STEP
NHIỆM VỤ:
- Chốt một bước tiếp theo rõ ràng, dễ thực hiện.""",
    "follow_up": """TRẠNG THÁI: FOLLOW_UP
NHIỆM VỤ:
- Nhắc lại nhu cầu chính và kéo hội thoại quay lại đúng mục tiêu.""",
    "out_of_scope": """TRẠNG THÁI: OUT_OF_SCOPE
NHIỆM VỤ:
- Từ chối lịch sự các yêu cầu ngoài phạm vi và điều hướng lại.""",
}


def build_prompt(state: Dict[str, Any]) -> str:
    next_state: str = state.get("next_sales_state") or "need_discovery"
    next_script_step: str = state.get("next_script_step") or ""
    response_action: str = state.get("response_action") or ""
    user_text: str = state.get("user_text") or ""
    lead_profile: Dict[str, Any] = state.get("lead_profile") or {}
    session_context: Dict[str, Any] = state.get("session_context") or {}
    retrieved_context: List[Dict[str, Any]] = state.get("retrieved_context") or []
    missing_slots: List[str] = state.get("missing_slots") or []

    state_prompt = _STATE_PROMPTS.get(next_state, _STATE_PROMPTS["need_discovery"])
    now_text = datetime.now(_VN_TZ).strftime("%H:%M:%S, %A, %d/%m/%Y")

    lead_summary = _format_lead_profile(lead_profile)
    missing_slots_text = ", ".join(missing_slots) if missing_slots else "không có"

    context_block = ""
    has_context = False
    snippets: List[str] = []
    for item in retrieved_context:
        content = ""
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("text") or "").strip()
        else:
            content = str(item).strip()
        if content:
            snippets.append(content)
    if snippets:
        has_context = True
        context_block = "\n\nNGỮ CẢNH RAG:\n" + "\n---\n".join(snippets[:6])

    discovery_guidance = _build_discovery_guidance(next_state, lead_profile, missing_slots)
    output_rules = _build_output_rules(next_state, has_context)
    pronoun_rules = _build_pronoun_rules(lead_profile, session_context)
    vision_machine_b_block = _format_vision_machine_b_block(session_context)

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Thời gian hệ thống hiện tại: {now_text}\n\n"
        f"{state_prompt}\n\n"
        f"SCRIPT STEP HIỆN TẠI: {next_script_step}\n"
        f"RESPONSE ACTION: {response_action}\n\n"
        f"HỒ SƠ KHÁCH HÀNG:\n{lead_summary}\n\n"
        f"{vision_machine_b_block}"
        f"THÔNG TIN CÒN THIẾU:\n{missing_slots_text}"
        f"{discovery_guidance}"
        f"{context_block}\n\n"
        f"Tin nhắn của khách: {user_text}\n\n"
        "YÊU CẦU TRẢ LỜI:\n"
        "- Trả lời đúng trọng tâm, ưu tiên câu hỏi đang được hỏi.\n"
        "- Nếu thiếu dữ kiện, nói rõ thiếu gì và đề nghị cách lấy thêm dữ liệu.\n"
        "- Không dùng dữ kiện ngoài ngữ cảnh retrieve.\n"
        "- Áp dụng nguyên văn khối xưng hô bên dưới; với GREETING, câu đầu phải chào đúng giới.\n"
        f"{pronoun_rules}\n"
        f"{output_rules}\n\n"
        "Trả lời:"
    )


def _fold_vn(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    decomp = unicodedata.normalize("NFD", text)
    no_mark = "".join(ch for ch in decomp if unicodedata.category(ch) != "Mn")
    return no_mark.replace("đ", "d")


def _normalize_gender_value(value: Any) -> str:
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
    if not folded:
        return "unknown"
    if folded in {"1", "01"}:
        return "male"
    if folded in {"2", "02"}:
        return "female"
    if folded in {"male", "nam", "man", "m", "anh", "ong"}:
        return "male"
    if folded in {"female", "nu", "woman", "f", "chi", "co"}:
        return "female"
    return "unknown"


_VISION_MACHINE_B_PROMPT_KEYS: tuple[str, ...] = (
    "gender_guess",
    "gioi_tinh",
    "gender_estimate",
    "gender",
    "age_range",
    "age_band",
    "nhom_tuoi",
    "age_group_estimate",
)


def _format_vision_machine_b_block(session_context: Dict[str, Any]) -> str:
    """Chỉ đưa 2 nhóm dữ liệu hợp đồng Máy B (giới + độ tuổi) vào prompt."""
    if not isinstance(session_context, dict):
        return ""
    context_json = session_context.get("context_json")
    if not isinstance(context_json, dict):
        return ""
    vision_context = context_json.get("vision_context")
    if not isinstance(vision_context, dict) or not vision_context:
        return ""
    filtered: Dict[str, Any] = {}
    for k in _VISION_MACHINE_B_PROMPT_KEYS:
        v = vision_context.get(k)
        if v in (None, "", [], "unknown"):
            continue
        filtered[k] = v
    if not filtered:
        return ""
    try:
        blob = json.dumps(filtered, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return ""
    return (
        "NHẬN DIỆN KHÁCH (MÁY B) — chỉ dùng các khóa sau cho xưng hô và gợi ý độ tuổi, không bịa thêm:\n"
        f"{blob}\n\n"
    )


def _build_pronoun_rules(lead_profile: Dict[str, Any], session_context: Dict[str, Any]) -> str:
    context_json = session_context.get("context_json") if isinstance(session_context, dict) else {}
    if not isinstance(context_json, dict):
        context_json = {}
    vision_context = context_json.get("vision_context") if isinstance(context_json, dict) else {}
    if not isinstance(vision_context, dict):
        vision_context = {}

    # Ưu tiên vision (camera / Máy B) trước lead/session để không bị hồ sơ cũ hoặc "unknown" che mất nhận diện.
    candidates = [
        vision_context.get("gender_guess"),
        vision_context.get("gender_estimate"),
        vision_context.get("gioi_tinh"),
        vision_context.get("gender"),
        vision_context.get("sex"),
        session_context.get("gender_estimate") if isinstance(session_context, dict) else None,
        session_context.get("gender_guess") if isinstance(session_context, dict) else None,
        session_context.get("gioi_tinh") if isinstance(session_context, dict) else None,
        lead_profile.get("gender_estimate"),
        lead_profile.get("gender_guess"),
        lead_profile.get("gioi_tinh"),
    ]

    resolved = "unknown"
    for item in candidates:
        gender = _normalize_gender_value(item)
        if gender != "unknown":
            resolved = gender
            break

    if resolved == "male":
        return (
            "XƯNG HÔ BẮT BUỘC (NAM — theo Máy B):\n"
            '- Gọi khách: "anh" trong câu, "Anh" đầu câu.\n'
            '- Xưng: "em".\n'
            '- Mở đầu chào hợp lệ: "Chào anh, em là Sunny..."\n'
            '- Cấm trong lượt này: "Anh/Chị", "anh/chị", "Chào anh/chị".'
        )
    if resolved == "female":
        return (
            "XƯNG HÔ BẮT BUỘC (NỮ — theo Máy B):\n"
            '- Gọi khách: "chị" / "Chị".\n'
            '- Xưng: "em".\n'
            '- Mở đầu chào hợp lệ: "Chào chị, em là Sunny..."\n'
            '- Cấm trong lượt này: "Anh/Chị", "anh/chị", "Chào anh/chị".'
        )
    return (
        "XƯNG HÔ (chưa có giới từ Máy B):\n"
        '- Gọi khách: "Anh/Chị" hoặc "anh/chị" trung tính.\n'
        '- Xưng: "em".\n'
        '- Mở đầu chào hợp lệ: "Chào anh/chị, em là Sunny..."'
    )


def _build_discovery_guidance(next_state: str, lead_profile: Dict[str, Any], missing_slots: List[str]) -> str:
    if next_state != "need_discovery":
        return ""

    hints: List[str] = []
    purpose = str(lead_profile.get("purpose") or "").strip().lower()
    family_size = lead_profile.get("family_member_count")
    children_count = lead_profile.get("children_count")

    if purpose == "mua_o":
        hints.append("Khách thiên về nhu cầu ở thực.")
    if isinstance(family_size, int) and family_size >= 3:
        hints.append("Gia đình đông người thường ưu tiên công năng và không gian.")
    if isinstance(children_count, int) and children_count > 0:
        hints.append("Có con nhỏ thường quan tâm trường học, công viên và môi trường an toàn.")
    if missing_slots:
        hints.append("Nếu có ngữ cảnh RAG, trả lời định hướng ngắn rồi hỏi tiếp 1 slot còn thiếu.")

    if not hints:
        return ""
    return "\n\nGỢI Ý KHÁM PHÁ NHU CẦU:\n- " + "\n- ".join(hints)


def _build_output_rules(next_state: str, has_context: bool) -> str:
    grounded_states = {"project_qa", "product_matching", "comparison", "objection_handling", "closing_next_step"}
    rules = ["RÀNG BUỘC HÌNH THỨC:"]

    if next_state == "greeting":
        rules.extend(
            [
                "- Tối đa 2 câu.",
                "- Chỉ 1 câu hỏi mở.",
                "- Câu chào đầu bám đúng giới Máy B (Chào anh / Chào chị / Chào anh/chị khi chưa rõ).",
            ]
        )
    elif next_state == "need_discovery":
        rules.extend(
            [
                "- Tối đa 3 câu.",
                "- Chỉ hỏi 1 câu chính ở cuối lượt.",
                "- Không hỏi lại thông tin đã có trong hồ sơ khách.",
            ]
        )

    if next_state in grounded_states:
        rules.append("- Chỉ sử dụng dữ liệu trong NGỮ CẢNH RAG.")
        if not has_context:
            rules.append("- Hiện chưa có ngữ cảnh phù hợp, không nêu số liệu hoặc dữ kiện cụ thể.")

    return "\n".join(rules)


def _format_lead_profile(profile: Dict[str, Any]) -> str:
    if not profile:
        return "(chưa có thông tin)"
    relevant_keys = [
        "customer_id",
        "full_name",
        "family_member_count",
        "children_count",
        "purpose",
        "property_type",
        "budget_text",
        "budget_min",
        "budget_max",
        "location_preference",
        "timeline",
        "financing_need",
        "key_concerns",
        "lead_temperature",
        "current_state",
        "gender_estimate",
        "gender_guess",
        "gioi_tinh",
        "age_group_estimate",
        "age_band",
        "nhom_tuoi",
        "identity_status",
        "face_match_score",
    ]
    filtered = {
        k: v
        for k, v in profile.items()
        if k in relevant_keys and v not in (None, [], "khong_ro", "unknown", "")
    }
    if not filtered:
        return "(chưa thu thập được thông tin)"
    return json.dumps(filtered, ensure_ascii=False, indent=2)
