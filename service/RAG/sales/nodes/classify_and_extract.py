"""Node 2+3 (merged): Classify intent AND extract lead slots in ONE LLM call.

Optimization: Replaces two sequential LLM calls (classify_intent + extract_lead_slots)
with a single call, reducing TTFT by ~40% (from ~5.7s to ~2.3s).

Output keys are identical to the two original nodes combined, so downstream nodes
(update_lead_profile, resolve_sales_state, ...) require zero changes.
"""

import json
import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from core.dependencies import llm_model_func
from utils.json_extract import extract_first_json_object

log = logging.getLogger("rag-service")

_COMBINED_PROMPT = """Bạn là bộ phân tích hội thoại bất động sản. Làm 2 việc cùng lúc từ tin nhắn khách:

1. Phân loại INTENT
2. Trích xuất THÔNG TIN LEAD

Trả về DUY NHẤT một JSON với cấu trúc chính xác sau:
{{
  "intent": "greeting" | "ask_recommendation" | "project_info" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "objection_type": null | "gia_cao" | "phap_ly" | "vi_tri" | "chua_du_tien" | "suy_nghi_them" | "khac",
  "buy_signal": true | false,
  "confidence": 0.0-1.0,
  "family_member_count": null | <số người trong gia đình>,
  "children_count": null | <số con nhỏ>,
  "purpose": null | "mua_o" | "kinh_doanh" | "dau_tu_cho_thue" | "dau_tu_tang_gia" | "giu_tai_san" | "tham_khao",
  "property_type": null | "can_ho" | "nha_pho" | "biet_thu" | "shophouse" | "dat_nen",
  "budget_text": null | "<chuỗi mô tả ngân sách>",
  "budget_min": null | <số tỷ đồng float>,
  "budget_max": null | <số tỷ đồng float>,
  "location_preference": [] | ["<khu vực 1>"],
  "timeline": null | "ngay_bay_gio" | "trong_6_thang" | "trong_1_nam" | "trong_2_nam",
  "financing_need": null | true | false,
  "key_concerns": [] | ["<băn khoăn>"],
  "contact_phone": null | "<số điện thoại>"
}}

Định nghĩa intent:
- greeting: chào hỏi, giới thiệu bản thân
- ask_recommendation: hỏi về dự án, muốn tư vấn, muốn giới thiệu sản phẩm
- project_info: hỏi thông tin cụ thể về dự án/sản phẩm/chính sách/pháp lý/tiện ích/vị trí mà cần trả lời trực tiếp theo dữ liệu có sẵn, không phải xin agent tư vấn shortlist
- comparison: muốn so sánh 2+ lựa chọn
- objection: phản đối, băn khoăn về giá/pháp lý/vị trí/tài chính
- buy_signal: hỏi bảng giá, còn căn không, muốn đặt cọc, muốn đi xem, muốn gặp sales
- follow_up: hỏi lại câu hỏi cũ, tiếp tục hội thoại trước
- out_of_scope: chủ đề không liên quan bất động sản
- other: không xác định rõ

Quy tắc trích xuất:
- "Gia đình 4 người", "vợ chồng và 2 con nhỏ" => family_member_count=4, children_count=2.
- "Sống 1 mình", "ở một mình", "độc thân" => family_member_count=1, children_count=0.
- "Vợ chồng chưa có con", "hai vợ chồng" => family_member_count=2, children_count=0 nếu người dùng không nhắc tới con.
- "Để ở" => purpose="mua_o".
- "Kinh doanh", "mở cửa hàng", "buôn bán" => purpose="kinh_doanh".
- Ưu tiên trích xuất location_preference khi khách nói "gần Long Biên", "ở Hà Nội", "quanh Cầu Giấy"...
- Nếu không có thông tin cho slot nào, để null hoặc [].
- Khi người dùng mô tả rõ tình trạng sống một mình hoặc chưa có con, hãy điền giá trị số 0 thay vì null cho children_count.

Lịch sử gần đây:
{history_str}

Tin nhắn khách: "{user_text}"
CHỈ TRẢ VỀ JSON."""

_INTENT_FIELDS = {"intent", "objection_type", "buy_signal", "confidence"}
_SLOT_FIELDS = {
    "family_member_count", "children_count", "purpose", "property_type", "budget_text", "budget_min", "budget_max",
    "location_preference", "timeline", "financing_need", "key_concerns", "contact_phone",
}


async def classify_and_extract(state: SalesAgentState) -> Dict[str, Any]:
    user_text: str = state.get("user_text") or ""
    history = state.get("chat_history") or []
    fast_slots = dict(state.get("extracted_slots") or {})

    history_str = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history[-4:]]
    ) if history else "(chưa có)"

    prompt = _COMBINED_PROMPT.format(history_str=history_str, user_text=user_text)

    try:
        response = await llm_model_func(
            prompt, enable_cot=False, response_format={"type": "json_object"}
        )
        payload = extract_first_json_object(str(response))
        if not payload:
            raise ValueError("No JSON in combined response")

        data = json.loads(payload)

        intent = str(data.get("intent", "other"))
        buy_signal = bool(data.get("buy_signal", False))
        objection_type = data.get("objection_type")
        slots = dict(fast_slots)
        for key in _SLOT_FIELDS:
            value = data.get(key)
            if value in (None, [], ""):
                continue
            slots[key] = value

        log.info(
            "classify_and_extract: intent=%s buy_signal=%s slots=%d",
            intent, buy_signal,
            sum(1 for v in slots.values() if v not in (None, [], "")),
        )
        return {
            # intent fields (same as classify_intent output)
            "detected_intent": intent,
            "buy_signal": buy_signal,
            "objection_type": objection_type,
            # slot fields (same as extract_lead_slots output)
            "extracted_slots": slots,
        }

    except Exception as e:
        log.error("classify_and_extract error: %s", e)
        return {
            "detected_intent": "other",
            "buy_signal": False,
            "objection_type": None,
            "extracted_slots": fast_slots,
        }
