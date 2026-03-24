"""Node 3: Extract lead profile slots from user text using LLM."""

import json
import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from core.dependencies import llm_model_func
from utils.json_extract import extract_first_json_object

log = logging.getLogger("rag-service")

_SLOT_PROMPT = """Bạn là bộ trích xuất thông tin lead bất động sản.
Từ tin nhắn khách, trích xuất các slot sau. Nếu không có thông tin, để null.
Trả về DUY NHẤT JSON:
{{
  "purpose": null | "mua_o" | "dau_tu_cho_thue" | "dau_tu_tang_gia" | "giu_tai_san" | "tham_khao",
  "property_type": null | "can_ho" | "nha_pho" | "biet_thu" | "shophouse" | "dat_nen",
  "budget_text": null | "<chuỗi mô tả ngân sách như '2-3 tỷ', 'dưới 5 tỷ'>",
  "budget_min": null | <số tỷ đồng float>,
  "budget_max": null | <số tỷ đồng float>,
  "location_preference": [] | ["<khu vực 1>", "<khu vực 2>"],
  "timeline": null | "ngay_bay_gio" | "trong_6_thang" | "trong_1_nam" | "trong_2_nam",
  "financing_need": null | true | false,
  "key_concerns": [] | ["<băn khoăn 1>"],
  "contact_phone": null | "<số điện thoại nếu khách cho>"
}}

Lịch sử gần đây:
{history_str}

Tin nhắn khách: "{user_text}"
CHỈ TRẢ VỀ JSON."""


async def extract_lead_slots(state: SalesAgentState) -> Dict[str, Any]:
    user_text: str = state.get("user_text") or ""
    history = state.get("chat_history") or []

    history_str = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history[-4:]]
    ) if history else "(chưa có)"

    prompt = _SLOT_PROMPT.format(history_str=history_str, user_text=user_text)

    try:
        response = await llm_model_func(
            prompt, enable_cot=False, response_format={"type": "json_object"}
        )
        payload = extract_first_json_object(str(response))
        if not payload:
            raise ValueError("No JSON in slot extraction")
        slots = json.loads(payload)
        log.info("extract_lead_slots: extracted %d non-null slots", sum(1 for v in slots.values() if v))
        return {"extracted_slots": slots}
    except Exception as e:
        log.error("extract_lead_slots error: %s", e)
        return {"extracted_slots": {}}
