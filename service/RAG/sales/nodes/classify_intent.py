"""Node 2: Classify user intent using LLM."""

import json
import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from core.dependencies import llm_model_func
from utils.json_extract import extract_first_json_object

log = logging.getLogger("rag-service")

_INTENT_PROMPT = """Bạn là bộ phân tích intent trong hệ thống tư vấn bất động sản.
Phân tích tin nhắn khách và trả về DUY NHẤT JSON:
{{
  "intent": "greeting" | "ask_recommendation" | "comparison" | "objection" | "buy_signal" | "follow_up" | "out_of_scope" | "other",
  "objection_type": null | "gia_cao" | "phap_ly" | "vi_tri" | "chua_du_tien" | "suy_nghi_them" | "khac",
  "buy_signal": true | false,
  "confidence": 0.0-1.0
}}

Định nghĩa:
- greeting: chào hỏi, giới thiệu bản thân
- ask_recommendation: hỏi về dự án, muốn tư vấn, muốn giới thiệu sản phẩm
- comparison: muốn so sánh 2+ lựa chọn
- objection: phản đối, băn khoăn về giá/pháp lý/vị trí/tài chính
- buy_signal: hỏi bảng giá, còn căn không, muốn đặt cọc, muốn đi xem, muốn gặp sales
- follow_up: hỏi lại câu hỏi cũ, tiếp tục hội thoại trước
- out_of_scope: chủ đề không liên quan bất động sản
- other: không xác định rõ

Lịch sử gần đây:
{history_str}

Tin nhắn khách: "{user_text}"
CHỈ TRẢ VỀ JSON."""


async def classify_intent(state: SalesAgentState) -> Dict[str, Any]:
    user_text: str = state.get("user_text") or ""
    history = state.get("chat_history") or []

    history_str = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history[-4:]]
    ) if history else "(chưa có)"

    prompt = _INTENT_PROMPT.format(
        history_str=history_str,
        user_text=user_text,
    )

    try:
        response = await llm_model_func(
            prompt, enable_cot=False, response_format={"type": "json_object"}
        )
        payload = extract_first_json_object(str(response))
        if not payload:
            raise ValueError("No JSON in intent response")
        data = json.loads(payload)
        intent = str(data.get("intent", "other"))
        buy_signal = bool(data.get("buy_signal", False))
        objection_type = data.get("objection_type")
        log.info("classify_intent: intent=%s buy_signal=%s", intent, buy_signal)
        return {
            "detected_intent": intent,
            "buy_signal": buy_signal,
            "objection_type": objection_type,
        }
    except Exception as e:
        log.error("classify_intent error: %s", e)
        return {"detected_intent": "other", "buy_signal": False, "objection_type": None}
