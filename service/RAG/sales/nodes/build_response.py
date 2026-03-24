"""Node 7: Build and generate the response using LLM."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from sales.prompt_builder import build_prompt
from core.dependencies import llm_model_func

log = logging.getLogger("rag-service")


async def build_response(state: SalesAgentState) -> Dict[str, Any]:
    prompt = build_prompt(state)
    history = state.get("chat_history") or []

    try:
        raw = await llm_model_func(prompt, history_messages=history)
        response_text = str(raw).strip()
        log.info(
            "build_response: state=%s response_len=%d",
            state.get("next_sales_state"),
            len(response_text),
        )
        return {"draft_response": response_text, "final_response": response_text}
    except Exception as e:
        log.error("build_response error: %s", e)
        fallback = "Xin lỗi, em gặp lỗi khi xử lý. Anh/Chị thử lại giúp em nhé."
        return {"draft_response": fallback, "final_response": fallback, "errors": [str(e)]}
