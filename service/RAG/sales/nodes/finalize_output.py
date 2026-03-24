"""Node 9: Finalise output for API response."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState

log = logging.getLogger("rag-service")


def finalize_output(state: SalesAgentState) -> Dict[str, Any]:
    final = (
        state.get("final_response")
        or state.get("draft_response")
        or "Xin lỗi, em chưa có câu trả lời phù hợp."
    )
    log.info(
        "finalize_output: session=%s state=%s",
        state.get("session_id"),
        state.get("next_sales_state"),
    )
    return {"final_response": final}
