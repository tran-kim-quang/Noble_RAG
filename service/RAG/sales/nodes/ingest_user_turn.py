"""Node 1: Load chat history and lead profile, normalise input."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from memory.chat_history_store import load_chat_history
from memory.lead_profile_store import load_lead_profile
from memory.session_store import load_session_context

log = logging.getLogger("rag-service")


async def ingest_user_turn(state: SalesAgentState) -> Dict[str, Any]:
    session_id: str = state.get("session_id", "default")
    user_text: str = (state.get("user_text") or state.get("raw_transcript") or "").strip()

    chat_history = await load_chat_history(session_id)
    lead_profile = await load_lead_profile(session_id)
    session_context = await load_session_context(session_id)

    if not lead_profile:
        lead_profile = {
            "lead_id": session_id,
            "current_state": "greeting",
            "lead_temperature": "cold",
        }

    log.info(
        "ingest_user_turn session=%s history_len=%d lead_state=%s",
        session_id,
        len(chat_history),
        lead_profile.get("current_state", "?"),
    )

    return {
        "user_text": user_text,
        "chat_history": chat_history,
        "lead_profile": lead_profile,
        "session_context": session_context,
        "current_sales_state": session_context.get("current_state", "greeting"),
        "errors": [],
    }
