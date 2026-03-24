"""Node 8: Persist chat history, lead profile, and session context."""

import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from memory.chat_history_store import save_chat_history
from memory.lead_profile_store import save_lead_profile
from memory.session_store import save_session_context

log = logging.getLogger("rag-service")


async def persist_turn(state: SalesAgentState) -> Dict[str, Any]:
    session_id: str = state.get("session_id") or "default"
    user_text: str = state.get("user_text") or ""
    final_response: str = state.get("final_response") or state.get("draft_response") or ""
    next_state: str = state.get("next_sales_state") or "greeting"

    # Append current turn to history
    history = list(state.get("chat_history") or [])
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": final_response})
    await save_chat_history(session_id, history)

    # Update and persist lead profile
    lead_profile = dict(state.get("lead_profile") or {})
    lead_profile["current_state"] = next_state
    lead_profile["last_user_intent"] = state.get("detected_intent")
    lead_profile["last_next_action"] = next_state
    await save_lead_profile(session_id, lead_profile)

    # Update session context
    session_context = dict(state.get("session_context") or {})
    session_context["previous_state"] = state.get("current_sales_state")
    session_context["current_state"] = next_state
    session_context["last_agent_action"] = next_state
    session_context["conversation_turn_count"] = (
        session_context.get("conversation_turn_count", 0) + 1
    )
    await save_session_context(session_id, session_context)

    log.info(
        "persist_turn: session=%s state=%s turn=%d",
        session_id,
        next_state,
        session_context["conversation_turn_count"],
    )

    return {
        "chat_history": history,
        "lead_profile": lead_profile,
        "session_context": session_context,
        "should_persist": False,
    }
