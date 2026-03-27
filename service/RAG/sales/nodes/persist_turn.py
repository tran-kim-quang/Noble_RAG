"""Node 8: Persist chat history, lead profile, and session context."""

import asyncio
import logging
from typing import Any, Dict

from sales.graph_state import SalesAgentState
from memory.chat_history_store import save_chat_history
from memory.lead_profile_store import save_lead_profile
from memory.local_snapshot_store import save_local_session_snapshot
from memory.session_store import save_session_context

log = logging.getLogger("rag-service")


async def _persist_worker(
    session_id: str,
    next_state: str,
    history: list[Dict[str, Any]],
    lead_profile: Dict[str, Any],
    session_context: Dict[str, Any],
) -> None:
    await save_chat_history(session_id, history)
    await save_lead_profile(session_id, lead_profile)
    await save_session_context(session_id, session_context)
    snapshot_path = save_local_session_snapshot(
        session_id=session_id,
        lead_profile=lead_profile,
        session_context=session_context,
        chat_history=history,
    )
    log.info(
        "persist_turn(async): session=%s state=%s turn=%d snapshot=%s",
        session_id,
        next_state,
        session_context.get("conversation_turn_count", 0),
        snapshot_path,
    )


async def persist_turn(state: SalesAgentState) -> Dict[str, Any]:
    session_id: str = state.get("session_id") or "default"
    user_text: str = state.get("user_text") or ""
    final_response: str = state.get("final_response") or state.get("draft_response") or ""
    next_state: str = state.get("next_sales_state") or "greeting"
    next_script_step: str = state.get("next_script_step") or "S1_opening"
    response_action: str = state.get("response_action") or next_state

    # Append current turn to history
    history = list(state.get("chat_history") or [])
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": final_response})
    lead_profile = dict(state.get("lead_profile") or {})
    lead_profile["current_state"] = next_state
    lead_profile["current_script_step"] = next_script_step
    lead_profile["last_user_intent"] = state.get("detected_intent")
    lead_profile["last_next_action"] = next_state
    lead_profile["last_action"] = response_action
    session_context = dict(state.get("session_context") or {})
    session_context["previous_state"] = state.get("current_sales_state")
    session_context["current_state"] = next_state
    session_context["current_script_step"] = next_script_step
    session_context["last_agent_action"] = response_action
    session_context["conversation_turn_count"] = (
        session_context.get("conversation_turn_count", 0) + 1
    )
    task = asyncio.create_task(
        _persist_worker(
            session_id=session_id,
            next_state=next_state,
            history=history,
            lead_profile=lead_profile,
            session_context=session_context,
        )
    )
    task.add_done_callback(
        lambda t: log.error("persist_turn async failed: %s", t.exception()) if t.exception() else None
    )

    return {
        "chat_history": history,
        "lead_profile": lead_profile,
        "session_context": session_context,
        "should_persist": False,
    }
