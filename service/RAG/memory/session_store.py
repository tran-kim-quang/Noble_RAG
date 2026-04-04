"""Session context storage backed by Redis.

Key: session_context:{session_id}
TTL: 24 hours
"""

from copy import deepcopy
from typing import Any, Dict, Optional

from core.config import get_settings
from memory.local_snapshot_store import load_local_session_snapshot
from memory.redis_store import redis_delete
from memory.redis_store import redis_get_json, redis_set_json

SESSION_TTL = 86400


def _redis_url() -> str:
    return get_settings().redis_url


def _key(session_id: str) -> str:
    return f"session_context:{session_id}"


def _default_conversation_context() -> Dict[str, Any]:
    return {
        "last_intent": None,
        "missing_slots": [],
        "last_question": None,
    }


def _default_context_json() -> Dict[str, Any]:
    return {
        "customer_profile": {},
        "vision_context": None,
        "conversation_context": _default_conversation_context(),
    }


def _default_context(session_id: str) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "customer_id": None,
        "current_state": "greeting",
        "current_script_step": "S1_opening",
        "previous_state": None,
        "last_agent_action": None,
        "conversation_turn_count": 0,
        "context_ready": False,
        "context_json": _default_context_json(),
    }


def ensure_session_context_shape(session_id: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _default_context(session_id)
    raw = dict(data or {})
    base.update(raw)

    raw_context_json = base.get("context_json")
    if not isinstance(raw_context_json, dict):
        raw_context_json = {}
    customer_profile = raw_context_json.get("customer_profile")
    if not isinstance(customer_profile, dict):
        customer_profile = {}
    vision_context = raw_context_json.get("vision_context")
    conversation_context = raw_context_json.get("conversation_context")
    if not isinstance(conversation_context, dict):
        conversation_context = {}
    merged_conversation = _default_conversation_context()
    merged_conversation.update(conversation_context)

    base["context_json"] = {
        "customer_profile": customer_profile,
        "vision_context": vision_context,
        "conversation_context": merged_conversation,
    }
    return base


async def get_existing_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    data = await redis_get_json(_redis_url(), _key(session_id))
    if isinstance(data, dict):
        return ensure_session_context_shape(session_id, data)

    snapshot = load_local_session_snapshot(session_id)
    snapshot_context = snapshot.get("session_context")
    if isinstance(snapshot_context, dict) and snapshot_context:
        return ensure_session_context_shape(session_id, snapshot_context)

    return None


async def load_session_context(session_id: str) -> Dict[str, Any]:
    existing = await get_existing_session_context(session_id)
    if existing:
        return existing
    return ensure_session_context_shape(session_id, None)


async def save_session_context(session_id: str, context: Dict[str, Any]) -> None:
    payload = ensure_session_context_shape(session_id, deepcopy(context))
    await redis_set_json(_redis_url(), _key(session_id), payload, ttl=SESSION_TTL)


async def increment_turn_count(session_id: str) -> int:
    ctx = await load_session_context(session_id)
    ctx["conversation_turn_count"] = ctx.get("conversation_turn_count", 0) + 1
    await save_session_context(session_id, ctx)
    return ctx["conversation_turn_count"]


async def delete_session_context(session_id: str) -> bool:
    return await redis_delete(_redis_url(), _key(session_id))
