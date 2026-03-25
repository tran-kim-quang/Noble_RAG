"""Session context storage backed by Redis.

Key: session_context:{session_id}
TTL: 24 hours
"""

import logging
from typing import Any, Dict, Optional

from memory.redis_store import redis_get_json, redis_set_json
from memory.redis_store import redis_delete
from core.config import get_settings

log = logging.getLogger("rag-service")
SESSION_TTL = 86400


def _redis_url() -> str:
    return get_settings().redis_url


def _key(session_id: str) -> str:
    return f"session_context:{session_id}"


async def load_session_context(session_id: str) -> Dict[str, Any]:
    data = await redis_get_json(_redis_url(), _key(session_id))
    if isinstance(data, dict):
        return data
    return {
        "session_id": session_id,
        "current_state": "greeting",
        "previous_state": None,
        "last_agent_action": None,
        "conversation_turn_count": 0,
    }


async def save_session_context(session_id: str, context: Dict[str, Any]) -> None:
    await redis_set_json(_redis_url(), _key(session_id), context, ttl=SESSION_TTL)


async def increment_turn_count(session_id: str) -> int:
    ctx = await load_session_context(session_id)
    ctx["conversation_turn_count"] = ctx.get("conversation_turn_count", 0) + 1
    await save_session_context(session_id, ctx)
    return ctx["conversation_turn_count"]


async def delete_session_context(session_id: str) -> bool:
    return await redis_delete(_redis_url(), _key(session_id))
