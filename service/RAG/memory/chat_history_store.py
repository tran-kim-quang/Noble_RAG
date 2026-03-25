"""Chat history storage backed by Redis.

Key pattern: chat_history:{session_id}
TTL: 24 hours (86400 s)
Max messages kept: HISTORY_MAX_TURNS * 2 (user + assistant)
"""

import logging
from typing import Any, Dict, List

from memory.redis_store import redis_get_json, redis_set_json
from memory.redis_store import redis_delete
from core.config import get_settings

HISTORY_MAX_TURNS = 10
log = logging.getLogger("rag-service")


def _redis_url() -> str:
    return get_settings().redis_url


def _key(session_id: str) -> str:
    return f"chat_history:{session_id}"


async def load_chat_history(session_id: str) -> List[Dict[str, Any]]:
    data = await redis_get_json(_redis_url(), _key(session_id))
    if isinstance(data, list):
        log.debug("chat_history loaded: session=%s msgs=%d", session_id, len(data))
        return data
    return []


async def save_chat_history(
    session_id: str,
    history: List[Dict[str, Any]],
    max_turns: int = HISTORY_MAX_TURNS,
) -> None:
    trimmed = history[-(max_turns * 2):]
    await redis_set_json(_redis_url(), _key(session_id), trimmed)
    log.debug("chat_history saved: session=%s msgs=%d", session_id, len(trimmed))


async def append_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    max_turns: int = HISTORY_MAX_TURNS,
) -> List[Dict[str, Any]]:
    history = await load_chat_history(session_id)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    await save_chat_history(session_id, history, max_turns)
    return history


async def delete_chat_history(session_id: str) -> bool:
    return await redis_delete(_redis_url(), _key(session_id))
