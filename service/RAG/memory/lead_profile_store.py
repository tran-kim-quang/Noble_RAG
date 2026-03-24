"""Lead profile storage.

Primary cache: Redis  key = lead_profile_cache:{session_id}  TTL 7 days
Source of truth: Postgres  table = sales.lead_profiles
"""

import json
import logging
from typing import Any, Dict, Optional

import asyncpg

from memory.redis_store import redis_get_json, redis_set_json
from core.config import get_settings

log = logging.getLogger("rag-service")
LEAD_TTL = 7 * 86400  # 7 days


def _redis_url() -> str:
    return get_settings().redis_url


def _postgres_url() -> str:
    return get_settings().postgres_url


def _key(session_id: str) -> str:
    return f"lead_profile_cache:{session_id}"


async def load_lead_profile(session_id: str) -> Optional[Dict[str, Any]]:
    data = await redis_get_json(_redis_url(), _key(session_id))
    if data:
        return data
    return await _load_from_postgres(session_id)


async def save_lead_profile(session_id: str, profile: Dict[str, Any]) -> None:
    await redis_set_json(_redis_url(), _key(session_id), profile, ttl=LEAD_TTL)
    try:
        await _upsert_to_postgres(session_id, profile)
    except Exception as e:
        log.error("Failed to persist lead_profile to Postgres session=%s: %s", session_id, e)


async def _load_from_postgres(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        conn = await asyncpg.connect(_postgres_url())
        row = await conn.fetchrow(
            "SELECT profile_data FROM sales.lead_profiles WHERE session_id = $1",
            session_id,
        )
        await conn.close()
        if row:
            raw = row["profile_data"]
            if isinstance(raw, str):
                return json.loads(raw)
            return dict(raw)
    except Exception as e:
        log.warning("Postgres load lead_profile failed session=%s: %s", session_id, e)
    return None


async def _upsert_to_postgres(session_id: str, profile: Dict[str, Any]) -> None:
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            INSERT INTO sales.lead_profiles (session_id, profile_data, updated_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET profile_data = EXCLUDED.profile_data, updated_at = NOW()
            """,
            session_id,
            json.dumps(profile, ensure_ascii=False),
        )
    finally:
        await conn.close()
