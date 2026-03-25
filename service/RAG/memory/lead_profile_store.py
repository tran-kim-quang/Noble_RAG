"""Lead profile storage.

Primary cache: Redis  key = lead_profile_cache:{session_id}  TTL 7 days
Source of truth: Postgres  table = sales.lead_profiles
"""

import json
import logging
from typing import Any, Dict, Optional

import asyncpg

from memory.redis_store import redis_delete, redis_get_json, redis_set_json
from memory.local_snapshot_store import load_local_session_snapshot
from core.config import get_settings

log = logging.getLogger("rag-service")
LEAD_TTL = 7 * 86400  # 7 days
_schema_ready = False


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
    await ensure_sales_schema()
    data = await _load_from_postgres(session_id)
    if data:
        return data
    snapshot = load_local_session_snapshot(session_id)
    profile = snapshot.get("lead_profile")
    return profile if isinstance(profile, dict) and profile else None


async def save_lead_profile(session_id: str, profile: Dict[str, Any]) -> None:
    await redis_set_json(_redis_url(), _key(session_id), profile, ttl=LEAD_TTL)
    try:
        await ensure_sales_schema()
        await _upsert_to_postgres(session_id, profile)
    except Exception as e:
        log.error("Failed to persist lead_profile to Postgres session=%s: %s", session_id, e)


async def delete_lead_profile_cache(session_id: str) -> bool:
    return await redis_delete(_redis_url(), _key(session_id))


async def ensure_sales_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            CREATE SCHEMA IF NOT EXISTS sales;

            CREATE TABLE IF NOT EXISTS sales.lead_profiles (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id VARCHAR(255) NOT NULL UNIQUE,
                profile_data JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_lead_profiles_session_id
                ON sales.lead_profiles(session_id);
            CREATE INDEX IF NOT EXISTS idx_lead_profiles_updated_at
                ON sales.lead_profiles(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_lead_profiles_data
                ON sales.lead_profiles USING GIN(profile_data);
            """
        )
        _schema_ready = True
    finally:
        await conn.close()


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
