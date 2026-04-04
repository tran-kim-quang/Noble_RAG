"""Persistence helpers for the v1 Sales RAG + Vision workflow."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import asyncpg

from core.config import get_settings

_schema_ready = False


def _postgres_url() -> str:
    return get_settings().postgres_url


async def ensure_pipeline_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            CREATE SCHEMA IF NOT EXISTS sales;

            CREATE TABLE IF NOT EXISTS sales.customer_sessions (
                session_id           VARCHAR(128) PRIMARY KEY,
                customer_id          TEXT,
                channel              TEXT,
                source               TEXT,
                session_status       TEXT DEFAULT 'active',
                short_memory_summary TEXT,
                created_from_vision  BOOLEAN DEFAULT FALSE,
                started_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_activity_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS sales.vision_context (
                vision_context_id    BIGSERIAL PRIMARY KEY,
                session_id           VARCHAR(128) REFERENCES sales.customer_sessions(session_id) ON DELETE CASCADE,
                customer_id          TEXT,
                image_asset_id       VARCHAR(128),
                matched              BOOLEAN,
                match_score          FLOAT,
                age_range            TEXT,
                gender_guess         TEXT,
                emotion              TEXT,
                dress_style          TEXT,
                visible_attributes   JSONB,
                scene_context        TEXT,
                raw_vision_summary   JSONB,
                created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS sales.session_context (
                session_id           VARCHAR(128) PRIMARY KEY REFERENCES sales.customer_sessions(session_id) ON DELETE CASCADE,
                customer_id          TEXT,
                context_json         JSONB NOT NULL,
                updated_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS sales.chat_messages (
                message_id           BIGSERIAL PRIMARY KEY,
                session_id           VARCHAR(128) REFERENCES sales.customer_sessions(session_id) ON DELETE CASCADE,
                customer_id          TEXT,
                role                 TEXT NOT NULL,
                content              TEXT NOT NULL,
                intent               TEXT,
                sales_stage          TEXT,
                created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS sales.documents (
                document_id          VARCHAR(128) PRIMARY KEY,
                collection_name      TEXT NOT NULL,
                project_id           TEXT,
                document_type        TEXT,
                file_name            TEXT,
                storage_url          TEXT,
                chunk_count          INT DEFAULT 0,
                ingest_status        TEXT DEFAULT 'pending',
                created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        )
        _schema_ready = True
    finally:
        await conn.close()


async def create_customer_session(
    *,
    session_id: str,
    customer_id: str,
    channel: Optional[str],
    source: Optional[str],
    created_from_vision: bool = False,
) -> None:
    await ensure_pipeline_schema()
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            INSERT INTO sales.customer_sessions (
                session_id, customer_id, channel, source, created_from_vision, last_activity_at
            ) VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                customer_id = EXCLUDED.customer_id,
                channel = COALESCE(EXCLUDED.channel, sales.customer_sessions.channel),
                source = COALESCE(EXCLUDED.source, sales.customer_sessions.source),
                created_from_vision = sales.customer_sessions.created_from_vision OR EXCLUDED.created_from_vision,
                last_activity_at = NOW()
            """,
            session_id,
            customer_id,
            channel,
            source,
            created_from_vision,
        )
    finally:
        await conn.close()


async def touch_customer_session_activity(session_id: str) -> None:
    await ensure_pipeline_schema()
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            "UPDATE sales.customer_sessions SET last_activity_at = NOW() WHERE session_id = $1",
            session_id,
        )
    finally:
        await conn.close()


async def get_latest_active_session_by_customer_id(customer_id: str) -> Optional[Dict[str, Any]]:
    await ensure_pipeline_schema()
    cid = (customer_id or "").strip()
    if not cid:
        return None

    conn = await asyncpg.connect(_postgres_url())
    try:
        row = await conn.fetchrow(
            """
            SELECT session_id, customer_id, channel, source, session_status,
                   created_from_vision, started_at, last_activity_at
            FROM sales.customer_sessions
            WHERE customer_id = $1
              AND COALESCE(session_status, 'active') = 'active'
            ORDER BY last_activity_at DESC NULLS LAST, started_at DESC NULLS LAST
            LIMIT 1
            """,
            cid,
        )
    finally:
        await conn.close()

    if not row:
        return None
    return dict(row)


async def load_session_context_row(session_id: str) -> Optional[Dict[str, Any]]:
    await ensure_pipeline_schema()
    sid = (session_id or "").strip()
    if not sid:
        return None

    conn = await asyncpg.connect(_postgres_url())
    try:
        row = await conn.fetchrow(
            """
            SELECT session_id, customer_id, context_json, updated_at
            FROM sales.session_context
            WHERE session_id = $1
            LIMIT 1
            """,
            sid,
        )
    finally:
        await conn.close()

    if not row:
        return None
    return dict(row)


async def upsert_document_metadata(
    *,
    document_id: str,
    collection_name: str,
    project_id: Optional[str],
    document_type: Optional[str],
    file_name: Optional[str],
    storage_url: Optional[str],
    chunk_count: int,
    ingest_status: str,
) -> None:
    await ensure_pipeline_schema()
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            INSERT INTO sales.documents (
                document_id, collection_name, project_id, document_type,
                file_name, storage_url, chunk_count, ingest_status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (document_id) DO UPDATE SET
                collection_name = EXCLUDED.collection_name,
                project_id = EXCLUDED.project_id,
                document_type = EXCLUDED.document_type,
                file_name = EXCLUDED.file_name,
                storage_url = EXCLUDED.storage_url,
                chunk_count = EXCLUDED.chunk_count,
                ingest_status = EXCLUDED.ingest_status
            """,
            document_id,
            collection_name,
            project_id,
            document_type,
            file_name,
            storage_url,
            chunk_count,
            ingest_status,
        )
    finally:
        await conn.close()


async def insert_vision_context(
    *,
    session_id: str,
    customer_id: str,
    image_asset_id: Optional[str],
    matched: bool,
    match_score: Optional[float],
    vision_summary: Optional[Dict[str, Any]],
) -> None:
    await ensure_pipeline_schema()
    summary = dict(vision_summary or {})
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            INSERT INTO sales.vision_context (
                session_id, customer_id, image_asset_id, matched, match_score,
                age_range, gender_guess, emotion, dress_style,
                visible_attributes, scene_context, raw_vision_summary
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10::jsonb, $11, $12::jsonb
            )
            """,
            session_id,
            customer_id,
            image_asset_id,
            matched,
            match_score,
            summary.get("age_range"),
            summary.get("gender_guess"),
            summary.get("emotion"),
            summary.get("dress_style"),
            json.dumps(summary.get("visible_attributes") or [], ensure_ascii=False),
            summary.get("scene_context"),
            json.dumps(summary, ensure_ascii=False),
        )
    finally:
        await conn.close()


async def upsert_session_context_row(
    *,
    session_id: str,
    customer_id: str,
    context_json: Dict[str, Any],
) -> None:
    await ensure_pipeline_schema()
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            INSERT INTO sales.session_context (session_id, customer_id, context_json, updated_at)
            VALUES ($1, $2, $3::jsonb, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                customer_id = EXCLUDED.customer_id,
                context_json = EXCLUDED.context_json,
                updated_at = NOW()
            """,
            session_id,
            customer_id,
            json.dumps(context_json, ensure_ascii=False),
        )
    finally:
        await conn.close()


async def insert_chat_message(
    *,
    session_id: str,
    customer_id: str,
    role: str,
    content: str,
    intent: Optional[str] = None,
    sales_stage: Optional[str] = None,
) -> None:
    await ensure_pipeline_schema()
    conn = await asyncpg.connect(_postgres_url())
    try:
        await conn.execute(
            """
            INSERT INTO sales.chat_messages (session_id, customer_id, role, content, intent, sales_stage)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            session_id,
            customer_id,
            role,
            content,
            intent,
            sales_stage,
        )
    finally:
        await conn.close()
