import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from core.config import get_settings

_schema_ready = False


def _postgres_url() -> str:
    return get_settings().postgres_url


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _json_or_default(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    a_norm = 0.0
    b_norm = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        a_norm += av * av
        b_norm += bv * bv
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(a_norm) * math.sqrt(b_norm))


@dataclass
class FaceMatch:
    customer_id: str
    score: float


class CustomerIdentityStore:
    async def ensure_customer_schema(self) -> None:
        global _schema_ready
        if _schema_ready:
            return
        conn = await asyncpg.connect(_postgres_url())
        try:
            await conn.execute(
                """
                CREATE SCHEMA IF NOT EXISTS customer;

                CREATE TABLE IF NOT EXISTS customer.customers (
                    id UUID PRIMARY KEY,
                    customer_code VARCHAR(64) NOT NULL UNIQUE,
                    display_name VARCHAR(255),
                    status VARCHAR(50) NOT NULL DEFAULT 'active',
                    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS customer.customer_faces (
                    id UUID PRIMARY KEY,
                    customer_id UUID NOT NULL REFERENCES customer.customers(id) ON DELETE CASCADE,
                    embedding JSONB NOT NULL,
                    embedding_model VARCHAR(100) NOT NULL,
                    face_hash VARCHAR(128),
                    quality_score FLOAT,
                    is_primary BOOLEAN NOT NULL DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS customer.customer_sessions (
                    id UUID PRIMARY KEY,
                    customer_id UUID NOT NULL REFERENCES customer.customers(id) ON DELETE CASCADE,
                    session_id VARCHAR(255) NOT NULL UNIQUE,
                    source VARCHAR(50) NOT NULL DEFAULT 'camera',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS customer.customer_identity_events (
                    id UUID PRIMARY KEY,
                    customer_id UUID,
                    session_id VARCHAR(255),
                    event_type VARCHAR(100) NOT NULL,
                    similarity_score FLOAT,
                    decision VARCHAR(50),
                    image_ref TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            _schema_ready = True
        finally:
            await conn.close()

    async def create_customer(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        await self.ensure_customer_schema()
        customer_id = str(uuid.uuid4())
        customer_code = f"CUS-{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}-{customer_id.split('-')[0].upper()}"
        conn = await asyncpg.connect(_postgres_url())
        try:
            await conn.execute(
                """
                INSERT INTO customer.customers (
                    id, customer_code, first_seen_at, last_seen_at, created_at, updated_at, metadata
                ) VALUES ($1::uuid, $2, NOW(), NOW(), NOW(), NOW(), $3::jsonb)
                """,
                customer_id,
                customer_code,
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        finally:
            await conn.close()
        return customer_id

    async def add_face_embedding(
        self,
        customer_id: str,
        embedding: List[float],
        embedding_model: str,
        quality_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_primary: bool = False,
    ) -> str:
        await self.ensure_customer_schema()
        embedding_id = str(uuid.uuid4())
        conn = await asyncpg.connect(_postgres_url())
        try:
            await conn.execute(
                """
                INSERT INTO customer.customer_faces (
                    id, customer_id, embedding, embedding_model, quality_score, is_primary, metadata, updated_at
                ) VALUES ($1::uuid, $2::uuid, $3::jsonb, $4, $5, $6, $7::jsonb, NOW())
                """,
                embedding_id,
                customer_id,
                json.dumps(embedding, ensure_ascii=False),
                embedding_model,
                quality_score,
                bool(is_primary),
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        finally:
            await conn.close()
        return embedding_id

    async def find_best_match(self, embedding: List[float]) -> Optional[FaceMatch]:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            rows = await conn.fetch("SELECT customer_id, embedding FROM customer.customer_faces")
        finally:
            await conn.close()

        best_customer_id: Optional[str] = None
        best_score = -1.0
        for row in rows:
            candidate = _json_or_default(row["embedding"], [])
            if not isinstance(candidate, list):
                continue
            score = _cosine_similarity(embedding, candidate)
            if score > best_score:
                best_score = score
                best_customer_id = str(row["customer_id"])
        if not best_customer_id:
            return None
        return FaceMatch(customer_id=best_customer_id, score=best_score)

    async def bind_session(
        self,
        customer_id: str,
        session_id: str,
        source: str = "camera",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            await conn.execute(
                """
                INSERT INTO customer.customer_sessions (id, customer_id, session_id, source, metadata)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb)
                ON CONFLICT (session_id) DO UPDATE SET
                    customer_id = EXCLUDED.customer_id,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata
                """,
                str(uuid.uuid4()),
                customer_id,
                session_id,
                source,
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        finally:
            await conn.close()

    async def touch_customer(self, customer_id: str) -> None:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            await conn.execute(
                "UPDATE customer.customers SET last_seen_at = NOW(), updated_at = NOW() WHERE id = $1::uuid",
                customer_id,
            )
        finally:
            await conn.close()

    async def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            row = await conn.fetchrow(
                "SELECT id, customer_code, first_seen_at, last_seen_at, metadata FROM customer.customers WHERE id = $1::uuid",
                customer_id,
            )
        finally:
            await conn.close()
        if not row:
            return None
        return {
            "customer_id": str(row["id"]),
            "customer_code": row["customer_code"],
            "first_seen_at": _to_iso(row["first_seen_at"]),
            "last_seen_at": _to_iso(row["last_seen_at"]),
            "metadata": _json_or_default(row["metadata"], {}),
        }

    async def get_customer_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            row = await conn.fetchrow(
                """
                SELECT cs.session_id, cs.source, cs.created_at, cs.metadata, c.id AS customer_id
                FROM customer.customer_sessions cs
                JOIN customer.customers c ON c.id = cs.customer_id
                WHERE cs.session_id = $1
                """,
                session_id,
            )
        finally:
            await conn.close()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "customer_id": str(row["customer_id"]),
            "source": row["source"],
            "created_at": _to_iso(row["created_at"]),
            "metadata": _json_or_default(row["metadata"], {}),
        }

    async def list_customer_sessions(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            rows = await conn.fetch(
                """
                SELECT session_id, source, created_at, metadata
                FROM customer.customer_sessions
                WHERE customer_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2
                """,
                customer_id,
                limit,
            )
        finally:
            await conn.close()
        return [
            {
                "session_id": row["session_id"],
                "source": row["source"],
                "created_at": _to_iso(row["created_at"]),
                "metadata": _json_or_default(row["metadata"], {}),
            }
            for row in rows
        ]

    async def log_identity_event(
        self,
        *,
        customer_id: Optional[str],
        session_id: str,
        event_type: str,
        similarity_score: Optional[float],
        decision: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.ensure_customer_schema()
        conn = await asyncpg.connect(_postgres_url())
        try:
            await conn.execute(
                """
                INSERT INTO customer.customer_identity_events (
                    id, customer_id, session_id, event_type, similarity_score, decision, metadata
                ) VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7::jsonb)
                """,
                str(uuid.uuid4()),
                customer_id,
                session_id,
                event_type,
                similarity_score,
                decision,
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        finally:
            await conn.close()
