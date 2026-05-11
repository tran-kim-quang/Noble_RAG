from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IdentitySnapshot:
    face_id: str
    face_session_key: str
    customer_kind: str
    name: str | None
    age: str | None
    gender: str | None
    source: str | None
    confidence: float | None
    vision_context: dict[str, Any] | None
    lead_state: dict[str, Any] | None
    session_id: str | None


class SharedIdentityStore:
    def __init__(self, db_path: str) -> None:
        resolved = Path(db_path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_profiles (
                    face_id TEXT PRIMARY KEY,
                    face_session_key TEXT NOT NULL,
                    customer_kind TEXT NOT NULL,
                    name TEXT,
                    age TEXT,
                    gender TEXT,
                    source TEXT,
                    confidence REAL,
                    embedding_json TEXT,
                    vision_context_json TEXT,
                    lead_state_json TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_identity_face_session ON identity_profiles(face_session_key)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_identity_session_id ON identity_profiles(session_id)")
            conn.commit()

    def upsert_identity(
        self,
        *,
        face_id: str,
        face_session_key: str,
        customer_kind: str,
        name: str | None = None,
        age: str | None = None,
        gender: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        embedding: list[float] | None = None,
        vision_context: dict[str, Any] | None = None,
        lead_state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM identity_profiles WHERE face_id = ?",
                (face_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO identity_profiles(
                        face_id, face_session_key, customer_kind, name, age, gender, source, confidence,
                        embedding_json, vision_context_json, lead_state_json, session_id, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        face_id,
                        face_session_key,
                        customer_kind,
                        name,
                        age,
                        gender,
                        source,
                        confidence,
                        json.dumps(embedding, ensure_ascii=False) if embedding is not None else None,
                        json.dumps(vision_context, ensure_ascii=False) if vision_context is not None else None,
                        json.dumps(lead_state, ensure_ascii=False) if lead_state is not None else None,
                        session_id,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return

            def pick(new_val: Any, old_key: str) -> Any:
                return new_val if new_val is not None else existing[old_key]

            conn.execute(
                """
                UPDATE identity_profiles
                SET face_session_key = ?,
                    customer_kind = ?,
                    name = ?,
                    age = ?,
                    gender = ?,
                    source = ?,
                    confidence = ?,
                    embedding_json = ?,
                    vision_context_json = ?,
                    lead_state_json = ?,
                    session_id = ?,
                    updated_at = ?
                WHERE face_id = ?
                """,
                (
                    pick(face_session_key, "face_session_key"),
                    pick(customer_kind, "customer_kind"),
                    pick(name, "name"),
                    pick(age, "age"),
                    pick(gender, "gender"),
                    pick(source, "source"),
                    pick(confidence, "confidence"),
                    json.dumps(embedding, ensure_ascii=False) if embedding is not None else existing["embedding_json"],
                    json.dumps(vision_context, ensure_ascii=False)
                    if vision_context is not None
                    else existing["vision_context_json"],
                    json.dumps(lead_state, ensure_ascii=False) if lead_state is not None else existing["lead_state_json"],
                    pick(session_id, "session_id"),
                    now,
                    face_id,
                ),
            )
            conn.commit()

    def load_by_face_session_key(self, face_session_key: str) -> IdentitySnapshot | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM identity_profiles WHERE face_session_key = ? ORDER BY updated_at DESC LIMIT 1",
                (face_session_key,),
            ).fetchone()
        return self._row_to_snapshot(row)

    def load_by_face_id(self, face_id: str) -> IdentitySnapshot | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM identity_profiles WHERE face_id = ?",
                (face_id,),
            ).fetchone()
        return self._row_to_snapshot(row)

    def load_guest_embeddings(self) -> list[tuple[str, list[float], dict[str, Any]]]:
        out: list[tuple[str, list[float], dict[str, Any]]] = []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT face_id, embedding_json, age, gender, updated_at
                FROM identity_profiles
                WHERE customer_kind = 'guest' AND embedding_json IS NOT NULL
                """
            ).fetchall()
        for row in rows:
            try:
                embedding = json.loads(str(row["embedding_json"]))
                if not isinstance(embedding, list) or not embedding:
                    continue
            except Exception:
                continue
            out.append(
                (
                    str(row["face_id"]),
                    [float(v) for v in embedding],
                    {
                        "age": row["age"],
                        "gender": row["gender"],
                        "updated_at": row["updated_at"],
                    },
                )
            )
        return out

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row | None) -> IdentitySnapshot | None:
        if row is None:
            return None

        def parse_json(raw: Any) -> dict[str, Any] | None:
            if raw is None:
                return None
            try:
                parsed = json.loads(str(raw))
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        return IdentitySnapshot(
            face_id=str(row["face_id"]),
            face_session_key=str(row["face_session_key"]),
            customer_kind=str(row["customer_kind"]),
            name=row["name"],
            age=row["age"],
            gender=row["gender"],
            source=row["source"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            vision_context=parse_json(row["vision_context_json"]),
            lead_state=parse_json(row["lead_state_json"]),
            session_id=row["session_id"],
        )
