from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any

log = logging.getLogger("sales-session-store")


@dataclass(frozen=True)
class SessionKeys:
    face_key: str
    session_key: str


class SessionStore:
    def __init__(self, redis_url: str, namespace: str = "noble_rag") -> None:
        self.namespace = namespace.strip() or "noble_rag"
        self._redis = None
        self._memory_face: dict[str, tuple[float, str]] = {}
        self._memory_session: dict[str, tuple[float, dict[str, Any]]] = {}
        if not redis_url:
            return
        try:
            import redis

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis = client
        except Exception as exc:  # pragma: no cover - defensive fallback for local envs
            log.warning("redis unavailable, fallback to in-memory session store: %s", exc)

    def _face_storage_key(self, face_session_key: str) -> str:
        return f"{self.namespace}:face:{face_session_key}"

    def _session_storage_key(self, session_id: str) -> str:
        return f"{self.namespace}:session:{session_id}"

    def load_by_face_key(self, face_session_key: str) -> dict[str, Any] | None:
        now = time.time()
        if self._redis is not None:
            raw = self._redis.get(self._face_storage_key(face_session_key))
            if not raw:
                return None
            return json.loads(raw)
        payload = self._memory_face.get(face_session_key)
        if payload is None:
            return None
        expires_at, raw = payload
        if expires_at <= now:
            self._memory_face.pop(face_session_key, None)
            return None
        return json.loads(raw)

    def load_by_session_id(self, session_id: str) -> dict[str, Any] | None:
        now = time.time()
        if self._redis is not None:
            raw = self._redis.get(self._session_storage_key(session_id))
            if not raw:
                return None
            return json.loads(raw)
        payload = self._memory_session.get(session_id)
        if payload is None:
            return None
        expires_at, record = payload
        if expires_at <= now:
            self._memory_session.pop(session_id, None)
            return None
        return json.loads(json.dumps(record))

    def save(self, record: dict[str, Any], ttl_sec: int) -> None:
        ttl = max(1, int(ttl_sec))
        face_session_key = str(record.get("face_session_key", "")).strip()
        session_id = str(record.get("session_id", "")).strip()
        if not face_session_key or not session_id:
            raise ValueError("record must contain face_session_key and session_id")

        encoded = json.dumps(record, ensure_ascii=False)
        if self._redis is not None:
            self._redis.setex(self._face_storage_key(face_session_key), ttl, encoded)
            self._redis.setex(self._session_storage_key(session_id), ttl, encoded)
            return

        expires_at = time.time() + ttl
        self._memory_face[face_session_key] = (expires_at, encoded)
        self._memory_session[session_id] = (expires_at, record)

    def delete(
        self,
        *,
        session_id: str | None = None,
        face_session_key: str | None = None,
    ) -> SessionKeys | None:
        resolved_face_key = str(face_session_key or "").strip() or None
        resolved_session_id = str(session_id or "").strip() or None

        if resolved_face_key is None and resolved_session_id is not None:
            record = self.load_by_session_id(resolved_session_id)
            if record is not None:
                resolved_face_key = str(record.get("face_session_key", "")).strip() or None

        if resolved_session_id is None and resolved_face_key is not None:
            record = self.load_by_face_key(resolved_face_key)
            if record is not None:
                resolved_session_id = str(record.get("session_id", "")).strip() or None

        if resolved_face_key is None and resolved_session_id is None:
            return None

        if self._redis is not None:
            if resolved_face_key is not None:
                self._redis.delete(self._face_storage_key(resolved_face_key))
            if resolved_session_id is not None:
                self._redis.delete(self._session_storage_key(resolved_session_id))
        else:
            if resolved_face_key is not None:
                self._memory_face.pop(resolved_face_key, None)
            if resolved_session_id is not None:
                self._memory_session.pop(resolved_session_id, None)

        return SessionKeys(
            face_key=resolved_face_key or "",
            session_key=resolved_session_id or "",
        )
