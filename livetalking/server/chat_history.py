from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from utils.logger import logger


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _isoformat(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="seconds")


def _append_unique(items: list[str], value: str) -> None:
    cleaned = _clean_text(value)
    if cleaned and cleaned not in items:
        items.append(cleaned)


def _render_text_block(text: str) -> str:
    content = str(text or "").rstrip()
    if not content:
        return "_(empty)_"
    return f"~~~text\n{content}\n~~~"


@dataclass
class ChatHistoryMessage:
    role: str
    text: str
    created_at: datetime
    raw_transcript: str | None = None


@dataclass
class ChatHistorySession:
    sessionid: int
    created_at: datetime
    avatar_id: str = ""
    model: str = ""
    transport: str = ""
    rag_session_ids: list[str] = field(default_factory=list)
    messages: list[ChatHistoryMessage] = field(default_factory=list)
    ended_at: datetime | None = None
    end_reason: str = ""
    file_path: Path | None = None


class ChatHistoryManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[int, ChatHistorySession] = {}
        self._base_dir = Path(__file__).resolve().parent.parent / "chat_history"

    def _derive_record_metadata(
        self,
        avatar_session: Any = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        params = params or {}
        opt = getattr(avatar_session, "opt", None)
        avatar_id = _clean_text(getattr(opt, "avatar_id", "")) or _clean_text(params.get("avatar"))
        model = _clean_text(getattr(opt, "model", ""))
        transport = _clean_text(getattr(opt, "transport", ""))
        return {
            "avatar_id": avatar_id,
            "model": model,
            "transport": transport,
        }

    def _ensure_session_locked(
        self,
        sessionid: int,
        avatar_session: Any = None,
        params: dict[str, Any] | None = None,
    ) -> ChatHistorySession:
        record = self._sessions.get(sessionid)
        if record is None:
            metadata = self._derive_record_metadata(avatar_session=avatar_session, params=params)
            record = ChatHistorySession(
                sessionid=sessionid,
                created_at=_now_local(),
                avatar_id=metadata["avatar_id"],
                model=metadata["model"],
                transport=metadata["transport"],
            )
            self._sessions[sessionid] = record
            return record

        metadata = self._derive_record_metadata(avatar_session=avatar_session, params=params)
        if metadata["avatar_id"] and not record.avatar_id:
            record.avatar_id = metadata["avatar_id"]
        if metadata["model"] and not record.model:
            record.model = metadata["model"]
        if metadata["transport"] and not record.transport:
            record.transport = metadata["transport"]
        return record

    def _track_rag_session_locked(self, record: ChatHistorySession, datainfo: dict[str, Any] | None = None) -> None:
        datainfo = datainfo or {}
        _append_unique(record.rag_session_ids, datainfo.get("rag_session_id", ""))
        _append_unique(record.rag_session_ids, datainfo.get("session_id", ""))

    def _build_file_path_locked(self, record: ChatHistorySession) -> Path:
        timestamp = record.created_at.strftime("%Y%m%d_%H%M%S")
        return self._base_dir / f"{timestamp}_session_{record.sessionid}.md"

    def _render_markdown_locked(self, record: ChatHistorySession) -> str:
        lines = [
            f"# LiveTalking Session {record.sessionid}",
            "",
            f"- Started at: {_isoformat(record.created_at)}",
        ]
        if record.ended_at is not None:
            lines.append(f"- Ended at: {_isoformat(record.ended_at)}")
        if record.end_reason:
            lines.append(f"- End reason: {record.end_reason}")
        if record.avatar_id:
            lines.append(f"- Avatar ID: {record.avatar_id}")
        if record.model:
            lines.append(f"- Model: {record.model}")
        if record.transport:
            lines.append(f"- Transport: {record.transport}")
        if record.rag_session_ids:
            lines.append(f"- RAG Session IDs: {', '.join(record.rag_session_ids)}")

        lines.extend(["", "## Conversation", ""])

        if not record.messages:
            lines.append("_No chat messages recorded._")
            lines.append("")
            return "\n".join(lines)

        role_labels = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }
        for idx, message in enumerate(record.messages, start=1):
            role_label = role_labels.get(message.role, message.role.title())
            lines.append(f"### {idx}. {role_label}")
            lines.append(f"- Time: {_isoformat(message.created_at)}")
            lines.append("")
            if (
                message.raw_transcript
                and _clean_text(message.raw_transcript)
                and _clean_text(message.raw_transcript) != _clean_text(message.text)
            ):
                lines.append("Raw transcript")
                lines.append("")
                lines.append(_render_text_block(message.raw_transcript))
                lines.append("")
            lines.append(_render_text_block(message.text))
            lines.append("")
        return "\n".join(lines)

    def _write_markdown(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    def start_session(
        self,
        sessionid: int,
        avatar_session: Any = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._sessions.get(sessionid)
            if record is not None and record.ended_at is not None:
                self._sessions.pop(sessionid, None)
            self._ensure_session_locked(sessionid, avatar_session=avatar_session, params=params)

    def add_message(
        self,
        sessionid: int,
        role: str,
        text: str,
        datainfo: dict[str, Any] | None = None,
    ) -> None:
        clean_text = _clean_text(text)
        if not clean_text:
            return

        rewrite_path: Path | None = None
        rewrite_content = ""
        with self._lock:
            record = self._ensure_session_locked(sessionid)
            self._track_rag_session_locked(record, datainfo)
            raw_transcript = ""
            if role == "user":
                raw_transcript = _clean_text((datainfo or {}).get("raw_transcript", ""))
            record.messages.append(
                ChatHistoryMessage(
                    role=_clean_text(role) or "system",
                    text=clean_text,
                    created_at=_now_local(),
                    raw_transcript=raw_transcript or None,
                )
            )
            if record.file_path is None:
                record.file_path = self._build_file_path_locked(record)
            rewrite_path = record.file_path
            rewrite_content = self._render_markdown_locked(record)

        if rewrite_path is not None:
            self._write_markdown(rewrite_path, rewrite_content)

    def finalize_session(self, sessionid: int, reason: str = "") -> Path | None:
        output_path: Path | None = None
        output_content = ""

        with self._lock:
            record = self._sessions.get(sessionid)
            if record is None:
                return None

            if record.ended_at is None:
                record.ended_at = _now_local()
            if reason and not record.end_reason:
                record.end_reason = _clean_text(reason)
            if not record.messages:
                return None

            if record.file_path is None:
                record.file_path = self._build_file_path_locked(record)
            output_path = record.file_path
            output_content = self._render_markdown_locked(record)

        self._write_markdown(output_path, output_content)
        logger.info("Saved chat history: session=%s path=%s", sessionid, output_path)
        return output_path


chat_history_manager = ChatHistoryManager()
