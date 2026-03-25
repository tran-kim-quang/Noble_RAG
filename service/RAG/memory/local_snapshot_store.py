"""Local text snapshot storage for MVP persistence.

This is a durable fallback/mirror so demo data survives Redis TTLs and
temporary database outages. Each session is stored in a single txt file.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from core.config import get_settings


def _snapshot_dir() -> Path:
    base_dir = Path(get_settings().session_export_dir)
    path = base_dir / "live_snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_session_id(session_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id).strip("_") or "session"


def _snapshot_path(session_id: str) -> Path:
    return _snapshot_dir() / f"{_safe_session_id(session_id)}.txt"


def get_local_snapshot_path(session_id: str) -> str:
    return str(_snapshot_path(session_id).resolve())


def save_local_session_snapshot(
    session_id: str,
    lead_profile: Dict[str, Any],
    session_context: Dict[str, Any],
    chat_history: List[Dict[str, Any]],
) -> str:
    snapshot = {
        "session_id": session_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "lead_profile": lead_profile or {},
        "session_context": session_context or {},
        "chat_history": chat_history or [],
    }

    lines = [
        "NOBLE SALES SESSION SNAPSHOT",
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        "",
        "CHAT HISTORY",
    ]
    if chat_history:
        for idx, message in enumerate(chat_history, 1):
            role = str(message.get("role") or "unknown")
            content = str(message.get("content") or "").strip()
            lines.append(f"{idx}. {role}: {content}")
    else:
        lines.append("(empty)")

    path = _snapshot_path(session_id)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def load_local_session_snapshot(session_id: str) -> Dict[str, Any]:
    path = _snapshot_path(session_id)
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace < 0 or last_brace < first_brace:
        return {}

    try:
        return json.loads(raw[first_brace:last_brace + 1])
    except Exception:
        return {}


def read_local_session_snapshot_text(session_id: str) -> str:
    path = _snapshot_path(session_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def delete_local_session_snapshot(session_id: str) -> bool:
    path = _snapshot_path(session_id)
    if not path.exists():
        return False
    path.unlink()
    return True
