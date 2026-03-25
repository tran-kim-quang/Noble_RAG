"""Export a completed sales session to a txt file."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from core.config import get_settings


def export_session_to_txt(
    session_id: str,
    lead_profile: Dict[str, Any],
    session_context: Dict[str, Any],
    chat_history: List[Dict[str, Any]],
) -> str:
    export_dir = Path(get_settings().session_export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_session_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", session_id).strip("_") or "session"
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    export_path = export_dir / f"{safe_session_id}_{timestamp}.txt"

    lines: List[str] = [
        "NOBLE SALES SESSION EXPORT",
        f"session_id: {session_id}",
        f"exported_at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "LEAD PROFILE",
        json.dumps(lead_profile or {}, ensure_ascii=False, indent=2),
        "",
        "SESSION CONTEXT",
        json.dumps(session_context or {}, ensure_ascii=False, indent=2),
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

    export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(export_path.resolve())
