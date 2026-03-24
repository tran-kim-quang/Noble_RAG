def validate_session_id(session_id: str) -> str:
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")
    return session_id.strip()[:128]


def validate_non_empty(text: str, field: str = "field") -> str:
    if not text or not text.strip():
        raise ValueError(f"{field} cannot be empty")
    return text.strip()
