import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences on common Vietnamese/Latin sentence endings."""
    parts = re.split(r'(?<=[.!?。！？\n])\s*', text)
    return [s.strip() for s in parts if s.strip()]


def truncate(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
