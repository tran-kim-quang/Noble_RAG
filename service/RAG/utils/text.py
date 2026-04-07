import re
from typing import Iterator, List


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


def normalize_user_text(text: str) -> str:
    """Normalize incoming user text to reduce encoding/transport artifacts.

    This function keeps behavior conservative:
    - trims control chars and collapses whitespace
    - removes replacement-char artifacts
    - attempts common mojibake repair (latin1/cp1252 -> utf-8)
    """
    raw = str(text or "")
    if not raw:
        return ""

    # Remove control chars except newline/tab, then normalize whitespace.
    raw = "".join(ch for ch in raw if ch in "\n\t" or ord(ch) >= 32)
    raw = raw.replace("\ufffd", "")
    normalized = normalize_whitespace(raw)
    if not normalized:
        return ""

    def _repair(candidate: str) -> str:
        for enc in ("latin1", "cp1252"):
            try:
                repaired = candidate.encode(enc).decode("utf-8")
            except Exception:
                continue
            repaired = normalize_whitespace(repaired)
            # Accept only when common mojibake markers are reduced.
            if repaired and (repaired.count("?") <= candidate.count("?") and repaired != candidate):
                return repaired
        return candidate

    return _repair(normalized)


def iter_stream_chunks(full_answer: str) -> Iterator[str]:
    text = (full_answer or "").strip()
    if not text:
        return

    if "\n" in text:
        for line in text.splitlines():
            clean = line.strip()
            if clean:
                yield clean
        return

    for sent in re.split(r"(?<=[.!?])\s+", text):
        clean = sent.strip()
        if clean:
            yield clean
