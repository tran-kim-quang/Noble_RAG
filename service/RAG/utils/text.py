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
