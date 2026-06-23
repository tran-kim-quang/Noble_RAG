from threading import Thread
import queue
from queue import Queue
from io import BytesIO
from enum import Enum
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    # Split on sentence-ending punctuation common in Vietnamese & Chinese text:
    # . ! ? … and their fullwidth variants, plus ; : and newlines as weaker breaks.
    _SENTENCE_PATTERN = re.compile(
        r"[^.!?\u3002\uff01\uff1f\u2026;:\n]+"
        r"(?:[.!?\u3002\uff01\uff1f\u2026]+|[;:\n]+|$)",
        re.UNICODE,
    )

    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        #self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        self.max_text_chars = max(0, int(getattr(opt, "TTS_MAX_TEXT_CHARS", 280)))

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        if len(msg) > 0:
            self.msgqueue.put((msg, datainfo))

    def _resolve_chunk_limit(self, datainfo: dict, max_chunk_chars: int | None) -> int:
        if max_chunk_chars is not None:
            try:
                return max(0, int(max_chunk_chars))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid max_chunk_chars=%r, fallback to default=%d",
                    max_chunk_chars,
                    self.max_text_chars,
                )
                return self.max_text_chars
        if isinstance(datainfo, dict):
            tts_opt = datainfo.get("tts")
            if isinstance(tts_opt, dict):
                for key in ("max_text_chars", "max_chars", "chunk_chars"):
                    value = tts_opt.get(key)
                    if value is None:
                        continue
                    try:
                        return max(0, int(value))
                    except (TypeError, ValueError):
                        logger.warning("Invalid tts.%s=%r, fallback to default", key, value)
                        break
        return self.max_text_chars

    def _split_sentence_to_limit(self, sentence: str, max_chars: int) -> list[str]:
        sentence = sentence.strip()
        if not sentence:
            return []
        if max_chars <= 0 or len(sentence) <= max_chars:
            return [sentence]

        parts: list[str] = []
        words = sentence.split()
        if len(words) <= 1:
            for i in range(0, len(sentence), max_chars):
                chunk = sentence[i:i + max_chars].strip()
                if chunk:
                    parts.append(chunk)
            return parts

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                parts.append(current)
                if len(word) <= max_chars:
                    current = word
                    continue
                for i in range(0, len(word), max_chars):
                    piece = word[i:i + max_chars].strip()
                    if piece:
                        parts.append(piece)
                current = ""
        if current:
            parts.append(current)
        return parts

    def split_text_for_tts(self, msg: str, datainfo: dict = {}, max_chunk_chars: int | None = None) -> list[str]:
        datainfo = datainfo or {}
        # Normalize whitespace but preserve single spaces between words
        text = " ".join((msg or "").split())
        if not text:
            return []

        max_chars = self._resolve_chunk_limit(datainfo, max_chunk_chars)
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        sentence_candidates = self._SENTENCE_PATTERN.findall(text)
        sentences = [part.strip() for part in sentence_candidates if part and part.strip()]
        if not sentences:
            # No sentence boundaries found – hard-split on word boundaries
            return self._split_sentence_to_limit(text, max_chars)

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            # If a single sentence is already too long, hard-split it first
            pieces = self._split_sentence_to_limit(sentence, max_chars)
            for piece in pieces:
                if not current:
                    current = piece
                    continue
                # Try merging with a space separator
                merged = f"{current} {piece}"
                if len(merged) <= max_chars:
                    # Still fits – keep accumulating
                    current = merged
                else:
                    # Flush current chunk and start a new one
                    chunks.append(current)
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    def put_msg_txt_chunked(self, msg: str, datainfo: dict = {}, max_chunk_chars: int | None = None):
        datainfo = datainfo or {}
        chunks = self.split_text_for_tts(msg, datainfo=datainfo, max_chunk_chars=max_chunk_chars)
        if not chunks:
            return
        for chunk in chunks:
            self.msgqueue.put((chunk, dict(datainfo)))

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self, quit_event):        
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        self.stop_tts()
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        pass

    def stop_tts(self):
        pass
