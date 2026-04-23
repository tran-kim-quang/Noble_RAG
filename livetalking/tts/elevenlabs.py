import json
import os
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from registry import register
from utils.logger import logger
from .base_tts import BaseTTS, State


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@register("tts", "elevenlabs")
class ElevenLabsTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        self.voice_id = (os.getenv("ELEVEN_VOICE") or "").strip()
        self.model_id = (os.getenv("ELEVEN_MODEL") or "eleven_multilingual_v2").strip()
        self.language_code = (os.getenv("ELEVEN_LANG") or "").strip()
        self.stability = float((os.getenv("ELEVEN_STABILITY") or "0.45").strip())
        self.similarity_boost = float((os.getenv("ELEVEN_SIMILARITY_BOOST") or "0.8").strip())
        self.speed = float((os.getenv("ELEVEN_SPEED") or "1.0").strip())
        self.use_speaker_boost = _env_bool("ELEVEN_USE_SPEAKER_BOOST", True)
        self.timeout_sec = float((os.getenv("ELEVEN_TIMEOUT_SEC") or "35").strip())
        # 16k raw PCM is easiest for avatar pipeline (no decode/resample overhead).
        self.output_format = (os.getenv("ELEVEN_OUTPUT_FORMAT") or "pcm_16000").strip()
        self._quota_exceeded = False
        self._last_quota_log_time = 0.0
        self.read_block_bytes = int((os.getenv("ELEVEN_READ_BLOCK_BYTES") or "4096").strip())

    def _build_request(self, text: str) -> Request | None:
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is empty. Cannot synthesize speech.")
            return None
        if not self.voice_id:
            logger.error("ELEVEN_VOICE is empty. Cannot synthesize speech.")
            return None

        query = urlencode({"output_format": self.output_format})
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?{query}"

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "speed": self.speed,
                "use_speaker_boost": self.use_speaker_boost,
            },
        }
        if self.language_code:
            payload["language_code"] = self.language_code

        return Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            },
            method="POST",
        )

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        if self._quota_exceeded:
            now = time.time()
            if now - self._last_quota_log_time > 10:
                logger.error(
                    "elevenlabs quota exceeded; skipping TTS requests until service restart or account quota top-up."
                )
                self._last_quota_log_time = now
            return

        req = self._build_request(text)
        if req is None:
            return

        t0 = time.time()
        first_audio_latency = None
        carry = b""
        stream = np.empty((0,), dtype=np.float32)
        sent_any = False
        sent_end = False
        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:
                while self.state == State.RUNNING:
                    chunk_bytes = resp.read(max(1024, self.read_block_bytes))
                    if not chunk_bytes:
                        break

                    if first_audio_latency is None:
                        first_audio_latency = time.time() - t0
                        logger.info("-------elevenlabs ttfb:%.4fs", first_audio_latency)

                    raw = carry + chunk_bytes
                    usable = len(raw) - (len(raw) % 2)
                    carry = raw[usable:]
                    if usable <= 0:
                        continue

                    pcm = np.frombuffer(raw[:usable], dtype=np.int16).astype(np.float32) / 32767.0
                    if pcm.size <= 0:
                        continue
                    if stream.size <= 0:
                        stream = pcm
                    else:
                        stream = np.concatenate((stream, pcm))

                    while stream.shape[0] >= self.chunk and self.state == State.RUNNING:
                        frame = stream[: self.chunk]
                        stream = stream[self.chunk :]
                        eventpoint = {"status": "start", "text": text} if not sent_any else {"text": text}
                        eventpoint.update(**textevent)
                        self.parent.put_audio_frame(frame, eventpoint)
                        sent_any = True

            if carry and self.state == State.RUNNING:
                # Keep sample alignment on odd-byte tail.
                padded = carry + (b"\x00" if len(carry) % 2 == 1 else b"")
                tail_pcm = np.frombuffer(padded, dtype=np.int16).astype(np.float32) / 32767.0
                if tail_pcm.size > 0:
                    if stream.size <= 0:
                        stream = tail_pcm
                    else:
                        stream = np.concatenate((stream, tail_pcm))
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            logger.error(
                "elevenlabs auth/request failed: status=%s voice=%s model=%s body=%s",
                e.code,
                self.voice_id,
                self.model_id,
                body[:500],
            )
            if e.code == 401 and "quota_exceeded" in body:
                self._quota_exceeded = True
                self._last_quota_log_time = time.time()
            return
        except Exception:
            logger.exception("elevenlabs tts failed")
            return

        if self.state != State.RUNNING:
            return

        if stream.shape[0] > 0:
            if stream.shape[0] < self.chunk:
                stream = np.pad(stream, (0, self.chunk - stream.shape[0]))
            eventpoint = {"text": text}
            eventpoint["status"] = "start" if not sent_any else "end"
            if eventpoint["status"] == "end":
                sent_end = True
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[: self.chunk], eventpoint)
            sent_any = True
            stream = stream[self.chunk :]

        if sent_any and not sent_end:
            # Emit a tiny trailing end frame when we already sent start-only chunks.
            end_event = {"status": "end", "text": text}
            end_event.update(**textevent)
            self.parent.put_audio_frame(np.zeros((self.chunk,), dtype=np.float32), end_event)

        logger.info("-------elevenlabs tts time:%.4fs", time.time() - t0)
        if not sent_any:
            logger.error("elevenlabs tts returned empty audio")
