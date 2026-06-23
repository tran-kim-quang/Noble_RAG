import http.client
import json
import os
import ssl
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import resampy
import soundfile as sf

from registry import register
from utils.logger import logger
from .base_tts import BaseTTS, State


_ENV_LOADED = False


def _load_env_file():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    _load_env_file()
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %.2f", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@register("tts", "elevenlabs")
class ElevenLabsTTS(BaseTTS):
    # Transient network / SSL errors worth retrying
    _RETRYABLE = (
        ssl.SSLError,
        ConnectionResetError,
        ConnectionAbortedError,
        http.client.RemoteDisconnected,
        TimeoutError,
    )

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.api_key = _env("ELEVENLABS_API_KEY")
        self.voice_id = _env("ELEVENLABS_VOICE_ID")
        self.model_id = (
            _env("ELEVENLABS_MODEL_ID")
            or _env("ELVENLABS_MODEL_ID")
            or "eleven_multilingual_v2"
        )
        self.output_format = _env("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
        self.timeout = _env_float("ELEVENLABS_TIMEOUT_SEC", 60.0)
        self.max_retries = max(1, int(_env("ELEVENLABS_MAX_RETRIES") or "3"))
        self.voice_settings = {
            "stability": _env_float("ELEVENLABS_STABILITY", 0.5),
            "similarity_boost": _env_float("ELEVENLABS_SIMILARITY_BOOST", 0.75),
            "style": _env_float("ELEVENLABS_STYLE", 0.0),
            "use_speaker_boost": _env_bool("ELEVENLABS_USE_SPEAKER_BOOST", True),
            "speed": _env_float("ELEVENLABS_SPEED", 1.0),
        }

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        text = " ".join((text or "").split())
        if not text:
            return

        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is empty")
            return

        event_tts = textevent.get("tts", {})
        voice_id = event_tts.get("voice_id") or event_tts.get("ref_file") or self.voice_id
        model_id = event_tts.get("model_id") or self.model_id
        if not voice_id:
            logger.error("ELEVENLABS_VOICE_ID is empty")
            return

        # Use streaming endpoint for PCM (same sample rate) – avatar starts speaking
        # within ~1-2 s instead of waiting for full audio to be generated.
        # Fall back to batch mode for other formats that need full decode (mp3, ogg...).
        use_stream = (
            self.output_format.startswith("pcm_")
            and self._pcm_sample_rate() == self.sample_rate
        )

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if use_stream:
                    self._stream_pcm_to_avatar(text, voice_id, model_id, textevent)
                else:
                    start = time.perf_counter()
                    audio_bytes = self._create_speech(text, voice_id, model_id)
                    stream = self._audio_bytes_to_stream(audio_bytes)
                    logger.info("-------elevenlabs tts time:%.4fs", time.perf_counter() - start)
                    self._push_audio_stream(stream, text, textevent)
                return  # success

            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                logger.error("elevenlabs HTTPError %s: %s", exc.code, detail[:500])
                # Only retry on rate-limit or server errors; fail fast on client errors
                if exc.code not in (429, 500, 502, 503, 504):
                    return
                last_exc = exc

            except Exception as exc:
                # Catches ssl.SSLError, ConnectionResetError, URLError, TimeoutError, etc.
                # Note: except (URLError, *tuple) is NOT valid Python 3.10 syntax,
                # so we catch all exceptions here and retry on every network/SSL error.
                last_exc = exc
                logger.warning(
                    "elevenlabs attempt %d/%d – %s: %s",
                    attempt, self.max_retries, type(exc).__name__, exc,
                )

            if attempt < self.max_retries:
                delay = 2 ** (attempt - 1)  # 1s → 2s → 4s
                logger.warning("elevenlabs retrying in %.0fs…", delay)
                time.sleep(delay)

        logger.error(
            "elevenlabs gave up after %d attempts. Last: %s: %s",
            self.max_retries, type(last_exc).__name__, last_exc,
        )

    def _build_tts_request(self, text: str, voice_id: str, model_id: str, stream: bool = False) -> Request:
        """Build a Request for the ElevenLabs TTS (batch or streaming) endpoint."""
        endpoint = "stream" if stream else ""
        path = f"text-to-speech/{voice_id}" + (f"/{endpoint}" if endpoint else "")
        query = urlencode({"output_format": self.output_format})
        url = f"https://api.elevenlabs.io/v1/{path}?{query}"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": self.voice_settings,
        }
        return Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/octet-stream",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            },
            method="POST",
        )

    def _stream_pcm_to_avatar(self, text: str, voice_id: str, model_id: str, textevent: dict):
        """Stream raw PCM audio from ElevenLabs directly into the avatar frame queue.

        Audio frames are pushed as soon as the first bytes arrive from the API,
        reducing first-audio latency from 15-25 s (batch) to ~1-2 s (streaming).
        Only works with PCM output formats (pcm_16000, pcm_22050, etc.).
        """
        req = self._build_tts_request(text, voice_id, model_id, stream=True)
        bytes_per_frame = self.chunk * 2  # int16 → 2 bytes per sample
        read_size = bytes_per_frame * 8   # read ~8 frames at a time

        t_start = time.perf_counter()
        first_frame = True
        buf = b""

        with urlopen(req, timeout=self.timeout) as response:
            while self.state == State.RUNNING:
                data = response.read(read_size)
                if not data:
                    break
                buf += data

                # Push every complete frame immediately
                while len(buf) >= bytes_per_frame and self.state == State.RUNNING:
                    frame_bytes, buf = buf[:bytes_per_frame], buf[bytes_per_frame:]
                    frame = (
                        np.frombuffer(frame_bytes, dtype=np.int16)
                        .astype(np.float32) / 32767.0
                    )
                    eventpoint = {}
                    if first_frame:
                        logger.info(
                            "-------elevenlabs stream first-byte latency:%.4fs",
                            time.perf_counter() - t_start,
                        )
                        eventpoint = {"status": "start", "text": text}
                        first_frame = False
                    eventpoint.update(**textevent)
                    self.parent.put_audio_frame(frame, eventpoint)

        # Flush any remaining partial frame
        if buf and self.state == State.RUNNING:
            if len(buf) % 2:
                buf = buf[:-1]
            if buf:
                raw = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32767.0
                padded = np.zeros(self.chunk, dtype=np.float32)
                padded[:min(len(raw), self.chunk)] = raw[:self.chunk]
                self.parent.put_audio_frame(padded, {})

        # End-of-speech signal
        if self.state == State.RUNNING:
            eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, dtype=np.float32), eventpoint)

        logger.info("-------elevenlabs stream total:%.4fs", time.perf_counter() - t_start)

    def _create_speech(self, text: str, voice_id: str, model_id: str) -> bytes:
        """Batch mode: fetch full audio then return. Used for non-PCM formats."""
        req = self._build_tts_request(text, voice_id, model_id, stream=False)
        with urlopen(req, timeout=self.timeout) as response:
            return response.read()

    def _audio_bytes_to_stream(self, audio_bytes: bytes) -> np.ndarray:
        if not audio_bytes:
            return np.array([], dtype=np.float32)

        if self.output_format.startswith("pcm_"):
            sample_rate = self._pcm_sample_rate()
            if len(audio_bytes) % 2:
                audio_bytes = audio_bytes[:-1]
            stream = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            byte_stream = BytesIO(audio_bytes)
            stream, sample_rate = sf.read(byte_stream)
            stream = stream.astype(np.float32)
            if stream.ndim > 1:
                stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(
                "[WARN] audio sample rate is %s, resampling into %s.",
                sample_rate,
                self.sample_rate,
            )
            stream = resampy.resample(
                x=stream,
                sr_orig=sample_rate,
                sr_new=self.sample_rate,
            )
        return stream.astype(np.float32, copy=False)

    def _pcm_sample_rate(self) -> int:
        parts = self.output_format.split("_", 1)
        if len(parts) != 2:
            return self.sample_rate
        try:
            return int(parts[1])
        except ValueError:
            return self.sample_rate

    def _push_audio_stream(self, stream: np.ndarray, text: str, textevent: dict):
        if stream.shape[0] <= 0:
            logger.error("elevenlabs returned empty audio")
            return

        idx = 0
        first = True
        while idx < stream.shape[0] and self.state == State.RUNNING:
            frame = stream[idx:idx + self.chunk]
            idx += self.chunk
            if frame.shape[0] < self.chunk:
                padded = np.zeros(self.chunk, dtype=np.float32)
                padded[:frame.shape[0]] = frame
                frame = padded

            eventpoint = {}
            if first:
                eventpoint = {"status": "start", "text": text}
                first = False
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(frame, eventpoint)

        if self.state == State.RUNNING:
            eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(np.zeros(self.chunk, dtype=np.float32), eventpoint)
