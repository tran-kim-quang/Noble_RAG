"""
STT HTTP API Service
--------------------
Endpoints:
  POST /transcribe         - Upload audio file, receive full transcript
  POST /transcribe/stream  - Upload audio, receive SSE stream of segments
  GET  /health             - Health check
  GET  /models             - Active model/provider info
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Generator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - handled at runtime when provider=whisper
    WhisperModel = None  # type: ignore[assignment]


def _env_bool(name: str, default: str = "false") -> bool:
    value = (os.getenv(name) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_float(name: str, default: str) -> float:
    try:
        return float((os.getenv(name) or default).strip())
    except ValueError:
        return float(default)


# Provider selection: deepgram | whisper | auto
STT_PROVIDER = (os.getenv("STT_PROVIDER") or "auto").strip().lower()

# Whisper config
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_BEAM = int((os.getenv("WHISPER_BEAM") or "5").strip())
WHISPER_BEST_OF = int((os.getenv("WHISPER_BEST_OF") or "2").strip())
WHISPER_TEMPERATURE = _env_float("WHISPER_TEMPERATURE", "0")
WHISPER_VAD_FILTER = _env_bool("WHISPER_VAD_FILTER", "true")
WHISPER_VAD_MIN_SILENCE_MS = int((os.getenv("WHISPER_VAD_MIN_SILENCE_MS") or "250").strip())
WHISPER_COND_PREV_TEXT = _env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")

# Deepgram config
DEEPGRAM_API_KEY = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
DEEPGRAM_API_URL = (os.getenv("DEEPGRAM_API_URL") or "https://api.deepgram.com/v2/listen").strip()
DEEPGRAM_MODEL = (os.getenv("DEEPGRAM_MODEL") or "flux-general-en").strip()
DEEPGRAM_DEFAULT_LANGUAGE = (os.getenv("DEEPGRAM_LANGUAGE") or "").strip()
DEEPGRAM_TIMEOUT_SEC = _env_float("DEEPGRAM_TIMEOUT_SEC", "45")
DEEPGRAM_SMART_FORMAT = _env_bool("DEEPGRAM_SMART_FORMAT", "true")
DEEPGRAM_PUNCTUATE = _env_bool("DEEPGRAM_PUNCTUATE", "true")
DEEPGRAM_UTTERANCES = _env_bool("DEEPGRAM_UTTERANCES", "true")
DEEPGRAM_DIARIZE = _env_bool("DEEPGRAM_DIARIZE", "false")
DEEPGRAM_EOT_THRESHOLD = (os.getenv("DEEPGRAM_EOT_THRESHOLD") or "").strip()
DEEPGRAM_EOT_TIMEOUT_MS = (os.getenv("DEEPGRAM_EOT_TIMEOUT_MS") or "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("whisper-service")


def _resolve_provider() -> str:
    if STT_PROVIDER in {"deepgram", "whisper"}:
        return STT_PROVIDER
    if STT_PROVIDER == "auto":
        return "deepgram" if DEEPGRAM_API_KEY else "whisper"
    log.warning("Unknown STT_PROVIDER='%s', fallback to auto strategy.", STT_PROVIDER)
    return "deepgram" if DEEPGRAM_API_KEY else "whisper"


ACTIVE_PROVIDER = _resolve_provider()

whisper_model: Optional[WhisperModel] = None  # type: ignore[type-arg]
_model_load_time = 0.0

if ACTIVE_PROVIDER == "whisper":
    if WhisperModel is None:
        raise RuntimeError("faster_whisper is not installed, cannot use STT_PROVIDER=whisper")
    log.info(
        "Loading whisper model '%s' on device='%s' compute='%s' ...",
        WHISPER_MODEL_SIZE,
        WHISPER_DEVICE,
        WHISPER_COMPUTE,
    )
    t0 = time.perf_counter()
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    _model_load_time = time.perf_counter() - t0
    log.info("Whisper model loaded in %.2fs", _model_load_time)
else:
    if DEEPGRAM_API_KEY:
        log.info("Using Deepgram STT model='%s'.", DEEPGRAM_MODEL)
    else:
        log.warning("STT provider is Deepgram but DEEPGRAM_API_KEY is empty.")


app = FastAPI(
    title="STT API",
    description="Speech-to-text service with provider switch: whisper/deepgram",
    version="2.0.0",
)

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
    ".mkv",
    ".opus",
    ".aac",
}


class WordToken(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: Optional[list[WordToken]] = None


class TranscribeResponse(BaseModel):
    language: str
    language_probability: float
    duration: float
    text: str
    model: str
    device: str
    segments: list[Segment]
    elapsed_seconds: float


@dataclass
class _Info:
    language: str
    language_probability: float
    duration: float


def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "audio")[1].lower() or ".wav"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = upload.file.read()
        tmp.write(content)
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def _transcribe_whisper(audio_path: str, language: Optional[str], word_timestamps: bool) -> tuple[list[Segment], _Info, str]:
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model is not initialized")

    segments_iter, info = whisper_model.transcribe(
        audio_path,
        language=language or None,
        beam_size=WHISPER_BEAM,
        best_of=WHISPER_BEST_OF,
        temperature=WHISPER_TEMPERATURE,
        vad_filter=WHISPER_VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS},
        condition_on_previous_text=WHISPER_COND_PREV_TEXT,
        word_timestamps=word_timestamps,
    )

    segments: list[Segment] = []
    for i, seg in enumerate(segments_iter):
        words = None
        if word_timestamps and seg.words:
            words = [
                WordToken(
                    word=w.word,
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    probability=round(w.probability, 4),
                )
                for w in seg.words
            ]
        segments.append(
            Segment(
                id=i,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                words=words,
            )
        )

    full_text = " ".join(seg.text for seg in segments if seg.text).strip()
    meta = _Info(
        language=str(getattr(info, "language", "") or ""),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=float(getattr(info, "duration", 0.0) or 0.0),
    )
    return segments, meta, full_text


def _deepgram_request(audio_path: str, language: Optional[str]) -> dict:
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=503, detail="DEEPGRAM_API_KEY is not configured")

    params: dict[str, str] = {
        "model": DEEPGRAM_MODEL,
        "smart_format": str(DEEPGRAM_SMART_FORMAT).lower(),
        "punctuate": str(DEEPGRAM_PUNCTUATE).lower(),
        "utterances": str(DEEPGRAM_UTTERANCES).lower(),
        "diarize": str(DEEPGRAM_DIARIZE).lower(),
    }
    lang = (language or DEEPGRAM_DEFAULT_LANGUAGE).strip()
    if lang:
        params["language"] = lang
    if DEEPGRAM_EOT_THRESHOLD:
        params["eot_threshold"] = DEEPGRAM_EOT_THRESHOLD
    if DEEPGRAM_EOT_TIMEOUT_MS:
        params["eot_timeout_ms"] = DEEPGRAM_EOT_TIMEOUT_MS

    req_url = f"{DEEPGRAM_API_URL}?{urlencode(params)}"
    with open(audio_path, "rb") as f:
        body = f.read()

    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    req = Request(req_url, data=body, method="POST")
    req.add_header("Authorization", f"Token {DEEPGRAM_API_KEY}")
    req.add_header("Content-Type", content_type)
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=DEEPGRAM_TIMEOUT_SEC) as resp:
            payload = resp.read().decode("utf-8", errors="ignore")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise HTTPException(status_code=e.code, detail=f"Deepgram request failed: {detail}")
    except URLError as e:
        raise HTTPException(status_code=502, detail=f"Deepgram network error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deepgram unknown error: {e}")

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Deepgram returned invalid JSON")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _word_token_from_deepgram(item: dict) -> WordToken:
    word = str(item.get("punctuated_word") or item.get("word") or "").strip()
    return WordToken(
        word=word,
        start=round(_safe_float(item.get("start"), 0.0), 3),
        end=round(_safe_float(item.get("end"), 0.0), 3),
        probability=round(_safe_float(item.get("confidence"), 0.0), 4),
    )


def _transcribe_deepgram(audio_path: str, language: Optional[str], word_timestamps: bool) -> tuple[list[Segment], _Info, str]:
    payload = _deepgram_request(audio_path, language)
    results = payload.get("results") or {}
    metadata = payload.get("metadata") or {}
    channels = results.get("channels") or []
    channel0 = channels[0] if channels else {}
    alternatives = channel0.get("alternatives") or []
    alt0 = alternatives[0] if alternatives else {}

    transcript = str(alt0.get("transcript") or "").strip()
    alt_words_raw = alt0.get("words") or []
    alt_words = [_word_token_from_deepgram(w) for w in alt_words_raw if isinstance(w, dict)]

    segments: list[Segment] = []
    utterances = results.get("utterances") or []
    if isinstance(utterances, list):
        for idx, utt in enumerate(utterances):
            if not isinstance(utt, dict):
                continue
            text = str(utt.get("transcript") or "").strip()
            if not text:
                continue
            words = None
            if word_timestamps:
                words_raw = utt.get("words") or []
                words = [_word_token_from_deepgram(w) for w in words_raw if isinstance(w, dict)]
            segments.append(
                Segment(
                    id=idx,
                    start=round(_safe_float(utt.get("start"), 0.0), 3),
                    end=round(_safe_float(utt.get("end"), 0.0), 3),
                    text=text,
                    words=words,
                )
            )

    if not segments and transcript:
        start = alt_words[0].start if alt_words else 0.0
        end = alt_words[-1].end if alt_words else _safe_float(metadata.get("duration"), 0.0)
        words = alt_words if word_timestamps and alt_words else None
        segments.append(
            Segment(
                id=0,
                start=round(start, 3),
                end=round(end, 3),
                text=transcript,
                words=words,
            )
        )

    if segments:
        full_text = " ".join(seg.text for seg in segments if seg.text).strip()
    else:
        full_text = transcript

    language_detected = str(
        channel0.get("detected_language") or channel0.get("language") or language or DEEPGRAM_DEFAULT_LANGUAGE or "unknown"
    )
    language_probability = _safe_float(channel0.get("language_confidence"), 0.0)
    duration = _safe_float(metadata.get("duration"), 0.0)
    if duration <= 0.0 and segments:
        duration = segments[-1].end

    meta = _Info(
        language=language_detected,
        language_probability=language_probability,
        duration=duration,
    )
    return segments, meta, full_text


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": ACTIVE_PROVIDER,
        "model": WHISPER_MODEL_SIZE if ACTIVE_PROVIDER == "whisper" else DEEPGRAM_MODEL,
        "device": WHISPER_DEVICE if ACTIVE_PROVIDER == "whisper" else "remote",
        "compute_type": WHISPER_COMPUTE if ACTIVE_PROVIDER == "whisper" else "api",
        "model_load_seconds": round(_model_load_time, 2),
    }


@app.get("/models")
async def list_models():
    if ACTIVE_PROVIDER == "whisper":
        return {
            "provider": "whisper",
            "loaded": WHISPER_MODEL_SIZE,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE,
        }
    return {
        "provider": "deepgram",
        "loaded": DEEPGRAM_MODEL,
        "api_url": DEEPGRAM_API_URL,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
):
    tmp_path = _save_upload(file)
    try:
        t0 = time.perf_counter()
        if ACTIVE_PROVIDER == "deepgram":
            segments, info, full_text = _transcribe_deepgram(tmp_path, language, word_timestamps)
            model_name = f"deepgram:{DEEPGRAM_MODEL}"
            device_name = "remote"
        else:
            segments, info, full_text = _transcribe_whisper(tmp_path, language, word_timestamps)
            model_name = WHISPER_MODEL_SIZE
            device_name = WHISPER_DEVICE
        elapsed = time.perf_counter() - t0

        log.info(
            "Transcribed '%s' provider=%s lang=%s (%.1fs audio) in %.2fs",
            file.filename,
            ACTIVE_PROVIDER,
            info.language,
            info.duration,
            elapsed,
        )

        return TranscribeResponse(
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration=round(info.duration, 3),
            text=full_text,
            model=model_name,
            device=device_name,
            segments=segments,
            elapsed_seconds=round(elapsed, 3),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/transcribe/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
):
    """
    Streams segments as SSE.
    For Deepgram provider, this endpoint performs a single API call and emits
    segment events from returned utterances.
    """
    tmp_path = _save_upload(file)

    def generate() -> Generator[str, None, None]:
        try:
            if ACTIVE_PROVIDER == "deepgram":
                segments, info, _ = _transcribe_deepgram(tmp_path, language, word_timestamps)
                meta = json.dumps(
                    {
                        "event": "meta",
                        "provider": "deepgram",
                        "language": info.language,
                        "language_probability": round(info.language_probability, 4),
                        "duration": round(info.duration, 3),
                    },
                    ensure_ascii=False,
                )
                yield f"data: {meta}\n\n"
                for seg in segments:
                    payload = json.dumps(
                        {
                            "event": "segment",
                            "id": seg.id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "words": [w.model_dump() for w in seg.words] if seg.words else None,
                        },
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
                yield 'data: {"event": "done"}\n\n'
                return

            if whisper_model is None:
                raise RuntimeError("Whisper model is not initialized")

            segments_iter, info = whisper_model.transcribe(
                tmp_path,
                language=language or None,
                beam_size=WHISPER_BEAM,
                word_timestamps=word_timestamps,
            )
            meta = json.dumps(
                {
                    "event": "meta",
                    "provider": "whisper",
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration": round(info.duration, 3),
                },
                ensure_ascii=False,
            )
            yield f"data: {meta}\n\n"

            for i, seg in enumerate(segments_iter):
                words = None
                if word_timestamps and seg.words:
                    words = [
                        {
                            "word": w.word,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4),
                        }
                        for w in seg.words
                    ]
                payload = json.dumps(
                    {
                        "event": "segment",
                        "id": i,
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "text": seg.text.strip(),
                        "words": words,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
            yield 'data: {"event": "done"}\n\n'

        except Exception as exc:
            err = json.dumps({"event": "error", "detail": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
