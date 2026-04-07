"""Faster-Whisper STT HTTP API Service."""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from typing import Generator, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "5"))

MODEL_SIZE = WHISPER_MODEL_NAME
DEVICE = WHISPER_DEVICE
COMPUTE_TYPE = WHISPER_COMPUTE

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("whisper-service")

_app_start = time.perf_counter()
_model: Optional[WhisperModel] = None
_model_lock = threading.Lock()

app = FastAPI(
    title="Faster-Whisper STT API",
    description="Speech-to-text service powered by faster-whisper",
    version="2.2.0",
)

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac",
    ".webm", ".mkv", ".opus", ".aac",
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
    text: str
    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    segments: list[Segment]
    elapsed_seconds: float


def _get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        log.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            WHISPER_MODEL_NAME,
            WHISPER_DEVICE,
            WHISPER_COMPUTE,
        )
        _model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        return _model


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


def _audio_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore").strip()
        return round(float(out), 3) if out else 0.0
    except Exception:
        return 0.0


def _normalize_transcript_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    raw = raw.replace("\ufffd", "")
    raw = " ".join(raw.split())
    return raw


def _run_whisper_transcribe(
    audio_path: str,
    language: Optional[str],
    word_timestamps: bool,
    vad_filter: Optional[bool],
) -> tuple[list[Segment], str, float, str, float]:
    model = _get_model()
    segments_iter, info = model.transcribe(
        audio=audio_path,
        language=(language or None),
        beam_size=max(1, WHISPER_BEAM),
        word_timestamps=bool(word_timestamps),
        vad_filter=(False if vad_filter is None else bool(vad_filter)),
    )

    segments: list[Segment] = []
    full_text_parts: list[str] = []

    for idx, seg in enumerate(segments_iter):
        txt = _normalize_transcript_text(seg.text)
        if txt:
            full_text_parts.append(txt)
        words = None
        if word_timestamps and getattr(seg, "words", None):
            words = [
                WordToken(
                    word=str(w.word or ""),
                    start=float(w.start or 0.0),
                    end=float(w.end or 0.0),
                    probability=float(w.probability or 0.0),
                )
                for w in seg.words
            ]

        segments.append(
            Segment(
                id=idx,
                start=float(seg.start or 0.0),
                end=float(seg.end or 0.0),
                text=txt,
                words=words,
            )
        )

    transcript = _normalize_transcript_text(" ".join(full_text_parts))
    language_detected = str(getattr(info, "language", "") or language or "auto")
    language_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    return segments, transcript, language_prob, language_detected, float(getattr(info, "duration", 0.0) or 0.0)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": STT_PROVIDER,
        "model": WHISPER_MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "uptime_seconds": round(time.perf_counter() - _app_start, 2),
    }


@app.get("/models")
async def list_models():
    return {
        "provider": STT_PROVIDER,
        "loaded": WHISPER_MODEL_NAME,
        "available": [WHISPER_MODEL_NAME],
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
    vad_filter: Optional[bool] = Form(None),
):
    tmp_path = _save_upload(file)
    try:
        duration = _audio_duration_seconds(tmp_path)
        t0 = time.perf_counter()
        segments, transcript, lang_prob, language_detected, inferred_duration = _run_whisper_transcribe(
            tmp_path,
            language,
            word_timestamps,
            vad_filter,
        )
        elapsed = time.perf_counter() - t0
        final_duration = duration if duration > 0 else round(inferred_duration, 3)

        log.info(
            "Transcribed '%s' model=%s duration=%.2fs in %.2fs text=%r",
            file.filename,
            WHISPER_MODEL_NAME,
            final_duration,
            elapsed,
            transcript[:120],
        )

        return TranscribeResponse(
            text=transcript,
            language=language_detected,
            language_probability=lang_prob,
            duration=final_duration,
            model=MODEL_SIZE,
            device=DEVICE,
            segments=segments,
            elapsed_seconds=round(elapsed, 3),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))
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
    vad_filter: Optional[bool] = Form(None),
):
    tmp_path = _save_upload(file)

    def generate() -> Generator[str, None, None]:
        try:
            duration = _audio_duration_seconds(tmp_path)
            segments, transcript, lang_prob, language_detected, inferred_duration = _run_whisper_transcribe(
                tmp_path,
                language,
                word_timestamps,
                vad_filter,
            )

            meta = json.dumps(
                {
                    "event": "meta",
                    "language": language_detected,
                    "language_probability": lang_prob,
                    "duration": duration if duration > 0 else inferred_duration,
                    "model": WHISPER_MODEL_NAME,
                }
            )
            yield f"data: {meta}\n\n"

            for seg in segments:
                payload = {
                    "event": "segment",
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [w.model_dump(mode="python") for w in seg.words] if seg.words else None,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            done = json.dumps({"event": "done", "text": transcript})
            yield f"data: {done}\n\n"
        except Exception as exc:
            yield f'data: {json.dumps({"event": "error", "detail": str(exc)})}\n\n'
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
