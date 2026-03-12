"""
Faster-Whisper HTTP API Service
--------------------------------
Endpoints:
  POST /transcribe   - Upload audio file, receive full transcript
  POST /transcribe/stream - Upload audio, receive SSE stream of segments
  GET  /health       - Health check
  GET  /models       - List available model sizes
"""

import io
import os
import time
import logging
import tempfile
from typing import Optional, Generator

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from faster_whisper import WhisperModel

# ── Config via env vars ────────────────────────────────────────────────
MODEL_SIZE   = os.getenv("WHISPER_MODEL",    "base")   # tiny/base/small/medium/large-v3
DEVICE       = os.getenv("WHISPER_DEVICE",   "cpu")    # cpu / cuda
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE",  "int8")   # int8 / float16 / float32
BEAM_SIZE    = int(os.getenv("WHISPER_BEAM", "5"))
LOG_LEVEL    = os.getenv("LOG_LEVEL",        "INFO")

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("whisper-service")

# ── Load model once at startup ─────────────────────────────────────────
log.info(f"Loading model '{MODEL_SIZE}' on device='{DEVICE}' compute='{COMPUTE_TYPE}' ...")
_model_load_start = time.perf_counter()
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
_model_load_time = time.perf_counter() - _model_load_start
log.info(f"Model loaded in {_model_load_time:.2f}s")

# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(
    title="Faster-Whisper API",
    description="Speech-to-text service powered by faster-whisper",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac",
    ".webm", ".mkv", ".opus", ".aac",
}

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]


# ── Response schemas ───────────────────────────────────────────────────
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
    model: str
    device: str
    segments: list[Segment]
    elapsed_seconds: float


# ── Helpers ────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
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


def _transcribe(
    audio_path: str,
    language: Optional[str],
    word_timestamps: bool,
) -> tuple:
    """Run faster-whisper transcription and return (segments_list, info)."""
    segments_iter, info = model.transcribe(
        audio_path,
        language=language or None,
        beam_size=BEAM_SIZE,
        word_timestamps=word_timestamps,
    )
    segments = []
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
    return segments, info


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model_load_seconds": round(_model_load_time, 2),
    }


@app.get("/models")
async def list_models():
    return {"available": AVAILABLE_MODELS, "loaded": MODEL_SIZE}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    word_timestamps: bool = Form(False),
):
    """
    Upload an audio file and receive the full transcription as JSON.

    - **file**: Audio file (wav, mp3, m4a, flac, ogg, webm…)
    - **language**: Force language code e.g. `vi`, `en`. Leave empty for auto-detect.
    - **word_timestamps**: Include per-word timestamps when `true`.
    """
    tmp_path = _save_upload(file)
    try:
        t0 = time.perf_counter()
        segments, info = _transcribe(tmp_path, language, word_timestamps)
        elapsed = time.perf_counter() - t0

        log.info(
            f"Transcribed '{file.filename}' lang={info.language} "
            f"({info.duration:.1f}s audio) in {elapsed:.2f}s"
        )

        return TranscribeResponse(
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration=round(info.duration, 3),
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
):
    """
    Same as /transcribe but streams segments as Server-Sent Events (SSE).

    Each event is a JSON object for one segment so you can display partial
    results while the audio is still being processed.
    """
    tmp_path = _save_upload(file)

    def generate() -> Generator[str, None, None]:
        try:
            segments_iter, info = model.transcribe(
                tmp_path,
                language=language or None,
                beam_size=BEAM_SIZE,
                word_timestamps=word_timestamps,
            )
            # Send metadata first
            import json
            meta = json.dumps({
                "event": "meta",
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 3),
            })
            yield f"data: {meta}\n\n"

            for i, seg in enumerate(segments_iter):
                words = None
                if word_timestamps and seg.words:
                    words = [
                        {"word": w.word, "start": round(w.start, 3),
                         "end": round(w.end, 3), "probability": round(w.probability, 4)}
                        for w in seg.words
                    ]
                payload = json.dumps({
                    "event": "segment",
                    "id": i,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "words": words,
                })
                yield f"data: {payload}\n\n"

            yield 'data: {"event": "done"}\n\n'
        except Exception as exc:
            import json
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
