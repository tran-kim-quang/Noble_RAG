# ──────────────────────────────────────────────────────────────────────────────
# Faster-Whisper API Service
#
# Build variants:
#   CPU only (default – smaller image, no GPU required):
#     docker build -t whisper-service .
#
#   GPU (CUDA 12.3 + cuDNN 9 – needs NVIDIA Container Toolkit on host):
#     docker build --build-arg BASE_IMAGE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 \
#                  --build-arg WHISPER_DEVICE=cuda \
#                  -t whisper-service:gpu .
# ──────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

# ── Build-time args (mirrored as env so uvicorn can read them) ─────────────
ARG  WHISPER_MODEL=base
ARG  WHISPER_DEVICE=cpu
ARG  WHISPER_COMPUTE=int8
ARG  WHISPER_BEAM=5

ENV  WHISPER_MODEL=${WHISPER_MODEL}  \
     WHISPER_DEVICE=${WHISPER_DEVICE} \
     WHISPER_COMPUTE=${WHISPER_COMPUTE} \
     WHISPER_BEAM=${WHISPER_BEAM}    \
     # Keep Python I/O unbuffered so logs stream immediately
     PYTHONUNBUFFERED=1              \
     PYTHONDONTWRITEBYTECODE=1       \
     # HuggingFace model cache inside the image layer
     HF_HOME=/app/.cache/hf

# ── System deps ────────────────────────────────────────────────────────────
# ffmpeg  → required by faster-whisper for audio decoding
# libgomp → OpenMP runtime needed by CTranslate2
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── App directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (layer-cached unless requirements.txt changes) ─────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────
COPY app.py /app/app.py

# ── (Optional) Pre-download model at build time ────────────────────────────
# Uncomment the line below to bake the model into the image.
# Trade-off: larger image, but container starts instantly without download.
# RUN python - <<'PYEOF'
# from faster_whisper import WhisperModel
# import os
# WhisperModel(os.environ["WHISPER_MODEL"], device="cpu", compute_type="int8")
# print("Model pre-cached OK")
# PYEOF

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8001

# Health probe (used by Docker / docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
