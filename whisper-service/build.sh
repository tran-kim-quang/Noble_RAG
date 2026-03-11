#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# build.sh – Build faster-whisper service image
#
# Usage:
#   ./build.sh              → Build GPU (mặc định, CUDA 12.3 + cuDNN9)
#   ./build.sh --cpu        → Build CPU only (python:3.11-slim)
#   ./build.sh --export     → Build GPU + xuất file .tar.gz để ship
#   ./build.sh --cpu --export → Build CPU + xuất file .tar.gz
#
# Yêu cầu build GPU:
#   - NVIDIA Container Toolkit trên máy BUILD (nếu muốn pre-download model)
#   - Máy đích cần NVIDIA Container Toolkit + driver để CHẠY
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Parse args ─────────────────────────────────────────────────────────
MODE="gpu"
EXPORT=false

for arg in "$@"; do
    case "$arg" in
        --cpu)    MODE="cpu" ;;
        --gpu)    MODE="gpu" ;;
        --export) EXPORT=true ;;
        -h|--help)
            grep '^#' "$0" | head -20 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $arg (use --cpu | --gpu | --export)"; exit 1 ;;
    esac
done

# ── Config theo mode ───────────────────────────────────────────────────
if [[ "$MODE" == "gpu" ]]; then
    BASE_IMAGE="nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04"
    WHISPER_DEVICE="cuda"
    WHISPER_COMPUTE="float16"
    IMAGE_TAG="whisper-service:gpu"
    EXPORT_FILE="whisper-service-gpu.tar.gz"
else
    BASE_IMAGE="python:3.11-slim-bookworm"
    WHISPER_DEVICE="cpu"
    WHISPER_COMPUTE="int8"
    IMAGE_TAG="whisper-service:cpu"
    EXPORT_FILE="whisper-service-cpu.tar.gz"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          🎙️  Faster-Whisper Service Builder                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Mode       : %-47s║\n" "$MODE (${WHISPER_DEVICE})"
printf "║  Base image : %-47s║\n" "$BASE_IMAGE"
printf "║  Compute    : %-47s║\n" "$WHISPER_COMPUTE"
printf "║  Tag        : %-47s║\n" "$IMAGE_TAG"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build ──────────────────────────────────────────────────────────────
echo "▶ Building Docker image..."
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg WHISPER_DEVICE="$WHISPER_DEVICE" \
    --build-arg WHISPER_COMPUTE="$WHISPER_COMPUTE" \
    --build-arg WHISPER_MODEL="base" \
    -t "$IMAGE_TAG" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$SCRIPT_DIR"

echo ""
echo "✅ Build thành công: $IMAGE_TAG"
docker image inspect "$IMAGE_TAG" --format \
    "   Size: {{.Size | printf \"%.0f\"}} bytes  |  Created: {{.Created}}"

# ── Export ─────────────────────────────────────────────────────────────
if [[ "$EXPORT" == true ]]; then
    echo ""
    echo "▶ Xuất image sang file $EXPORT_FILE ..."
    docker save "$IMAGE_TAG" | gzip > "$SCRIPT_DIR/$EXPORT_FILE"
    echo "✅ Đã tạo: $SCRIPT_DIR/$EXPORT_FILE"
    echo "   Kích thước: $(du -sh "$SCRIPT_DIR/$EXPORT_FILE" | cut -f1)"
    echo ""
    echo "📦 Cách ship sang máy khác:"
    echo "   scp $EXPORT_FILE user@remote:/path/"
    echo ""
    echo "   # Trên máy đích:"
    echo "   docker load < $EXPORT_FILE"
fi

# ── Hướng dẫn chạy ────────────────────────────────────────────────────
echo ""
echo "▶ Để chạy service:"
if [[ "$MODE" == "gpu" ]]; then
    echo "   docker run -d \\"
    echo "     --gpus all \\"
    echo "     --name whisper-service \\"
    echo "     -p 8001:8001 \\"
    echo "     -e WHISPER_MODEL=base \\"
    echo "     -v whisper_cache:/app/.cache/hf \\"
    echo "     $IMAGE_TAG"
else
    echo "   docker run -d \\"
    echo "     --name whisper-service \\"
    echo "     -p 8001:8001 \\"
    echo "     -e WHISPER_MODEL=base \\"
    echo "     -v whisper_cache:/app/.cache/hf \\"
    echo "     $IMAGE_TAG"
fi
echo ""
echo "   # Health check:"
echo "   curl http://localhost:8001/health"
