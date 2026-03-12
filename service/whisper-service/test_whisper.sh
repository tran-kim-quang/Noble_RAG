#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# Test nhanh whisper-service sau khi service đã chạy
# Usage:
#   chmod +x test_whisper.sh
#   ./test_whisper.sh [audio_file] [language]
#
# Examples:
#   ./test_whisper.sh jfk.flac en
#   ./test_whisper.sh audio.m4a vi
# ──────────────────────────────────────────────────────
set -euo pipefail

SERVICE_URL="${WHISPER_URL:-http://localhost:8001}"
AUDIO_FILE="${1:-}"
LANGUAGE="${2:-}"        # Bỏ trống = tự detect

# ── Health check ──────────────────────────────────────
echo "=== Health Check ==="
curl -s "$SERVICE_URL/health" | python3 -m json.tool
echo ""

# ── Transcribe test ───────────────────────────────────
if [[ -n "$AUDIO_FILE" ]]; then
    echo "=== Transcribing: $AUDIO_FILE ==="
    LANG_ARG=""
    if [[ -n "$LANGUAGE" ]]; then
        LANG_ARG="-F language=$LANGUAGE"
    fi

    curl -s -X POST "$SERVICE_URL/transcribe" \
        -F "file=@${AUDIO_FILE}" \
        $LANG_ARG \
        | python3 -m json.tool
else
    echo "Tip: Pass an audio file to transcribe, e.g.:"
    echo "  $0 audio.m4a vi"
fi
