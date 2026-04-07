#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

python_exe=""
for candidate in \
  "$project_root/venv/bin/python" \
  "$project_root/.venv/bin/python" \
  "$(command -v python3.12 || true)" \
  "$(command -v python3.10 || true)" \
  "$(command -v python3 || true)" \
  "$(command -v python || true)"; do
  if [[ -n "${candidate:-}" && -x "$candidate" ]]; then
    python_exe="$candidate"
    break
  fi
done

if [[ -z "$python_exe" ]]; then
  echo "Cannot find Python executable. Expected venv/.venv python or python3 in PATH."
  exit 1
fi

env_file="$project_root/.env"
if [[ -f "$env_file" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    if [[ "$line" == *=* ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      key="$(echo "$key" | xargs)"
      value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      if [[ -n "$key" && -n "$value" && -z "${!key:-}" ]]; then
        export "$key=$value"
      fi
    fi
  done < "$env_file"
fi

avatar_id="half_avatar"
requested_tts="elevenlabs"
enable_transition="0"
transition_duration="0.06"
transport="${LIVETALKING_TRANSPORT:-webrtc}"
push_url="${LIVETALKING_PUSH_URL:-http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream}"
listen_port="${LIVETALKING_PORT:-18010}"

export NOBLE_RAG_API_URL="${NOBLE_RAG_API_URL:-http://127.0.0.1:8010}"
export NOBLE_VISION_API_URL="${NOBLE_VISION_API_URL:-http://127.0.0.1:8020}"
export NOBLE_WHISPER_API_URL="${NOBLE_WHISPER_API_URL:-http://127.0.0.1:8001}"
export NOBLE_VISION_SOURCE="${NOBLE_VISION_SOURCE:-browser}"
export RAG_CHAT_MODE="${RAG_CHAT_MODE:-stream}"
export RAG_CHAT_ENDPOINT="${RAG_CHAT_ENDPOINT:-/query/stream}"
export RAG_STREAM_METHOD="${RAG_STREAM_METHOD:-POST}"
export LIGHTRAG_URL="${LIGHTRAG_URL:-$NOBLE_RAG_API_URL}"
export RAG_STREAM_ENDPOINT="${RAG_STREAM_ENDPOINT:-/query/stream}"

echo "Project root: $project_root"
echo "Python: $python_exe"
echo "Avatar: $avatar_id"
echo "Hold video: disabled"
echo "TTS requested: $requested_tts"
echo "Transport: $transport"
echo "Listen port: $listen_port"
if [[ "$transport" == "rtcpush" ]]; then
  echo "RTCPush URL: $push_url"
fi

vae_config_candidates=(
  "$project_root/models/sd-vae-ft-mse/config.json"
  "$project_root/models/sd-vae/config.json"
)

vae_config_found=""
for candidate in "${vae_config_candidates[@]}"; do
  if [[ -e "$candidate" ]]; then
    vae_config_found="$candidate"
    break
  fi
done

required_model_paths=(
  "$project_root/models/musetalkV15/unet.pth"
  "$project_root/models/musetalkV15/musetalk.json"
  "$project_root/models/whisper/config.json"
)

missing_paths=()
if [[ -z "$vae_config_found" ]]; then
  missing_paths+=("${vae_config_candidates[0]}")
fi
for required_path in "${required_model_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    missing_paths+=("$required_path")
  fi
done

if (( ${#missing_paths[@]} > 0 )); then
  echo "Missing required MuseTalk model files:"
  printf ' - %s\n' "${missing_paths[@]}"
  echo "Place the downloaded MuseTalk model bundle under $project_root/models and try again."
  exit 1
fi

arguments=(
  "app.py"
  "--avatar_id" "$avatar_id"
  "--tts" "$requested_tts"
  "--transport" "$transport"
  "--listenport" "$listen_port"
  "--transition_duration" "$transition_duration"
)

if [[ "$enable_transition" == "1" ]]; then
  arguments+=("--enable_transition")
fi

if [[ "$transport" == "rtcpush" ]]; then
  arguments+=("--push_url" "$push_url")
fi

exec "$python_exe" "${arguments[@]}"
