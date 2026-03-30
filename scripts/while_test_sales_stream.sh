#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SESSION_ID="${SESSION_ID:-manual_stream_test_$(date +%s)}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required"
  exit 1
fi

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

echo "BASE_URL=$BASE_URL"
echo "SESSION_ID=$SESSION_ID"
echo "Streaming mode: POST /sales/chat/stream"
echo "Commands: exit, quit, :q"
echo

while true; do
  printf "You> "
  IFS= read -r message || break

  case "$message" in
    exit|quit|:q)
      echo "Bye."
      break
      ;;
  esac

  if [[ -z "${message// }" ]]; then
    continue
  fi

  payload=$(printf '{"session_id":"%s","message":"%s"}' \
    "$(json_escape "$SESSION_ID")" \
    "$(json_escape "$message")")

  stream_file=$(mktemp)
  elapsed=$(curl -sS -N -o "$stream_file" -w "%{time_total}" \
    -H "Content-Type: application/json" \
    -X POST "$BASE_URL/sales/chat/stream" \
    -d "$payload")

  echo "Stream>"
  cat "$stream_file"
  printf "[time=%ss]\n\n" "$elapsed"

  rm -f "$stream_file"
done
