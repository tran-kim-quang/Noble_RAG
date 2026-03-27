#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"
SESSION_PREFIX="${SESSION_PREFIX:-sales_test}"
USE_STREAM="${USE_STREAM:-1}"
USE_SALES_STREAM="${USE_SALES_STREAM:-1}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required"
  exit 1
fi

if command -v jq >/dev/null 2>&1; then
  HAS_JQ=1
else
  HAS_JQ=0
fi

run_sales_chat() {
  local session_id="$1"
  local message="$2"

  local payload
  payload=$(printf '{"session_id":"%s","message":"%s"}' "$(json_escape "$session_id")" "$(json_escape "$message")")

  local body_file
  body_file=$(mktemp)

  local elapsed
  elapsed=$(curl -sS -o "$body_file" -w "%{time_total}" \
    -H "Content-Type: application/json" \
    -X POST "$BASE_URL/sales/chat" \
    -d "$payload")

  local response sales_state missing_slots
  if [[ "$HAS_JQ" -eq 1 ]]; then
    response=$(jq -r '.response // ""' "$body_file")
    sales_state=$(jq -r '.sales_state // ""' "$body_file")
    missing_slots=$(jq -c '.missing_slots // []' "$body_file")
  else
    response=$(cat "$body_file")
    sales_state="(need jq)"
    missing_slots="(need jq)"
  fi

  printf "[sales/chat] %.2fs | state=%s | missing=%s\n" "$elapsed" "$sales_state" "$missing_slots"
  printf "Q: %s\n" "$message"
  printf "A: %s\n\n" "$response"

  rm -f "$body_file"
}

run_query_stream() {
  local session_id="$1"
  local message="$2"

  local payload
  payload=$(printf '{"query":"%s","session_id":"%s"}' "$(json_escape "$message")" "$(json_escape "$session_id")")

  local stream_file
  stream_file=$(mktemp)

  local elapsed
  elapsed=$(curl -sS -N -o "$stream_file" -w "%{time_total}" \
    -H "Content-Type: application/json" \
    -X POST "$BASE_URL/query/stream" \
    -d "$payload")

  local first_chunk
  first_chunk=$(grep -m1 '"chunk":"' "$stream_file" || true)

  printf "[query/stream] %.2fs\n" "$elapsed"
  printf "Q: %s\n" "$message"
  printf "First stream chunk: %s\n" "${first_chunk:-<empty>}"
  printf "Raw stream saved: %s\n\n" "$stream_file"
}

run_sales_chat_stream() {
  local session_id="$1"
  local message="$2"

  local payload
  payload=$(printf '{"session_id":"%s","message":"%s"}' "$(json_escape "$session_id")" "$(json_escape "$message")")

  local stream_file
  stream_file=$(mktemp)

  local elapsed
  elapsed=$(curl -sS -N -o "$stream_file" -w "%{time_total}" \
    -H "Content-Type: application/json" \
    -X POST "$BASE_URL/sales/chat/stream" \
    -d "$payload")

  local first_chunk
  first_chunk=$(grep -m1 '"chunk":"' "$stream_file" || true)

  printf "[sales/chat/stream] %.2fs\n" "$elapsed"
  printf "Q: %s\n" "$message"
  printf "First stream chunk: %s\n" "${first_chunk:-<empty>}"
  printf "Raw stream saved: %s\n\n" "$stream_file"
}

run_suite_sales_chat() {
  local sid="$SESSION_PREFIX-$(date +%s)-chat"
  echo "=== SALES CHAT SUITE | session=$sid ==="

  run_sales_chat "$sid" "chào em"
  run_sales_chat "$sid" "4 người"
  run_sales_chat "$sid" "2 con nhỏ"
  run_sales_chat "$sid" "mua để ở"
  run_sales_chat "$sid" "quanh Tây Hồ"

  run_sales_chat "$sid" "pháp lý Noble Crystal Tây Hồ thế nào?"
  run_sales_chat "$sid" "so sánh Noble Crystal và Noble Empire"
  run_sales_chat "$sid" "giá hơi cao quá"
  run_sales_chat "$sid" "anh muốn đi xem nhà mẫu"
}

run_suite_stream() {
  local sid="$SESSION_PREFIX-$(date +%s)-stream"
  echo "=== STREAM SUITE | session=$sid ==="

  run_query_stream "$sid" "chào em"
  run_query_stream "$sid" "4 người"
  run_query_stream "$sid" "pháp lý Noble Crystal Tây Hồ thế nào?"
  run_query_stream "$sid" "so sánh Noble Crystal và Noble Empire"
}

run_suite_sales_stream() {
  local sid="$SESSION_PREFIX-$(date +%s)-sales-stream"
  echo "=== SALES STREAM SUITE | session=$sid ==="

  run_sales_chat_stream "$sid" "chào em"
  run_sales_chat_stream "$sid" "4 người"
  run_sales_chat_stream "$sid" "pháp lý Noble Crystal Tây Hồ thế nào?"
  run_sales_chat_stream "$sid" "so sánh Noble Crystal và Noble Empire"
}

main() {
  echo "BASE_URL=$BASE_URL"
  run_suite_sales_chat
  if [[ "$USE_SALES_STREAM" == "1" ]]; then
    run_suite_sales_stream
  fi
  if [[ "$USE_STREAM" == "1" ]]; then
    run_suite_stream
  fi
  echo "Done."
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

main "$@"
