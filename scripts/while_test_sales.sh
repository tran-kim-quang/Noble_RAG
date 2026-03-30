#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
SESSION_ID="${SESSION_ID:-manual_test_$(date +%s)}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required"
  exit 1
fi

if command -v jq >/dev/null 2>&1; then
  HAS_JQ=1
else
  HAS_JQ=0
fi

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

echo "BASE_URL=$BASE_URL"
echo "SESSION_ID=$SESSION_ID"
echo "Type a message and press Enter."
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

  body_file=$(mktemp)
  elapsed=$(curl -sS -o "$body_file" -w "%{time_total}" \
    -H "Content-Type: application/json" \
    -X POST "$BASE_URL/sales/chat" \
    -d "$payload")

  if [[ "$HAS_JQ" -eq 1 ]]; then
    response=$(jq -r '.response // ""' "$body_file")
    sales_state=$(jq -r '.sales_state // ""' "$body_file")
    missing_slots=$(jq -c '.missing_slots // []' "$body_file")
  else
    response=$(cat "$body_file")
    sales_state="(need jq)"
    missing_slots="(need jq)"
  fi

  printf "Bot> %s\n" "$response"
  printf "[time=%ss state=%s missing=%s]\n\n" "$elapsed" "$sales_state" "$missing_slots"

  rm -f "$body_file"
done
