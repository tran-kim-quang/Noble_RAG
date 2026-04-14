#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_COMPOSE_FILE="${SCRIPT_DIR}/../docker-compose.yml"
LEGACY_COMPOSE_FILE="${SCRIPT_DIR}/../docker-compose.internal.yml"
if [[ -f "${DEFAULT_COMPOSE_FILE}" ]]; then
  COMPOSE_FILE="${DEFAULT_COMPOSE_FILE}"
else
  COMPOSE_FILE="${LEGACY_COMPOSE_FILE}"
fi
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "docker compose/docker-compose not found"
  exit 1
fi

"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" up -d --build

echo "started"
"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" ps
