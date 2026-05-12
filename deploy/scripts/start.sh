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

# Prefer GPU by default when NVIDIA runtime is available; fallback to CPU.
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  export OLLAMA_RUNTIME="nvidia"
  echo "[start] NVIDIA runtime detected -> Ollama will run with GPU"
else
  export OLLAMA_RUNTIME="runc"
  echo "[start] NVIDIA runtime not found -> Ollama will run with CPU (runc)"
fi

"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" up -d --build

echo "started"
"${COMPOSE_CMD[@]}" -f "${COMPOSE_FILE}" ps
