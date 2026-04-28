#!/usr/bin/env bash
set -euo pipefail

ORCH_HEALTH_URL="${ORCH_HEALTH_URL:-http://127.0.0.1:8021/health}"
ORCH_QUERY_URL="${ORCH_QUERY_URL:-http://127.0.0.1:8021/sales/query}"
VISION_HEALTH_URL="${VISION_HEALTH_URL:-http://127.0.0.1:8031/health}"

echo "[orchestrator]"
curl -fsS "${ORCH_HEALTH_URL}" | python3 -m json.tool

echo "[vision]"
curl -fsS "${VISION_HEALTH_URL}" | python3 -m json.tool

echo "[query-smoke]"
curl -fsS -X POST "${ORCH_QUERY_URL}" \
  -H "Content-Type: application/json" \
  -d '{"message":"gia can 2 phong ngu", "need_retrieval": true}' | python3 -m json.tool
