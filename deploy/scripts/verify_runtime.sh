#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_COMPOSE_FILE="${ROOT_DIR}/deploy/docker-compose.yml"
LEGACY_COMPOSE_FILE="${ROOT_DIR}/deploy/docker-compose.internal.yml"
ORCH_BASE_URL="${ORCH_BASE_URL:-http://127.0.0.1:8021}"

if [[ -f "${DEFAULT_COMPOSE_FILE}" ]]; then
  COMPOSE_FILE="${DEFAULT_COMPOSE_FILE}"
else
  COMPOSE_FILE="${LEGACY_COMPOSE_FILE}"
fi

STEP=0
pass() { echo "[PASS] $1"; }
fail() {
  echo "[FAIL] $1"
  echo "next: ./deploy/scripts/logs.sh orchestrator-service 200"
  echo "next: ./deploy/scripts/logs.sh retrieval-service 200"
  echo "next: ./deploy/scripts/logs.sh qdrant 200"
  echo "VERIFY FAIL"
  exit 1
}
step() {
  STEP=$((STEP+1))
  echo "[$STEP/10] $1"
}

step "detect compose command"
if docker compose version >/dev/null 2>&1; then
  COMPOSE_BIN="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_BIN="docker-compose"
else
  fail "docker compose/docker-compose not found"
fi
pass "compose=${COMPOSE_BIN}"

step "compose config"
if ! ${COMPOSE_BIN} -f "${COMPOSE_FILE}" config >/dev/null; then
  fail "compose config invalid (${COMPOSE_FILE})"
fi
pass "compose config"

step "build stack"
if ! ${COMPOSE_BIN} -f "${COMPOSE_FILE}" build; then
  fail "build failed (likely network/disk/dependency)"
fi
pass "build"

step "up -d"
if ! ${COMPOSE_BIN} -f "${COMPOSE_FILE}" up -d; then
  fail "up failed"
fi
pass "up"

step "ps running services"
RUNNING_COUNT="$(${COMPOSE_BIN} -f "${COMPOSE_FILE}" ps --services --filter status=running | wc -l | tr -d ' ')"
if [[ "${RUNNING_COUNT}" -lt 3 ]]; then
  ${COMPOSE_BIN} -f "${COMPOSE_FILE}" ps || true
  fail "expected >=3 running services, got ${RUNNING_COUNT}"
fi
pass "services running=${RUNNING_COUNT}"

step "health check"
HEALTH_JSON="$(curl -fsS "${ORCH_BASE_URL}/health" 2>/dev/null || true)"
if [[ -z "${HEALTH_JSON}" ]]; then
  fail "orchestrator health unreachable at ${ORCH_BASE_URL}/health"
fi
if ! echo "${HEALTH_JSON}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status") in {"ok","healthy"}' >/dev/null 2>&1; then
  echo "health=${HEALTH_JSON}"
  fail "orchestrator health invalid"
fi
pass "orchestrator health"

step "ingest markdown docs"
if ! ${COMPOSE_BIN} -f "${COMPOSE_FILE}" exec -T retrieval-service sh -lc "python /app/scripts/ingest_markdown_to_retrieval.py --base-url http://retrieval-service:8011 --data-dir /app/data --files 02_12_2025_CSBH_574_NOBLE_PALACE_TAY_THANG_LONG_HDBM.md CONCEPT_THIET_KE_08_02_2025_only_hang_muc_noi_dung.md"; then
  fail "ingest markdown docs failed"
fi
pass "ingest markdown docs"

step "query orchestrator (3 cases)"
Q1="$(curl -fsS -X POST "${ORCH_BASE_URL}/sales/query" -H 'Content-Type: application/json' -d '{"message":"Giá căn 2 phòng ngủ là bao nhiêu?","need_retrieval":true}' 2>/dev/null || true)"
Q2="$(curl -fsS -X POST "${ORCH_BASE_URL}/sales/query" -H 'Content-Type: application/json' -d '{"message":"Xin chào"}' 2>/dev/null || true)"
Q3="$(curl -fsS -X POST "${ORCH_BASE_URL}/sales/query" -H 'Content-Type: application/json' -d '{"message":"Dự án có sân golf trên mây không?","need_retrieval":true}' 2>/dev/null || true)"

if [[ -z "${Q1}" || -z "${Q2}" || -z "${Q3}" ]]; then
  fail "query failed (empty response)"
fi

if ! echo "${Q1}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("route") in {"retrieval_confident","retrieval_low_confidence"}; assert d.get("action"); assert d.get("decision_reason")' >/dev/null 2>&1; then
  echo "q1=${Q1}"
  fail "query case 1 invalid"
fi
if ! echo "${Q2}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("route")=="no_retrieval_needed"' >/dev/null 2>&1; then
  echo "q2=${Q2}"
  fail "query case 2 invalid"
fi
if ! echo "${Q3}" | python3 -c 'import json,sys; d=json.load(sys.stdin); r=d.get("route"); low=(d.get("retrieval") or {}).get("low_confidence",False); assert (r=="retrieval_low_confidence") or bool(low)' >/dev/null 2>&1; then
  echo "q3=${Q3}"
  fail "query case 3 invalid"
fi
pass "query cases"

echo "VERIFY PASS"
