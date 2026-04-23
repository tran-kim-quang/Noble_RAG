# Internal Deploy Runbook (1 Linux Server)

## 1) Architecture (minimal)
- `orchestrator-service`: public on host (`127.0.0.1:8021` by default).
- `retrieval-service`: internal-only (Docker network, no host port publish).
- `qdrant`: internal-only (Docker network, no host port publish).
- Qdrant data persisted at `deploy/volumes/qdrant`.
- Runtime config:
  - `deploy/.env.retrieval`
  - `deploy/.env.orchestrator`

## 2) Files
- `deploy/docker-compose.yml`
- `deploy/.env.retrieval`
- `deploy/.env.orchestrator`
- `deploy/volumes/qdrant/`
- `deploy/scripts/start.sh`
- `deploy/scripts/stop.sh`
- `deploy/scripts/restart.sh`
- `deploy/scripts/status.sh`
- `deploy/scripts/logs.sh`
- `deploy/scripts/health.sh`

## 3) Deploy (copy-paste)
```bash
cd /path/to/Noble_RAG

# choose compose command
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  COMPOSE="docker-compose"
fi

# optional: tune config
vi deploy/.env.retrieval
vi deploy/.env.orchestrator

# build images
$COMPOSE -f deploy/docker-compose.yml build

# up
$COMPOSE -f deploy/docker-compose.yml up -d

# check running containers
./deploy/scripts/status.sh

# health check (orchestrator + query smoke)
./deploy/scripts/health.sh
```

## 4) Operations
```bash
# tail logs
./deploy/scripts/logs.sh orchestrator-service 200
./deploy/scripts/logs.sh retrieval-service 200
./deploy/scripts/logs.sh qdrant 200

# restart one service
$COMPOSE -f deploy/docker-compose.yml restart retrieval-service
$COMPOSE -f deploy/docker-compose.yml restart orchestrator-service

# restart all
./deploy/scripts/restart.sh

# stop stack (keep data)
./deploy/scripts/stop.sh

# remove containers/network (keep data volume dir)
$COMPOSE -f deploy/docker-compose.yml down
```

## 5) Post-deploy checks
### 5.1 Ingest sample docs
```bash
$COMPOSE -f deploy/docker-compose.yml exec -T retrieval-service sh -lc \
  "python - <<'PY'
import json, urllib.request
docs = json.load(open('/app/data/sample_docs.json', 'r', encoding='utf-8'))
body = json.dumps({'documents': docs}).encode('utf-8')
req = urllib.request.Request(
    'http://retrieval-service:8011/ingest',
    data=body,
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.read().decode())
PY"
```

### 5.2 Test orchestrator
```bash
curl -fsS -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Giá căn 2 phòng ngủ là bao nhiêu?", "need_retrieval": true}' | python3 -m json.tool
```

## 6) Security defaults
- Qdrant is not published to host.
- Retrieval service is not published to host.
- Only orchestrator is published to host.
- Default bind is loopback:
```bash
ORCH_BIND_IP=127.0.0.1 ./deploy/scripts/restart.sh
```
- To bind orchestrator to private LAN IP:
```bash
ORCH_BIND_IP=10.10.10.20 ./deploy/scripts/restart.sh
```

## 7) Rollback plan (minimal)
### 7.1 Rollback code only
```bash
git checkout <previous_commit_or_tag>
./deploy/scripts/restart.sh
./deploy/scripts/health.sh
```

### 7.2 Stop all but keep Qdrant data
```bash
./deploy/scripts/stop.sh
```

### 7.3 Full reset (delete Qdrant data too)
```bash
./deploy/scripts/stop.sh
rm -rf deploy/volumes/qdrant/*
./deploy/scripts/start.sh
```

### 7.4 Back to local uvicorn mode
```bash
# terminal 1
uvicorn retrieval_service.app:app --host 0.0.0.0 --port 8011

# terminal 2
uvicorn orchestrator_service.app:app --host 0.0.0.0 --port 8021
```
