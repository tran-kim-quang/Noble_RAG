# Internal Deploy Runbook (1 Linux Server)

## 1) Architecture (minimal)
- `orchestrator-service`: public on host (`127.0.0.1:8021` by default).
- `retrieval-service`: internal-only (Docker network, no host port publish).
- `vision-service`: internal-only (Docker network, no host port publish).
- `redis`: internal-only session store for known/guest face sessions.
- `qdrant`: internal-only (Docker network, no host port publish).
- Qdrant data persisted at `deploy/volumes/qdrant`.
- Runtime config:
  - `deploy/.env.retrieval`
  - `deploy/.env.orchestrator`

## 2) Files
- `deploy/docker-compose.yml`
- `deploy/.env.retrieval`
- `deploy/.env.orchestrator`
- `deploy/.env.vision`
- `docs/huong_dan_ket_noi_host_model_cho_retrieval_service.md`
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

# if using local Ollama on host machine
# set in deploy/.env.retrieval:
#   EMBEDDING_BACKEND=remote
#   EMBEDDING_API_URL=http://host.docker.internal:11434/api/embeddings
#   EMBEDDING_API_FORMAT=ollama
#   EMBEDDING_MODEL=qwen3-embedding:8b
#   EMBEDDING_DIM=4096
#   LLM_WARM_API_URL=http://host.docker.internal:11434/api/generate
#   LLM_WARM_MODEL=gemma4:latest

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
### 5.1 Ingest markdown project docs (Haystack -> Qdrant)
```bash
$COMPOSE -f deploy/docker-compose.yml exec -T retrieval-service sh -lc \
  "python /app/scripts/ingest_markdown_to_retrieval.py \
    --base-url http://retrieval-service:8011 \
    --data-dir /app/data \
    --files 02_12_2025_CSBH_574_NOBLE_PALACE_TAY_THANG_LONG_HDBM.md CONCEPT_THIET_KE_08_02_2025_only_hang_muc_noi_dung.md"
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
