# Orchestrator Service Local Runbook (2 Routes)

## Terminal 1 - Start Retrieval Service
```bash
cd /home/ncthang/CongThang/Noble_RAG
source .venv/bin/activate

QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=retrieval_bench_qwen3_4b \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B \
EMBEDDING_DIM=2560 \
MIN_RETRIEVE_SCORE=0.68 \
uvicorn retrieval_service.app:app --host 127.0.0.1 --port 8011
```

## Terminal 2 - Ingest Data + Start Orchestrator Service
```bash
cd /home/ncthang/CongThang/Noble_RAG
source .venv/bin/activate

curl -s -X POST http://127.0.0.1:8011/ingest \
  -H "Content-Type: application/json" \
  --data @data/sample_docs.json

curl -s -X POST http://127.0.0.1:8011/ingest \
  -H "Content-Type: application/json" \
  --data @data/sample_docs_project_layers.json

RETRIEVAL_SERVICE_URL=http://127.0.0.1:8011 \
ORCHESTRATOR_DEFAULT_TOP_K=5 \
uvicorn orchestrator_service.app:app --host 127.0.0.1 --port 8021
```

## Terminal 3 - E2E Checks
```bash
cd /home/ncthang/CongThang/Noble_RAG
source .venv/bin/activate

python3 scripts/e2e_smoke_test.py --base-url http://127.0.0.1:8021
```

## Optional - Run Unit Tests
```bash
cd /home/ncthang/CongThang/Noble_RAG
source .venv/bin/activate
pytest -q tests/test_orchestrator_e2e.py
```

## Curl End-to-End Suite

### 1) Health orchestrator
```bash
curl -s http://127.0.0.1:8021/health
```

### 2) consult_discovery
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{
    "message":"Mua de dau tu thi nen bat dau tu dau?",
    "lead_state":{
      "name":null,
      "phone_contact":null,
      "need":{"summary":"","topics":[],"evidence":[],"last_updated_at":null},
      "painpoint":{"summary":"","topics":[],"evidence":[],"last_updated_at":null}
    }
  }'
```

### 3) project_grounded (auto routing)
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{
    "message":"Co can nao gan truong hoc va benh vien?",
    "lead_state":{
      "name":null,
      "phone_contact":null,
      "need":{"summary":"","topics":[],"evidence":[],"last_updated_at":null},
      "painpoint":{"summary":"","topics":[],"evidence":[],"last_updated_at":null}
    }
  }'
```

### 4) force project_grounded
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{
    "message":"Cho minh thong tin phap ly du an",
    "force_route":"project_grounded",
    "lead_state":{
      "name":null,
      "phone_contact":null,
      "need":{"summary":"","topics":[],"evidence":[],"last_updated_at":null},
      "painpoint":{"summary":"","topics":[],"evidence":[],"last_updated_at":null}
    }
  }'
```

### 5) blank message -> 422
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"   "}'
```
