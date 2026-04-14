# Orchestrator Service Local Runbook

## Terminal 1 - Start Retrieval Service
```bash
cd /home/ncthang/CongThang/Noble_RAG
source .venv/bin/activate

QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=retrieval_bench_v2 \
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
EMBEDDING_DIM=384 \
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

### 1) Health retrieval
```bash
curl -s http://127.0.0.1:8011/health
```

### 2) Health orchestrator
```bash
curl -s http://127.0.0.1:8021/health
```

### 3) no_retrieval_needed (explicit false)
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"xin chao", "need_retrieval": false}'
```

### 4) retrieval_confident (explicit true)
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"gia can 2 phong ngu", "need_retrieval": true, "top_k": 5}'
```

### 5) retrieval_confident (auto heuristic)
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"phap ly du an hien tai"}'
```

### 6) retrieval_low_confidence (no-answer style)
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"du an co san truot tuyet trong nha khong", "need_retrieval": true}'
```

### 7) retrieval_low_confidence (mơ hồ)
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"cho minh thong tin tong quan du an", "need_retrieval": true}'
```

### 8) blank message -> 422
```bash
curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"   "}'
```

### 9) retrieval service down test
```bash
pkill -f "uvicorn retrieval_service.app:app"

curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"gia can 2 phong ngu", "need_retrieval": true}'
```

### 10) restart retrieval after down test
```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=retrieval_bench_v2 \
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
EMBEDDING_DIM=384 \
MIN_RETRIEVE_SCORE=0.68 \
uvicorn retrieval_service.app:app --host 127.0.0.1 --port 8011
```
