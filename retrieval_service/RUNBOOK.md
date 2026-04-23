# Retrieval Service Local Runbook

## 1) Start Qdrant
```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}' | grep 6333 || \
  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

## 2) Create venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install dependencies
```bash
pip install -e ./haystack
pip install -r retrieval_service/requirements.txt
```

## 4) Run server
```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=retrieval_bench_v2 \
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
EMBEDDING_DIM=384 \
MIN_RETRIEVE_SCORE=0.68 \
uvicorn retrieval_service.app:app --host 127.0.0.1 --port 8011
```

Notes:
- `MIN_RETRIEVE_SCORE`: confidence threshold for low-confidence flag.
- `LOW_CONFIDENCE_EMPTY_RESULTS=true`: optional strict mode to return empty `results` when low confidence.

## 5) Run pytest (new terminal)
```bash
source .venv/bin/activate
pytest -q tests/test_retrieval_service_api.py
```

## 6) Ingest + curl sanity tests (new terminal)
```bash
source .venv/bin/activate

curl -s http://127.0.0.1:8011/health

curl -s -X POST http://127.0.0.1:8011/ingest \
  -H "Content-Type: application/json" \
  --data @data/sample_docs.json

curl -s -X POST http://127.0.0.1:8011/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"giá bán khoảng bao nhiêu", "top_k": 5}'
```

## 7) Run benchmark (new terminal)
```bash
source .venv/bin/activate
python3 scripts/benchmark_retrieval.py \
  --base-url http://127.0.0.1:8011 \
  --cases data/sample_queries.json
```
