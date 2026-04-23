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
QDRANT_COLLECTION=retrieval_bench_qwen3_4b \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B \
EMBEDDING_DIM=2560 \
MIN_RETRIEVE_SCORE=0.68 \
uvicorn retrieval_service.app:app --host 127.0.0.1 --port 8011
```

Notes:
- `MIN_RETRIEVE_SCORE`: confidence threshold for low-confidence flag.
- `LOW_CONFIDENCE_EMPTY_RESULTS=true`: optional strict mode to return empty `results` when low confidence.
- Current pipeline uses Haystack SentenceTransformers embedders, so use the HF model id (`Qwen/Qwen3-Embedding-4B`) instead of Ollama tag format (`qwen3-embedding:4b`).
- If you previously used a 384-dim collection (e.g. MiniLM), keep `QDRANT_COLLECTION` as a new name (example above) to avoid dimension mismatch.

### Run with local Ollama embedding/LLM (`/api/...`)
```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=retrieval_bench_qwen3_8b_local \
EMBEDDING_BACKEND=remote \
EMBEDDING_API_URL=http://127.0.0.1:11434/api/embeddings \
EMBEDDING_API_FORMAT=ollama \
EMBEDDING_API_KEY= \
EMBEDDING_API_KEY_HEADER=X-API-Key \
EMBEDDING_MODEL=qwen3-embedding:8b \
EMBEDDING_DIM=4096 \
MODEL_WARM_ENABLED=true \
MODEL_WARM_INTERVAL_SEC=120 \
LLM_WARM_ENABLED=true \
LLM_WARM_API_URL=http://127.0.0.1:11434/api/generate \
LLM_WARM_MODEL=gemma4:latest \
MIN_RETRIEVE_SCORE=0.68 \
uvicorn retrieval_service.app:app --host 127.0.0.1 --port 8011
```

Prerequisites:
- local Ollama serves APIs at `http://127.0.0.1:11434`
- models `qwen3-embedding:8b` and `gemma4:latest` are available locally
- `MODEL_WARM_ENABLED=true` keeps remote models warm on a fixed interval (embedding + optional LLM)

## 5) Run pytest (new terminal)
```bash
source .venv/bin/activate
pytest -q tests/test_retrieval_service_api.py
```

## 6) Ingest + curl sanity tests (new terminal)
```bash
source .venv/bin/activate

curl -s http://127.0.0.1:8011/health

python scripts/ingest_markdown_to_retrieval.py \
  --base-url http://127.0.0.1:8011 \
  --data-dir data \
  --files 02_12_2025_CSBH_574_NOBLE_PALACE_TAY_THANG_LONG_HDBM.md CONCEPT_THIET_KE_08_02_2025_only_hang_muc_noi_dung.md

curl -s -X POST http://127.0.0.1:8011/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"giá bán khoảng bao nhiêu", "top_k": 5}'

curl -s -X POST http://127.0.0.1:8011/retrieve/project-grounded \
  -H "Content-Type: application/json" \
  -d '{"query":"co can nao gan truong hoc va benh vien", "retrieval_intent":"family + daily convenience", "top_k": 5}'
```

## 7) Run benchmark (new terminal)
```bash
source .venv/bin/activate
python3 scripts/benchmark_retrieval.py \
  --base-url http://127.0.0.1:8011 \
  --cases data/sample_queries.json
```
