# Huong dan goi endpoint server va su dung model

Tai lieu nay huong dan client goi den server model (Neolab) sau refactor gateway moi.

## 1) Thong tin ket noi

- Server LAN IP: `10.50.56.103`
- Base URL: `http://10.50.56.103:8000`
- API key header: `X-API-Key`
- API key mau: `change-this-to-a-strong-key`

> Luu y: sau refactor, nen su dung endpoint `/ollama/...` thay vi `/chat` va `/embed` cu.

## 2) Danh sach endpoint can dung

- `GET /health`: kiem tra server song
- `GET /models`: lay danh sach model dang cung cap
- `POST /ollama/api/generate`: goi LLM (chat/text generation)
- `POST /ollama/api/embeddings`: lay embedding vector

## 3) Test nhanh bang curl

### 3.1 Kiem tra health

```bash
curl -sS -H "X-API-Key: change-this-to-a-strong-key" \
  http://10.50.56.103:8000/health
```

### 3.2 Liet ke models

```bash
curl -sS -H "X-API-Key: change-this-to-a-strong-key" \
  http://10.50.56.103:8000/models
```

Ky vong co it nhat 2 model:

- `gemma4:26b` (LLM)
- `qwen3-embedding:8b` (embedding)

### 3.3 Goi LLM (khong stream)

```bash
curl -sS -H "X-API-Key: change-this-to-a-strong-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:26b",
    "prompt": "xin chao",
    "stream": false
  }' \
  http://10.50.56.103:8000/ollama/api/generate
```

### 3.4 Goi LLM (streaming)

```bash
curl -sS -N -H "X-API-Key: change-this-to-a-strong-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:26b",
    "prompt": "xin chao",
    "stream": true
  }' \
  http://10.50.56.103:8000/ollama/api/generate
```

Neu stream on, response se ve tung chunk token.

### 3.5 Goi embedding

```bash
curl -sS -H "X-API-Key: change-this-to-a-strong-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding:8b",
    "prompt": "xin chao"
  }' \
  http://10.50.56.103:8000/ollama/api/embeddings
```

Response se co truong `embedding` (list float).

## 4) Mau goi bang Python

```python
import requests

BASE_URL = "http://10.50.56.103:8000"
HEADERS = {"X-API-Key": "change-this-to-a-strong-key"}

# 1) models
models = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=20)
models.raise_for_status()
print(models.json())

# 2) generate
gen_payload = {
    "model": "gemma4:26b",
    "prompt": "xin chao",
    "stream": False,
}
gen = requests.post(
    f"{BASE_URL}/ollama/api/generate",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=gen_payload,
    timeout=120,
)
gen.raise_for_status()
print(gen.json().get("response"))

# 3) embedding
emb_payload = {
    "model": "qwen3-embedding:8b",
    "prompt": "xin chao",
}
emb = requests.post(
    f"{BASE_URL}/ollama/api/embeddings",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=emb_payload,
    timeout=120,
)
emb.raise_for_status()
print("embedding_dim =", len(emb.json().get("embedding", [])))
```

## 5) Cau hinh Noble_RAG de dung embedding tren server

Cap nhat ca 2 file:

- `deploy/.env.retrieval`
- `deploy/env/retrieval.env`

Gia tri khuyen nghi:

```env
EMBEDDING_BACKEND=remote
EMBEDDING_API_URL=http://10.50.56.103:8000/ollama/api/embeddings
EMBEDDING_API_FORMAT=ollama
EMBEDDING_API_KEY=change-this-to-a-strong-key
EMBEDDING_API_KEY_HEADER=X-API-Key
EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIM=4096
```

Sau do restart retrieval-service:

```bash
docker compose -f deploy/docker-compose.yml up -d --force-recreate retrieval-service
```

## 6) Loi thuong gap va cach xu ly

- `401/403`: sai API key hoac server bat auth key khac.
- `404` voi `/chat` hoac `/embed`: dang goi endpoint cu, doi sang `/ollama/api/generate` va `/ollama/api/embeddings`.
- `connection refused/timeout`: sai IP, server chua mo port `8000`, hoac firewall chan.
- `dimension mismatch`: `EMBEDDING_DIM` khong khop model embedding dang dung.

## 7) Streaming va warm model (quan trong cho UX)

- Streaming can bat o request phia client:
  - `POST /ollama/api/generate` voi `"stream": true`
- Server can ho tro stream passthrough (gateway -> ollama).
- Retrieval/embedding khong can stream.

Neu muon giam cold start, bat warm mechanism o retrieval-service:

```env
MODEL_WARM_ENABLED=true
MODEL_WARM_INTERVAL_SEC=120
LLM_WARM_ENABLED=true
LLM_WARM_API_URL=http://10.50.56.103:8000/ollama/api/generate
LLM_WARM_MODEL=gemma4:26b
```

## 8) Checklist nhanh truoc khi go live

- Goi duoc `GET /health`
- Goi duoc `GET /models`
- LLM tra loi duoc voi `/ollama/api/generate`
- Embedding tra ve vector dung dim
- Retrieval-service `healthy` sau khi restart
