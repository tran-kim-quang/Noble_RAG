# Noble RAG - Huong dan start he thong

Tai lieu nay huong dan start nhanh he thong Noble RAG trong repo nay, gom:
- `orchestrator-service` (API chinh): `:8021`
- `retrieval-service` (noi bo): `:8011`
- `vision-service` (noi bo): `:8031`
- `qdrant` (vector DB, noi bo)
- `ollama` (embedding backend, noi bo)

## 1) Yeu cau moi truong

- Linux/WSL2
- Docker Engine + Docker Compose plugin
- Neu dung `deploy/docker-compose.yml` mac dinh: can NVIDIA GPU + NVIDIA Container Toolkit (vi `ollama` dang cau hinh `runtime: nvidia`)

Kiem tra nhanh:

```bash
docker --version
docker compose version
```

## 2) Quick start (khuyen nghi, bang Docker)

### Buoc 1: vao thu muc du an

```bash
cd /home/neolab/noble/Noble_RAG
```

### Buoc 2: chuan bi file env (lan dau)

Neu chua co file env runtime, copy tu file mau:

```bash
cp -n deploy/retrieval.example.env deploy/.env.retrieval
cp -n deploy/orchestrator.example.env deploy/.env.orchestrator
cp -n deploy/vision.example.env deploy/.env.vision
```

### Buoc 3: cap nhat cau hinh bat buoc

Sua file `deploy/.env.orchestrator`:

- `ORCHESTRATOR_DECIDER_API_KEY`
- `ORCHESTRATOR_SYNTHESIS_API_KEY`
- Cac bien timeout/history dang so phai la so hop le (khong them ky tu)

Kiem tra file `deploy/.env.retrieval`:

- `EMBEDDING_API_URL=http://ollama:11434/api/embeddings`
- `EMBEDDING_MODEL=qwen3-embedding:8b`
- `EMBEDDING_DIM=4096`

Neu dung vision service:

- Dat anh khach hang theo cau truc `face_db/<ten_khach>/image1.jpg`
- Kiem tra `deploy/.env.vision`
- Kiem tra `deploy/.env.orchestrator` co:
  - `VISION_ENABLED=true`
  - `VISION_SERVICE_URL=http://vision-service:8031`

### Buoc 4: start stack

```bash
./deploy/scripts/start.sh
./deploy/scripts/status.sh
```

### Buoc 5: pull model embedding cho Ollama (nen lam 1 lan)

```bash
docker exec -it noble_ollama ollama pull qwen3-embedding:8b
```

### Buoc 6: ingest du lieu markdown vao retrieval

```bash
docker compose -f deploy/docker-compose.yml exec -T retrieval-service sh -lc \
  "python /app/scripts/ingest_markdown_to_retrieval.py \
    --base-url http://retrieval-service:8011 \
    --data-dir /app/data"
```

Neu may dang dung `docker-compose` legacy, thay `docker compose` bang `docker-compose`.

### Buoc 7: health check va smoke test

```bash
./deploy/scripts/health.sh
```

Hoac test tay:

```bash
curl -s http://127.0.0.1:8021/health | python3 -m json.tool

curl -s -X POST http://127.0.0.1:8021/sales/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Gia can 2 phong ngu la bao nhieu?"}' | python3 -m json.tool

curl -s -X POST http://127.0.0.1:8021/sales/query-with-vision \
  -H "Content-Type: application/json" \
  -d '{"message":"Tu van tong quan cho toi","image_base64":"<base64-or-data-url>"}' | python3 -m json.tool
```

## 3) Van hanh hang ngay

```bash
# xem trang thai
./deploy/scripts/status.sh

# xem logs
./deploy/scripts/logs.sh orchestrator-service 200
./deploy/scripts/logs.sh retrieval-service 200
./deploy/scripts/logs.sh vision-service 200
./deploy/scripts/logs.sh qdrant 200

# restart
./deploy/scripts/restart.sh

# stop toan bo stack
./deploy/scripts/stop.sh
```

Bind IP public cho orchestrator:

```bash
ORCH_BIND_IP=0.0.0.0 ./deploy/scripts/restart.sh
```

## 4) Start local dev (khong dung docker-compose full stack)

Che do nay phu hop khi debug nhanh service bang `uvicorn`.

### Buoc 1: tao virtualenv va cai dependencies

```bash
cd /home/neolab/noble/Noble_RAG
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r retrieval_service/requirements.local.txt
pip install -r orchestrator_service/requirements.txt
pip install -r vision_service/requirements.txt
```

### Buoc 2: chay Qdrant local

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}' | grep 6333 || \
  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Buoc 3: chay Ollama local

```bash
ollama serve
```

Mo terminal khac:

```bash
ollama pull qwen3-embedding:8b
```

### Buoc 4: chay retrieval service (terminal 1)

```bash
source .venv/bin/activate

QDRANT_URL=http://127.0.0.1:6333 \
QDRANT_COLLECTION=retrieval_bench_qwen3_8b_local \
EMBEDDING_BACKEND=remote \
EMBEDDING_API_URL=http://127.0.0.1:11434/api/embeddings \
EMBEDDING_API_FORMAT=ollama \
EMBEDDING_MODEL=qwen3-embedding:8b \
EMBEDDING_DIM=4096 \
uvicorn retrieval_service.app:app --host 127.0.0.1 --port 8011
```

### Buoc 5: ingest va chay orchestrator (terminal 2)

```bash
source .venv/bin/activate

python scripts/ingest_markdown_to_retrieval.py \
  --base-url http://127.0.0.1:8011 \
  --data-dir data

RETRIEVAL_SERVICE_URL=http://127.0.0.1:8011 \
ORCHESTRATOR_DEFAULT_TOP_K=5 \
uvicorn orchestrator_service.app:app --host 127.0.0.1 --port 8021
```

### Buoc 6: chay vision service (terminal 3)

```bash
source .venv/bin/activate

VISION_FACE_DB=face_db \
VISION_MATCH_THRESHOLD=0.45 \
uvicorn vision_service.app:app --host 127.0.0.1 --port 8031
```

## 5) Loi thuong gap

- `docker compose/docker-compose not found`
  - Cai Docker Compose plugin hoac `docker-compose`.
- `unknown runtime nvidia` hoac container `ollama` khong len
  - Cai NVIDIA Container Toolkit, hoac doi sang cau hinh embedding remote khong can GPU.
- `401/403` khi goi model API o orchestrator
  - Kiem tra lai `ORCHESTRATOR_DECIDER_API_KEY` va `ORCHESTRATOR_SYNTHESIS_API_KEY`.
- `invalid literal for int()` luc orchestrator start
  - Kiem tra bien so trong `deploy/.env.orchestrator` (vi du `*_TIMEOUT_SEC`, `*_HISTORY_TURNS`) chi duoc de so.
- Khong ra ket qua retrieval nhu ky vong
  - Kiem tra da ingest du lieu chua va model embedding co dung `EMBEDDING_DIM` hay khong.

## 6) Lenh verify runtime tu dong

```bash
./deploy/scripts/verify_runtime.sh
```

Script nay se tu dong: build -> up -> health -> ingest -> query smoke test.
