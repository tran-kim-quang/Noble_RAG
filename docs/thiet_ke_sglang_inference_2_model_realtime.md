# Thiet ke SGLang cho inference 2 model (LLM + Embedding) gan realtime

## 1) Scope va muc tieu

Project hien tai dang dung 2 model runtime:
- LLM decider/synthesis: `gemma4:*` (qua `ORCHESTRATOR_DECIDER_*`)
- Embedding retrieval: `qwen3-embedding:*` (qua `EMBEDDING_*`)

Muc tieu:
- Giu nguyen logic app (`orchestrator_service`, `retrieval_service`)
- Chuyen serving backend sang SGLang
- Toi uu TTFT/throughput de UX gan realtime

## 2) Kien truc de xuat

### 2.1. Runtime topology (khuyen nghi)

Tach thanh 2 SGLang worker doc lap:
- `sglang-llm` (generate/chat): port `31000`
- `sglang-embedding` (`--is-embedding`): port `32000`

Ly do:
- Decode (LLM) va embedding co profile tai nguyen khac nhau
- Tranh chen lan queue/KV cache
- De scale rieng theo nhu cau

### 2.2. Optional HA/Gateway

Neu can HA/load balancing:
- Dat `SGLang Model Gateway` truoc cac worker
- Dung policy `cache_aware` (default) hoac `power_of_two`
- Bat retry, queue timeout, readiness health checks

## 3) Mapping endpoint voi code hien tai

### 3.1. Orchestrator (LLM)

Code dang goi payload kieu Ollama (`/api/generate`, field `response`).

=> Dung SGLang Ollama-compatible API cho nhanh, it doi code:
- `ORCHESTRATOR_DECIDER_API_URL=http://host.docker.internal:31000/api/generate`
- `ORCHESTRATOR_DECIDER_MODEL=<model-name-da-serve>`

### 3.2. Retrieval (embedding)

`retrieval_service` da support `EMBEDDING_API_FORMAT=openai`.

=> Chuyen sang OpenAI embeddings endpoint cua SGLang:
- `EMBEDDING_API_FORMAT=openai`
- `EMBEDDING_API_URL=http://host.docker.internal:32000/v1/embeddings`
- `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B`
- `EMBEDDING_DIM=4096`

## 4) Lenh launch SGLang (single-node)

> Ghi chu: docs khuyen nghi entrypoint `sglang serve`.

### 4.1. LLM worker

```bash
CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path google/gemma-3-27b-it \
  --host 0.0.0.0 \
  --port 31000 \
  --tp-size 1 \
  --mem-fraction-static 0.86 \
  --chunked-prefill-size 4096 \
  --max-running-requests 256 \
  --schedule-policy lpm \
  --schedule-conservativeness 0.7 \
  --stream-interval 1 \
  --api-key <SGLANG_API_KEY>
```

Neu may co >=2 GPU danh cho LLM:
- uu tien tang `--dp-size` de tang throughput
- ket hop `--tp-size` khi model lon

### 4.2. Embedding worker

```bash
CUDA_VISIBLE_DEVICES=1 sglang serve \
  --model-path Qwen/Qwen3-Embedding-8B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 32000 \
  --tp-size 1 \
  --mem-fraction-static 0.90 \
  --max-running-requests 1024 \
  --api-key <SGLANG_API_KEY>
```

## 5) Cap nhat env trong project

### 5.1. `deploy/env/orchestrator.env`

```env
ORCHESTRATOR_DECIDER_ENABLED=true
ORCHESTRATOR_DECIDER_API_URL=http://host.docker.internal:31000/api/generate
ORCHESTRATOR_DECIDER_MODEL=google/gemma-3-27b-it
ORCHESTRATOR_DECIDER_TIMEOUT_SEC=30
ORCHESTRATOR_DECIDER_TEMPERATURE=0.2
ORCHESTRATOR_DECIDER_KEEP_ALIVE=30m
```

### 5.2. `deploy/env/retrieval.env`

```env
EMBEDDING_BACKEND=remote
EMBEDDING_API_URL=http://host.docker.internal:32000/v1/embeddings
EMBEDDING_API_FORMAT=openai
EMBEDDING_API_TIMEOUT_SEC=10
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
EMBEDDING_DIM=4096

LLM_WARM_ENABLED=true
LLM_WARM_API_URL=http://host.docker.internal:31000/api/generate
LLM_WARM_MODEL=google/gemma-3-27b-it
LLM_WARM_TIMEOUT_SEC=10
LLM_WARM_KEEP_ALIVE=30m
```

## 6) Tuning de dat gan realtime

Thu tu tuning khuyen nghi:
1. Chot `--mem-fraction-static` (uu tien tang KV cache pool nhung van de lai 5-8GB available GPU mem)
2. Khi OOM prefill: giam `--chunked-prefill-size` (4096 -> 2048)
3. Khi OOM decode: giam `--max-running-requests`
4. Neu server qua "bao thu" (`queue > 0`, token usage thap): giam `--schedule-conservativeness` (vd `0.3`)
5. Neu bi retract request vi day KV cache: tang `--schedule-conservativeness` (vd `1.3`)
6. GPU du bo nho: uu tien DP de tang throughput

## 7) Speculative decoding (optional)

Cho workload LLM nhe-to-vua, co the thu:
- `--speculative-algorithm EAGLE`/`EAGLE3`
- `--speculative-draft-model-path ...`

Khuyen nghi benchmark A/B truoc khi bat production vi loi ich phu thuoc model/hardware.

## 8) Health check va readiness

Kiem tra worker:
- `GET /health`
- `GET /v1/models`

Neu dung gateway:
- `GET /liveness`
- `GET /readiness`
- `GET /engine_metrics`

## 9) KPI de xac nhan "gan realtime"

De xuat dat SLO ban dau:
- P50 TTFT <= 700ms
- P95 TTFT <= 1500ms
- P95 end-to-end `/sales/query` <= 2500ms (khong tinh network xa)
- Embedding P95 <= 120ms/query ngan

Chay benchmark theo 3 muc tai: 10, 30, 60 concurrent users.

## 10) Luu y tuong thich model

- SGLang docs list ho tro Gemma (v1/v2/v3) va Qwen3-Embedding.
- Neu ban can dung dung `gemma4:*`, can test startup truoc trong moi truong cua ban.
- Neu `gemma4` gap loi runtime/transformers, fallback nhanh la dung Gemma3 hoac Qwen3-Instruct cho route decider/synthesis.

