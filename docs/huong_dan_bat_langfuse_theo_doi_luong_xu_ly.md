# Huong dan bat Langfuse de theo doi luong xu ly

## 1) Da duoc tich hop trong code

Project da duoc gan tracing tai:
- `orchestrator_service/app.py`: trace theo tung request `/sales/query`
  - `orchestrator.analyze_turn`
  - `orchestrator.consult_discovery`
  - `orchestrator.project_grounded.direct|chain`
  - `orchestrator.sales_state_engine`
  - `orchestrator.reply_planning`
  - `orchestrator.reply_synthesis`
- `retrieval_service/app.py`: trace endpoint `/retrieve`, `/retrieve/project-grounded`, `/ingest`
- `retrieval_service/pipeline.py`: trace chi tiet multi-pass grounded retrieval
  - `candidate_pass`
  - `proximity_pass`
  - `evidence_pass`
  - `shape_response`

## 2) Bien moi truong

Da co san trong cac file env:
- `deploy/.env.orchestrator`
- `deploy/.env.retrieval`
- `deploy/env/orchestrator.env`
- `deploy/env/retrieval.env`

Can dien:

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_FLUSH_AT_REQUEST_END=false
```

## 3) Chay docker

Vi da them dependency Python (`langfuse`), can rebuild image it nhat 1 lan:

```bash
docker compose -f deploy/docker-compose.yml build retrieval-service orchestrator-service
docker compose -f deploy/docker-compose.yml up -d
```

## 4) Theo doi trace

- Moi request `/sales/query` se tao trace o Orchestrator.
- Retrieval nhan header trace/session/user tu Orchestrator de lien ket ngu canh canh xu ly.
- Neu chua cau hinh key hoac `LANGFUSE_ENABLED=false`, he thong chay binh thuong (no-op).
