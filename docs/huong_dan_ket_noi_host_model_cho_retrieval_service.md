# Huong dan ket noi host_model cho retrieval_service

Tai lieu nay dua tren repo `https://github.com/tran-kim-quang/host_model`.

> Cap nhat: server da chuyen sang gateway `/ollama/...`. Vui long uu tien doc tai lieu moi:
> `docs/huong_dan_goi_endpoint_server_va_su_dung_model.md`

## 1) Chuan bi may host_model

Tren may server model:

1. Clone va chay host_model
2. Dam bao `.env` cua host_model co:
   - `API_HOST=0.0.0.0`
   - `API_PORT=8000`
   - `SECURITY_ENABLED=true`
   - `API_KEYS=<api-key-cua-ban>`
3. Start server:

```bash
python main.py
```

Kiem tra:

```bash
curl http://<HOST_MODEL_IP>:8000/health
```

## 2) Cau hinh Noble_RAG (retrieval_service)

Cap nhat `deploy/.env.retrieval` (hoac `deploy/env/retrieval.env`):

```env
EMBEDDING_BACKEND=remote
EMBEDDING_API_URL=http://host.docker.internal:8000/embed
EMBEDDING_API_FORMAT=host_model
EMBEDDING_API_KEY=<api-key-cua-host_model>
EMBEDDING_API_KEY_HEADER=X-API-Key
EMBEDDING_MODEL=qwen3-embedding:4b
EMBEDDING_DIM=2560
```

Neu host_model nam o may LAN khac (khong phai host cung Docker engine), doi:

```env
EMBEDDING_API_URL=http://<HOST_MODEL_IP>:8000/embed
```

## 3) Deploy lai stack

```bash
cd /path/to/Noble_RAG
docker compose -f deploy/docker-compose.yml up -d --build
```

## 4) Test ket noi embedding endpoint

Test truc tiep tu retrieval container:

```bash
docker compose -f deploy/docker-compose.yml exec -T retrieval-service sh -lc \
  "python - <<'PY'
import json, urllib.request
url='http://host.docker.internal:8000/embed'
payload={'text':'ket noi host_model','model':'qwen3-embedding:4b'}
req=urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST', headers={'Content-Type':'application/json','X-API-Key':'<api-key-cua-host_model>'})
with urllib.request.urlopen(req, timeout=20) as r:
    data=json.loads(r.read().decode())
print('dim=', len(data.get('embedding', [])))
PY"
```

Neu `dim` dung voi `EMBEDDING_DIM` thi retrieval service se khoi dong on dinh.

## 5) Loi thuong gap

- 401/403: sai `EMBEDDING_API_KEY` hoac bi chan allowlist IP/hostname ben host_model.
- timeout/unreachable: sai IP, firewall chua mo cong `8000`, hoac server chua bind `0.0.0.0`.
- dimension mismatch: `EMBEDDING_DIM` khong trung kich thuoc embedding cua model.
