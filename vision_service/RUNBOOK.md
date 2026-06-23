# Vision Service Runbook

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r vision_service/requirements.txt
```

Run service:

```bash
VISION_FACE_DB=face_db \
VISION_MATCH_THRESHOLD=0.45 \
uvicorn vision_service.app:app --host 127.0.0.1 --port 8031
```

Face DB structure:

```text
face_db/
  Linh/
    01.jpg
    02.jpg
  Nam/
    01.jpg
```

Health check:

```bash
curl -s http://127.0.0.1:8031/health
```

Identify by JSON:

```bash
curl -s -X POST http://127.0.0.1:8031/vision/identify \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<base64-or-data-url>"}'
```

Call orchestrator unified endpoint:

```bash
curl -s -X POST http://127.0.0.1:8021/sales/query-with-vision \
  -H "Content-Type: application/json" \
  -d '{"message":"Tu van tong quan cho toi","image_base64":"<base64-or-data-url>"}'
```
