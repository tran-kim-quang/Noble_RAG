# Vision Service (FastAPI)

Service này cho phép máy khác gọi tới model vision chạy trên máy hiện tại.

## 1) Cai dat

```bash
pip install -r service/vision/requirements.txt
```

## 2) Chay service

```bash
uvicorn service.vision.app:app --host 0.0.0.0 --port 8002
```

## 3) Endpoint cho may khac goi

- Method: `POST`
- URL: `http://<host-name-hoac-ip>:8002/api/vision/generate`
- Body:

```json
{
  "prompt": "Mo ta hinh anh theo nganh bat dong san",
  "host_name": "vision-a.local",
  "images": [],
  "stream": false,
  "options": {
    "temperature": 0.2
  }
}
```

`host_name` la tuy chon. Neu khong gui, service se tu lay tu header/request host.

## 4) Cau hinh theo hostname de mo rong

Co the cau hinh map `hostname -> model + ollama url` qua bien moi truong:

```env
MODEL_UNDERSTAND_VISION=gemma4:latest
VISION_DEFAULT_OLLAMA_BASE_URL=http://127.0.0.1:11434
VISION_DEFAULT_TIMEOUT_SECONDS=120
VISION_HOST_CONFIG={"vision-a.local":{"model":"gemma4:latest","ollama_base_url":"http://127.0.0.1:11434","timeout_seconds":120},"vision-b.local":{"model":"llava:latest","ollama_base_url":"http://127.0.0.1:11434","timeout_seconds":180}}
```

Thu tu resolve host:
1. `host_name` trong body.
2. Header `X-Vision-Host`.
3. Header `X-Forwarded-Host`.
4. Host trong URL request.
5. Fallback `default`.

## 5) Vi du goi tu may khac

```bash
curl -X POST "http://vision-a.local:8002/api/vision/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Tom tat noi dung anh theo van phong sales\",\"host_name\":\"vision-a.local\"}"
```

## 6) Health check

```bash
curl http://<host>:8002/healthz
```
