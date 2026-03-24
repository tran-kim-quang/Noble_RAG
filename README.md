## Noble RAG

### Cài đặt và chạy

```bash
# Cài đặt
docker-compose up --build

# Chạy
docker-compose up

# Dừng
docker-compose down
```

### Web UI cho RAG Client

```bash
# Chạy UI (mặc định http://localhost:8501)
poetry run python main.py
```

Biến môi trường tùy chọn:

- `RAG_SERVICE_URL` (mặc định: `http://localhost:8000`)
- `WEB_UI_HOST` (mặc định: `0.0.0.0`)
- `WEB_UI_PORT` (mặc định: `8501`)

