## Noble RAG

### Tài liệu nên đọc trước

- [System Usage Guide](./docs/SYSTEM_USAGE_GUIDE.md): hướng dẫn endpoint, payload, URL, và workflow tích hợp toàn hệ thống
- [Project Overview](./docs/PROJECT_OVERVIEW.md): tổng quan kiến trúc và module
- [Vision Guide](./service/vision/guide.md): chi tiết luồng vision identity

### Cài đặt và chạy

```bash
# Cài đặt
docker compose up --build

# Chạy
docker compose up

# Dừng
docker compose down
```

### Đóng gói service + chạy test trong Docker

```bash
# Build toàn bộ image service theo docker-compose
docker compose build

# Chạy riêng bộ test knowledge_base (container one-shot)
docker compose --profile test run --rm rag-tests
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
