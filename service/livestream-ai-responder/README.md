# Livestream AI Responder

Service tự động đọc comment từ Redis queue (đẩy bởi `livestream-tiktok-service`) và gửi vào **RAG Sales Agent** để nhận phản hồi AI, log ra console theo thời gian thực.

## Kiến trúc

```
TikTok Live ──comment──▶ livestream-service ──LPUSH──▶ Redis Queue
                                                              │
                                                       BRPOP (FIFO)
                                                              │
                                               livestream-ai-responder
                                                              │
                                                    POST /sales/chat
                                                              │
                                                       rag-service
                                                     (LightRAG + LangGraph)
                                                              │
                                                       AI Reply → Log
```

## File structure

```
service/livestream-ai-responder/
├── Dockerfile
├── requirements.txt
├── config.py          # Settings với pydantic-settings
├── redis_consumer.py  # BRPOP từ Redis queue
├── rag_client.py      # HTTP client gọi /sales/chat
└── main.py            # Vòng lặp chính
```

## Cài đặt (tích hợp vào noble_rag docker-compose)

### 1. Đặt TikTok username vào `.env`

```env
# .env (root của Noble_RAG)
TIKTOK_USERNAME=ten_kenh_tiktok   # ← thay bằng username thật, không có @
```

### 2. Build và khởi động

```bash
# Build 2 service mới + khởi động toàn bộ stack
docker compose build livestream-service livestream-ai-responder
docker compose up -d livestream-service livestream-ai-responder
```

Hoặc khởi động toàn bộ stack:
```bash
docker compose up -d
```

### 3. Kiểm tra

```bash
# Health check livestream collector
curl http://localhost:8002/health

# Xem log AI responder (phản hồi comment theo thời gian thực)
docker compose logs -f livestream-ai-responder

# Xem số comment đang chờ trong queue
curl http://localhost:8002/api/queue/length
```

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | URL Redis (Docker network nội bộ) |
| `REDIS_QUEUE_KEY` | `livestream:comments` | Key queue trong Redis |
| `POP_TIMEOUT` | `5` | Giây BRPOP chờ comment mới |
| `RAG_SERVICE_URL` | `http://rag-service:8000` | URL RAG service |
| `LIVESTREAM_SESSION_ID` | `livestream_session` | Prefix session ID (thêm `_{room_id}`) |
| `RAG_REQUEST_TIMEOUT` | `60` | Timeout gọi RAG agent (giây) |
| `MAX_RAG_RETRIES` | `2` | Số lần retry khi RAG lỗi |
| `RETRY_DELAY_SEC` | `2.0` | Giây chờ giữa retry |
| `LOG_LEVEL` | `INFO` | Mức log |

## Mở rộng

AI reply hiện tại được log ra console (`stdout`). Có thể mở rộng trong `main.py` tại phần `if reply:` để:

- **TTS (ElevenLabs)**: Gọi ElevenLabs API để đọc reply thành âm thanh
- **Webhook**: POST reply lên hệ thống bên ngoài
- **Bot TikTok**: Tích hợp TikTok Live Comment API để tự động trả lời trên stream

## Tắt service

```bash
# Dừng chỉ 2 service livestream
docker compose stop livestream-service livestream-ai-responder

# Dừng + xoá container
docker compose rm -f livestream-service livestream-ai-responder
```
