# Noble RAG System Usage Guide

## Mục tiêu

Tài liệu này mô tả:

- các service đang chạy trong hệ thống
- các endpoint HTTP chính
- request/response mẫu
- workflow khuyến nghị cho document ingestion, vision identity, sales chat, và whisper

Tài liệu này bám theo code hiện tại trong repo.

## 1. Thành phần hệ thống

Hệ thống hiện có các service chính:

- `rag-service`: API chính cho document ingestion, sales agent, query stream, health/status
- `vision-service`: nhận keyframe ảnh từ browser/mobile để nhận diện khách hàng theo `session_id`
- `whisper-service`: speech-to-text cho audio
- `rag-ui`: UI quản trị tài liệu cơ bản
- `postgres`, `qdrant`, `redis`, `ollama`: hạ tầng lưu trữ và embedding/runtime

## 2. URL mặc định khi chạy bằng Docker Compose

Theo `docker-compose.yml` và `.env` hiện tại:

- RAG API: `http://localhost:8010`
- Vision API: `http://localhost:8020`
- Whisper API: `http://localhost:8001`
- RAG UI: `http://localhost:8501`
- pgAdmin: `http://localhost:5050`
- Ollama: `http://localhost:11434`
- Qdrant: `http://localhost:6333`

Lưu ý:

- `RAG_SERVICE_PORT` hiện đang được đặt là `8010` trong `.env`
- nếu đổi `.env` hoặc override env khi chạy compose, các URL có thể thay đổi

## 3. Khuyến nghị workflow tổng thể

Workflow khuyến nghị cho frontend/browser:

1. Tạo `session_id` ở frontend.
2. Nếu có nhận diện khách hàng bằng camera browser, chụp 1 keyframe và gửi tới `POST /vision/identify`.
3. Sau khi có kết quả vision, gọi `POST /sales/chat/stream` với cùng `session_id`.
4. Các turn chat sau tiếp tục dùng lại đúng `session_id`.
5. Khi cần đọc trạng thái hoặc hồ sơ lead, gọi các endpoint `/sales/state/*`, `/sales/lead/*`, `/sales/history/*`, `/sales/session/*/snapshot`.
6. Khi cần đóng phiên, gọi `POST /sales/session/{session_id}/close`.

Workflow cho document ingestion:

1. Upload file UTF-8 text vào `POST /upload-document`.
2. Kiểm tra tài liệu bằng `GET /documents`.
3. Nếu cần theo dõi `track_id` thì dùng `GET /documents/track/{track_id}`.
4. Khi cần xóa tài liệu, gọi `DELETE /documents/{doc_id}`.

Workflow cho voice:

1. Upload audio tới Whisper.
2. Lấy transcript.
3. Gửi transcript sang `POST /sales/chat/stream` hoặc `POST /sales/chat`.

## 4. RAG Service Endpoints

Base URL:

```text
http://localhost:8010
```

### 4.1 Health và model info

#### `GET /health`

Dùng để health check service.

Ví dụ:

```bash
curl http://localhost:8010/health
```

Response mẫu:

```json
{
  "status": "healthy",
  "service": "rag-service",
  "llm_provider": "gemini",
  "llm_model": "gemini-2.5-flash",
  "embedding_model": "embeddinggemma:300m"
}
```

#### `GET /models`

Trả về thông tin LLM, embedding, và storage backend.

```bash
curl http://localhost:8010/models
```

#### `GET /status`

Trả về trạng thái runtime tổng quan.

```bash
curl http://localhost:8010/status
```

### 4.2 Document endpoints

#### `POST /upload-document`

Upload tài liệu text UTF-8 vào RAG.

Request:

- `multipart/form-data`
- field `file`: file text UTF-8
- field `storage_type`: `graph`, `vector`, hoặc `both`
- field `metadata`: JSON string optional

Ví dụ:

```bash
curl -X POST http://localhost:8010/upload-document \
  -F "file=@./sample.txt" \
  -F "storage_type=graph" \
  -F 'metadata={"source":"manual_upload"}'
```

Response mẫu:

```json
{
  "status": "success",
  "document_id": "abc123",
  "chunks_count": 12,
  "storage_type": "graph",
  "message": "Document processed (12 chunks in 1.24s)"
}
```

Lưu ý:

- file phải decode được bằng UTF-8
- file rỗng sẽ bị `400`
- field `metadata` được nhận ở API, nhưng hiện không nên coi là contract metadata hoàn chỉnh cho downstream flow

#### `GET /documents`

Liệt kê tài liệu đã ingest.

Ví dụ:

```bash
curl "http://localhost:8010/documents?limit=50"
```

#### `GET /documents/track/{track_id}`

Kiểm tra trạng thái ingest theo `track_id`.

```bash
curl http://localhost:8010/documents/track/<track_id>
```

#### `DELETE /documents/{doc_id}`

Xóa tài liệu theo `document_id`.

```bash
curl -X DELETE http://localhost:8010/documents/<doc_id>
```

### 4.3 Sales endpoints

#### `POST /sales/chat`

Endpoint non-streaming cho sales agent.

Request JSON:

```json
{
  "session_id": "sales_demo_001",
  "message": "Chào em",
  "raw_transcript": "Chào em"
}
```

Ví dụ:

```bash
curl -X POST http://localhost:8010/sales/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sales_demo_001",
    "message": "Chào em"
  }'
```

Response mẫu:

```json
{
  "session_id": "sales_demo_001",
  "response": "Em chào Anh/Chị...",
  "sales_state": "greeting",
  "lead_profile": {},
  "missing_slots": ["budget"]
}
```

#### `POST /sales/chat/stream`

Endpoint khuyến nghị cho frontend chat.

Request JSON:

```json
{
  "session_id": "sales_demo_001",
  "message": "Anh đang tìm căn 3 phòng ngủ ở Tây Hồ",
  "raw_transcript": "Anh đang tìm căn 3 phòng ngủ ở Tây Hồ"
}
```

Ví dụ:

```bash
curl -N -X POST http://localhost:8010/sales/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sales_demo_001",
    "message": "Anh đang tìm căn 3 phòng ngủ ở Tây Hồ"
  }'
```

Stream trả về `application/x-ndjson`.

Các event quan trọng:

- `phase=thinking_ack`: câu trả lời đệm ngay khi bắt đầu xử lý
- `phase=response`: các chunk nội dung chính
- `phase=error`: lỗi xử lý
- `phase=complete`: event cuối, có `done=true`

Event cuối thường có dạng:

```json
{
  "chunk": "",
  "done": true,
  "phase": "complete",
  "session_id": "sales_demo_001",
  "sales_state": "need_discovery",
  "missing_slots": ["budget"],
  "lead_profile": {},
  "latency_sec": 1.234
}
```

#### `GET /sales/lead/{session_id}`

Đọc lead profile hiện tại.

```bash
curl http://localhost:8010/sales/lead/sales_demo_001
```

#### `PATCH /sales/lead/{session_id}`

Update thủ công lead profile.

Ví dụ:

```bash
curl -X PATCH http://localhost:8010/sales/lead/sales_demo_001 \
  -H "Content-Type: application/json" \
  -d '{
    "updates": {
      "budget": "15 ty",
      "family_size": 4
    }
  }'
```

#### `GET /sales/state/{session_id}`

Đọc sales state hiện tại.

```bash
curl http://localhost:8010/sales/state/sales_demo_001
```

#### `GET /sales/history/{session_id}`

Đọc lịch sử chat đã lưu.

```bash
curl http://localhost:8010/sales/history/sales_demo_001
```

#### `GET /sales/session/{session_id}/snapshot`

Đọc snapshot local của session.

```bash
curl http://localhost:8010/sales/session/sales_demo_001/snapshot
```

#### `POST /sales/recommendations/refresh`

Trigger lại một vòng gợi ý sản phẩm cho session.

Ví dụ:

```bash
curl -X POST "http://localhost:8010/sales/recommendations/refresh?session_id=sales_demo_001"
```

#### `POST /sales/followup/generate`

Sinh câu follow-up cho lead hiện tại.

```bash
curl -X POST "http://localhost:8010/sales/followup/generate?session_id=sales_demo_001"
```

#### `POST /sales/session/{session_id}/close`

Export session ra file txt rồi clear cache tạm thời.

```bash
curl -X POST http://localhost:8010/sales/session/sales_demo_001/close
```

### 4.4 Query alias endpoint

#### `POST /query/stream`

Đây là endpoint stream backward-compatible.

Nó có thể:

- route sang search nếu câu hỏi ngoài knowledge base
- hoặc route sang sales/RAG flow

Ví dụ:

```bash
curl -N -X POST http://localhost:8010/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sales_demo_001",
    "query": "Pháp lý Noble Crystal Tây Hồ thế nào?"
  }'
```

Request fields:

- `query`
- `message`
- `session_id`
- `top_k`
- `return_structured_output`

Lưu ý:

- frontend mới nên ưu tiên `POST /sales/chat/stream`
- `POST /query/stream` phù hợp cho compatibility hoặc các client cũ

## 5. Vision Service Endpoints

Base URL:

```text
http://localhost:8020
```

Vision service hiện được thiết kế để nhận keyframe từ browser/mobile, không phụ thuộc vào camera local trong backend.

### 5.1 `POST /vision/identify`

Nhận một ảnh và bind kết quả identity vào `session_id`.

Request:

- `multipart/form-data`
- `session_id`: bắt buộc
- `source`: optional, ví dụ `browser`, `mobile_web`, `kiosk`
- `image`: file ảnh

Ví dụ:

```bash
curl -X POST http://localhost:8020/vision/identify \
  -F "session_id=sales_demo_001" \
  -F "source=browser" \
  -F "image=@./face.jpg"
```

Response mẫu:

```json
{
  "session_id": "sales_demo_001",
  "customer_id": "2a8c...",
  "is_existing_customer": true,
  "gender_estimate": "male",
  "age_group_estimate": "30-39",
  "match_score": 0.87,
  "customer_context": {
    "lead_profile": {},
    "recent_chat_history": [],
    "recent_session_context": {},
    "purchase_history": []
  }
}
```

Lưu ý:

- ảnh phải có `content-type` bắt đầu bằng `image/`
- service kỳ vọng ảnh có đúng 1 khuôn mặt hợp lệ
- nếu ảnh không decode được hoặc có quá nhiều mặt, service sẽ trả `400`

### 5.2 `GET /vision/session/{session_id}`

Đọc session đã bind với customer nào.

```bash
curl http://localhost:8020/vision/session/sales_demo_001
```

Đây là endpoint mà `rag-service` gọi để hydrate identity vào sales flow.

### 5.3 `GET /vision/customer/{customer_id}`

Đọc thông tin tổng hợp của một customer đã được Vision nhận diện.

```bash
curl http://localhost:8020/vision/customer/<customer_id>
```

## 6. Whisper Service Endpoints

Base URL:

```text
http://localhost:8001
```

### 6.1 `GET /health`

```bash
curl http://localhost:8001/health
```

### 6.2 `GET /models`

```bash
curl http://localhost:8001/models
```

### 6.3 `POST /transcribe`

Upload audio và nhận full transcript JSON.

Ví dụ:

```bash
curl -X POST http://localhost:8001/transcribe \
  -F "file=@./sample.wav" \
  -F "language=vi" \
  -F "word_timestamps=false"
```

### 6.4 `POST /transcribe/stream`

Upload audio và nhận transcript dạng SSE.

```bash
curl -N -X POST http://localhost:8001/transcribe/stream \
  -F "file=@./sample.wav" \
  -F "language=vi"
```

## 7. Workflow khuyến nghị cho frontend

### 7.1 Browser capture + sales chat

Đây là workflow nên dùng cho frontend mới.

1. Frontend tạo `session_id`.
2. Browser mở camera bằng `getUserMedia`.
3. User chụp 1 keyframe.
4. Frontend gửi keyframe tới `POST /vision/identify` với cùng `session_id`.
5. Frontend bắt đầu chat bằng `POST /sales/chat/stream`.
6. Mỗi turn tiếp theo tiếp tục gửi cùng `session_id`.
7. Khi cần khôi phục UI hoặc dashboard, frontend đọc thêm:
   - `GET /sales/state/{session_id}`
   - `GET /sales/lead/{session_id}`
   - `GET /sales/history/{session_id}`
   - `GET /sales/session/{session_id}/snapshot`

### 7.2 Voice workflow

1. Frontend/mobile upload audio tới Whisper.
2. Nhận transcript.
3. Gửi transcript sang `POST /sales/chat/stream`.
4. Nếu muốn lưu transcript gốc, gửi thêm `raw_transcript`.

### 7.3 Search fallback workflow

Nếu vẫn đang dùng client cũ:

1. Gọi `POST /query/stream`.
2. Kiểm tra event cuối:
   - `route_category=SEARCH` nếu câu hỏi đi ra ngoài knowledge base nội bộ
   - `route_category=RAG` nếu đi qua sales/RAG flow

## 8. Các lưu ý tích hợp

- `session_id` là khóa quan trọng nhất để nối vision, sales, history, và snapshot.
- Nếu frontend dùng vision, hãy gửi ảnh trước turn chat đầu tiên để sales flow có thể hydrate identity kịp thời.
- `POST /sales/chat/stream` là endpoint chat nên ưu tiên cho UI mới.
- `POST /sales/chat` phù hợp cho backend integration cần response JSON một lần.
- `POST /query/stream` nên coi là endpoint compatibility.
- `upload-document` hiện phù hợp nhất cho file text UTF-8.
- Vision service đang xử lý theo keyframe upload, không còn cần backend tự bật camera local.

## 9. Checklist test nhanh

### Kiểm tra RAG API

```bash
curl http://localhost:8010/health
curl http://localhost:8010/models
```

### Kiểm tra Vision API

```bash
curl http://localhost:8020/vision/session/test_session
```

Nếu chưa bind session nào, endpoint này có thể trả `404`, điều đó vẫn bình thường.

### Kiểm tra Whisper API

```bash
curl http://localhost:8001/health
```

### Test sales stream nhanh

```bash
curl -N -X POST http://localhost:8010/sales/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "manual_test_001",
    "message": "Chào em"
  }'
```

## 10. Tài liệu liên quan

- `README.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/RAG_SERVICE_GUIDE.txt`
- `service/vision/guide.md`
