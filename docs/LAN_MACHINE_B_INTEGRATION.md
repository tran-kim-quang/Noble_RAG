# Hướng Dẫn Tích Hợp Máy B -> Máy A Qua LAN

## 1) Mục tiêu

Máy A chạy backend RAG (FastAPI).  
Máy B (vision/client) gửi dữ liệu user qua LAN để:

1. Mở phiên chat mới hoặc quay lại phiên cũ theo `customer_id`.
2. Gửi tin nhắn hội thoại vào đúng session.

---

## 2) Thông tin kết nối LAN

- Base URL máy A: `http://<MAY_A_IP>:8010`
- Ví dụ trong LAN hiện tại: `http://192.168.100.7:8010`

Lưu ý:

- Máy A và B phải cùng subnet LAN.
- Mở firewall inbound port `8010` trên máy A.

---

## 3) Endpoint chính cho máy B

### `POST /api/v1/session/open`

Dùng để mở session mới hoặc resume session cũ theo `customer_id`.

Request body mẫu:

```json
{
  "customer_id": "cust_001",
  "allow_resume": true,
  "channel": "kiosk",
  "source": "machine_b",
  "customer_profile": {
    "full_name": "Nguyen Van A",
    "phone": "0900000000",
    "project_interest": "Noble Tây Thăng Long"
  },
  "vision_summary": {
    "age_range": "30-45",
    "gender_guess": "male",
    "emotion": "neutral",
    "dress_style": "formal",
    "visible_attributes": ["glasses"],
    "scene_context": "indoor"
  }
}
```

Response mẫu:

```json
{
  "session_id": "ses_9f143ea70d32",
  "customer_id": "cust_001",
  "reused_session": true,
  "context_ready": true,
  "context_used": true
}
```

Ý nghĩa:

- `reused_session=true`: đã tìm thấy session active cũ theo `customer_id`.
- `reused_session=false`: tạo session mới.

---

### `POST /api/v1/sales/chat`

Gửi tin nhắn hội thoại vào session đã mở/resume.

Request body mẫu:

```json
{
  "session_id": "ses_9f143ea70d32",
  "customer_id": "cust_001",
  "channel": "kiosk",
  "message": "xin chào, cho mình biết pháp lý dự án",
  "stream": false
}
```

Response mẫu:

```json
{
  "response": "Tên pháp lý của dự án ...",
  "intent": null,
  "sales_stage": "new",
  "missing_slots": [],
  "context_used": true
}
```

---

## 4) Luồng tích hợp chuẩn (khuyến nghị)

1. Máy B nhận dữ liệu camera/vision local.
2. Máy B gọi `POST /api/v1/session/open` với `customer_id` + metadata user/vision.
3. Lưu `session_id` trả về.
4. Mỗi user utterance, gọi `POST /api/v1/sales/chat` với `session_id` đó.
5. Khi cần debug lịch sử, gọi `GET /sales/history/{session_id}`.

---

## 4.1) Luồng một bước cho nút Mic/Chat (khuyến nghị mới)

Mỗi lần user bấm Mic/Chat trên máy B:

1. Chụp 1 frame camera.
2. Gọi `POST /api/v1/sales/chat-with-camera` sang máy A.
3. Endpoint này tự động:
4. Forward ảnh sang API face của máy B.
5. Nhận `customer_id` từ máy B.
6. Open/resume `session_id` theo `customer_id`.
7. Chạy chat và trả response trong cùng 1 request.

Multipart fields:

- `file` (required)
- `message` (required)
- `channel` (optional, default `kiosk`)
- `source` (optional, default `machine_b_auto`)
- `session_id` (optional)
- `customer_id_hint` (optional)
- `allow_resume` (optional, default `true`)

```bash
curl -X POST "http://192.168.100.7:8010/api/v1/sales/chat-with-camera" ^
  -F "file=@C:\\temp\\snap.jpg" ^
  -F "message=xin chao, cho minh biet phap ly du an noble tay thang long" ^
  -F "allow_resume=true"
```

---

## 5) Ví dụ gọi nhanh bằng cURL (từ máy B)

```bash
curl -X POST "http://192.168.100.7:8010/api/v1/session/open" ^
  -H "Content-Type: application/json" ^
  -d "{\"customer_id\":\"cust_001\",\"allow_resume\":true,\"channel\":\"kiosk\",\"source\":\"machine_b\"}"
```

```bash
curl -X POST "http://192.168.100.7:8010/api/v1/sales/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"ses_9f143ea70d32\",\"customer_id\":\"cust_001\",\"channel\":\"kiosk\",\"message\":\"chi tiết shophouse\",\"stream\":false}"
```

---

## 6) Lỗi thường gặp

- `connection timeout/refused`:
  - Kiểm tra IP máy A.
  - Kiểm tra firewall port `8010`.
  - Kiểm tra container/service trên máy A đang `healthy`.

- `400 customer_id is required`:
  - Thiếu `customer_id` trong `session/open`.

- `409 session_context_not_found` (ở flow cũ):
  - Gọi `session/open` trước khi `sales/chat`.

---

## 7) Endpoint tham chiếu thêm

- `GET /health`
- `GET /models`
- `GET /status`
- `GET /sales/history/{session_id}`
- `POST /query/stream` (alias stream)
