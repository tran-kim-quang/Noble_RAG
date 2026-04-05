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
    "gender_guess": "male",
    "gioi_tinh": "nam",
    "age_band": "mid",
    "nhom_tuoi": "trung niên"
  }
}
```

Máy B (vision) chỉ gửi các khóa trên trong `vision_summary`: **giới tính** (`gender_guess` + `gioi_tinh` để xưng hô), **độ tuổi** (`age_band` + `nhom_tuoi` để tư vấn). Trường **có người** (`co_nguoi`) chỉ dùng nội bộ trên Máy B — **không** gửi sang Máy A; chỉ khi face detect thành công Máy B mới gọi `session/open` (kèm `customer_id` ở cấp body để Máy A gắn lịch sử chat).

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

---

## 8) Cấu hình Máy B (vision / repo này) để nhận dữ liệu và truyền lên Máy A

Máy B chạy API `api_customer_embedding` (FastAPI), có thể:

1. Lưu **embedding** khuôn mặt theo `customer_id` (local SQLite).
2. **Gọi** `POST /api/v1/session/open` trên Máy A sau khi đăng ký ảnh (relay).
3. **Proxy** thủ công `session/open` và `sales/chat` qua các route tích hợp.

### Biến môi trường trên Máy B

| Biến | Ý nghĩa |
|------|--------|
| `MACHINE_A_BASE_URL` | Base URL Máy A, **không** có slash cuối. Ví dụ: `http://192.168.100.7:8010` |
| `RELAY_SESSION_AFTER_FACE_REGISTER` | `1` / `true`: sau mỗi `POST /v1/face/register` thành công, **tự** gọi `session/open` trên Máy A (không cần form `relay_to_machine_a`). |
| `PORT` | Cổng HTTP của API trên Máy B (mặc định 8000). |

**PowerShell (phiên hiện tại):**

```powershell
$env:MACHINE_A_BASE_URL = "http://192.168.100.7:8010"
$env:RELAY_SESSION_AFTER_FACE_REGISTER = "1"   # tùy chọn: tự relay sau đăng ký mặt
.\venv\Scripts\python.exe -m uvicorn api_customer_embedding:app --host 0.0.0.0 --port 8001
```

### Firewall & mạng

- **Máy B → Máy A:** cần mở **outbound** tới `TCP <MAY_A_IP>:8010` (thường mặc định cho phép).
- **Máy A → Máy B** (nếu A cần gọi API embedding trên B): mở **inbound** trên Máy B cho cổng đang chạy (ví dụ `8001`).
- Kiểm tra: từ Máy B: `curl http://192.168.100.7:8010/health`

### Endpoint tích hợp trên Máy B

| Phương thức | Đường dẫn | Mô tả |
|-------------|-----------|--------|
| GET | `/v1/integration/machine-a` | Kiểm tra `MACHINE_A_BASE_URL` và `GET .../health` trên Máy A |
| POST | `/v1/integration/session-open` | Body JSON giống mục 3 — proxy tới Máy A |
| POST | `/v1/integration/sales/chat` | Body JSON giống mục 3 — proxy tới Máy A |
| POST | `/v1/face/register` | Form `relay_to_machine_a=true` + `full_name` / `phone` / `project_interest` để vừa lưu embedding vừa gọi `session/open` trên A |

Response `POST /v1/face/register` có thêm field `machine_a`: kết quả relay (HTTP status + body từ Máy A) hoặc lỗi.

### Định dạng JSON body từ `POST /v1/face/register` (Máy A đọc để xưng hô / tuổi)

Dữ liệu lưu SQLite `customer_faces` (cột `meta`) thường được **nhúng nguyên** vào response HTTP dưới key **`meta`** (hoặc tương đương `metadata`). Máy A (`_default_vision_summary` trong `routes_pipeline_v1.py`) đọc các khóa sau:

| Nhóm | Khóa trong `meta` (hoặc top-level) | Ví dụ | Ghi chú |
|------|-------------------------------------|-------|---------|
| Giới tính | `gioi_tinh` | `nam`, `nữ` | Ưu tiên cho tiếng Việt; có thể thêm `gender_guess`: `male` / `female` hoặc `1` / `2`. |
| Độ tuổi | `nhom_tuoi` | `trẻ`, `trung niên`, … | Map sang `age_range` gợi ý (vd. `trẻ` → `20-30`). |
| Độ tuổi | `age_band` | `young`, `mid`, `elderly` | Tương đương nhóm tuổi. |
| Độ tuổi | `age_range` | `30-45` | Nếu có sẵn khoảng số, giữ nguyên. |
| Nội bộ B | `co_nguoi` | `true` / `false` | Chỉ cờ có người; Máy A có thể lưu kèm, **không** bắt buộc cho xưng hô. |
| Chất lượng | `face_score` | `0.9997` | Tuỳ chọn; Máy A có thể passthrough vào `vision_context`. |

Ví dụ payload tối thiểu (đủ cho Sunny gọi đúng “anh/chị”):

```json
{
  "customer_id": "cam_ebabd4e00c01",
  "meta": {
    "co_nguoi": true,
    "gioi_tinh": "nam",
    "nhom_tuoi": "trẻ",
    "face_score": 0.9997
  }
}
```

Sau khi chuẩn hóa, `vision_context` trên Máy A sẽ có dạng `gender_guess: "male"`, `gioi_tinh: "nam"`, `age_range: "20-30"`, `nhom_tuoi: "trẻ"`, v.v.

### Ví dụ: đăng ký mặt và relay session lên Máy A

```powershell
curl -X POST "http://127.0.0.1:8001/v1/face/register" `
  -F "customer_id=cust_001" `
  -F "file=@C:\anh.jpg" `
  -F "relay_to_machine_a=true" `
  -F "full_name=Nguyen Van A" `
  -F "phone=0900000000" `
  -F "project_interest=Noble Tay Thang Long"
```

Phụ thuộc `httpx` (đã có trong `requirements.txt`).

