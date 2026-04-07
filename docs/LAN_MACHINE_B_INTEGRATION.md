# Vision Ingest — Thiết kế nhận stream từ Máy B

## Mục tiêu
- Máy A (RAG) có thể nhận "presence" và video stream từ Máy B (camera) để xử lý và lưu trạng thái.
- Tối thiểu yêu cầu: Máy B phục vụ MJPEG; Máy A kéo (pull) stream.
- Yêu cầu bảo mật, khả năng khởi động tự động và hồi phục khi lỗi.

## Các endpoint (FastAPI)
- `POST /api/v1/vision/register`
  - Mô tả: Máy B đăng ký stream URL để Máy A bắt đầu pull.
  - Body (JSON):
    ```json
    {
      "camera_id": "cam01",
      "stream_url": "http://192.168.100.6:8090/stream.mjpg",
      "token": "optional-bearer-token",
      "secret": "shared-secret"
    }
    ```
  - Trả về: `{ "status": "registered", "camera_id": "cam01" }`

- `POST /api/v1/vision/presence`
  - Mô tả: Máy B gửi sự kiện presence (nhanh, dùng để báo có/không người).
  - Body (JSON): `{ "camera": "cam01", "ts_unix": 167..., "co_nguoi": true, "secret": "..." }`
  - Trả về: `{ "status": "ok", "camera": "cam01" }`

- `GET /api/v1/vision/status`
  - Mô tả: Danh sách camera đã đăng ký + `last_seen`, `status`, `has_frame`.

- `GET /health`
  - Mô tả: health check service.

## Kiến trúc hoạt động (khuyến nghị)
1. Máy B chạy MJPEG: `http://<B_IP>:8090/stream.mjpg`.
2. Máy B gọi `POST /api/v1/vision/register` trên Máy A với `camera_id` + `stream_url` (+ token/secret).
3. Máy A tạo background task để pull MJPEG, decode các JPEG frame và lưu `last_frame` + `last_seen` trong bộ nhớ.
4. Máy A có thể tiêu thụ `last_frame` cho RAG, hiển thị dashboard, hoặc ghi HLS/ngắn hạn.

Lý do chọn pull (Máy A kéo): giảm yêu cầu mở firewall/NAT trên Máy A, Máy A chủ động quản lý kết nối, reconnect/backoff tốt hơn.

## Luồng push (nếu cần)
- Nếu muốn Máy B push trực tiếp, có thể mở endpoint trên Máy A (`/api/v1/vision/push_mjpeg` hoặc WebSocket) và Máy B POST multipart MJPEG hoặc stream qua WS. Lưu ý: cần mở port và xử lý auth.

## Bảo mật
- Dùng `MACHINE_A_INGEST_SECRET` (shared secret) để xác thực register/presence.
- Khi kéo stream, Máy A dùng header `Authorization: Bearer <token>` nếu Máy B cung cấp token.
- Dùng TLS (nginx reverse proxy + certbot) để mã hóa traffic giữa Máy A ↔ Máy B khi qua mạng không tin cậy.

## Firewall & tự động mở cổng
- Linux (systemd + ufw):
  - `ufw allow 8010/tcp`
  - Tạo systemd unit cho uvicorn service.
- Windows (PowerShell Admin):
  - `New-NetFirewallRule -DisplayName "RAG Ingest 8010" -Direction Inbound -LocalPort 8010 -Protocol TCP -Action Allow`
  - Hoặc `netsh advfirewall firewall add rule name="RAG Ingest 8010" dir=in action=allow protocol=TCP localport=8010`
- Khuyến nghị: chạy lệnh mở firewall 1 lần trong bước provision (cần quyền admin).

## Chạy nhanh (Máy A)
```powershell
# trong venv
pip install fastapi uvicorn aiohttp
uvicorn api_vision_ingest:app --host 0.0.0.0 --port 8010
```

## Ví dụ gọi từ Máy B
```powershell
$body = @{ camera_id='cam01'; stream_url='http://192.168.100.6:8090/stream.mjpg'; secret='<shared_secret>' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://192.168.100.6:8010/api/v1/vision/register' -Body $body -ContentType 'application/json'
```

## Xử lý lỗi & reliability
- Worker reconnect/backoff (exponential) nếu stream trả lỗi.
- Giữ `presence_history` ngắn (ví dụ 50 entry) để debug.
- Expose `/metrics` (Prometheus) và `/health` để giám sát.

## Nâng cấp tương lai
- Thay MJPEG bằng RTSP hoặc WebRTC cho hiệu suất/băng thông tốt hơn.
- Lưu `last_frame` vào Redis hoặc object store để chia sẻ giữa process/service.
- Thêm authentication mạnh hơn: mTLS hoặc OAuth2.

## Troubleshooting nhanh
- Nếu không connect: kiểm tra firewall trên Máy B và Máy A, kiểm tra `uvicorn` logs, kiểm tra `network route` (cùng subnet).
- Kiểm tra camera stream trực tiếp: `curl http://<B_IP>:8090/stream.mjpg` hoặc mở bằng browser.

---

File này nằm trong `docs/architecture/vision_ingest.md` — bạn có muốn mình cập nhật `docs/LAN_MACHINE_B.md` để trỏ tới file này không?