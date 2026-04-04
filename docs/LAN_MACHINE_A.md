# LAN Integration Guide (Machine A RAG Backend)

This guide configures Machine A so Machine B can send/receive data over the same LAN.

## 1) Machine A role

Machine A hosts RAG backend on port `8010` and provides:

- Session open/resume by `customer_id`
- Sales chat APIs
- LAN bridge APIs to call Machine B face endpoints

## 2) Required env on Machine A

Set in `.env`:

```env
RAG_SERVICE_HOST=0.0.0.0
RAG_SERVICE_PORT=8010
CORS_ORIGINS=*

# Machine B endpoint (face embedding service)
MACHINE_B_BASE_URL=http://<MACHINE_B_IP>:8001
MACHINE_B_TIMEOUT_SEC=8
MACHINE_B_API_KEY=
```

Current local config example:

- `MACHINE_B_BASE_URL=http://10.10.0.20:8001`

## 3) Start/restart services on Machine A

```powershell
docker compose up -d --build --force-recreate rag-service
```

Check health:

```powershell
curl http://127.0.0.1:8010/health
```

## 4) APIs Machine B should call on Machine A

### 4.1 Open or resume chat session

`POST /api/v1/session/open`

Example:

```json
{
  "customer_id": "cust_001",
  "allow_resume": true,
  "channel": "kiosk",
  "source": "machine_b",
  "customer_profile": {
    "full_name": "Nguyen Van A"
  },
  "vision_summary": {
    "age_range": "30-45",
    "gender_guess": "male"
  }
}
```

Response includes:

- `session_id`
- `reused_session`

### 4.2 Chat on existing session

`POST /api/v1/sales/chat`

Example:

```json
{
  "session_id": "ses_xxx",
  "customer_id": "cust_001",
  "channel": "kiosk",
  "message": "xin chao, tu van du an",
  "stream": false
}
```

### 4.3 One-shot auto vision + open/resume + chat

`POST /api/v1/sales/chat-with-camera`

Use this endpoint when user presses mic/chat on Machine B:

1. capture one camera frame on Machine B
2. send frame + message to Machine A
3. Machine A forwards frame to Machine B face API, gets `customer_id`
4. Machine A open/resume session by `customer_id`
5. Machine A runs chat in that session

Multipart fields:

- `file` (image, required)
- `message` (required)
- `channel` (optional, default `kiosk`)
- `source` (optional, default `machine_b_auto`)
- `session_id` (optional)
- `customer_id_hint` (optional)
- `allow_resume` (optional, default `true`)
- `customer_profile_json` (optional JSON object string)
- `vision_summary_json` (optional JSON object string)

Example:

```bash
curl -X POST "http://<MACHINE_A_IP>:8010/api/v1/sales/chat-with-camera" ^
  -F "file=@C:\\temp\\snap.jpg" ^
  -F "message=xin chao, tu van phap ly du an" ^
  -F "channel=kiosk" ^
  -F "allow_resume=true"
```

## 5) New LAN bridge APIs on Machine A (A -> B)

These APIs let Machine A call Machine B directly.

### 5.1 Check Machine B health

`GET /api/v1/lan/machine-b/health`

### 5.2 Fetch face embedding by customer_id from Machine B

`GET /api/v1/lan/machine-b/face/{customer_id}`

### 5.3 Register face on Machine B through Machine A

`POST /api/v1/lan/machine-b/face/register`

Multipart form fields:

- `customer_id` (text)
- `file` (image)
- optional tuning: `threshold`, `center_w`, `center_h`, `margin`

## 6) End-to-end LAN check

From Machine A:

1. Verify B reachable:

```powershell
curl http://<MACHINE_B_IP>:8001/health
```

2. Verify A bridge can reach B:

```powershell
curl http://127.0.0.1:8010/api/v1/lan/machine-b/health
```

3. Verify B -> A open session:

```powershell
curl -X POST http://<MACHINE_A_IP>:8010/api/v1/session/open ^
  -H "Content-Type: application/json" ^
  -d "{\"customer_id\":\"cust_001\",\"allow_resume\":true,\"source\":\"machine_b\"}"
```

## 7) Firewall/network notes

- Open inbound TCP `8010` on Machine A.
- Open inbound TCP `8001` on Machine B (if B face service runs there).
- Both machines must be in same LAN/subnet and route to each other.
