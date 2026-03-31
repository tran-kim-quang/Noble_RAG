# Vision Customer Personalization Implementation Guide

## Mục tiêu

Bổ sung chức năng **vision-based customer identification** cho hệ thống Noble RAG để:

1. Nhận ảnh khách hàng từ camera ở bước chụp ảnh.
2. Trích xuất face embedding và đối sánh với khách đã tồn tại.
3. Nếu khách đã tồn tại:
   - lấy `customer_id`
   - truy cập các dữ liệu liên quan của khách:
     - lịch sử chat
     - lead profile
     - session context gần nhất
     - lịch sử mua hàng (giai đoạn hiện tại dùng file dummy)
4. Nếu khách chưa tồn tại:
   - tạo mới `customer_id`
   - khởi tạo các cấu trúc dữ liệu liên quan
5. **Không tạo mới database / collection cho knowledge base của dự án**, vì đây là data source phục vụ RAG chung cho toàn hệ thống.

---

## Hiện trạng storage trong repo

### 1) PostgreSQL
Hệ thống đang có 2 schema chính:

#### `rag`
Dùng cho knowledge base / RAG metadata:
- `rag.documents`
- `rag.chunks`
- `rag.entities`
- `rag.relations`
- `rag.query_history`

#### `sales`
Dùng cho sales agent / session:
- `sales.lead_profiles`
- `sales.conversation_sessions`
- `sales.conversation_turns`
- `sales.sales_events`
- `sales.appointment_requests`

### 2) Redis
Đang dùng làm cache / ephemeral session store:
- `chat_history:{session_id}`
- `session_context:{session_id}`
- `lead_profile_cache:{session_id}`

### 3) Local snapshot txt
Đang có fallback durable snapshot theo session:
- thư mục `session_export_dir/live_snapshots`
- mỗi session lưu trong 1 file txt

### 4) Qdrant
Đang dùng cho vector retrieval của knowledge base.
**Không nên reuse Qdrant cho face identity ở bước đầu**, vì face identity là dữ liệu cá nhân có lifecycle khác với KB.
Nên ưu tiên lưu face embedding trong PostgreSQL để đồng bộ transactional metadata và dễ quản trị.

### 5) Purchase history
Hiện **chưa có** storage chính thức cho lịch sử mua hàng.
Yêu cầu hiện tại: tạo **1 file dummy** để hệ thống có thể đọc / ghi tạm trong giai đoạn MVP.

---

## Nguyên tắc thiết kế

1. **Tách customer identity khỏi knowledge base**
   - Không đụng vào schema `rag`
   - Không tạo collection KB mới
   - Không đẩy dữ liệu nhận diện khách vào kho tri thức RAG chung

2. **Customer identity là lớp riêng**
   - thêm schema mới: `customer`
   - nối customer với `session_id` của sales flow

3. **Session vẫn là đơn vị hội thoại**
   - `session_id` vẫn dùng cho từng cuộc hội thoại / livestream / lần tương tác
   - `customer_id` là định danh bền vững của khách hàng qua nhiều session

4. **Face matching chỉ là bước định danh**
   - Không dùng LLM để quyết định match face
   - Dùng embedding + ngưỡng similarity
   - LLM chỉ dùng sau đó để cá nhân hóa hội thoại

5. **Privacy by design**
   - Không lưu ảnh gốc nếu không thật sự cần
   - Nếu cần audit, chỉ lưu ảnh đã mã hóa hoặc ảnh thumbnail có TTL / retention rõ ràng
   - Ưu tiên lưu:
     - face embedding
     - metadata nhận diện
     - đường dẫn ảnh tạm nếu có

---

## Thiết kế dữ liệu đề xuất

## A. PostgreSQL schema mới: `customer`

Tạo schema mới riêng cho identity.

### 1. `customer.customers`
Thông tin master của khách hàng.

```sql
CREATE SCHEMA IF NOT EXISTS customer;

CREATE TABLE IF NOT EXISTS customer.customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_code VARCHAR(64) NOT NULL UNIQUE,
    display_name VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

### 2. `customer.customer_faces`
Lưu embedding và thông tin định danh khuôn mặt.

```sql
CREATE TABLE IF NOT EXISTS customer.customer_faces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES customer.customers(id) ON DELETE CASCADE,
    embedding vector(512),
    embedding_model VARCHAR(100) NOT NULL,
    face_hash VARCHAR(128),
    quality_score FLOAT,
    is_primary BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

> Ghi chú:
> - chiều embedding `512` chỉ là ví dụ, phải đổi theo model thực tế
> - nếu model trả về 384 hoặc 768 thì sửa đúng dimension

### 3. `customer.customer_sessions`
Map customer với session hiện tại / lịch sử session.

```sql
CREATE TABLE IF NOT EXISTS customer.customer_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES customer.customers(id) ON DELETE CASCADE,
    session_id VARCHAR(255) NOT NULL UNIQUE,
    source VARCHAR(50) NOT NULL DEFAULT 'camera',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

### 4. `customer.customer_profile_cache`
Không bắt buộc, chỉ dùng nếu muốn gom profile cá nhân hóa lâu dài ở cấp customer thay vì chỉ cấp session.

```sql
CREATE TABLE IF NOT EXISTS customer.customer_profile_cache (
    customer_id UUID PRIMARY KEY REFERENCES customer.customers(id) ON DELETE CASCADE,
    profile_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 5. `customer.customer_identity_events`
Audit log cho nhận diện.

```sql
CREATE TABLE IF NOT EXISTS customer.customer_identity_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID,
    session_id VARCHAR(255),
    event_type VARCHAR(100) NOT NULL,
    similarity_score FLOAT,
    decision VARCHAR(50),
    image_ref TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

---

## B. Dummy purchase history file

Do hiện chưa có purchase DB, tạo file JSON làm nguồn tạm:

**Đề xuất path**
```text
service/RAG/data/customer_purchase_history.json
```

**Format**
```json
{
  "customers": {
    "customer_uuid_1": [
      {
        "order_id": "ORD-001",
        "project_code": "NOBLE-Q7",
        "product_code": "APT-12A-01",
        "product_type": "apartment",
        "status": "reserved",
        "amount": 3500000000,
        "created_at": "2026-03-31T10:00:00Z",
        "metadata": {
          "sales_channel": "livestream"
        }
      }
    ]
  }
}
```

### Yêu cầu với dummy file
- Nếu customer chưa có lịch sử mua hàng, trả về `[]`
- Đọc / ghi atomic
- Khi create customer mới thì **không cần tạo entry rỗng ngay**, có thể lazy init khi đọc
- Tách hẳn khỏi knowledge base

---

## Kiến trúc service đề xuất

## Flow tổng quát

```text
Camera capture
  -> Vision API nhận ảnh
  -> Face detect / validate
  -> Embedding extraction
  -> Similarity search
      -> match found => lấy customer_id
      -> no match => tạo customer_id mới + lưu embedding
  -> bind customer_id <-> session_id
  -> hydrate personalization context
      -> lead profile hiện có
      -> chat history gần nhất
      -> session context gần nhất / active
      -> purchase history từ file dummy
  -> trả về customer context cho UI / sales flow
```

---

## API đề xuất

## 1. POST `/vision/identify`
Nhận ảnh và xác định khách hàng.

### Request
- multipart upload hoặc base64 image
- fields:
  - `session_id`
  - `source` = `camera`
  - `image`

### Response mẫu
```json
{
  "session_id": "session_abc",
  "customer_id": "2a8c...",
  "is_existing_customer": true,
  "match_score": 0.87,
  "customer_context": {
    "lead_profile": {},
    "recent_chat_history": [],
    "recent_session_context": {},
    "purchase_history": []
  }
}
```

## 2. GET `/vision/customer/{customer_id}`
Xem tổng hợp dữ liệu cá nhân hóa của khách.

## 3. GET `/vision/session/{session_id}`
Xem session hiện đang gắn với customer nào.

## 4. POST `/vision/customer/{customer_id}/faces`
Bổ sung thêm embedding / ảnh mẫu cho khách đã tồn tại.
Không cần làm ở phase 1 nếu muốn scope nhỏ.

---

## Cách nối với sales flow hiện tại

## Bước 1: thêm `customer_id` vào session context
Trong `session_context`, lưu thêm:
```json
{
  "customer_id": "uuid",
  "identity_status": "matched|new|unknown"
}
```

## Bước 2: thêm `customer_id` vào lead profile
Khi đã match được khách:
- merge vào `lead_profile`
- ví dụ:
```json
{
  "lead_id": "session_abc",
  "customer_id": "uuid",
  "identity_status": "matched"
}
```

## Bước 3: hydrate context trước khi gọi sales graph
Sau khi vision identify xong:
- load lead profile cũ liên quan customer
- load chat history gần nhất của session trước đó nếu cần
- load purchase history dummy
- gắn summary vào initial state hoặc session context

### Lưu ý quan trọng
Không nên đổ toàn bộ lịch sử cũ trực tiếp vào prompt.
Nên tạo 1 bước **customer context summarization** trước khi inject vào sales graph.

Ví dụ:
```python
customer_context = {
    "customer_id": customer_id,
    "known_preferences": {...},
    "purchase_history_summary": "...",
    "last_sessions_summary": "...",
}
```

---

## Mapping dữ liệu cũ và mới

### Dữ liệu hiện có theo session
- `sales.lead_profiles.session_id`
- `sales.conversation_sessions.session_id`
- `sales.conversation_turns.session_id`
- Redis keys theo `session_id`

### Dữ liệu mới theo customer
- `customer.customers.id`
- `customer.customer_faces.customer_id`
- `customer.customer_sessions.customer_id <-> session_id`

### Quy tắc mapping
- 1 customer có thể có nhiều session
- 1 session chỉ map tới 1 customer
- session mới của khách cũ phải bind vào customer cũ
- dữ liệu sales hiện tại chưa cần migrate toàn bộ sang `customer_id`
- phase đầu chỉ cần thêm bảng map `customer_sessions`

---

## Các module nên tạo

## 1. `service/RAG/vision/face_service.py`
Chịu trách nhiệm:
- detect face
- validate face quality
- extract embedding
- compare similarity

### Interface gợi ý
```python
class FaceService:
    async def extract_embedding(self, image_bytes: bytes) -> list[float]:
        ...

    async def identify_customer(self, image_bytes: bytes, threshold: float = 0.75) -> dict:
        ...
```

## 2. `service/RAG/vision/customer_identity_store.py`
Chịu trách nhiệm:
- create customer
- load customer by id
- upsert face embedding
- find nearest face embedding
- bind customer với session

## 3. `service/RAG/vision/purchase_history_store.py`
Chịu trách nhiệm:
- đọc / ghi file dummy JSON
- `get_purchase_history(customer_id)`
- `append_purchase_event(customer_id, purchase_event)`

## 4. `service/RAG/api/routes_vision.py`
Expose API cho vision identity.

## 5. `service/RAG/models/vision_models.py`
Request / response models.

## 6. `service/RAG/vision/customer_context_builder.py`
Build customer context để inject vào sales agent.

---

## Lựa chọn face recognition

## MVP khuyến nghị
Dùng một face embedding model ổn định, có thể self-host hoặc local inference:
- InsightFace
- FaceNet
- ArcFace-compatible embedding stack

### Không nên
- dùng LLM vision để “nhận diện người này có phải khách cũ không”
- dùng OCR / caption model thay cho face embedding

### Pipeline tối thiểu
1. detect đúng 1 khuôn mặt
2. crop face
3. quality check:
   - ảnh không quá mờ
   - kích thước tối thiểu
   - không bị che quá nhiều
4. extract embedding
5. cosine similarity search
6. threshold decision

---

## Similarity strategy

## Matching rule
- lấy top-1 similarity
- nếu `score >= MATCH_THRESHOLD` => existing customer
- nếu thấp hơn => tạo customer mới

### Config gợi ý
```python
VISION_MATCH_THRESHOLD=0.75
VISION_MIN_FACE_SIZE=112
VISION_MAX_FACES=1
VISION_EMBEDDING_DIM=512
```

### Cần log
- score
- threshold
- model version
- thời điểm nhận diện
- session_id
- customer_id được chọn

---

## Redis cache đề xuất

Có thể thêm cache để giảm truy vấn DB:

- `customer_identity:{customer_id}`
- `customer_session:{session_id}`
- `customer_context:{customer_id}`

TTL gợi ý:
- 1h đến 24h tùy loại data

---

## Tích hợp với UI / camera flow

## Luồng frontend đề xuất
1. UI mở camera
2. user chụp ảnh
3. gửi ảnh tới `POST /vision/identify`
4. backend trả về:
   - `customer_id`
   - `is_existing_customer`
   - `customer_context`
5. UI mới bắt đầu mở chat sales hoặc prefill thông tin
6. những message sau tiếp tục dùng `session_id` như hiện tại

### UX gợi ý
- nếu match được khách cũ:
  - “Đã nhận diện khách hàng cũ”
- nếu là khách mới:
  - “Tạo hồ sơ khách hàng mới”
- nếu ảnh lỗi:
  - yêu cầu chụp lại

---

## Tích hợp với sales agent

## Phase 1
Chỉ cần làm 2 việc:
1. bind `customer_id` vào `session_id`
2. load purchase/chat/profile context trước lượt chat đầu tiên

## Phase 2
- tạo personalization summary
- dùng summary để điều chỉnh:
  - opening
  - recommendation
  - follow-up
  - objection handling

### Ví dụ personalization fields
```json
{
  "customer_id": "...",
  "is_returning_customer": true,
  "last_interaction_at": "2026-03-30T14:00:00Z",
  "known_preferences": {
    "location_preference": "Q7",
    "purpose": "investment"
  },
  "purchase_history_summary": "Đã từng giữ chỗ 1 căn..."
}
```

---

## Không sửa / không tạo mới ở đâu

### Không tạo mới trong `rag.*`
Không thêm bảng customer vào schema `rag`.

### Không tạo collection KB mới trong Qdrant
Qdrant hiện dành cho retrieval / chunk embeddings của tri thức dự án.

### Không trộn purchase history vào knowledge base
Purchase history là customer data vận hành, không phải project knowledge.

---

## Kế hoạch implement đề xuất

## Phase 1 - nền tảng
- tạo `customer` schema
- tạo `customer_identity_store.py`
- tạo `purchase_history_store.py` với dummy JSON
- tạo `vision_models.py`
- tạo `routes_vision.py`
- bind `customer_id` vào `session_id`

## Phase 2 - face recognition
- tích hợp face detector + embedding model
- implement similarity search
- thêm quality check
- thêm identity event log

## Phase 3 - personalization
- build `customer_context_builder.py`
- inject summary vào sales flow
- ưu tiên prefill lead profile nếu có dữ liệu cũ

## Phase 4 - hardening
- rate limit
- image validation
- retention policy
- monitoring / metrics
- manual relink customer tools

---

## Pseudocode backend

```python
async def identify_or_create_customer(session_id: str, image_bytes: bytes) -> dict:
    face = await face_service.detect_single_face(image_bytes)
    if not face.ok:
        raise VisionError("invalid_face")

    embedding = await face_service.extract_embedding(face.cropped_bytes)

    match = await customer_identity_store.find_best_match(embedding)

    if match and match.score >= settings.vision_match_threshold:
        customer_id = match.customer_id
        is_existing = True
        await customer_identity_store.touch_customer(customer_id)
    else:
        customer_id = await customer_identity_store.create_customer(
            metadata={"created_from": "vision_identify"}
        )
        await customer_identity_store.add_face_embedding(
            customer_id=customer_id,
            embedding=embedding,
            embedding_model=settings.vision_embedding_model,
            quality_score=face.quality_score,
        )
        is_existing = False

    await customer_identity_store.bind_session(
        customer_id=customer_id,
        session_id=session_id,
        source="camera",
    )

    customer_context = await customer_context_builder.build(customer_id=customer_id)

    return {
        "session_id": session_id,
        "customer_id": customer_id,
        "is_existing_customer": is_existing,
        "match_score": match.score if match else None,
        "customer_context": customer_context,
    }
```

---

## Cần sửa trong app startup

Trong `app.py`, ngoài `ensure_sales_schema()` cần thêm:
```python
await ensure_customer_schema()
```

---

## Acceptance criteria

1. Upload 1 ảnh hợp lệ:
   - trả về `customer_id`
   - bind được vào `session_id`

2. Ảnh của khách cũ:
   - nhận lại đúng `customer_id`
   - trả được lead profile / purchase history / chat context liên quan

3. Ảnh của khách mới:
   - tạo `customer_id` mới
   - khởi tạo dữ liệu cần thiết mà không đụng vào KB

4. Purchase history dummy:
   - đọc được theo `customer_id`
   - nếu chưa có thì trả `[]`

5. Sales flow vẫn hoạt động như cũ với `session_id`
   - chỉ được enrich thêm `customer_id` và personalization context

---

## Rủi ro cần chú ý

1. **False positive face match**
   - cần threshold đủ chặt
   - có thể thêm bước xác nhận tay ở UI nếu score nằm trong vùng xám

2. **Nhiều khuôn mặt trong ảnh**
   - phase đầu chỉ chấp nhận đúng 1 mặt
   - nếu >1 thì reject

3. **Ảnh kém chất lượng**
   - reject sớm trước khi extract embedding

4. **Privacy / compliance**
   - định nghĩa retention ảnh rõ ràng
   - không log raw image vào text logs

5. **Prompt injection qua customer context**
   - không đưa raw free-text dài từ history cũ vào prompt
   - phải summarize / sanitize trước

---

## Kết luận kiến trúc

Giải pháp đúng với hệ thống hiện tại là:

- giữ nguyên `rag` knowledge base
- giữ nguyên sales flow theo `session_id`
- thêm lớp `customer` riêng để định danh bền vững
- map `customer_id <-> session_id`
- dùng PostgreSQL cho identity metadata + face embeddings
- dùng file JSON dummy cho purchase history giai đoạn đầu
- chỉ inject customer context đã được làm sạch vào sales agent

Cách này ít phá vỡ kiến trúc cũ, dễ rollback, và đủ tốt để MVP hóa tính năng vision personalization.