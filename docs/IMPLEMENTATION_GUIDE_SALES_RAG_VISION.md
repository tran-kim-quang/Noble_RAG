# Hướng dẫn implement hệ thống Sales RAG + Vision Customer Context

## 1. Mục đích tài liệu

Tài liệu này mô tả đầy đủ kiến trúc và kế hoạch implement code cho hệ thống:

- nhận ảnh từ vision
- detect khuôn mặt
- đối chiếu khuôn mặt với database
- nếu có trong database thì gắn với khách hàng cũ
- nếu chưa có thì tạo hồ sơ tạm
- tạo session mới
- gọi `vision understand` để phân tích đặc điểm/ngữ cảnh người dùng từ ảnh
- ghi toàn bộ vào `context database`
- dùng context đó để bắt đầu bước hỏi đáp sales qua text với RAG

Tài liệu được viết theo hướng có thể bắt đầu code ngay, bám sát kiến trúc service hiện tại đang có `postgres`, `qdrant`, `rag-service` làm core phía Machine A.

---

## 2. Phạm vi hệ thống

Hệ thống được chia thành 2 lane chính:

### Lane A: Vision lane
Nhận ảnh, định danh khách, lấy ngữ cảnh thị giác, tạo session.

### Lane B: Sales RAG lane
Dùng session context đã được làm giàu bởi vision để hỏi đáp sales bằng text.

---

## 3. Workflow bắt buộc

## 3.1 Workflow tổng thể

```mermaid
flowchart TD
  A[Client gửi ảnh] --> B[Detect khuôn mặt]
  B --> C{Có khuôn mặt?}
  C -- Không --> Z[Trả no_face_detected]
  C -- Có --> D[Chọn khuôn mặt tốt nhất]
  D --> E[Tạo face embedding]
  E --> F[Tìm trong face database]
  F --> G{Có match?}
  G -- Có --> H[Tải customer profile]
  G -- Không --> I[Tạo unknown customer]
  H --> J[Tạo session]
  I --> J
  J --> K[Gọi Vision Understand]
  K --> L[Trích đặc điểm user từ ảnh]
  L --> M[Ghi vision_context]
  M --> N[Merge vào session_context]
  N --> O[Cho phép sales chat bắt đầu]
```

## 3.2 Quy tắc nghiệp vụ

1. Ảnh phải đi trước text nếu flow bắt đầu từ vision.
2. Detect face là bước bắt buộc.
3. Chỉ được gọi `vision understand` sau khi đã xác định được ảnh có khuôn mặt hợp lệ.
4. Chỉ được cho phép `/sales/chat` nếu đã có `session_context`.
5. `vision understand` không dùng để định danh khách hàng.
6. `face matching` là nguồn quyết định identity.
7. `session_context` là nguồn context chính để text chat sử dụng.

---

## 4. Kiến trúc tổng thể

## 4.1 Các service chính

### 4.1.1 `hayhooks-api`
Service chính expose các pipeline Haystack thành REST API.

Chịu trách nhiệm:
- expose endpoint ingest knowledge
- expose endpoint vision enroll
- expose endpoint vision identify + context
- expose endpoint sales chat

### 4.1.2 `postgres`
Lưu dữ liệu nghiệp vụ:
- khách hàng
- session
- chat history
- vision context
- session context
- document metadata

### 4.1.3 `qdrant`
Lưu vector:
- text embeddings cho knowledge base
- face embeddings cho customer recognition

### 4.1.4 `object storage`
Lưu:
- file tài liệu gốc
- ảnh gốc
- ảnh crop mặt

Có thể dùng:
- MinIO
- S3
- local storage trong phase đầu

### 4.1.5 `vision runtime`
Bao gồm:
- face detector
- face embedder
- face matcher
- vision understand client

### 4.1.6 `LLM runtime`
Dùng cho:
- slot extraction
- response generation
- vision understand nếu dùng model vision-capable

---

## 5. 4 endpoint cần implement

## 5.1 `POST /api/v1/knowledge/ingest`

### Mục tiêu
Nạp tài liệu vào knowledge base.

### Input
`multipart/form-data`

Các field:
- `file`
- `project_id`
- `collection`
- `document_type`
- `source_name` optional

### Output
```json
{
  "document_id": "doc_001",
  "status": "indexed",
  "chunks_created": 84,
  "collection": "noble_tay_thang_long"
}
```

### Flow xử lý
1. nhận file
2. lưu file vào object storage
3. parse text
4. làm sạch text
5. chunk theo rule
6. tạo embeddings
7. upsert vào collection `knowledge_chunks`
8. lưu metadata vào Postgres

---

## 5.2 `POST /api/v1/vision/enroll`

### Mục tiêu
Đăng ký khuôn mặt cho một khách hàng đã biết `customer_id`.

### Input
`multipart/form-data`

Các field:
- `image`
- `customer_id`
- `source`
- `note` optional

### Output
```json
{
  "customer_id": "cus_001",
  "face_detected": true,
  "face_count": 1,
  "best_face_saved": true,
  "embedding_saved": true,
  "image_asset_id": "img_001",
  "face_embedding_id": "femb_001"
}
```

### Flow xử lý
1. lưu ảnh gốc
2. detect face
3. nếu không có face thì trả lỗi nghiệp vụ
4. chọn best face
5. crop face
6. tạo embedding
7. lưu embedding vào vector DB
8. lưu metadata ảnh vào Postgres
9. liên kết face với `customer_id`

---

## 5.3 `POST /api/v1/vision/identify-and-context`

### Mục tiêu
Đây là endpoint trung tâm của workflow.

### Nhiệm vụ
- nhận ảnh
- detect mặt
- identify khách hàng
- tạo session
- gọi `vision understand`
- merge vào context DB
- trả `session_id` để bước chat dùng tiếp

### Input
`multipart/form-data`

Các field:
- `image`
- `channel`
- `source`
- `allow_create_unknown`
- `client_trace_id` optional

### Output
```json
{
  "matched": true,
  "customer_id": "cus_001",
  "session_id": "ses_1001",
  "vision_context_saved": true,
  "vision_summary": {
    "age_range": "30-40",
    "gender_guess": "male",
    "emotion": "neutral",
    "dress_style": "formal",
    "visible_attributes": ["ao_so_mi", "deo_kinh"],
    "scene_context": "indoor"
  }
}
```

### Flow xử lý chi tiết
1. lưu ảnh gốc
2. detect tất cả khuôn mặt
3. nếu không có khuôn mặt:
   - trả `no_face_detected`
4. chọn `best_face`
5. crop mặt
6. tạo face embedding
7. similarity search trong `face_embeddings`
8. nếu score >= threshold:
   - load customer cũ
9. nếu score < threshold:
   - nếu `allow_create_unknown=true` thì tạo `unknown customer`
   - nếu không thì trả `unknown_face`
10. tạo session mới
11. gọi `vision understand`
12. lấy summary từ ảnh
13. lưu vào bảng `vision_context`
14. merge vào `session_context`
15. trả `session_id`, `customer_id`, `vision_summary`

---

## 5.4 `POST /api/v1/sales/chat`

### Mục tiêu
Bắt đầu và duy trì hội thoại sales sử dụng context đã được tạo từ vision.

### Input
```json
{
  "session_id": "ses_1001",
  "customer_id": "cus_001",
  "message": "Xin chao",
  "channel": "camera-kiosk",
  "stream": false
}
```

### Output
```json
{
  "response": "Em chao Anh. De em ho tro nhanh hon, Anh dang tim san pham de o hay dau tu a?",
  "intent": "greeting",
  "sales_stage": "discovery",
  "missing_slots": ["purpose", "budget"],
  "context_used": true
}
```

### Flow xử lý
1. load `customers`
2. load `customer_sessions`
3. load `session_context`
4. merge prompt context
5. classify intent
6. extract sales slots
7. update customer profile
8. nếu thiếu slot quan trọng:
   - hỏi tiếp theo template
9. nếu đủ slot:
   - retrieve knowledge
   - rerank
   - generate câu trả lời sales
10. persist chat message
11. update `session_context`

---

## 6. Schema database

## 6.1 Bảng `customers`

```sql
CREATE TABLE customers (
    customer_id          VARCHAR(64) PRIMARY KEY,
    full_name            TEXT,
    phone                TEXT,
    email                TEXT,
    source_channel       TEXT,
    status               TEXT DEFAULT 'active',
    sales_stage          TEXT DEFAULT 'new',
    family_size          INT,
    children_count       INT,
    purpose              TEXT,
    location_preference  TEXT,
    budget_min           NUMERIC,
    budget_max           NUMERIC,
    budget_text          TEXT,
    project_interest     TEXT,
    interest_summary     TEXT,
    created_at           TIMESTAMP DEFAULT NOW(),
    updated_at           TIMESTAMP DEFAULT NOW()
);
```

### Ghi chú
- Đây là hồ sơ bền vững của khách hàng.
- Không lưu dữ liệu tạm của riêng 1 session vào đây nếu chưa được xác nhận.

---

## 6.2 Bảng `customer_faces`

```sql
CREATE TABLE customer_faces (
    face_embedding_id     VARCHAR(64) PRIMARY KEY,
    customer_id           VARCHAR(64) REFERENCES customers(customer_id),
    image_asset_id        VARCHAR(64),
    vector_collection     TEXT NOT NULL,
    quality_score         FLOAT,
    is_primary            BOOLEAN DEFAULT FALSE,
    created_at            TIMESTAMP DEFAULT NOW()
);
```

### Ghi chú
- Mỗi khách có thể có nhiều face embeddings.
- Chỉ nên có 1 embedding tốt nhất làm `is_primary=true`.

---

## 6.3 Bảng `image_assets`

```sql
CREATE TABLE image_assets (
    image_asset_id        VARCHAR(64) PRIMARY KEY,
    customer_id           VARCHAR(64),
    storage_url           TEXT NOT NULL,
    crop_storage_url      TEXT,
    source                TEXT,
    face_count            INT DEFAULT 0,
    width                 INT,
    height                INT,
    created_at            TIMESTAMP DEFAULT NOW()
);
```

### Ghi chú
- `customer_id` có thể null lúc đầu nếu ảnh chưa identify xong.
- Sau khi identify xong có thể update lại.

---

## 6.4 Bảng `customer_sessions`

```sql
CREATE TABLE customer_sessions (
    session_id            VARCHAR(64) PRIMARY KEY,
    customer_id           VARCHAR(64) REFERENCES customers(customer_id),
    channel               TEXT,
    source                TEXT,
    session_status        TEXT DEFAULT 'active',
    short_memory_summary  TEXT,
    created_from_vision   BOOLEAN DEFAULT FALSE,
    started_at            TIMESTAMP DEFAULT NOW(),
    last_activity_at      TIMESTAMP DEFAULT NOW()
);
```

### Ghi chú
- Mỗi lần vision identify thành công hoặc tạo unknown customer sẽ mở một session mới.
- Session này là cầu nối giữa vision và sales chat.

---

## 6.5 Bảng `vision_context`

```sql
CREATE TABLE vision_context (
    vision_context_id     BIGSERIAL PRIMARY KEY,
    session_id            VARCHAR(64) REFERENCES customer_sessions(session_id),
    customer_id           VARCHAR(64) REFERENCES customers(customer_id),
    image_asset_id        VARCHAR(64) REFERENCES image_assets(image_asset_id),
    matched               BOOLEAN,
    match_score           FLOAT,
    age_range             TEXT,
    gender_guess          TEXT,
    emotion               TEXT,
    dress_style           TEXT,
    visible_attributes    JSONB,
    scene_context         TEXT,
    raw_vision_summary    TEXT,
    created_at            TIMESTAMP DEFAULT NOW()
);
```

### Ghi chú
- Dữ liệu ở đây mang tính session-level.
- Không nên dùng bảng này làm profile chính của khách.

---

## 6.6 Bảng `session_context`

```sql
CREATE TABLE session_context (
    session_id            VARCHAR(64) PRIMARY KEY REFERENCES customer_sessions(session_id),
    customer_id           VARCHAR(64) REFERENCES customers(customer_id),
    context_json          JSONB NOT NULL,
    updated_at            TIMESTAMP DEFAULT NOW()
);
```

### Vai trò
Đây là bảng quan trọng nhất với workflow hiện tại.

### `context_json` đề xuất
```json
{
  "customer_profile": {
    "sales_stage": "new",
    "family_size": null,
    "purpose": null
  },
  "vision_context": {
    "age_range": "30-40",
    "gender_guess": "male",
    "emotion": "neutral",
    "dress_style": "formal",
    "visible_attributes": ["ao_so_mi", "deo_kinh"],
    "scene_context": "indoor"
  },
  "conversation_context": {
    "last_intent": null,
    "missing_slots": [],
    "last_question": null
  }
}
```

### Quy tắc
- Text chat chỉ đọc `session_context` để lấy context runtime.
- Tránh join quá nhiều bảng ở mỗi lượt hỏi đáp.

---

## 6.7 Bảng `chat_messages`

```sql
CREATE TABLE chat_messages (
    message_id            BIGSERIAL PRIMARY KEY,
    session_id            VARCHAR(64) REFERENCES customer_sessions(session_id),
    customer_id           VARCHAR(64) REFERENCES customers(customer_id),
    role                  TEXT NOT NULL,
    content               TEXT NOT NULL,
    intent                TEXT,
    sales_stage           TEXT,
    created_at            TIMESTAMP DEFAULT NOW()
);
```

---

## 6.8 Bảng `documents`

```sql
CREATE TABLE documents (
    document_id           VARCHAR(64) PRIMARY KEY,
    collection_name       TEXT NOT NULL,
    project_id            TEXT,
    document_type         TEXT,
    file_name             TEXT,
    storage_url           TEXT,
    chunk_count           INT DEFAULT 0,
    ingest_status         TEXT DEFAULT 'pending',
    created_at            TIMESTAMP DEFAULT NOW()
);
```

---

## 6.9 Bảng `document_chunks` optional

```sql
CREATE TABLE document_chunks (
    chunk_id              VARCHAR(64) PRIMARY KEY,
    document_id           VARCHAR(64) REFERENCES documents(document_id),
    chunk_index           INT,
    content               TEXT,
    metadata_json         JSONB,
    created_at            TIMESTAMP DEFAULT NOW()
);
```

---

## 7. Vector collections

## 7.1 Collection `knowledge_chunks`

Payload:
```json
{
  "chunk_id": "chk_001",
  "document_id": "doc_001",
  "project_id": "noble_tay_thang_long",
  "document_type": "faq",
  "content": "..."
}
```

## 7.2 Collection `face_embeddings`

Payload:
```json
{
  "face_embedding_id": "femb_001",
  "customer_id": "cus_001",
  "image_asset_id": "img_001",
  "is_primary": true
}
```

---

## 8. Thiết kế pipeline

## 8.1 Pipeline `knowledge_ingest_pipeline`

### Mục tiêu
Index tài liệu vào KB.

### Các bước
1. `save_document_asset`
2. `parse_document`
3. `clean_text`
4. `chunk_document`
5. `embed_chunks`
6. `upsert_knowledge_vectors`
7. `persist_document_metadata`
8. `return_ingest_result`

---

## 8.2 Pipeline `vision_enroll_pipeline`

### Mục tiêu
Lưu face embedding cho khách đã biết.

### Các bước
1. `save_image_asset`
2. `detect_faces`
3. `select_best_face`
4. `crop_face`
5. `embed_face`
6. `save_face_vector`
7. `persist_face_metadata`
8. `return_enroll_result`

---

## 8.3 Pipeline `vision_identify_and_context_pipeline`

### Mục tiêu
Định danh + tạo session + tạo context.

### Các bước
1. `save_image_asset`
2. `detect_faces`
3. `select_best_face`
4. `crop_face`
5. `embed_face`
6. `search_face_db`
7. `resolve_customer`
8. `create_session`
9. `call_vision_understand`
10. `persist_vision_context`
11. `merge_session_context`
12. `return_identify_context_result`

### Rule bắt buộc
- `create_session` phải chạy sau `resolve_customer`
- `vision_understand` chỉ chạy sau `detect_faces` thành công
- `merge_session_context` phải ghi 1 bản context hợp nhất dùng ngay cho chat

---

## 8.4 Pipeline `sales_chat_pipeline`

### Mục tiêu
Hỏi đáp sales có dùng context vision.

### Các bước
1. `load_customer_profile`
2. `load_session_context`
3. `build_chat_context`
4. `classify_intent`
5. `extract_slots`
6. `merge_profile_updates`
7. `check_missing_slots`
8. route:
   - `render_template_question`
   - hoặc `retrieve_knowledge`
9. `rerank_documents`
10. `generate_sales_response`
11. `persist_chat_history`
12. `update_session_context`
13. `return_chat_result`

---

## 9. Logic `vision understand`

## 9.1 Mục tiêu
Lấy những đặc điểm hữu ích để mở đầu hội thoại sales, không dùng cho định danh.

## 9.2 Output schema chuẩn

```json
{
  "age_range": "30-40",
  "gender_guess": "male",
  "emotion": "neutral",
  "dress_style": "formal",
  "visible_attributes": ["ao_so_mi", "deo_kinh"],
  "scene_context": "indoor"
}
```

## 9.3 Những gì nên lấy
- độ tuổi ước lượng
- giới tính ước lượng
- trạng thái cảm xúc tổng quát
- phong cách trang phục
- vật thể nhìn thấy rõ
- bối cảnh tổng quát

## 9.4 Những gì không nên lấy
- suy đoán nhạy cảm
- suy đoán nghề nghiệp hoặc thu nhập
- suy luận sức khỏe hoặc chủng tộc
- kết luận chắc chắn từ ảnh mờ

---

## 10. Contract giữa vision và text

## 10.1 Đầu ra của `vision_identify_and_context`
Bắt buộc phải có:
- `session_id`
- `customer_id`
- `vision_summary`

## 10.2 Đầu vào của `sales_chat`
Bắt buộc phải có:
- `session_id`
- `customer_id`
- `message`

## 10.3 Quy tắc
- `/sales/chat` không tự đi tìm face match
- `/sales/chat` chỉ dựa trên `session_context`
- mọi dữ liệu từ vision phải được đổ vào `session_context` trước khi bước chat bắt đầu

---

## 11. Sales state và slot model

## 11.1 Sales stage
- `new`
- `discovery`
- `qualified`
- `recommending`
- `objection_handling`
- `follow_up`
- `closed`

## 11.2 Các slot cần thu thập
- `purpose`
- `budget_min`
- `budget_max`
- `budget_text`
- `location_preference`
- `family_size`
- `children_count`
- `project_interest`

## 11.3 Điều kiện đủ để chuyển sang retrieve mạnh
Cần ít nhất:
- `purpose`
- và một trong các trường:
  - `budget`
  - `location_preference`
  - `project_interest`

Nếu chưa đủ thì dùng template hỏi tiếp, không retrieve để tiết kiệm độ trễ.

---

## 12. Merge context rule

## 12.1 Tư duy
Có 3 lớp context:

### A. `customer_profile`
Dữ liệu bền vững của khách.

### B. `vision_context`
Dữ liệu theo session mới nhất từ ảnh.

### C. `conversation_context`
Dữ liệu runtime của cuộc chat.

## 12.2 Merge mẫu

```python
merged_context = {
    "customer_profile": customer_profile,
    "vision_context": vision_summary,
    "conversation_context": existing_conversation_context or {
        "last_intent": None,
        "missing_slots": [],
        "last_question": None,
    },
}
```

## 12.3 Nguyên tắc
- không ghi đè bừa vào `customers`
- dữ liệu vision nên ưu tiên ở session-level
- chỉ khi user xác nhận hoặc hệ thống chắc chắn mới update profile bền vững

---

## 13. Session state machine

```mermaid
flowchart TD
  A[vision_received] --> B[face_identified]
  B --> C[context_ready]
  C --> D[chat_started]
  D --> E[qualified]
  E --> F[recommending]
  F --> G[closed]
```

## Rule
- chỉ khi `context_ready` mới cho phép `/sales/chat`
- nếu face không match nhưng `allow_create_unknown=true`, vẫn có thể vào `context_ready`

---

## 14. Cấu trúc thư mục code đề xuất

```text
services/
  ai_api/
    app.py
    wrappers/
      knowledge_ingest_wrapper.py
      vision_enroll_wrapper.py
      vision_identify_context_wrapper.py
      sales_chat_wrapper.py
    pipelines/
      knowledge_ingest.py
      vision_enroll.py
      vision_identify_context.py
      sales_chat.py
    repositories/
      customer_repo.py
      customer_face_repo.py
      session_repo.py
      session_context_repo.py
      vision_context_repo.py
      document_repo.py
      image_asset_repo.py
      chat_message_repo.py
    db/
      postgres.py
      qdrant.py
    vision/
      detector.py
      face_selector.py
      face_embedder.py
      face_matcher.py
      vision_understand.py
    sales/
      intent_classifier.py
      slot_extractor.py
      context_loader.py
      response_builder.py
      template_router.py
    knowledge/
      parser.py
      cleaner.py
      chunker.py
      embedder.py
      retriever.py
      reranker.py
    schemas/
      api_models.py
      db_models.py
      enums.py
      constants.py
```

---

## 15. Interface code đề xuất

## 15.1 Face detector

```python
class FaceDetector:
    def detect(self, image_bytes: bytes) -> list[dict]:
        ...
```

Mỗi face nên trả:
- bbox
- score
- landmarks
- blur score optional

---

## 15.2 Face selector

```python
class FaceSelector:
    def select_best(self, faces: list[dict], image_width: int, image_height: int) -> dict:
        ...
```

### Rule chọn best face
Ưu tiên:
- confidence cao
- diện tích lớn
- gần trung tâm
- ít blur

---

## 15.3 Face embedder

```python
class FaceEmbedder:
    def embed(self, face_crop_bytes: bytes) -> list[float]:
        ...
```

---

## 15.4 Face matcher

```python
class FaceMatcher:
    def search(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        ...
```

Kết quả:
- customer_id
- face_embedding_id
- score

---

## 15.5 Vision understand service

```python
class VisionUnderstandService:
    async def analyze(self, image_bytes: bytes) -> dict:
        ...
```

---

## 15.6 Session context service

```python
class SessionContextService:
    async def build_and_save(
        self,
        session_id: str,
        customer_id: str,
        customer_profile: dict,
        vision_summary: dict,
        existing_context: dict | None = None,
    ) -> dict:
        ...
```

---

## 15.7 Sales chat service

```python
class SalesChatService:
    async def chat(
        self,
        session_id: str,
        customer_id: str,
        message: str,
        channel: str | None = None,
    ) -> dict:
        ...
```

---

## 16. Pseudocode cho flow chính

## 16.1 Vision identify and context

```python
async def identify_and_create_context(
    image_bytes: bytes,
    channel: str,
    source: str,
    allow_create_unknown: bool = True,
) -> dict:
    image_asset = await image_asset_repo.save_raw_image(image_bytes=image_bytes, source=source)

    faces = face_detector.detect(image_bytes)
    if not faces:
        return {
            "matched": False,
            "error": "no_face_detected",
        }

    best_face = face_selector.select_best(faces=faces, image_width=0, image_height=0)
    face_crop_bytes = crop_face_from_image(image_bytes, best_face)

    embedding = face_embedder.embed(face_crop_bytes)
    matches = face_matcher.search(embedding=embedding, top_k=1)

    best_match = matches[0] if matches else None
    if best_match and best_match["score"] >= FACE_MATCH_THRESHOLD:
        matched = True
        customer_id = best_match["customer_id"]
        match_score = best_match["score"]
    else:
        if not allow_create_unknown:
            return {
                "matched": False,
                "error": "unknown_face",
            }
        matched = False
        match_score = None
        customer_id = await customer_repo.create_unknown_customer(channel=channel)

    session_id = await session_repo.create_session(
        customer_id=customer_id,
        channel=channel,
        source=source,
        created_from_vision=True,
    )

    vision_summary = await vision_understand_service.analyze(image_bytes=image_bytes)

    await vision_context_repo.insert(
        session_id=session_id,
        customer_id=customer_id,
        image_asset_id=image_asset["image_asset_id"],
        matched=matched,
        match_score=match_score,
        vision_summary=vision_summary,
    )

    customer_profile = await customer_repo.get_by_id(customer_id)
    merged_context = await session_context_service.build_and_save(
        session_id=session_id,
        customer_id=customer_id,
        customer_profile=customer_profile,
        vision_summary=vision_summary,
    )

    return {
        "matched": matched,
        "customer_id": customer_id,
        "session_id": session_id,
        "vision_context_saved": True,
        "vision_summary": vision_summary,
        "session_context": merged_context,
    }
```

---

## 16.2 Sales chat

```python
async def sales_chat(session_id: str, customer_id: str, message: str) -> dict:
    customer_profile = await customer_repo.get_by_id(customer_id)
    session_context = await session_context_repo.get(session_id)

    if not session_context:
        raise ValueError("session_context_not_found")

    chat_context = build_chat_context(
        customer_profile=customer_profile,
        session_context=session_context,
        user_message=message,
    )

    intent = await intent_classifier.classify(chat_context)
    extracted_slots = await slot_extractor.extract(chat_context)

    profile_updates = merge_profile_updates(customer_profile, extracted_slots)
    await customer_repo.update(customer_id, profile_updates)

    missing_slots = compute_missing_slots(profile_updates)

    if missing_slots:
        response = render_template_question(
            missing_slots=missing_slots,
            chat_context=chat_context,
        )
    else:
        retrieved_docs = await retriever.search(chat_context)
        reranked_docs = await reranker.rerank(chat_context, retrieved_docs)
        response = await response_builder.generate(
            chat_context=chat_context,
            retrieved_docs=reranked_docs,
        )

    await chat_message_repo.insert(
        session_id=session_id,
        customer_id=customer_id,
        role="user",
        content=message,
        intent=intent,
        sales_stage=profile_updates.get("sales_stage", "discovery"),
    )
    await chat_message_repo.insert(
        session_id=session_id,
        customer_id=customer_id,
        role="assistant",
        content=response,
        intent=intent,
        sales_stage=profile_updates.get("sales_stage", "discovery"),
    )

    await session_context_repo.update_conversation_context(
        session_id=session_id,
        last_intent=intent,
        missing_slots=missing_slots,
        last_question=response if missing_slots else None,
    )

    return {
        "response": response,
        "intent": intent,
        "sales_stage": profile_updates.get("sales_stage", "discovery"),
        "missing_slots": missing_slots,
        "context_used": True,
    }
```

---

## 17. Chiến lược implement theo sprint

## Sprint 1: nền tảng dữ liệu
Mục tiêu:
- chạy được DB schema
- repo layer đọc/ghi Postgres
- vector collections trong Qdrant
- image storage hoạt động

Checklist:
- [ ] migration SQL
- [ ] kết nối Postgres
- [ ] kết nối Qdrant
- [ ] service lưu file
- [ ] id generator

---

## Sprint 2: vision enroll
Mục tiêu:
- enroll được khuôn mặt cho customer đã có

Checklist:
- [ ] detect face
- [ ] select best face
- [ ] crop face
- [ ] embed face
- [ ] upsert face vector
- [ ] lưu `customer_faces`
- [ ] lưu `image_assets`

---

## Sprint 3: vision identify and context
Mục tiêu:
- identify được khách hoặc tạo unknown
- tạo session
- gọi vision understand
- lưu `vision_context`
- tạo `session_context`

Checklist:
- [ ] search face embedding
- [ ] threshold match
- [ ] create unknown customer
- [ ] create session
- [ ] vision understand call
- [ ] merge session context
- [ ] endpoint response chuẩn

---

## Sprint 4: sales chat fast path
Mục tiêu:
- chat dùng được session context
- hỏi tiếp bằng template
- update customer profile

Checklist:
- [ ] classify intent
- [ ] extract slots
- [ ] render question template
- [ ] persist chat messages
- [ ] update conversation context

---

## Sprint 5: knowledge ingest + RAG
Mục tiêu:
- ingest tài liệu
- retrieve
- rerank
- generate grounded answer

Checklist:
- [ ] parse docs
- [ ] chunk + embed
- [ ] retrieve top-k
- [ ] rerank
- [ ] answer generation
- [ ] citations optional

---

## Sprint 6: tuning và hardening
Checklist:
- [ ] tune face threshold
- [ ] tune timeout
- [ ] add structured logs
- [ ] add metrics
- [ ] add retry policy
- [ ] add fallback when vision understand fails
- [ ] add dedup session rule

---

## 18. Config và constants đề xuất

## 18.1 Environment variables

```bash
POSTGRES_URL=postgresql://user:pass@postgres:5432/app
QDRANT_URL=http://qdrant:6333
OBJECT_STORAGE_BUCKET=customer-assets
FACE_COLLECTION=face_embeddings
KNOWLEDGE_COLLECTION=knowledge_chunks

FACE_MATCH_THRESHOLD=0.75
FACE_TOP_K=3

VISION_UNDERSTAND_TIMEOUT_SEC=8
SALES_CHAT_TIMEOUT_SEC=12
RAG_RETRIEVE_TOP_K=5
RAG_RERANK_TOP_K=3
```

## 18.2 Constants Python

```python
FACE_MATCH_THRESHOLD = 0.75
RAG_RETRIEVE_TOP_K = 5
RAG_RERANK_TOP_K = 3

SESSION_STATUS_ACTIVE = "active"

SALES_STAGE_NEW = "new"
SALES_STAGE_DISCOVERY = "discovery"
SALES_STAGE_QUALIFIED = "qualified"
SALES_STAGE_RECOMMENDING = "recommending"
```

---

## 19. Error handling

## 19.1 Vision endpoint errors

### `no_face_detected`
HTTP 200 hoặc 422 tùy design, nhưng khuyến nghị 200 với error code business.

```json
{
  "success": false,
  "error_code": "no_face_detected",
  "message": "Khong tim thay khuon mat hop le trong anh"
}
```

### `unknown_face`
```json
{
  "success": false,
  "error_code": "unknown_face",
  "message": "Khong tim thay khach hang phu hop trong database"
}
```

### `vision_understand_failed`
Không nên fail toàn bộ flow nếu identify xong.
Có thể degrade:
- vẫn tạo session
- vẫn lưu session_context với `vision_context = null`

---

## 19.2 Sales chat errors

### `session_context_not_found`
Không cho chat chạy tiếp.

### `customer_not_found`
Dừng flow.

### `llm_timeout`
Dùng fallback:
- câu hỏi template ngắn
- hoặc tin nhắn an toàn kiểu "Anh/Chi cho em xin them thong tin"

---

## 20. Logging và observability

Mỗi request nên có:
- `request_id`
- `session_id`
- `customer_id`
- `client_trace_id`
- `pipeline_name`
- `latency_ms`

## Log bắt buộc ở vision flow
- số face detect được
- quality score của best face
- match score
- matched hay unknown
- thời gian vision understand

## Log bắt buộc ở sales flow
- intent
- slots extracted
- missing slots
- có retrieve hay không
- số docs retrieve
- response latency

---

## 21. Test plan

## 21.1 Unit test
- face selector
- session context merge
- missing slot logic
- intent -> route template/retrieve
- threshold face matching

## 21.2 Integration test
- enroll face -> identify face -> create session -> sales chat
- unknown face -> create unknown customer -> sales chat
- no face -> return business error
- sales chat with existing session context
- ingest doc -> retrieve answer

## 21.3 E2E test case bắt buộc

### Case 1: khách cũ
1. enroll ảnh customer
2. gửi ảnh mới
3. system match thành công
4. system tạo session
5. system gọi vision understand
6. system lưu session_context
7. gửi text "xin chao"
8. bot hỏi mở đầu phù hợp

### Case 2: khách mới
1. gửi ảnh lạ
2. system không match
3. system tạo unknown customer
4. system tạo session
5. system gọi vision understand
6. system ghi session_context
7. bắt đầu chat discovery

### Case 3: ảnh lỗi
1. gửi ảnh không có mặt
2. system trả `no_face_detected`

---

## 22. Cấu trúc wrappers Hayhooks đề xuất

## 22.1 `vision_identify_context_wrapper.py`

```python
from hayhooks import BasePipelineWrapper

class VisionIdentifyContextWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = build_vision_identify_context_pipeline()

    async def run_api_async(
        self,
        image: bytes,
        channel: str,
        source: str,
        allow_create_unknown: bool = True,
        client_trace_id: str | None = None,
    ) -> dict:
        return await self.pipeline.run(
            image=image,
            channel=channel,
            source=source,
            allow_create_unknown=allow_create_unknown,
            client_trace_id=client_trace_id,
        )
```

## 22.2 `sales_chat_wrapper.py`

```python
from hayhooks import BasePipelineWrapper

class SalesChatWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = build_sales_chat_pipeline()

    async def run_api_async(
        self,
        session_id: str,
        customer_id: str,
        message: str,
        channel: str | None = None,
        stream: bool = False,
    ) -> dict:
        return await self.pipeline.run(
            session_id=session_id,
            customer_id=customer_id,
            message=message,
            channel=channel,
            stream=stream,
        )
```

---

## 23. Thứ tự code file cụ thể

Ưu tiên code theo thứ tự này:

1. `schemas/db_models.py`
2. `db/postgres.py`
3. `db/qdrant.py`
4. `repositories/customer_repo.py`
5. `repositories/session_repo.py`
6. `repositories/session_context_repo.py`
7. `repositories/vision_context_repo.py`
8. `repositories/image_asset_repo.py`
9. `vision/detector.py`
10. `vision/face_selector.py`
11. `vision/face_embedder.py`
12. `vision/face_matcher.py`
13. `vision/vision_understand.py`
14. `pipelines/vision_enroll.py`
15. `pipelines/vision_identify_context.py`
16. `sales/intent_classifier.py`
17. `sales/slot_extractor.py`
18. `sales/template_router.py`
19. `sales/response_builder.py`
20. `pipelines/sales_chat.py`
21. `knowledge/parser.py`
22. `knowledge/chunker.py`
23. `knowledge/embedder.py`
24. `knowledge/retriever.py`
25. `pipelines/knowledge_ingest.py`
26. `wrappers/*.py`

---

## 24. Những lỗi kiến trúc cần tránh

1. Không gọi sales chat trước khi `session_context` tồn tại.
2. Không dùng vision understand để suy ra identity.
3. Không cho retrieve chạy ở mọi turn chat.
4. Không cập nhật bừa profile bền vững từ dữ liệu vision suy đoán.
5. Không lưu nhiều face embedding chất lượng thấp làm primary.
6. Không join quá nhiều bảng mỗi lượt chat, hãy đọc từ `session_context`.
7. Không để endpoint vision fail toàn bộ chỉ vì `vision understand` lỗi.

---

## 25. Kết luận

Kiến trúc đúng cho use case này là:

- Vision đi trước
- Detect mặt
- Match database
- Tạo session
- Gọi vision understand
- Ghi vào `vision_context`
- Hợp nhất vào `session_context`
- Sales chat sử dụng `session_context` để mở đầu và duy trì hỏi đáp

Điểm quan trọng nhất của toàn bộ thiết kế là:

**`session_context` là cầu nối giữa vision và sales RAG.**

Nếu implement đúng bảng này và pipeline `vision_identify_and_context`, phần còn lại sẽ đơn giản hơn rất nhiều.

---

## 26. Checklist hoàn thành MVP

- [ ] ingest được tài liệu vào knowledge base
- [ ] enroll được face cho customer
- [ ] identify được face từ ảnh mới
- [ ] tạo được unknown customer khi không match
- [ ] tạo session sau vision flow
- [ ] gọi được vision understand
- [ ] lưu được `vision_context`
- [ ] merge được `session_context`
- [ ] sales chat dùng được session context
- [ ] bot hỏi tiếp bằng template khi thiếu slot
- [ ] bot retrieve knowledge khi đủ slot
- [ ] persist được chat history



---

# 27. Bộ skeleton code FastAPI/Hayhooks + SQLAlchemy models

Phần này cung cấp skeleton code để bắt đầu implement ngay.  
Code ở mức khung chuẩn hóa, chưa gắn chặt với model/provider cụ thể.

## 27.1 Cấu trúc thư mục đề xuất

```text
services/
  ai_api/
    app.py
    wrappers/
      knowledge_ingest_wrapper.py
      vision_enroll_wrapper.py
      vision_identify_context_wrapper.py
      sales_chat_wrapper.py
    pipelines/
      knowledge_ingest.py
      vision_enroll.py
      vision_identify_context.py
      sales_chat.py
    repositories/
      customer_repo.py
      customer_face_repo.py
      session_repo.py
      session_context_repo.py
      vision_context_repo.py
      document_repo.py
      image_asset_repo.py
      chat_message_repo.py
    db/
      base.py
      postgres.py
      qdrant.py
    schemas/
      api_models.py
      db_models.py
      enums.py
      constants.py
    services/
      id_service.py
      storage_service.py
      session_context_service.py
    vision/
      detector.py
      face_selector.py
      face_embedder.py
      face_matcher.py
      vision_understand.py
      image_utils.py
    sales/
      intent_classifier.py
      slot_extractor.py
      template_router.py
      response_builder.py
      context_loader.py
    knowledge/
      parser.py
      cleaner.py
      chunker.py
      embedder.py
      retriever.py
      reranker.py
```

---

## 27.2 `schemas/enums.py`

```python
from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


class SalesStage(str, Enum):
    NEW = "new"
    DISCOVERY = "discovery"
    QUALIFIED = "qualified"
    RECOMMENDING = "recommending"
    OBJECTION_HANDLING = "objection_handling"
    FOLLOW_UP = "follow_up"
    CLOSED = "closed"


class ImageSource(str, Enum):
    CAMERA = "camera"
    UPLOAD = "upload"
    LIVESTREAM = "livestream"


class DocumentType(str, Enum):
    FAQ = "faq"
    BROCHURE = "brochure"
    POLICY = "policy"
    LEGAL = "legal"
    PRICING = "pricing"
    PRODUCT_CATALOG = "product_catalog"
```

---

## 27.3 `schemas/constants.py`

```python
FACE_MATCH_THRESHOLD = 0.75
FACE_TOP_K = 3

RAG_RETRIEVE_TOP_K = 5
RAG_RERANK_TOP_K = 3

DEFAULT_COLLECTION_KNOWLEDGE = "knowledge_chunks"
DEFAULT_COLLECTION_FACES = "face_embeddings"
```

---

## 27.4 `db/base.py`

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

---

## 27.5 `db/postgres.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+asyncpg://user:pass@localhost:5432/app",
)

engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

---

## 27.6 `schemas/db_models.py`

```python
from datetime import datetime
from sqlalchemy import (
    String,
    Text,
    Integer,
    Numeric,
    Float,
    Boolean,
    ForeignKey,
    BigInteger,
    JSON,
    DateTime,
)
from sqlalchemy.orm import Mapped, mapped_column
from db.base import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    full_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    phone: Mapped[str | None] = mapped_column(Text, nullable=True)
    email: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_channel: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="active")
    sales_stage: Mapped[str] = mapped_column(Text, default="new")
    family_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    children_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    location_preference: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_min: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    budget_max: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    budget_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_interest: Mapped[str | None] = mapped_column(Text, nullable=True)
    interest_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CustomerFace(Base):
    __tablename__ = "customer_faces"

    face_embedding_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    image_asset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vector_collection: Mapped[str] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ImageAsset(Base):
    __tablename__ = "image_assets"

    image_asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    storage_url: Mapped[str] = mapped_column(Text)
    crop_storage_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CustomerSession(Base):
    __tablename__ = "customer_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    channel: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_status: Mapped[str] = mapped_column(Text, default="active")
    short_memory_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_from_vision: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class VisionContext(Base):
    __tablename__ = "vision_context"

    vision_context_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("customer_sessions.session_id"))
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    image_asset_id: Mapped[str | None] = mapped_column(ForeignKey("image_assets.image_asset_id"), nullable=True)
    matched: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    age_range: Mapped[str | None] = mapped_column(Text, nullable=True)
    gender_guess: Mapped[str | None] = mapped_column(Text, nullable=True)
    emotion: Mapped[str | None] = mapped_column(Text, nullable=True)
    dress_style: Mapped[str | None] = mapped_column(Text, nullable=True)
    visible_attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scene_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_vision_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SessionContext(Base):
    __tablename__ = "session_context"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("customer_sessions.session_id"), primary_key=True
    )
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    context_json: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("customer_sessions.session_id"))
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    sales_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    collection_name: Mapped[str] = mapped_column(Text)
    project_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingest_status: Mapped[str] = mapped_column(Text, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

---

## 27.7 `schemas/api_models.py`

```python
from pydantic import BaseModel, Field
from typing import Any


class SalesChatRequest(BaseModel):
    session_id: str
    customer_id: str
    message: str
    channel: str | None = None
    stream: bool = False


class SalesChatResponse(BaseModel):
    response: str
    intent: str | None = None
    sales_stage: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    context_used: bool = True


class VisionIdentifyContextResponse(BaseModel):
    matched: bool
    customer_id: str | None = None
    session_id: str | None = None
    vision_context_saved: bool = False
    vision_summary: dict[str, Any] | None = None
    error: str | None = None


class VisionEnrollResponse(BaseModel):
    customer_id: str
    face_detected: bool
    face_count: int
    best_face_saved: bool
    embedding_saved: bool
    image_asset_id: str | None = None
    face_embedding_id: str | None = None
```

---

## 27.8 `services/id_service.py`

```python
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
```

---

## 27.9 `services/storage_service.py`

```python
from pathlib import Path
from services.id_service import new_id


class StorageService:
    def __init__(self, base_dir: str = "/tmp/customer-assets") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save_bytes(self, data: bytes, suffix: str = ".bin") -> str:
        file_id = new_id("asset")
        path = self.base_dir / f"{file_id}{suffix}"
        path.write_bytes(data)
        return str(path)
```

---

## 27.10 Repository skeletons

### `repositories/customer_repo.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.db_models import Customer
from services.id_service import new_id


class CustomerRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, customer_id: str) -> Customer | None:
        result = await self.db.execute(
            select(Customer).where(Customer.customer_id == customer_id)
        )
        return result.scalar_one_or_none()

    async def create_unknown_customer(self, channel: str | None = None) -> str:
        customer_id = new_id("cus")
        entity = Customer(
            customer_id=customer_id,
            source_channel=channel,
            status="active",
            sales_stage="new",
            interest_summary="unknown_customer_created_from_vision",
        )
        self.db.add(entity)
        await self.db.commit()
        return customer_id

    async def update_fields(self, customer: Customer, updates: dict) -> Customer:
        for key, value in updates.items():
            if hasattr(customer, key) and value is not None:
                setattr(customer, key, value)
        await self.db.commit()
        await self.db.refresh(customer)
        return customer
```

### `repositories/session_repo.py`

```python
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.db_models import CustomerSession
from services.id_service import new_id


class SessionRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_session(
        self,
        customer_id: str,
        channel: str | None,
        source: str | None,
        created_from_vision: bool = False,
    ) -> str:
        session_id = new_id("ses")
        entity = CustomerSession(
            session_id=session_id,
            customer_id=customer_id,
            channel=channel,
            source=source,
            created_from_vision=created_from_vision,
            session_status="active",
            started_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow(),
        )
        self.db.add(entity)
        await self.db.commit()
        return session_id
```

### `repositories/session_context_repo.py`

```python
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.db_models import SessionContext


class SessionContextRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get(self, session_id: str) -> SessionContext | None:
        result = await self.db.execute(
            select(SessionContext).where(SessionContext.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def upsert(self, session_id: str, customer_id: str, context_json: dict) -> None:
        existing = await self.get(session_id)
        if existing:
            existing.customer_id = customer_id
            existing.context_json = context_json
            existing.updated_at = datetime.utcnow()
        else:
            self.db.add(
                SessionContext(
                    session_id=session_id,
                    customer_id=customer_id,
                    context_json=context_json,
                    updated_at=datetime.utcnow(),
                )
            )
        await self.db.commit()

    async def update_conversation_context(
        self,
        session_id: str,
        last_intent: str | None,
        missing_slots: list[str],
        last_question: str | None,
    ) -> None:
        existing = await self.get(session_id)
        if not existing:
            return
        payload = existing.context_json or {}
        payload.setdefault("conversation_context", {})
        payload["conversation_context"]["last_intent"] = last_intent
        payload["conversation_context"]["missing_slots"] = missing_slots
        payload["conversation_context"]["last_question"] = last_question
        existing.context_json = payload
        existing.updated_at = datetime.utcnow()
        await self.db.commit()
```

### `repositories/vision_context_repo.py`

```python
import json
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.db_models import VisionContext


class VisionContextRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def insert(
        self,
        session_id: str,
        customer_id: str,
        image_asset_id: str | None,
        matched: bool,
        match_score: float | None,
        vision_summary: dict,
    ) -> None:
        entity = VisionContext(
            session_id=session_id,
            customer_id=customer_id,
            image_asset_id=image_asset_id,
            matched=matched,
            match_score=match_score,
            age_range=vision_summary.get("age_range"),
            gender_guess=vision_summary.get("gender_guess"),
            emotion=vision_summary.get("emotion"),
            dress_style=vision_summary.get("dress_style"),
            visible_attributes={"items": vision_summary.get("visible_attributes", [])},
            scene_context=vision_summary.get("scene_context"),
            raw_vision_summary=json.dumps(vision_summary, ensure_ascii=False),
        )
        self.db.add(entity)
        await self.db.commit()
```

### `repositories/chat_message_repo.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.db_models import ChatMessage


class ChatMessageRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def insert(
        self,
        session_id: str,
        customer_id: str,
        role: str,
        content: str,
        intent: str | None = None,
        sales_stage: str | None = None,
    ) -> None:
        self.db.add(
            ChatMessage(
                session_id=session_id,
                customer_id=customer_id,
                role=role,
                content=content,
                intent=intent,
                sales_stage=sales_stage,
            )
        )
        await self.db.commit()
```

---

## 27.11 `services/session_context_service.py`

```python
class SessionContextService:
    def build_context(
        self,
        customer_profile: dict,
        vision_summary: dict | None,
        existing_context: dict | None = None,
    ) -> dict:
        existing_context = existing_context or {}
        return {
            "customer_profile": customer_profile,
            "vision_context": vision_summary or {},
            "conversation_context": existing_context.get("conversation_context", {
                "last_intent": None,
                "missing_slots": [],
                "last_question": None,
            }),
        }
```

---

## 27.12 Vision interfaces

### `vision/detector.py`

```python
class FaceDetector:
    def detect(self, image_bytes: bytes) -> list[dict]:
        raise NotImplementedError
```

### `vision/face_selector.py`

```python
class FaceSelector:
    def select_best(self, faces: list[dict], image_width: int, image_height: int) -> dict:
        if not faces:
            raise ValueError("No faces to select")
        return sorted(
            faces,
            key=lambda f: (f.get("score", 0.0), f.get("area", 0.0)),
            reverse=True,
        )[0]
```

### `vision/face_embedder.py`

```python
class FaceEmbedder:
    def embed(self, face_crop_bytes: bytes) -> list[float]:
        raise NotImplementedError
```

### `vision/face_matcher.py`

```python
class FaceMatcher:
    def search(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        raise NotImplementedError
```

### `vision/vision_understand.py`

```python
class VisionUnderstandService:
    async def analyze(self, image_bytes: bytes) -> dict:
        return {
            "age_range": "30-40",
            "gender_guess": "male",
            "emotion": "neutral",
            "dress_style": "formal",
            "visible_attributes": ["ao_so_mi", "deo_kinh"],
            "scene_context": "indoor",
        }
```

---

## 27.13 Sales skeletons

### `sales/intent_classifier.py`

```python
class IntentClassifier:
    async def classify(self, chat_context: dict) -> str:
        user_message = (chat_context.get("user_message") or "").lower()
        if "xin chao" in user_message or "chao" in user_message:
            return "greeting"
        if "gia" in user_message or "can" in user_message:
            return "ask_recommendation"
        return "other"
```

### `sales/slot_extractor.py`

```python
class SlotExtractor:
    async def extract(self, chat_context: dict) -> dict:
        # TODO: thay bằng LLM hoặc hybrid extractor
        return {}
```

### `sales/template_router.py`

```python
def compute_missing_slots(profile: dict) -> list[str]:
    missing = []
    if not profile.get("purpose"):
        missing.append("purpose")
    if not (profile.get("budget_min") or profile.get("budget_max") or profile.get("budget_text")):
        missing.append("budget")
    return missing


def render_template_question(missing_slots: list[str], chat_context: dict) -> str:
    if "purpose" in missing_slots:
        return "Em ho tro nhanh hon nhe, Anh/Chi dang tim san pham de o hay dau tu a?"
    if "budget" in missing_slots:
        return "Anh/Chi du tru khoang ngan sach nao de em goi y sat hon a?"
    return "Anh/Chi co the chia se them nhu cau de em tu van chinh xac hon khong a?"
```

### `sales/response_builder.py`

```python
class ResponseBuilder:
    async def generate(self, chat_context: dict, retrieved_docs: list[dict]) -> str:
        return "Em da tim thay mot so thong tin phu hop. De em goi y phuong an sat nhu cau cua Anh/Chi."
```

### `sales/context_loader.py`

```python
def build_chat_context(customer_profile: dict, session_context: dict, user_message: str) -> dict:
    return {
        "customer_profile": customer_profile,
        "session_context": session_context,
        "user_message": user_message,
    }
```

---

## 27.14 Knowledge skeletons

### `knowledge/parser.py`

```python
class DocumentParser:
    async def parse(self, file_bytes: bytes, file_name: str) -> str:
        return file_bytes.decode("utf-8", errors="ignore")
```

### `knowledge/cleaner.py`

```python
def clean_text(text: str) -> str:
    return " ".join(text.split())
```

### `knowledge/chunker.py`

```python
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks
```

### `knowledge/embedder.py`

```python
class KnowledgeEmbedder:
    async def embed_many(self, chunks: list[str]) -> list[list[float]]:
        # TODO: nối model embedding thật
        return [[0.0] * 8 for _ in chunks]
```

### `knowledge/retriever.py`

```python
class Retriever:
    async def search(self, chat_context: dict) -> list[dict]:
        return []
```

### `knowledge/reranker.py`

```python
class Reranker:
    async def rerank(self, chat_context: dict, docs: list[dict]) -> list[dict]:
        return docs[:3]
```

---

## 27.15 Pipeline skeletons

### `pipelines/vision_identify_context.py`

```python
from schemas.constants import FACE_MATCH_THRESHOLD, FACE_TOP_K
from services.session_context_service import SessionContextService


class VisionIdentifyContextPipeline:
    def __init__(
        self,
        image_asset_repo,
        customer_repo,
        session_repo,
        session_context_repo,
        vision_context_repo,
        face_detector,
        face_selector,
        face_embedder,
        face_matcher,
        vision_understand_service,
        storage_service,
    ) -> None:
        self.image_asset_repo = image_asset_repo
        self.customer_repo = customer_repo
        self.session_repo = session_repo
        self.session_context_repo = session_context_repo
        self.vision_context_repo = vision_context_repo
        self.face_detector = face_detector
        self.face_selector = face_selector
        self.face_embedder = face_embedder
        self.face_matcher = face_matcher
        self.vision_understand_service = vision_understand_service
        self.storage_service = storage_service
        self.session_context_service = SessionContextService()

    async def run(
        self,
        image: bytes,
        channel: str,
        source: str,
        allow_create_unknown: bool = True,
        client_trace_id: str | None = None,
    ) -> dict:
        raw_path = await self.storage_service.save_bytes(image, suffix=".jpg")

        faces = self.face_detector.detect(image)
        if not faces:
            return {
                "matched": False,
                "error": "no_face_detected",
            }

        best_face = self.face_selector.select_best(faces, image_width=0, image_height=0)
        face_crop = image  # TODO: crop thật bằng bbox

        embedding = self.face_embedder.embed(face_crop)
        matches = self.face_matcher.search(embedding, top_k=FACE_TOP_K)
        best_match = matches[0] if matches else None

        if best_match and best_match.get("score", 0.0) >= FACE_MATCH_THRESHOLD:
            matched = True
            customer_id = best_match["customer_id"]
            match_score = best_match["score"]
        else:
            if not allow_create_unknown:
                return {
                    "matched": False,
                    "error": "unknown_face",
                }
            matched = False
            match_score = None
            customer_id = await self.customer_repo.create_unknown_customer(channel=channel)

        session_id = await self.session_repo.create_session(
            customer_id=customer_id,
            channel=channel,
            source=source,
            created_from_vision=True,
        )

        vision_summary = await self.vision_understand_service.analyze(image)

        await self.vision_context_repo.insert(
            session_id=session_id,
            customer_id=customer_id,
            image_asset_id=None,
            matched=matched,
            match_score=match_score,
            vision_summary=vision_summary,
        )

        customer = await self.customer_repo.get_by_id(customer_id)
        customer_profile = {
            "customer_id": customer.customer_id,
            "sales_stage": customer.sales_stage,
            "purpose": customer.purpose,
            "budget_text": customer.budget_text,
            "location_preference": customer.location_preference,
        }

        merged_context = self.session_context_service.build_context(
            customer_profile=customer_profile,
            vision_summary=vision_summary,
            existing_context=None,
        )

        await self.session_context_repo.upsert(
            session_id=session_id,
            customer_id=customer_id,
            context_json=merged_context,
        )

        return {
            "matched": matched,
            "customer_id": customer_id,
            "session_id": session_id,
            "vision_context_saved": True,
            "vision_summary": vision_summary,
        }
```

### `pipelines/sales_chat.py`

```python
from sales.context_loader import build_chat_context
from sales.template_router import compute_missing_slots, render_template_question


class SalesChatPipeline:
    def __init__(
        self,
        customer_repo,
        session_context_repo,
        chat_message_repo,
        intent_classifier,
        slot_extractor,
        retriever,
        reranker,
        response_builder,
    ) -> None:
        self.customer_repo = customer_repo
        self.session_context_repo = session_context_repo
        self.chat_message_repo = chat_message_repo
        self.intent_classifier = intent_classifier
        self.slot_extractor = slot_extractor
        self.retriever = retriever
        self.reranker = reranker
        self.response_builder = response_builder

    async def run(
        self,
        session_id: str,
        customer_id: str,
        message: str,
        channel: str | None = None,
        stream: bool = False,
    ) -> dict:
        customer = await self.customer_repo.get_by_id(customer_id)
        if not customer:
            raise ValueError("customer_not_found")

        session_ctx_row = await self.session_context_repo.get(session_id)
        if not session_ctx_row:
            raise ValueError("session_context_not_found")

        customer_profile = {
            "customer_id": customer.customer_id,
            "sales_stage": customer.sales_stage,
            "purpose": customer.purpose,
            "budget_min": customer.budget_min,
            "budget_max": customer.budget_max,
            "budget_text": customer.budget_text,
            "location_preference": customer.location_preference,
        }
        session_context = session_ctx_row.context_json

        chat_context = build_chat_context(
            customer_profile=customer_profile,
            session_context=session_context,
            user_message=message,
        )

        intent = await self.intent_classifier.classify(chat_context)
        extracted_slots = await self.slot_extractor.extract(chat_context)

        customer = await self.customer_repo.update_fields(customer, extracted_slots)
        customer_profile.update(extracted_slots)

        missing_slots = compute_missing_slots(customer_profile)

        if missing_slots:
            response = render_template_question(missing_slots, chat_context)
        else:
            docs = await self.retriever.search(chat_context)
            reranked = await self.reranker.rerank(chat_context, docs)
            response = await self.response_builder.generate(chat_context, reranked)

        await self.chat_message_repo.insert(
            session_id=session_id,
            customer_id=customer_id,
            role="user",
            content=message,
            intent=intent,
            sales_stage=customer.sales_stage,
        )
        await self.chat_message_repo.insert(
            session_id=session_id,
            customer_id=customer_id,
            role="assistant",
            content=response,
            intent=intent,
            sales_stage=customer.sales_stage,
        )

        await self.session_context_repo.update_conversation_context(
            session_id=session_id,
            last_intent=intent,
            missing_slots=missing_slots,
            last_question=response if missing_slots else None,
        )

        return {
            "response": response,
            "intent": intent,
            "sales_stage": customer.sales_stage,
            "missing_slots": missing_slots,
            "context_used": True,
        }
```

### `pipelines/vision_enroll.py`

```python
class VisionEnrollPipeline:
    def __init__(self, storage_service, face_detector, face_selector, face_embedder):
        self.storage_service = storage_service
        self.face_detector = face_detector
        self.face_selector = face_selector
        self.face_embedder = face_embedder

    async def run(self, image: bytes, customer_id: str, source: str, note: str | None = None) -> dict:
        raw_path = await self.storage_service.save_bytes(image, suffix=".jpg")
        faces = self.face_detector.detect(image)
        if not faces:
            return {
                "customer_id": customer_id,
                "face_detected": False,
                "face_count": 0,
                "best_face_saved": False,
                "embedding_saved": False,
            }

        best_face = self.face_selector.select_best(faces, image_width=0, image_height=0)
        face_crop = image
        _embedding = self.face_embedder.embed(face_crop)

        return {
            "customer_id": customer_id,
            "face_detected": True,
            "face_count": len(faces),
            "best_face_saved": True,
            "embedding_saved": True,
            "image_asset_id": raw_path,
            "face_embedding_id": "TODO_SAVE_ID",
        }
```

### `pipelines/knowledge_ingest.py`

```python
from knowledge.cleaner import clean_text
from knowledge.chunker import chunk_text


class KnowledgeIngestPipeline:
    def __init__(self, parser, embedder):
        self.parser = parser
        self.embedder = embedder

    async def run(
        self,
        file_bytes: bytes,
        file_name: str,
        project_id: str,
        collection: str,
        document_type: str,
        source_name: str | None = None,
    ) -> dict:
        text = await self.parser.parse(file_bytes=file_bytes, file_name=file_name)
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)
        _vectors = await self.embedder.embed_many(chunks)

        return {
            "document_id": "TODO_DOCUMENT_ID",
            "status": "indexed",
            "chunks_created": len(chunks),
            "collection": collection,
        }
```

---

## 27.16 Hayhooks wrappers

### `wrappers/vision_identify_context_wrapper.py`

```python
from hayhooks import BasePipelineWrapper


class VisionIdentifyContextWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = build_vision_identify_context_pipeline()

    async def run_api_async(
        self,
        image: bytes,
        channel: str,
        source: str,
        allow_create_unknown: bool = True,
        client_trace_id: str | None = None,
    ) -> dict:
        return await self.pipeline.run(
            image=image,
            channel=channel,
            source=source,
            allow_create_unknown=allow_create_unknown,
            client_trace_id=client_trace_id,
        )
```

### `wrappers/sales_chat_wrapper.py`

```python
from hayhooks import BasePipelineWrapper


class SalesChatWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = build_sales_chat_pipeline()

    async def run_api_async(
        self,
        session_id: str,
        customer_id: str,
        message: str,
        channel: str | None = None,
        stream: bool = False,
    ) -> dict:
        return await self.pipeline.run(
            session_id=session_id,
            customer_id=customer_id,
            message=message,
            channel=channel,
            stream=stream,
        )
```

### `wrappers/vision_enroll_wrapper.py`

```python
from hayhooks import BasePipelineWrapper


class VisionEnrollWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = build_vision_enroll_pipeline()

    async def run_api_async(
        self,
        image: bytes,
        customer_id: str,
        source: str,
        note: str | None = None,
    ) -> dict:
        return await self.pipeline.run(
            image=image,
            customer_id=customer_id,
            source=source,
            note=note,
        )
```

### `wrappers/knowledge_ingest_wrapper.py`

```python
from hayhooks import BasePipelineWrapper


class KnowledgeIngestWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.pipeline = build_knowledge_ingest_pipeline()

    async def run_api_async(
        self,
        file_bytes: bytes,
        file_name: str,
        project_id: str,
        collection: str,
        document_type: str,
        source_name: str | None = None,
    ) -> dict:
        return await self.pipeline.run(
            file_bytes=file_bytes,
            file_name=file_name,
            project_id=project_id,
            collection=collection,
            document_type=document_type,
            source_name=source_name,
        )
```

---

## 27.17 FastAPI bootstrap skeleton

### `app.py`

```python
from fastapi import FastAPI

app = FastAPI(title="Sales RAG Vision API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
```

### Gợi ý triển khai
Có 2 cách:

- Cách 1: chạy Hayhooks độc lập, deploy các wrapper thành endpoint
- Cách 2: dùng FastAPI làm entrypoint ngoài, gọi vào pipeline nội bộ hoặc reverse proxy sang Hayhooks

Khuyến nghị cho MVP:
- dùng Hayhooks cho pipeline endpoints
- giữ FastAPI `/health`, `/metrics`, `/internal/debug` nếu cần

---

## 27.18 Factory functions

### `pipelines/__init__.py` hoặc `builders.py`

```python
from vision.detector import FaceDetector
from vision.face_selector import FaceSelector
from vision.face_embedder import FaceEmbedder
from vision.face_matcher import FaceMatcher
from vision.vision_understand import VisionUnderstandService

from sales.intent_classifier import IntentClassifier
from sales.slot_extractor import SlotExtractor
from sales.response_builder import ResponseBuilder
from knowledge.retriever import Retriever
from knowledge.reranker import Reranker
from knowledge.parser import DocumentParser
from knowledge.embedder import KnowledgeEmbedder

from services.storage_service import StorageService
from pipelines.vision_identify_context import VisionIdentifyContextPipeline
from pipelines.vision_enroll import VisionEnrollPipeline
from pipelines.sales_chat import SalesChatPipeline
from pipelines.knowledge_ingest import KnowledgeIngestPipeline


def build_vision_identify_context_pipeline():
    # TODO: inject thật từ db/repo container
    raise NotImplementedError


def build_vision_enroll_pipeline():
    raise NotImplementedError


def build_sales_chat_pipeline():
    raise NotImplementedError


def build_knowledge_ingest_pipeline():
    return KnowledgeIngestPipeline(
        parser=DocumentParser(),
        embedder=KnowledgeEmbedder(),
    )
```

---

## 27.19 Alembic gợi ý

Khuyến nghị thêm:
- `alembic init migrations`
- sinh migration cho toàn bộ models
- không tạo bảng thủ công ở production

---

## 27.20 Danh sách TODO sau khi dán skeleton

### Phần vision
- [ ] gắn detector thật, ví dụ SCRFD
- [ ] gắn embedder thật, ví dụ ArcFace/InsightFace
- [ ] implement crop face từ bbox
- [ ] implement Qdrant search cho face vector
- [ ] implement save face vector

### Phần sales
- [ ] thay `IntentClassifier` mock bằng model thật
- [ ] thay `SlotExtractor` mock bằng hybrid extractor
- [ ] implement retriever/reranker thật
- [ ] viết prompt grounded response

### Phần data
- [ ] implement repositories còn thiếu
- [ ] thêm transaction control
- [ ] thêm soft delete nếu cần
- [ ] thêm updated_at auto-update

### Phần serving
- [ ] deploy wrappers bằng Hayhooks
- [ ] thêm auth nội bộ nếu cần
- [ ] thêm structured logging
- [ ] thêm timeout và retry

---

## 27.21 Gợi ý thứ tự code thực tế

1. `db/base.py`
2. `schemas/db_models.py`
3. `db/postgres.py`
4. repositories
5. `services/storage_service.py`
6. vision interfaces
7. `pipelines/vision_identify_context.py`
8. `pipelines/sales_chat.py`
9. Hayhooks wrappers
10. tích hợp detector/embedder thật
11. retriever/reranker thật
12. test integration

