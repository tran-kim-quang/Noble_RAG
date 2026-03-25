# Noble_RAG Project Overview

## 1. Mục tiêu dự án

`Noble_RAG` là một hệ thống AI hỗ trợ tư vấn bất động sản theo 2 lớp chính:

- `RAG service`: nhận tài liệu, lưu vào hệ thống LightRAG, phục vụ truy xuất ngữ nghĩa.
- `Sales agent`: điều phối hội thoại bán hàng theo state machine, thu thập nhu cầu khách, truy xuất ngữ cảnh phù hợp và phản hồi theo flow sales.

Ngoài backend chính, repo còn có:

- `Whisper service`: chuyển giọng nói thành văn bản.
- `Web UI`: giao diện đơn giản để upload tài liệu và quản lý documents.
- `Voice workflow`: luồng demo nói chuyện qua micro, gọi Whisper rồi gửi sang RAG service.

---

## 2. Kiến trúc hiện tại

### 2.1 Thành phần chính

#### `service/RAG`

Backend FastAPI chính của dự án.

Chức năng:

- expose API cho health, documents, query, sales
- quản lý LightRAG
- điều phối sales graph bằng LangGraph
- lưu session, lead profile, chat history
- snapshot local phiên chat ra file `.txt`

#### `service/whisper-service`

Service FastAPI riêng cho speech-to-text bằng `faster-whisper`.

Chức năng:

- nhận file audio
- trả transcript đầy đủ hoặc stream từng segment

#### `ui`

Streamlit UI phục vụ:

- upload document
- xem danh sách documents đã ingest
- kiểm tra trạng thái track
- xoá document

#### `client`

HTTP client đơn giản để gọi:

- RAG service
- Whisper service

#### `input`

Luồng microphone/streaming audio cho demo voice workflow.

#### `docs`

Tài liệu kỹ thuật, quickstart, analysis, và mô tả nghiệp vụ.

#### `scripts`

Script hỗ trợ setup Postgres/Qdrant và khởi tạo hạ tầng.

---

## 3. Cấu trúc thư mục quan trọng

```text
Noble_RAG/
├── service/
│   ├── RAG/
│   │   ├── api/                # FastAPI routes
│   │   ├── core/               # config, logging, dependencies
│   │   ├── memory/             # Redis/Postgres/local snapshot persistence
│   │   ├── models/             # Pydantic models, state enums
│   │   ├── rag/                # query routing và LightRAG retrieval
│   │   ├── sales/              # LangGraph sales agent
│   │   │   ├── nodes/          # từng bước trong graph
│   │   │   ├── graph.py        # build và compile graph
│   │   │   ├── state_machine.py
│   │   │   └── prompt_builder.py
│   │   ├── storage/            # utility làm việc với storage
│   │   └── app.py              # entrypoint FastAPI
│   └── whisper-service/
│       └── app.py              # Whisper FastAPI app
├── ui/
│   └── rag_web_ui.py           # Streamlit UI
├── client/
│   ├── rag_client.py
│   └── whisper_client.py
├── input/
│   └── streaming_voice.py
├── scripts/
├── infra/
├── docs/
├── docker-compose.yml
└── main.py                     # web mode / voice mode launcher
```

---

## 4. Luồng xử lý chính

### 4.1 Luồng tài liệu RAG

1. Client gọi `POST /upload-document`
2. File text được đọc vào backend
3. Backend gọi `rag.ainsert(...)`
4. LightRAG xử lý chunking, indexing, cập nhật document status
5. Khi query, hệ thống dùng `rag.aquery(...)` để truy xuất ngữ cảnh

### 4.2 Luồng sales agent

Sales agent được build bằng LangGraph tại [graph.py](/home/meoconlonton/Noble_RAG/service/RAG/sales/graph.py).

Pipeline hiện tại:

1. `ingest_user_turn`
2. `classify_and_extract`
3. `update_lead_profile`
4. `resolve_sales_state`
5. `retrieve_context` nếu state cần grounding
6. `build_response`
7. `persist_turn`
8. `finalize_output`

Ý nghĩa:

- `ingest_user_turn`: nạp history, session context, lead profile cũ
- `classify_and_extract`: phân loại intent và trích xuất slot chỉ trong 1 lần gọi LLM
- `update_lead_profile`: merge slot mới vào hồ sơ khách
- `resolve_sales_state`: quyết định state tiếp theo bằng luật code
- `retrieve_context`: gọi RAG để lấy ngữ cảnh dự án hoặc playbook
- `build_response`: sinh câu trả lời theo prompt + guardrail
- `persist_turn`: lưu lịch sử, profile, state, snapshot local
- `finalize_output`: chuẩn hoá output cuối trả về API

### 4.3 Luồng voice demo

1. `main.py` chạy ở `APP_MODE=voice`
2. Microphone ghi âm từng segment
3. Gửi sang Whisper service để transcript
4. Gửi transcript sang RAG service
5. In phản hồi của sales agent ra terminal

---

## 5. Cơ chế lưu dữ liệu hiện tại

### 5.1 Lead profile

- cache Redis: key `lead_profile_cache:{session_id}`, TTL 7 ngày
- source of truth nếu backend DB chạy: Postgres `sales.lead_profiles`
- fallback demo bền: file local `.txt`

Code liên quan:

- [lead_profile_store.py](/home/meoconlonton/Noble_RAG/service/RAG/memory/lead_profile_store.py)

### 5.2 Session context

- Redis TTL 24 giờ
- fallback local snapshot

Code:

- [session_store.py](/home/meoconlonton/Noble_RAG/service/RAG/memory/session_store.py)

### 5.3 Chat history

- Redis TTL 24 giờ
- fallback local snapshot

Code:

- [chat_history_store.py](/home/meoconlonton/Noble_RAG/service/RAG/memory/chat_history_store.py)

### 5.4 Local snapshot cho MVP

Mỗi session hiện được mirror ra:

- `./exports/sessions/live_snapshots/<session_id>.txt`

Nội dung snapshot gồm:

- `lead_profile`
- `session_context`
- `chat_history`
- `updated_at`

Code:

- [local_snapshot_store.py](/home/meoconlonton/Noble_RAG/service/RAG/memory/local_snapshot_store.py)

---

## 6. Chức năng theo module

### 6.1 `service/RAG/api`

- định nghĩa toàn bộ REST API của hệ thống

### 6.2 `service/RAG/core`

- load config từ `.env`
- bootstrap LightRAG, embedding, LLM, logging

### 6.3 `service/RAG/rag`

- route query giữa RAG / SEARCH / OTHER
- gọi trực tiếp `rag.aquery(...)`
- phục vụ retrieval cho sales agent

### 6.4 `service/RAG/sales`

- state machine hội thoại sales
- prompt builder
- LangGraph orchestration
- node-level logic cho classify, retrieve, respond, persist

### 6.5 `service/RAG/memory`

- quản lý Redis
- lưu lead vào Postgres
- snapshot local `.txt`

### 6.6 `service/whisper-service`

- speech-to-text HTTP API

### 6.7 `ui`

- web UI upload document và kiểm tra tài liệu

---

## 7. Danh sách endpoint của RAG service

Base app: `service/RAG/app.py`

### 7.1 Nhóm Health

#### `GET /health`

Mục đích:

- kiểm tra service còn sống
- trả thông tin model hiện dùng

Khi dùng:

- healthcheck container
- monitoring
- smoke test khi deploy

#### `GET /models`

Mục đích:

- xem cấu hình LLM, embedding, storage backend

Khi dùng:

- debug môi trường
- xác nhận model đang chạy đúng

#### `GET /status`

Mục đích:

- xem trạng thái service và loại storage hệ thống đang dùng

Khi dùng:

- kiểm tra nhanh trạng thái runtime

---

### 7.2 Nhóm Documents

#### `POST /upload-document`

Mục đích:

- upload file text vào LightRAG để ingest

Input:

- `file`
- `storage_type`
- `metadata` JSON string

Khi dùng:

- nạp tài liệu dự án, policy, FAQ, playbook vào kho RAG

#### `GET /documents`

Mục đích:

- liệt kê documents đã được ingest trong workspace hiện tại

Khi dùng:

- quản trị kho tài liệu
- kiểm tra tài liệu đã vào hệ thống hay chưa

#### `GET /documents/track/{track_id}`

Mục đích:

- kiểm tra tiến độ ingest theo `track_id`

Khi dùng:

- polling trạng thái xử lý tài liệu lớn

#### `DELETE /documents/{doc_id}`

Mục đích:

- trigger xoá một document khỏi hệ thống RAG

Khi dùng:

- dọn tài liệu cũ
- xoá tài liệu ingest nhầm

---

### 7.3 Nhóm Query

#### `POST /query/stream`

Mục đích:

- nhận query/message và trả lời theo dạng stream `ndjson`
- hiện đang đi qua `sales_graph`

Khi dùng:

- chat UI cần stream từng câu
- demo hội thoại realtime

Output cuối stream có thêm:

- `session_id`
- `sales_state`
- `missing_slots`
- `lead_profile`

---

### 7.4 Nhóm Sales

Base prefix: `/sales`

#### `POST /sales/chat`

Mục đích:

- endpoint chính cho sales agent
- nhận 1 lượt chat và trả lại phản hồi đầy đủ

Input:

- `session_id`
- `message`
- `raw_transcript` optional

Khi dùng:

- web app hoặc CRM tích hợp hội thoại sales

#### `GET /sales/lead/{session_id}`

Mục đích:

- lấy lead profile đã lưu của một session

Khi dùng:

- kiểm tra thông tin khách đã thu thập được
- hiển thị hồ sơ khách trên UI/CRM

#### `PATCH /sales/lead/{session_id}`

Mục đích:

- chỉnh tay lead profile

Khi dùng:

- sales hoặc admin cập nhật thông tin khách thủ công

#### `GET /sales/state/{session_id}`

Mục đích:

- xem state hiện tại của hội thoại

Khi dùng:

- debug flow sales
- đồng bộ UI theo stage hiện tại

#### `GET /sales/session/{session_id}/snapshot`

Mục đích:

- đọc trực tiếp snapshot `.txt` local của session

Trả về:

- path file snapshot
- lead profile
- session context
- chat history
- raw text của file

Khi dùng:

- demo MVP
- audit nhanh dữ liệu khách mà không cần Redis/Postgres sống

#### `POST /sales/recommendations/refresh`

Mục đích:

- ép agent chạy lại vòng `product_matching`

Khi dùng:

- khách muốn xem lại shortlist
- CRM cần refresh đề xuất

#### `POST /sales/followup/generate`

Mục đích:

- sinh follow-up message cho lead

Khi dùng:

- hỗ trợ sales gửi tin nhắn chăm lại khách

#### `POST /sales/session/{session_id}/close`

Mục đích:

- export session ra `.txt`
- clear cache tạm trong Redis

Khi dùng:

- kết thúc một phiên chăm sóc khách
- xuất dữ liệu lưu trữ hoặc bàn giao

Lưu ý:

- snapshot local live hiện vẫn được giữ lại để không mất dữ liệu demo

---

## 8. Endpoint của Whisper service

Base app: `service/whisper-service/app.py`

#### `GET /health`

- health check cho Whisper service

#### `GET /models`

- xem model whisper đang load và các model hỗ trợ

#### `POST /transcribe`

- upload audio và nhận transcript đầy đủ dạng JSON

#### `POST /transcribe/stream`

- upload audio và nhận transcript dạng SSE theo từng segment

---

## 9. Cách dùng phù hợp theo ngữ cảnh

### Nếu cần ingest tài liệu

- dùng `POST /upload-document`
- sau đó kiểm tra bằng `GET /documents`

### Nếu cần chat sales chuẩn API

- dùng `POST /sales/chat`

### Nếu cần stream phản hồi

- dùng `POST /query/stream`

### Nếu cần xem hồ sơ khách

- dùng `GET /sales/lead/{session_id}`
- hoặc `GET /sales/session/{session_id}/snapshot`

### Nếu cần xem raw session đã lưu local

- dùng `GET /sales/session/{session_id}/snapshot`

### Nếu cần voice demo

- chạy `main.py` ở `APP_MODE=voice`
- Whisper service phải chạy trước

---

## 10. Trạng thái thực tế của dự án hiện tại

Theo code hiện tại:

- sales flow đã có state machine tương đối rõ
- đã có guardrail để tránh pitch sớm khi chưa đủ slot
- đã có fallback lưu local `.txt` cho MVP
- documents API đã có đủ để ingest và quản trị tài liệu
- query stream hiện dùng chung sales graph

Các điểm cần lưu ý:

- nếu Redis/Postgres/Qdrant không chạy, nhiều chức năng RAG đầy đủ sẽ không hoạt động như production
- local snapshot hiện phù hợp cho demo/MVP hơn là production
- retrieval runtime vẫn phụ thuộc LightRAG backend, không đọc trực tiếp file nguồn trong repo

---

## 11. File tham chiếu quan trọng

- [app.py](/home/meoconlonton/Noble_RAG/service/RAG/app.py)
- [graph.py](/home/meoconlonton/Noble_RAG/service/RAG/sales/graph.py)
- [state_machine.py](/home/meoconlonton/Noble_RAG/service/RAG/sales/state_machine.py)
- [prompt_builder.py](/home/meoconlonton/Noble_RAG/service/RAG/sales/prompt_builder.py)
- [retriever.py](/home/meoconlonton/Noble_RAG/service/RAG/rag/retriever.py)
- [routes_documents.py](/home/meoconlonton/Noble_RAG/service/RAG/api/routes_documents.py)
- [routes_query.py](/home/meoconlonton/Noble_RAG/service/RAG/api/routes_query.py)
- [routes_sales.py](/home/meoconlonton/Noble_RAG/service/RAG/api/routes_sales.py)
- [local_snapshot_store.py](/home/meoconlonton/Noble_RAG/service/RAG/memory/local_snapshot_store.py)
