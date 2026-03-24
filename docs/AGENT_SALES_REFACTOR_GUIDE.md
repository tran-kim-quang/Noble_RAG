# AI Sales Agent Refactor Guide

## Mục tiêu

Tài liệu này mô tả hướng refactor branch `quang-dev` để hệ thống chuyển từ một RAG assistant chuyên domain bất động sản thành một **AI sales agent bất động sản có kịch bản**.

Mục tiêu sau refactor:
- Hiểu nhu cầu khách hàng theo quy trình sales.
- Thu thập và lưu hồ sơ lead có cấu trúc.
- Đề xuất sản phẩm phù hợp theo budget, khu vực, loại hình và mục đích mua.
- Xử lý objection theo playbook.
- Chốt bước tiếp theo như gửi shortlist, bảng giá, hẹn call hoặc hẹn đi xem dự án.

---

## Hiện trạng branch `quang-dev`

Branch hiện tại đã có các thành phần nền tảng tốt:
- `whisper-service` cho speech-to-text.
- `rag-service` dùng LightRAG.
- `redis` cho chat history.
- `postgres` cho storage quan hệ.
- `qdrant` cho vector retrieval.
- `ollama` cho embedding runtime local.
- `rag-ui` cho giao diện chat cơ bản.

Hệ thống hiện tại phù hợp với bài toán **domain RAG assistant**, nhưng chưa đủ để trở thành **sales agent** vì còn thiếu:
- sales state machine
- lead profile memory có cấu trúc
- product schema cho bất động sản
- objection playbooks
- closing playbooks
- prompt builder theo state
- sales orchestration layer

---

## Nguyên tắc kiến trúc sau refactor

1. Không nhồi toàn bộ logic sales vào một system prompt dài.
2. Giữ LightRAG làm retrieval engine, không biến nó thành nơi lưu lead profile chính.
3. Dùng Redis cho working memory ngắn hạn.
4. Dùng Postgres làm nguồn sự thật chính cho lead, session và sales events.
5. Dùng Qdrant/LightRAG cho knowledge retrieval: brochure, FAQ, chính sách, pháp lý, playbook.
6. Tách rõ bốn lớp:
   - Retrieval layer
   - Sales orchestration layer
   - Memory layer
   - Response generation layer

---

## Folder structure đề xuất

```bash
Noble_RAG/
├── service/
│   ├── RAG/
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   ├── api/
│   │   │   ├── routes_query.py
│   │   │   ├── routes_documents.py
│   │   │   ├── routes_health.py
│   │   │   └── routes_sales.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── dependencies.py
│   │   ├── rag/
│   │   │   ├── lightrag_client.py
│   │   │   ├── retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── reranker.py
│   │   │   └── document_ingest.py
│   │   ├── sales/
│   │   │   ├── orchestrator.py
│   │   │   ├── state_machine.py
│   │   │   ├── policy.py
│   │   │   ├── intent_classifier.py
│   │   │   ├── lead_extractor.py
│   │   │   ├── objection_handler.py
│   │   │   ├── product_matcher.py
│   │   │   ├── closing_strategy.py
│   │   │   └── prompt_builder.py
│   │   ├── memory/
│   │   │   ├── chat_history_store.py
│   │   │   ├── lead_profile_store.py
│   │   │   ├── session_store.py
│   │   │   └── redis_store.py
│   │   ├── models/
│   │   │   ├── api_models.py
│   │   │   ├── lead_schema.py
│   │   │   ├── sales_state.py
│   │   │   ├── product_schema.py
│   │   │   └── response_schema.py
│   │   ├── tools/
│   │   │   ├── tavily_tool.py
│   │   │   ├── weather_tool.py
│   │   │   ├── calculator_tool.py
│   │   │   └── schedule_tool.py
│   │   └── utils/
│   │       ├── text.py
│   │       ├── time.py
│   │       ├── json_extract.py
│   │       └── validators.py
│   └── whisper-service/
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
├── data/
│   ├── raw/
│   │   ├── brochures/
│   │   ├── policies/
│   │   ├── legal_docs/
│   │   └── faqs/
│   ├── curated/
│   │   ├── projects/
│   │   ├── personas/
│   │   ├── objection_playbooks/
│   │   ├── closing_playbooks/
│   │   └── sales_prompts/
│   └── eval/
│       ├── conversations/
│       ├── golden_cases/
│       └── scoring_rubric.yaml
├── scripts/
│   ├── ingest_projects.py
│   ├── ingest_documents.py
│   ├── backfill_embeddings.py
│   ├── eval_sales_agent.py
│   └── seed_demo_data.py
├── ui/
│   ├── rag_web_ui.py
│   ├── sales_dashboard.py
│   └── components/
├── tests/
│   ├── test_state_machine.py
│   ├── test_lead_extractor.py
│   ├── test_product_matcher.py
│   ├── test_objection_handler.py
│   ├── test_prompt_builder.py
│   └── test_routes_sales.py
└── docs/
    └── AGENT_SALES_REFACTOR_GUIDE.md
```

---

## Sales state machine

### Danh sách state

- `greeting`
- `qualification`
- `need_discovery`
- `budget_alignment`
- `product_matching`
- `comparison`
- `objection_handling`
- `buy_signal`
- `closing_next_step`
- `follow_up`
- `out_of_scope`

### Mục tiêu từng state

#### 1. `greeting`
Mục tiêu:
- Mở đầu tự nhiên.
- Không pitch sản phẩm ngay.
- Mời khách chia sẻ nhu cầu.

#### 2. `qualification`
Mục tiêu:
- Xác định user mua để ở, đầu tư cho thuê, đầu tư tăng giá, giữ tài sản hay chỉ tham khảo.

#### 3. `need_discovery`
Mục tiêu:
- Thu các slot còn thiếu.

Slot quan trọng:
- `purpose`
- `budget_range`
- `location_preference`
- `property_type`
- `timeline`

#### 4. `budget_alignment`
Mục tiêu:
- Cân chỉnh kỳ vọng sản phẩm với mức ngân sách thực tế.

#### 5. `product_matching`
Mục tiêu:
- Đề xuất 1 đến 3 lựa chọn phù hợp.
- Luôn nêu vì sao hợp.
- Không ném danh sách dài.

#### 6. `comparison`
Mục tiêu:
- So sánh 2 hoặc 3 lựa chọn theo tiêu chí cụ thể: giá, vị trí, pháp lý, tiến độ, thanh toán, tiềm năng cho thuê.

#### 7. `objection_handling`
Mục tiêu:
- Xử lý phản đối như giá cao, pháp lý, vị trí, chưa đủ tiền, muốn suy nghĩ thêm.

#### 8. `buy_signal`
Mục tiêu:
- Nhận diện tín hiệu mua như hỏi bảng giá, hỏi lịch đi xem, hỏi còn căn hay không.

#### 9. `closing_next_step`
Mục tiêu:
- Chốt một bước tiếp theo nhỏ như gửi shortlist, gửi bảng giá, hẹn call, hẹn đi xem.

#### 10. `follow_up`
Mục tiêu:
- Nhắc lại mối quan tâm chính và kéo lead quay lại funnel.

#### 11. `out_of_scope`
Mục tiêu:
- Từ chối lịch sự và đưa hội thoại quay lại bất động sản.

---

## Luật chuyển state

Các rule cốt lõi:
- Không được vào `product_matching` nếu chưa có đủ dữ liệu tối thiểu.
- Dữ liệu tối thiểu trước khi pitch gồm:
  - mục đích mua
  - ngân sách
  - khu vực quan tâm hoặc mức linh hoạt khu vực
  - loại hình sản phẩm
- Khi phát hiện objection thì chuyển sang `objection_handling`.
- Khi phát hiện buy signal thì chuyển sang `closing_next_step`.
- Khi user yêu cầu so sánh thì chuyển sang `comparison`.

Pseudo logic:

```python
if out_of_scope:
    return "out_of_scope"
if missing_required_slots:
    return "need_discovery"
if missing_budget:
    return "budget_alignment"
if detected_intent == "comparison":
    return "comparison"
if detected_intent == "objection":
    return "objection_handling"
if detected_intent == "buy_signal":
    return "closing_next_step"
return "product_matching"
```

---

## Lead schema

### `LeadProfile`

Đề xuất schema:

```python
class LeadProfile(BaseModel):
    lead_id: str
    name: Optional[str] = None
    purpose: Optional[str] = "khong_ro"
    property_type: Optional[str] = "khong_ro"
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    budget_text: Optional[str] = None
    location_preference: List[str] = []
    region_flexibility: Optional[bool] = None
    financing_need: Optional[bool] = None
    financing_ratio: Optional[float] = None
    timeline: Optional[str] = "khong_ro"
    legal_sensitivity: Optional[str] = None
    risk_appetite: Optional[str] = None
    key_needs: List[str] = []
    key_concerns: List[str] = []
    objections: List[str] = []
    recommended_projects: List[str] = []
    shortlisted_projects: List[str] = []
    rejected_projects: List[str] = []
    current_state: str
    lead_temperature: Optional[str] = "cold"
    preferred_contact_channel: Optional[str] = "chat"
    contact_phone: Optional[str] = None
    last_user_intent: Optional[str] = None
    last_next_action: Optional[str] = None
```

### `SessionContext`

```python
class SessionContext(BaseModel):
    session_id: str
    current_state: str
    previous_state: Optional[str] = None
    last_agent_action: Optional[str] = None
    last_retrieved_context_ids: List[str] = []
    conversation_turn_count: int = 0
```

---

## Lưu dữ liệu ở đâu

### Redis
Dùng cho dữ liệu nóng, phục vụ session đang chạy:
- `chat_history:{session_id}`
- `session_context:{session_id}`
- `lead_profile_cache:{session_id}`
- `last_recommendations:{session_id}`

### Postgres
Dùng làm nguồn sự thật chính cho dữ liệu sales:
- `lead_profiles`
- `conversation_sessions`
- `conversation_turns`
- `sales_events`
- `appointment_requests`

### Qdrant / LightRAG
Dùng cho retrieval knowledge:
- brochure
- FAQ
- pháp lý
- chính sách bán hàng
- objection playbooks
- closing playbooks
- comparison knowledge

Không dùng Qdrant làm nơi chính để lưu lead profile.

---

## Product schema

Nên có dữ liệu dự án dạng có cấu trúc song song với tài liệu text.

Ví dụ:

```python
class ProductProject(BaseModel):
    project_id: str
    project_name: str
    city: str
    district: str
    ward: Optional[str] = None
    property_types: List[str]
    target_personas: List[str]
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    area_min: Optional[float] = None
    area_max: Optional[float] = None
    legal_status: Optional[str] = None
    handover_time: Optional[str] = None
    payment_policy: Optional[str] = None
    bank_support: Optional[str] = None
    strengths: List[str] = []
    weaknesses: List[str] = []
    fit_rules: List[str] = []
    disqualify_rules: List[str] = []
    objection_answers: dict = {}
    cta_recommendation: Optional[str] = None
```

---

## Prompt sales

### System prompt chính

```text
Bạn là AI Sales Agent bất động sản chuyên tư vấn các sản phẩm/dự án trong hệ thống Noble.

MỤC TIÊU CHÍNH
- Hiểu đúng nhu cầu thật của khách hàng.
- Đề xuất sản phẩm phù hợp nhất dựa trên hồ sơ khách và dữ liệu retrieve được.
- Xử lý băn khoăn một cách trung thực, tinh tế, không ép mua.
- Dẫn hội thoại tới bước tiếp theo rõ ràng như gửi shortlist, gửi bảng giá, hẹn call, hẹn xem dự án.

VAI TRÒ
- Bạn không phải chatbot FAQ thuần túy.
- Bạn là chuyên viên sales tư vấn theo quy trình.
- Bạn cần chủ động dẫn dắt nhưng vẫn tự nhiên và tôn trọng khách.

NGUYÊN TẮC BẮT BUỘC
1. Không bịa thông tin về giá, pháp lý, ưu đãi, tiến độ, tồn kho.
2. Chỉ sử dụng thông tin có trong context, knowledge base hoặc tool results.
3. Nếu thiếu dữ liệu, hãy nói rõ chưa đủ thông tin và hỏi thêm hoặc đề xuất bước tiếp theo phù hợp.
4. Không cam kết lợi nhuận, không hứa chắc tăng giá, không dùng lời lẽ thao túng.
5. Không pitch sản phẩm cụ thể khi chưa hiểu tối thiểu:
   - mục đích mua
   - ngân sách
   - khu vực quan tâm hoặc mức linh hoạt khu vực
   - loại hình sản phẩm
6. Sau mỗi lượt trả lời, cố gắng tạo ra một bước tiến nhỏ trong sales funnel.

PHONG CÁCH
- Xưng "em", gọi khách là "Anh/Chị".
- Tự nhiên, gọn, rõ.
- Không chào hỏi máy móc ở mọi lượt.
- Không trả lời như robot.
- Không liệt kê quá nhiều lựa chọn cùng lúc.
- Ưu tiên 1 đến 3 phương án tốt nhất.

CÁCH TƯ VẤN
- Nếu đang ở giai đoạn khám phá nhu cầu: hỏi 1 đến 2 câu ngắn, trọng tâm.
- Nếu đang ở giai đoạn match sản phẩm: nêu lựa chọn + vì sao phù hợp + điểm cần lưu ý + gợi ý bước tiếp theo.
- Nếu đang ở giai đoạn xử lý phản đối: đồng cảm, làm rõ, giải thích có cơ sở, rồi mở ra lựa chọn tiếp theo.
- Nếu có buy signal: ưu tiên chốt bước tiếp theo.

KHI THIẾU THÔNG TIN
- Không đoán bừa.
- Hỏi lại thông minh, ngắn gọn.
- Nếu khách quá mơ hồ, hãy giúp khách thu hẹp tiêu chí.

KHI NGOÀI PHẠM VI
- Từ chối lịch sự và đưa hội thoại quay lại bất động sản/dự án.

ĐỊNH DẠNG TRẢ LỜI
- Ngắn gọn, thường 2 đến 5 câu.
- Nếu đề xuất sản phẩm: luôn nói "vì sao hợp".
- Nếu phù hợp, kết thúc bằng một CTA nhẹ.
```

### Prompt theo state

#### NEED_DISCOVERY
```text
TRẠNG THÁI HIỆN TẠI: NEED_DISCOVERY
NHIỆM VỤ
- Thu thêm thông tin còn thiếu để có thể tư vấn chính xác.
- Chỉ hỏi tối đa 2 câu ngắn.
- Không đề xuất dự án cụ thể trừ khi đã có đủ dữ liệu cơ bản.
```

#### PRODUCT_MATCHING
```text
TRẠNG THÁI HIỆN TẠI: PRODUCT_MATCHING
NHIỆM VỤ
- Đề xuất tối đa 3 lựa chọn phù hợp.
- Mỗi lựa chọn phải có:
  1. tên dự án/sản phẩm
  2. vì sao phù hợp với khách
  3. 1 điểm cần lưu ý
```

#### OBJECTION_HANDLING
```text
TRẠNG THÁI HIỆN TẠI: OBJECTION_HANDLING
NHIỆM VỤ
- Xác định phản đối chính.
- Đồng cảm trước, giải thích sau.
- Không tranh luận tay đôi.
- Không ép mua.
- Sau khi xử lý, mở ra 1 bước tiếp theo hợp lý.
```

#### CLOSING_NEXT_STEP
```text
TRẠNG THÁI HIỆN TẠI: CLOSING_NEXT_STEP
NHIỆM VỤ
- Chốt một bước tiếp theo nhỏ, rõ ràng.
- Ưu tiên CTA phù hợp với mức nóng của lead.
- Không tạo cảm giác ép.
```

---

## Orchestrator flow

Sales agent cần một orchestration layer riêng, không dồn hết logic vào `app.py`.

Pipeline một lượt hội thoại:
1. Nhận query hoặc transcript.
2. Load chat history.
3. Load lead profile.
4. Detect intent và signal.
5. Extract slot update.
6. Update lead profile.
7. Resolve next state.
8. Retrieve đúng loại context theo state.
9. Build prompt theo state.
10. Generate structured output.
11. Persist history, lead profile và event.
12. Trả response cho UI hoặc TTS/avatar.

Pseudo-code:

```python
async def handle_sales_turn(session_id: str, user_text: str):
    history = await history_store.get(session_id)
    lead = await lead_store.get(session_id)
    signals = await detect_turn_signals(user_text, history, lead)
    lead = await extract_and_merge_lead_profile(lead, user_text, history)
    next_state = resolve_state(lead.current_state, lead, signals)
    retrieved_context = await retrieve_context_for_state(next_state, lead, user_text)
    prompt = build_sales_prompt(next_state, lead, history, retrieved_context, user_text)
    result = await llm_generate_structured(prompt)
    await persist_all(session_id, lead, history, result)
    return result.response_text
```

---

## Retrieval strategy theo state

### `need_discovery`
Retrieve nhẹ:
- discovery question bank
- buyer persona prompts

### `product_matching`
Retrieve:
- project schema
- product catalog
- pricing band
- project docs liên quan

### `comparison`
Retrieve:
- 2 hoặc 3 project schemas
- comparison snippets
- khác biệt về giá, pháp lý, vị trí, thanh toán

### `objection_handling`
Retrieve:
- objection playbook
- FAQ liên quan
- policy hoặc legal docs nếu cần

### `closing_next_step`
Retrieve:
- CTA playbook
- shortlist templates
- bảng giá hoặc brochure phù hợp

---

## API endpoints nên thêm

Tạo `routes_sales.py` với các endpoint:
- `POST /sales/chat`
- `POST /sales/chat/stream`
- `GET /sales/lead/{session_id}`
- `PATCH /sales/lead/{session_id}`
- `GET /sales/state/{session_id}`
- `POST /sales/recommendations/refresh`
- `POST /sales/followup/generate`

---

## Tách trách nhiệm khỏi `service/RAG/app.py`

Hiện `app.py` đang ôm quá nhiều việc. Sau refactor:

- `app.py`: chỉ init FastAPI, dependency injection, route registration.
- `sales/orchestrator.py`: điều phối luồng sales.
- `sales/state_machine.py`: resolve state.
- `sales/prompt_builder.py`: dựng prompt theo state.
- `memory/chat_history_store.py`: đọc ghi history.
- `memory/lead_profile_store.py`: đọc ghi lead profile.
- `rag/retriever.py`: retrieval logic.
- `tools/tavily_tool.py`: xử lý search ngoài.

---

## Dữ liệu curated nên có

### 1. Project data
Mỗi dự án nên có file JSON riêng trong `data/curated/projects/`.

### 2. Objection playbooks
Ví dụ:
- `gia_cao.yaml`
- `phap_ly.yaml`
- `vi_tri.yaml`
- `chua_du_tien.yaml`
- `suy_nghi_them.yaml`

### 3. Closing playbooks
Ví dụ:
- `book_site_visit.yaml`
- `book_call.yaml`
- `send_shortlist.yaml`

### 4. Sales prompts
Tách riêng prompt hệ thống, prompt theo state, refusal rules.

---

## Roadmap triển khai

### Phase 1
- Tách `app.py`.
- Thêm `SalesState`.
- Thêm `LeadProfile`.
- Thêm `sales/orchestrator.py`.
- Thêm `POST /sales/chat`.

### Phase 2
- Thêm `project_schema.py`.
- Thêm `product_matcher.py`.
- Thêm `lead_profile_store.py`.
- Thêm persist Postgres cho lead profile và sales events.

### Phase 3
- Thêm objection playbooks.
- Thêm closing playbooks.
- Thêm prompt builder theo state.
- Thêm evaluation test cases.

### Phase 4
- UI hiển thị state, lead profile và retrieved products.
- Kết nối TTS/avatar.
- Theo dõi conversion funnel.

---

## Definition of Done

Một bản refactor được coi là hoàn tất khi:
- Agent biết hỏi đủ nhu cầu trước khi pitch.
- Agent lưu lead profile có cấu trúc.
- Agent đề xuất sản phẩm bằng logic phù hợp, không trả lời kiểu brochure bot.
- Agent xử lý objection bằng playbook.
- Agent biết chốt bước tiếp theo.
- Retrieval được dùng theo state, không dùng một kiểu chung cho mọi case.
- Prompt được tách thành system prompt + state prompt + retrieved context.

---

## Kết luận

Refactor đúng cho branch `quang-dev` không phải là kéo dài prompt hiện tại, mà là thêm một **sales brain layer** phía trên LightRAG.

Công thức đúng là:

**Whisper + LightRAG + Redis history + Lead Profile + Sales State Machine + Prompt Builder + Playbooks + Postgres persistence**

Khi hoàn thành, hệ thống sẽ chuyển từ một **RAG assistant bất động sản** sang một **AI sales agent bất động sản có kịch bản và khả năng dẫn dắt hội thoại thực sự**.
